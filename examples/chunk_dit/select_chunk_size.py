"""Offline two-stage chunkDiT tuning, following section 3.3 of the chunkDiT paper."""

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import statistics
import time
from pathlib import Path

import torch
from huggingface_hub import model_info

import diffusers
from diffusers import CogVideoXTransformer3DModel, HunyuanVideoTransformer3DModel
from diffusers.models.chunk_dit_utils import ChunkDiTWorkspace
from diffusers.models.embeddings import get_3d_rotary_pos_embed


GIB = 2**30


def workload_shape(kind, config, height, width, frames, batch, text_tokens):
    patch = config["patch_size"]
    temporal_patch = config.get("patch_size_t") or 1
    if min(height, width, frames, batch, text_tokens) <= 0 or height % (8 * patch) or width % (8 * patch):
        raise ValueError("Positive dimensions are required; height/width must align to VAE scale 8 times patch size")
    latent_frames = (frames - 1) // 4 + 1
    if kind == "cogvideo":
        latent_frames = math.ceil(latent_frames / temporal_patch) * temporal_patch
    elif latent_frames % temporal_patch:
        raise ValueError("HunyuanVideo latent frames must align to the temporal patch size")
    video_tokens = (latent_frames // temporal_patch) * (height // (8 * patch)) * (width // (8 * patch))
    dim = config["num_attention_heads"] * config["attention_head_dim"]
    return {
        "height": height,
        "width": width,
        "frames": frames,
        "batch": batch,
        "text_tokens": text_tokens,
        "latent_frames": latent_frames,
        "video_tokens": video_tokens,
        "total_tokens": video_tokens + text_tokens,
        "hidden_dim": dim,
        "bytes_per_element": 2,
    }


def aligned_candidates(minimum, maximum, alignment, sequence_length):
    if min(minimum, maximum, alignment, sequence_length) <= 0 or minimum > maximum:
        raise ValueError("Chunk bounds and alignment must be positive, with min <= max")
    lower = math.ceil(minimum / alignment) * alignment
    upper = min(maximum, math.ceil(sequence_length / alignment) * alignment)
    return list(range(lower, upper + 1, alignment))


def fit_memory_model(points, weights, kv, token_bytes):
    if not points:
        raise ValueError("No successful calibration samples")
    xs = [p["chunk_size"] for p in points]
    ys = [p["peak_allocated"] - weights - kv for p in points]
    mean_x, mean_y = statistics.mean(xs), statistics.mean(ys)
    variance = sum((x - mean_x) ** 2 for x in xs)
    slope = max(0.0, sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / variance) if variance else 0.0
    fixed = max(0.0, max(y - slope * x for x, y in zip(xs, ys)))
    return {
        "fixed_bytes": fixed,
        "alpha": slope / token_bytes,
        "bytes_per_token": slope,
        "weights_bytes": weights,
        "kv_bytes": kv,
        "sample_count": len(points),
    }


def predict_peak(model, chunk):
    return model["weights_bytes"] + model["kv_bytes"] + model["fixed_bytes"] + model["bytes_per_token"] * chunk


def maximum_chunk(model, budget, upper):
    remaining = budget - predict_peak(model, 0)
    if remaining < 0:
        return 0
    slope = model["bytes_per_token"]
    return min(upper, math.floor(math.nextafter(remaining / slope, math.inf))) if slope else upper


def rank_candidates(candidates, budget):
    return sorted(
        (row for row in candidates if row["status"] == "ok" and row["peak_allocated"] <= budget),
        key=lambda row: (row["estimated_dit_seconds"], row["peak_allocated"], row["chunk_size"]),
    )


@torch.inference_mode()
def measure(call, warmups, repeats):
    records = []
    try:
        for index in range(warmups + repeats):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            output = call()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            records.append(
                {
                    "warmup": index < warmups,
                    "seconds": elapsed,
                    "peak_allocated": torch.cuda.max_memory_allocated(),
                    "peak_reserved": torch.cuda.max_memory_reserved(),
                }
            )
            del output
    except torch.cuda.OutOfMemoryError:
        return {"status": "oom", "error": "CUDA out of memory"}
    return {
        "status": "ok",
        "median_seconds": statistics.median(r["seconds"] for r in records if not r["warmup"]),
        "peak_allocated": max(r["peak_allocated"] for r in records),
        "peak_reserved": max(r["peak_reserved"] for r in records),
        "measurements": records,
    }


@torch.inference_mode()
def make_inputs(model, kind, shape):
    batch, length, text, dim = (shape[k] for k in ("batch", "video_tokens", "text_tokens", "hidden_dim"))
    dtype, device = torch.bfloat16, torch.device("cuda")
    latent_shape = (batch, model.config.in_channels, shape["latent_frames"], shape["height"] // 8, shape["width"] // 8)
    if kind == "cogvideo":
        latent_shape = (
            batch,
            shape["latent_frames"],
            model.config.in_channels,
            shape["height"] // 8,
            shape["width"] // 8,
        )
    forward = {
        "hidden_states": torch.randn(latent_shape, device=device, dtype=dtype),
        "encoder_hidden_states": torch.randn(batch, text, model.config.text_embed_dim, device=device, dtype=dtype),
        "timestep": torch.full((batch,), 500.0, device=device, dtype=dtype),
        "return_dict": False,
    }
    block_args = {
        "hidden_states": torch.randn(batch, length, dim, device=device, dtype=dtype),
        "encoder_hidden_states": torch.randn(batch, text, dim, device=device, dtype=dtype),
    }
    if kind == "hunyuan":
        forward["encoder_attention_mask"] = torch.ones(batch, text, device=device, dtype=torch.bool)
        forward["pooled_projections"] = torch.randn(
            batch, model.config.pooled_projection_dim, device=device, dtype=dtype
        )
        forward["guidance"] = torch.full((batch,), 6000.0, device=device, dtype=dtype)
        temb, _ = model.time_text_embed(forward["timestep"], forward["pooled_projections"], forward["guidance"])
        rotary = model.rope(forward["hidden_states"])
        mask = torch.ones(batch, 1, 1, length + text, device=device, dtype=torch.bool)
        specs = []
        for name, blocks, rope_name in [
            ("double", model.transformer_blocks, "freqs_cis"),
            ("single", model.single_transformer_blocks, "image_rotary_emb"),
        ]:
            if len(blocks):
                specs.append(
                    (
                        name,
                        blocks[min(1, len(blocks) - 1)],
                        len(blocks),
                        {**block_args, "temb": temb, "attention_mask": mask, rope_name: rotary},
                    )
                )
    else:
        if model.config.patch_size_t != 2:
            raise ValueError("This tuner supports CogVideoX 1.5 with patch_size_t=2")
        rotary = get_3d_rotary_pos_embed(
            embed_dim=model.config.attention_head_dim,
            crops_coords=None,
            grid_size=(
                shape["height"] // (8 * model.config.patch_size),
                shape["width"] // (8 * model.config.patch_size),
            ),
            temporal_size=shape["latent_frames"] // 2,
            grid_type="slice",
            max_size=(
                model.config.sample_height // model.config.patch_size,
                model.config.sample_width // model.config.patch_size,
            ),
            device=device,
        )
        forward["image_rotary_emb"] = rotary
        temb = model.time_embedding(model.time_proj(forward["timestep"]).to(dtype))
        specs = [
            (
                "block",
                model.transformer_blocks[min(1, len(model.transformer_blocks) - 1)],
                len(model.transformer_blocks),
                {**block_args, "temb": temb, "image_rotary_emb": rotary},
            )
        ]
    return forward, specs


def profile_candidate(specs, chunk, warmups, repeats):
    workspace = ChunkDiTWorkspace(chunk)
    blocks = {}
    for name, block, count, inputs in specs:
        measurement = measure(lambda: block(**inputs, chunk_dit_workspace=workspace), warmups, repeats)
        blocks[name] = {**measurement, "layer_count": count}
        if measurement["status"] != "ok":
            break
    workspace.key = workspace.value = None
    gc.collect()
    torch.cuda.empty_cache()
    if any(block["status"] != "ok" for block in blocks.values()):
        return {"chunk_size": chunk, "status": "oom", "blocks": blocks}
    return {
        "chunk_size": chunk,
        "status": "ok",
        "blocks": blocks,
        "estimated_dit_seconds": sum(b["layer_count"] * b["median_seconds"] for b in blocks.values()),
        "peak_allocated": max(b["peak_allocated"] for b in blocks.values()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["hunyuan", "cogvideo"], required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--revision")
    parser.add_argument(
        "--memory-budget-gib", type=float, required=True, help="DiT allocated peak budget, including weights"
    )
    parser.add_argument("--reserve-gib", type=float, default=1.0, help="Subtract this headroom from the given budget")
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--frames", type=int)
    parser.add_argument("--batch-size", type=int, help="Effective DiT batch: defaults to Hunyuan 1, CogVideoX CFG 2")
    parser.add_argument("--text-tokens", type=int)
    parser.add_argument("--min-chunk-size", type=int, default=1024)
    parser.add_argument("--max-chunk-size", type=int, default=16384)
    parser.add_argument("--alignment", type=int, default=256)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--validate-forward", action="store_true", help="Also validate complete Transformer forwards")
    parser.add_argument("--validation-repeats", type=int, default=1, help="Only used with --validate-forward")
    parser.add_argument("--output", required=True, help="Result JSON; exact matching successful results are reused")
    parser.add_argument("--force", action="store_true", help="Ignore an existing tuning result")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required for offline profiling")
    if not math.isfinite(args.memory_budget_gib) or args.memory_budget_gib <= 0:
        parser.error("memory-budget-gib must be finite and positive")
    if not math.isfinite(args.reserve_gib) or not 0 <= args.reserve_gib < args.memory_budget_gib:
        parser.error("reserve-gib must be finite, non-negative and smaller than the budget")
    if args.warmups < 1 or args.repeats < 1 or (args.validate_forward and args.validation_repeats < 1):
        parser.error("At least one warmup and one measured repetition are required")
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    hunyuan = args.model == "hunyuan"
    cls = HunyuanVideoTransformer3DModel if hunyuan else CogVideoXTransformer3DModel
    location = Path(args.model_path)
    revision = args.revision if location.exists() else model_info(args.model_path, revision=args.revision).sha
    config = cls.load_config(args.model_path, subfolder="transformer", revision=revision)
    shape = workload_shape(
        args.model,
        config,
        args.height if args.height is not None else (720 if hunyuan else 768),
        args.width if args.width is not None else (1280 if hunyuan else 1360),
        args.frames if args.frames is not None else (129 if hunyuan else 81),
        args.batch_size if args.batch_size is not None else (1 if hunyuan else 2),
        args.text_tokens
        if args.text_tokens is not None
        else (256 if hunyuan else config.get("max_text_seq_length", 226)),
    )
    grid = aligned_candidates(
        args.min_chunk_size, args.max_chunk_size, args.alignment, max(shape["video_tokens"], shape["text_tokens"])
    )
    if not grid:
        parser.error("No aligned chunk sizes satisfy the requested bounds and sequence length")
    budget = math.floor((args.memory_budget_gib - args.reserve_gib) * GIB)
    free, total = torch.cuda.mem_get_info()
    if budget > free:
        parser.error(f"Usable budget exceeds current free GPU memory ({free / GIB:.2f} GiB)")
    properties = torch.cuda.get_device_properties(0)
    source = Path(diffusers.__file__).parent
    source_paths = [
        "models/chunk_dit_utils.py",
        "models/transformers/transformer_hunyuan_video.py",
        "models/transformers/cogvideox_transformer_3d.py",
        "models/normalization.py",
    ]
    identity = {
        "model": str(location.resolve()) if location.exists() else args.model_path,
        "revision": revision,
        "config": config,
        "weight_files": [
            (f.name, f.stat().st_size, f.stat().st_mtime_ns)
            for f in sorted((location / "transformer").iterdir())
            if f.suffix in {".safetensors", ".bin", ".json"}
        ]
        if location.exists()
        else None,
        "gpu": {
            "name": properties.name,
            "capability": [properties.major, properties.minor],
            "total_memory": total,
            "sm_count": properties.multi_processor_count,
        },
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "diffusers": diffusers.__version__,
        "runtime": {
            "python": platform.python_version(),
            "cpu_threads": torch.get_num_threads(),
            "allocator_backend": torch.cuda.get_allocator_backend(),
            "allocator_configuration": os.environ.get("PYTORCH_ALLOC_CONF", os.environ.get("PYTORCH_CUDA_ALLOC_CONF")),
            "cuda_launch_blocking": os.environ.get("CUDA_LAUNCH_BLOCKING"),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "sdpa_enabled": [
                torch.backends.cuda.flash_sdp_enabled(),
                torch.backends.cuda.mem_efficient_sdp_enabled(),
                torch.backends.cuda.cudnn_sdp_enabled(),
                torch.backends.cuda.math_sdp_enabled(),
            ],
        },
        "dtype": "bfloat16",
        "workload": shape,
        "constraints": {
            "effective_budget_bytes": budget,
            "memory_budget_gib": args.memory_budget_gib,
            "reserve_gib": args.reserve_gib,
            "validate_forward": args.validate_forward,
            "grid": grid,
        },
        "sampling": {
            "warmups": args.warmups,
            "repeats": args.repeats,
            "validation_repeats": args.validation_repeats if args.validate_forward else None,
        },
        "source_sha256": {p: hashlib.sha256((source / p).read_bytes()).hexdigest() for p in source_paths},
        "tuner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    cache_key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    if path.exists() and not args.force:
        cached = json.loads(path.read_text())
        if cached.get("cache_key") == cache_key and cached.get("status") == "passed":
            print(
                f"Cache hit: chunk_size={cached['chunk_size']}; "
                f"full_forward_validated={cached['full_forward_validated']}\n"
                f"pipe.transformer.enable_chunk_dit({cached['chunk_size']})"
            )
            return
    result = {
        "identity": {
            "model_type": args.model,
            **{key: identity[key] for key in ("dtype", "workload", "constraints", "sampling")},
        },
        "cache_key": cache_key,
        "status": "running",
        "candidates": [],
        "validations": [],
        "validation_mode": "full_forward" if args.validate_forward else "block_only",
        "full_forward_validated": False,
        "scope": "BF16 Transformer only; synthetic conditioning, real weights; no text encoder, VAE or scheduler",
        "memory_model": "weights + KV + fixed_live_tensors + alpha * B * C * D * bytes",
        "selection_objective": "lowest layer-count-weighted representative block latency; optional full-forward validation",
    }
    path.write_text(json.dumps(result, indent=2))
    model = cls.from_pretrained(
        args.model_path, subfolder="transformer", revision=revision, torch_dtype=torch.bfloat16
    ).eval()
    model._validate_chunk_dit()
    weights = sum(p.numel() * p.element_size() for p in list(model.parameters()) + list(model.buffers()))
    token_bytes = shape["batch"] * shape["hidden_dim"] * 2
    kv = 2 * shape["total_tokens"] * token_bytes
    lower_bound = weights + 2 * kv
    result.update(weights_bytes=weights, kv_bytes=kv, implementation_memory_lower_bound=lower_bound)
    if lower_bound > budget:
        result.update(
            status="infeasible",
            reason="Weights, global KV, and full block input/output already exceed the usable budget",
        )
        path.write_text(json.dumps(result, indent=2))
        raise SystemExit("No feasible chunk size: the fixed memory lower bound exceeds the budget")
    torch.manual_seed(42)
    model.to("cuda")
    forward, specs = make_inputs(model, args.model, shape)
    calibration = sorted({min(grid, key=lambda c: abs(c - grid[0] * factor)) for factor in (1, 2, 4)})
    rows = {}
    for chunk in calibration:
        print(f"Calibrating C={chunk}", flush=True)
        rows[chunk] = profile_candidate(specs, chunk, args.warmups, args.repeats)
        result["candidates"] = list(rows.values())
        path.write_text(json.dumps(result, indent=2))
    memory_models = {}
    for name in [spec[0] for spec in specs]:
        points = [
            {"chunk_size": c, "peak_allocated": r["blocks"][name]["peak_allocated"]}
            for c, r in rows.items()
            if name in r["blocks"] and r["blocks"][name]["status"] == "ok"
        ]
        if not points:
            result.update(status="infeasible", reason=f"All calibration samples OOM for {name}")
            path.write_text(json.dumps(result, indent=2))
            raise SystemExit(result["reason"])
        memory_models[name] = fit_memory_model(points, weights, kv, token_bytes)
    upper = min(maximum_chunk(m, budget, grid[-1]) for m in memory_models.values())
    result.update(
        calibration_chunks=calibration,
        fitted_memory_models=memory_models,
        theoretical_max_chunk=upper,
        theoretical_aligned_max_chunk=(upper // args.alignment) * args.alignment,
    )
    for chunk in grid:
        if chunk in rows:
            continue
        if chunk > upper:
            rows[chunk] = {
                "chunk_size": chunk,
                "status": "pruned_by_memory_model",
                "predicted_peak_allocated": max(predict_peak(m, chunk) for m in memory_models.values()),
            }
        else:
            print(f"Profiling C={chunk}", flush=True)
            rows[chunk] = profile_candidate(specs, chunk, args.warmups, args.repeats)
        result["candidates"] = sorted(rows.values(), key=lambda r: r["chunk_size"])
        path.write_text(json.dumps(result, indent=2))
    ranked = rank_candidates(rows.values(), budget)
    del specs
    gc.collect()
    torch.cuda.empty_cache()
    if not args.validate_forward:
        if ranked:
            result.update(status="passed", chunk_size=ranked[0]["chunk_size"])
        else:
            result.update(status="infeasible", reason="No candidate fits the block-profile memory budget")
    else:
        for candidate in ranked:
            chunk = candidate["chunk_size"]
            print(f"Validating complete Transformer C={chunk}", flush=True)
            model.enable_chunk_dit(chunk)
            validation = measure(lambda: model(**forward), args.warmups, args.validation_repeats)
            result["validations"].append({"chunk_size": chunk, **validation})
            gc.collect()
            torch.cuda.empty_cache()
            if validation["status"] == "ok" and validation["peak_allocated"] <= budget:
                result.update(
                    status="passed",
                    chunk_size=chunk,
                    full_forward_validated=True,
                    verified_peak_allocated=validation["peak_allocated"],
                    verified_dit_seconds=validation["median_seconds"],
                )
                break
        else:
            result.update(status="infeasible", reason="No profiled candidate passed the full-forward memory budget")
    if result["status"] == "passed":
        selected = next(candidate for candidate in ranked if candidate["chunk_size"] == result["chunk_size"])
        result.update(
            selected_block_peak_allocated=selected["peak_allocated"],
            estimated_dit_seconds=selected["estimated_dit_seconds"],
        )
    path.write_text(json.dumps(result, indent=2))
    if result["status"] != "passed":
        raise SystemExit(result["reason"])
    metrics = (
        f"verified_peak={result['verified_peak_allocated'] / GIB:.3f} GiB; measured_DiT={result['verified_dit_seconds']:.3f}s"
        if result["full_forward_validated"]
        else f"block_peak={result['selected_block_peak_allocated'] / GIB:.3f} GiB; estimated_DiT={result['estimated_dit_seconds']:.3f}s"
    )
    print(
        f"Selected chunk_size={result['chunk_size']}; {metrics}; "
        f"full_forward_validated={result['full_forward_validated']}\n"
        f"pipe.transformer.enable_chunk_dit({result['chunk_size']})"
    )


if __name__ == "__main__":
    main()
