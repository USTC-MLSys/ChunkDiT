"""Run one configuration per process; compare outputs against a dense reference run."""

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from huggingface_hub import model_info

import diffusers
from diffusers import CogVideoXPipeline, HunyuanVideoPipeline
from diffusers.utils import export_to_video


def prepare_vae_decode(pipe):
    modules = {name: module for name, module in pipe.components.items() if isinstance(module, torch.nn.Module)}
    for name, module in modules.items():
        if name != "vae":
            module.to("cpu")
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    pipe.vae._hf_hook.pre_forward(pipe.vae)
    torch.cuda.synchronize()
    return verify_model_devices(pipe, "vae")


def verify_model_devices(pipe, active):
    devices = {}
    for name, module in pipe.components.items():
        if not isinstance(module, torch.nn.Module):
            continue
        tensors = list(module.parameters()) + list(module.buffers())
        devices[name] = sorted({str(tensor.device) for tensor in tensors})
        expected = "cuda" if name == active else "cpu"
        if any(tensor.device.type != expected for tensor in tensors):
            raise RuntimeError(f"Unexpected device at {active} compute entry: {name}={devices[name]}")
    return devices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["hunyuan", "cogvideo"], required=True)
    parser.add_argument("--model-path")
    parser.add_argument("--revision")
    parser.add_argument("--chunk-size", type=int, default=0)
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--frames", type=int)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", default="A cat walks on the grass, realistic style.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--reference", help="Dense output directory, produced using the same arguments")
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--vae-conv-chunk-size", type=int, default=0)
    parser.add_argument("--latent-only", action="store_true")
    parser.add_argument(
        "--vae-tile-height", type=int, help="Spatial tile height in decoded pixels; unset uses the VAE default"
    )
    parser.add_argument(
        "--vae-tile-width", type=int, help="Spatial tile width in decoded pixels; unset uses the VAE default"
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    if args.runs < 1 or args.warmups < 0:
        parser.error("At least one measured run and a non-negative warmup count are required")
    for size in (args.vae_tile_height, args.vae_tile_width):
        if size is not None and (size < 32 or size % 8):
            parser.error("VAE tile dimensions must be multiples of 8 and at least 32 pixels")
    if args.vae_conv_chunk_size < 0:
        parser.error("vae-conv-chunk-size must be non-negative")
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=True)
    hunyuan = args.model == "hunyuan"
    model_id = args.model_path or ("hunyuanvideo-community/HunyuanVideo" if hunyuan else "zai-org/CogVideoX1.5-5B")
    revision = args.revision if Path(model_id).exists() else model_info(model_id, revision=args.revision).sha
    result = {
        "arguments": vars(args),
        "model_id": model_id,
        "revision": revision,
        "torch": torch.__version__,
        "diffusers": diffusers.__version__,
        "python": platform.python_version(),
        "gpu": torch.cuda.get_device_name(),
        "phase_measurement_scope": "compute_after_verified_onload" if args.cpu_offload else "module_calls",
        "measurement_metric": "torch.cuda.max_memory_allocated",
    }
    try:
        pipeline_class = HunyuanVideoPipeline if hunyuan else CogVideoXPipeline
        pipe = pipeline_class.from_pretrained(model_id, revision=revision, torch_dtype=torch.bfloat16)
        pipe.transformer.eval()
        if args.chunk_size:
            pipe.transformer.enable_chunk_dit(args.chunk_size)
        tiling = {}
        if args.vae_tile_height is not None:
            tiling["tile_sample_min_height"] = args.vae_tile_height
            if hunyuan:
                tiling["tile_sample_stride_height"] = args.vae_tile_height * 3 // 4
        if args.vae_tile_width is not None:
            tiling["tile_sample_min_width"] = args.vae_tile_width
            if hunyuan:
                tiling["tile_sample_stride_width"] = args.vae_tile_width * 3 // 4
        pipe.vae.enable_tiling(**tiling)
        if args.vae_conv_chunk_size:
            pipe.vae.enable_memory_efficient_decode(args.vae_conv_chunk_size)
        if hunyuan:
            if pipe.vae.tile_sample_stride_height % 8 or pipe.vae.tile_sample_stride_width % 8:
                raise ValueError("Hunyuan tile strides must align to the spatial compression ratio (8)")
        else:
            for dimension in ("height", "width"):
                sample = getattr(pipe.vae, "tile_sample_min_" + dimension)
                latent = getattr(pipe.vae, "tile_latent_min_" + dimension)
                overlap = getattr(pipe.vae, "tile_overlap_factor_" + dimension)
                if int(latent * (1 - overlap)) * 8 != sample - int(sample * overlap):
                    raise ValueError(
                        "CogVideoX tile size must align latent strides with output blending; use height 96, width 120"
                    )
        result["vae_tiling"] = {
            name: getattr(pipe.vae, name)
            for name in (
                "tile_sample_min_height",
                "tile_sample_min_width",
                "tile_sample_stride_height",
                "tile_sample_stride_width",
                "tile_sample_min_num_frames",
                "tile_sample_stride_num_frames",
                "tile_overlap_factor_height",
                "tile_overlap_factor_width",
                "num_latent_frames_batch_size",
            )
            if hasattr(pipe.vae, name)
        }
        if args.cpu_offload:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cuda")
        call = {
            "prompt": args.prompt,
            "height": args.height or (720 if hunyuan else 768),
            "width": args.width or (1280 if hunyuan else 1360),
            "num_frames": args.frames or (129 if hunyuan else 81),
            "num_inference_steps": args.steps,
            "output_type": "latent" if args.latent_only else "np",
        }
        result["generation"] = call
        # Correctness run is separate from timed runs to exclude host copies and video export.
        first_output = []
        step_outputs = []

        def capture(module, inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output.sample
            step_outputs.append(tensor.detach().cpu())

        base_decode = pipe.vae.decode

        def checked_decode(*decode_args, **decode_kwargs):
            result["correctness_decode_entry_devices"] = prepare_vae_decode(pipe)
            return base_decode(*decode_args, **decode_kwargs)

        if args.cpu_offload:
            pipe.vae.decode = checked_decode
        handle = pipe.transformer.register_forward_hook(capture)
        video = pipe(**call, generator=torch.Generator("cuda").manual_seed(args.seed))
        handle.remove()
        pipe.vae.decode = base_decode
        first_output = step_outputs[0]
        torch.save({"first": first_output, "steps": step_outputs}, directory / "outputs.pt")
        if args.reference:
            metadata = json.loads((Path(args.reference) / "result.json").read_text())
            if metadata.get("generation") != call or metadata["arguments"]["seed"] != args.seed:
                raise ValueError("Reference generation arguments and seed must match")
            if metadata.get("model_id") != model_id or metadata.get("revision") != revision:
                raise ValueError("Reference checkpoint and revision must match")
            if metadata.get("vae_tiling") != result["vae_tiling"]:
                raise ValueError("Reference VAE tiling settings must match")
            baseline = torch.load(Path(args.reference) / "outputs.pt", map_location="cpu", weights_only=True)
            errors = []
            for actual, expected in zip(step_outputs, baseline["steps"], strict=True):
                actual, expected = actual.double(), expected.double()
                errors.append(
                    {
                        "relative_l2": ((actual - expected).norm() / expected.norm().clamp_min(1e-12)).item(),
                        "cosine": F.cosine_similarity(actual.flatten(), expected.flatten(), dim=0).item(),
                    }
                )
            result["step_output_errors"] = errors
            result["first_step_pass"] = errors[0]["relative_l2"] <= 0.01 and errors[0]["cosine"] >= 0.999
        if not args.latent_only:
            export_to_video(video.frames[0], str(directory / "sample.mp4"), fps=24 if hunyuan else 16)
        step_outputs.clear()
        del first_output, video
        measurements = []
        for run in range(args.warmups + args.runs):
            intervals = []
            peaks = []
            e2e_peaks = []
            e2e_reserved = []
            other_peaks = []
            vae_measurements = []
            offload_handoff_measurements = []
            transformer_onload_measurements = []
            decoder_measurements = []
            decoder_entry = []
            vae_segment_peaks = []
            vae_segment_reserved = []
            original_decode = pipe.vae.decode

            def before(module, inputs, kwargs):
                torch.cuda.synchronize()
                e2e_peaks.append(torch.cuda.max_memory_allocated())
                e2e_reserved.append(torch.cuda.max_memory_reserved())
                other_peaks.append(torch.cuda.max_memory_allocated())
                torch.cuda.reset_peak_memory_stats()
                if args.cpu_offload:
                    onload_start = time.perf_counter()
                    inputs, kwargs = module._hf_hook.pre_forward(module, *inputs, **kwargs)
                    torch.cuda.synchronize()
                    devices = verify_model_devices(pipe, "transformer")
                    transformer_onload_measurements.append(
                        {
                            "seconds": time.perf_counter() - onload_start,
                            "peak_allocated": torch.cuda.max_memory_allocated(),
                            "model_devices_before_compute": devices,
                        }
                    )
                    e2e_peaks.append(torch.cuda.max_memory_allocated())
                    e2e_reserved.append(torch.cuda.max_memory_reserved())
                    torch.cuda.reset_peak_memory_stats()
                intervals.append([time.perf_counter(), torch.cuda.memory_allocated()])
                return inputs, kwargs

            def after(module, inputs, output):
                torch.cuda.synchronize()
                intervals[-1].append(time.perf_counter() - intervals[-1][0])
                peaks.append((torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()))
                e2e_peaks.append(torch.cuda.max_memory_allocated())
                e2e_reserved.append(torch.cuda.max_memory_reserved())
                torch.cuda.reset_peak_memory_stats()

            def before_decoder(module, inputs):
                torch.cuda.synchronize()
                vae_segment_peaks.append(torch.cuda.max_memory_allocated())
                vae_segment_reserved.append(torch.cuda.max_memory_reserved())
                torch.cuda.reset_peak_memory_stats()
                decoder_entry.append((time.perf_counter(), torch.cuda.memory_allocated()))

            def after_decoder(module, inputs, output):
                torch.cuda.synchronize()
                started_decoder, entry = decoder_entry.pop()
                decoder_measurements.append(
                    {
                        "seconds": time.perf_counter() - started_decoder,
                        "entry_allocated": entry,
                        "peak_allocated": torch.cuda.max_memory_allocated(),
                    }
                )
                vae_segment_peaks.append(torch.cuda.max_memory_allocated())
                vae_segment_reserved.append(torch.cuda.max_memory_reserved())
                torch.cuda.reset_peak_memory_stats()

            def decode(*decode_args, **decode_kwargs):
                torch.cuda.synchronize()
                other_peaks.append(torch.cuda.max_memory_allocated())
                e2e_peaks.append(torch.cuda.max_memory_allocated())
                e2e_reserved.append(torch.cuda.max_memory_reserved())
                torch.cuda.reset_peak_memory_stats()
                devices = None
                if args.cpu_offload:
                    handoff_start = time.perf_counter()
                    devices = prepare_vae_decode(pipe)
                    offload_handoff_measurements.append(
                        {
                            "seconds": time.perf_counter() - handoff_start,
                            "peak_allocated": torch.cuda.max_memory_allocated(),
                            "peak_reserved": torch.cuda.max_memory_reserved(),
                        }
                    )
                    e2e_peaks.append(torch.cuda.max_memory_allocated())
                    e2e_reserved.append(torch.cuda.max_memory_reserved())
                    torch.cuda.reset_peak_memory_stats()
                entry = torch.cuda.memory_allocated()
                decode_start = time.perf_counter()
                decoded = original_decode(*decode_args, **decode_kwargs)
                torch.cuda.synchronize()
                vae_segment_peaks.append(torch.cuda.max_memory_allocated())
                vae_segment_reserved.append(torch.cuda.max_memory_reserved())
                vae_peak = max(vae_segment_peaks)
                vae_reserved = max(vae_segment_reserved)
                vae_measurements.append(
                    {
                        "seconds": time.perf_counter() - decode_start,
                        "entry_allocated": entry,
                        "model_devices_before_decode": devices,
                        "non_vae_models_verified_on_cpu": devices is not None,
                        "peak_allocated": vae_peak,
                        "peak_reserved": vae_reserved,
                    }
                )
                e2e_peaks.append(vae_peak)
                e2e_reserved.append(vae_reserved)
                torch.cuda.reset_peak_memory_stats()
                return decoded

            pipe.vae.decode = decode
            pre = pipe.transformer.register_forward_pre_hook(before, with_kwargs=True)
            post = pipe.transformer.register_forward_hook(after)
            decoder_pre = pipe.vae.decoder.register_forward_pre_hook(before_decoder)
            decoder_post = pipe.vae.decoder.register_forward_hook(after_decoder)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter()
            video = pipe(**call, generator=torch.Generator("cuda").manual_seed(args.seed))
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            if not args.latent_only:
                expected_shape = (call["num_frames"], call["height"], call["width"], 3)
                if tuple(video.frames[0].shape) != expected_shape:
                    raise ValueError(f"Decoded shape {video.frames[0].shape} != expected {expected_shape}")
                result["decoded_shape"] = list(video.frames[0].shape)
            pre.remove()
            post.remove()
            decoder_pre.remove()
            decoder_post.remove()
            pipe.vae.decode = original_decode
            e2e_peaks.append(torch.cuda.max_memory_allocated())
            e2e_reserved.append(torch.cuda.max_memory_reserved())
            phase_peaks = {
                "transformer": max(x[0] for x in peaks),
                "vae": max((x["peak_allocated"] for x in vae_measurements), default=0),
                "transformer_onload": max((x["peak_allocated"] for x in transformer_onload_measurements), default=0),
                "offload_handoff": max((x["peak_allocated"] for x in offload_handoff_measurements), default=0),
                "other": max(other_peaks, default=0),
                "postprocess": torch.cuda.max_memory_allocated(),
            }
            if run >= args.warmups:
                measurements.append(
                    {
                        "seconds": elapsed,
                        "phase_peaks_allocated": phase_peaks,
                        "peak_phase": max(phase_peaks, key=phase_peaks.get),
                        "vae_measurements": vae_measurements,
                        "offload_handoff_measurements": offload_handoff_measurements,
                        "transformer_onload_measurements": transformer_onload_measurements,
                        "decoder_measurements": decoder_measurements,
                        "decoder_peak_allocated": max((x["peak_allocated"] for x in decoder_measurements), default=0),
                        "decoder_call_count": len(decoder_measurements),
                        "transformer_seconds": sum(x[2] for x in intervals),
                        "peak_allocated": max(e2e_peaks),
                        "peak_reserved": max(e2e_reserved),
                        "transformer_peak_allocated": max(x[0] for x in peaks),
                        "transformer_peak_reserved": max(x[1] for x in peaks),
                        "transformer_entry_allocated": [x[1] for x in intervals],
                        "transformer_activation_peak": max(p[0] - x[1] for p, x in zip(peaks, intervals)),
                    }
                )
            del video
        result["measurements"] = measurements
        result["median_seconds"] = statistics.median(x["seconds"] for x in measurements)
        result["status"] = "passed" if result.get("first_step_pass", True) else "numerical_failure"
    except torch.cuda.OutOfMemoryError as error:
        result.update(status="oom", error=str(error))
    except Exception as error:
        result.update(status="error", error=str(error))
        raise
    finally:
        (directory / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
