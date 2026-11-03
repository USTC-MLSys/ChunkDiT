"""Measure real pipeline DiT forwards after onload, serially on one GPU per model."""

import argparse
import hashlib
import json
import platform
import statistics
import time
from pathlib import Path

import torch
from benchmark import verify_model_devices

import diffusers
from diffusers import CogVideoXPipeline, HunyuanVideoPipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["hunyuan", "cogvideo"], required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not 0 < args.warmup_steps < args.steps:
        parser.error("warmup-steps must be positive and less than steps")
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    hunyuan = args.model == "hunyuan"
    cls = HunyuanVideoPipeline if hunyuan else CogVideoXPipeline
    pipe = cls.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
    pipe.transformer.eval()
    pipe.vae.enable_tiling()
    pipe.vae.enable_memory_efficient_decode(4)
    pipe.enable_model_cpu_offload()
    call = {
        "prompt": "A cat walks on the grass, realistic style.",
        "height": 720 if hunyuan else 768,
        "width": 1280 if hunyuan else 1360,
        "num_frames": 129 if hunyuan else 81,
        "num_inference_steps": args.steps,
        "guidance_scale": 6.0,
        "output_type": "latent",
    }
    source_files = [
        "src/diffusers/models/chunk_dit_utils.py",
        "src/diffusers/models/transformers/transformer_hunyuan_video.py",
        "src/diffusers/models/transformers/cogvideox_transformer_3d.py",
    ]
    result = {
        "arguments": vars(args),
        "generation": call,
        "seed": 42,
        "precision": "bfloat16",
        "torch": torch.__version__,
        "diffusers": diffusers.__version__,
        "python": platform.python_version(),
        "gpu": torch.cuda.get_device_name(),
        "source_sha256": {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in source_files},
        "scope": "transformer forward after verified onload; excludes text encoding, scheduler, transfers and VAE",
        "cpu_offload": "model",
        "vae_conv_chunk_size": 4,
        "vae_decode_executed": False,
        "timing": "synchronized perf_counter wall time and CUDA events on current stream",
        "attention": {
            "flash_enabled": torch.backends.cuda.flash_sdp_enabled(),
            "efficient_enabled": torch.backends.cuda.mem_efficient_sdp_enabled(),
            "cudnn_enabled": torch.backends.cuda.cudnn_sdp_enabled(),
            "math_enabled": torch.backends.cuda.math_sdp_enabled(),
            "chunk_cuda_math_disabled": True,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
        },
        "phases": [],
        "status": "running",
    }
    path.write_text(json.dumps(result, indent=2))
    for label, chunk in [("dense-start", 0), ("2048", 2048), ("4096", 4096), ("8192", 8192), ("dense-end", 0)]:
        if chunk:
            pipe.transformer.enable_chunk_dit(chunk)
        else:
            pipe.transformer.disable_chunk_dit()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        phase = {"label": label, "chunk_size": chunk, "steps": []}
        pending = {}

        def before(module, inputs, kwargs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            inputs, kwargs = module._hf_hook.pre_forward(module, *inputs, **kwargs)
            torch.cuda.synchronize()
            onload_seconds = time.perf_counter() - start
            devices = verify_model_devices(pipe, "transformer")
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            pending.update(
                onload_seconds=onload_seconds, devices=devices, start_event=start_event, end_event=end_event
            )
            pending["wall_start"] = time.perf_counter()
            start_event.record()
            return inputs, kwargs

        def after(module, inputs, output):
            pending["end_event"].record()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - pending["wall_start"]
            index = len(phase["steps"])
            phase["steps"].append(
                {
                    "index": index,
                    "warmup": index < args.warmup_steps,
                    "wall_seconds": elapsed,
                    "cuda_seconds": pending["start_event"].elapsed_time(pending["end_event"]) / 1000,
                    "onload_seconds_excluded": pending["onload_seconds"],
                    "devices_at_compute_entry": pending["devices"],
                }
            )
            print(f"{args.model} {label} step {index + 1}/{args.steps}: {elapsed:.6f}s", flush=True)
            pending.clear()

        pre = pipe.transformer.register_forward_pre_hook(before, with_kwargs=True)
        post = pipe.transformer.register_forward_hook(after)
        try:
            started = time.perf_counter()
            with torch.inference_mode():
                output = pipe(**call, generator=torch.Generator("cuda").manual_seed(42))
            torch.cuda.synchronize()
            phase["latent_pipeline_seconds"] = time.perf_counter() - started
            del output
        finally:
            pre.remove()
            post.remove()
        assert len(phase["steps"]) == args.steps
        measured = [step for step in phase["steps"] if not step["warmup"]]
        for metric in ("wall_seconds", "cuda_seconds"):
            values = [step[metric] for step in measured]
            phase["median_" + metric] = statistics.median(values)
            phase["range_" + metric] = [min(values), max(values)]
        result["phases"].append(phase)
        path.write_text(json.dumps(result, indent=2))
    dense = [
        s["wall_seconds"] for p in result["phases"] if p["chunk_size"] == 0 for s in p["steps"] if not s["warmup"]
    ]
    result["dense_median_wall_seconds"] = statistics.median(dense)
    result["summary"] = [
        {
            "chunk_size": p["chunk_size"],
            "median_wall_seconds": p["median_wall_seconds"],
            "median_cuda_seconds": p["median_cuda_seconds"],
            "latency_change_percent": (p["median_wall_seconds"] / result["dense_median_wall_seconds"] - 1) * 100,
        }
        for p in result["phases"]
        if p["chunk_size"]
    ]
    result["dense_bracket_change_percent"] = (
        result["phases"][-1]["median_wall_seconds"] / result["phases"][0]["median_wall_seconds"] - 1
    ) * 100
    result["status"] = "passed"
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
