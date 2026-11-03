"""Compare production-sized decoder tiles with identical weights and temporal caches."""

import argparse
import json
import time
from pathlib import Path

import torch

from diffusers import AutoencoderKLCogVideoX, AutoencoderKLHunyuanVideo


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["hunyuan", "cogvideo"], required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--chunk-size", type=int, default=0)
    p.add_argument("--output", required=True)
    p.add_argument("--reference")
    args = p.parse_args()
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=True)
    cls = AutoencoderKLHunyuanVideo if args.model == "hunyuan" else AutoencoderKLCogVideoX
    vae = cls.from_pretrained(args.model_path, subfolder="vae", torch_dtype=torch.bfloat16).eval().cuda()
    if args.chunk_size:
        vae.enable_memory_efficient_decode(args.chunk_size)
    torch.manual_seed(42)
    shape = (1, 16, 5, 32, 32) if args.model == "hunyuan" else (1, 16, 3, 30, 45)
    inputs = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    next_inputs = (
        None if args.model == "hunyuan" else torch.randn(1, 16, 2, 30, 45, device="cuda", dtype=torch.bfloat16)
    )
    records = []
    with torch.inference_mode():
        for run in range(4):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            if next_inputs is None:
                output = vae.decoder(inputs)
            else:
                first, cache = vae.decoder(inputs)
                second, cache = vae.decoder(next_inputs, conv_cache=cache)
                output = torch.cat((first, second), dim=2)
                del first, second, cache
            torch.cuda.synchronize()
            if run:
                records.append(
                    {"seconds": time.perf_counter() - start, "peak_allocated": torch.cuda.max_memory_allocated()}
                )
            if run == 0:
                torch.save(output.cpu(), directory / "output.pt")
            del output
    result = {"model": args.model, "chunk_size": args.chunk_size, "measurements": records, "status": "passed"}
    if args.reference:
        baseline = torch.load(Path(args.reference) / "output.pt", weights_only=True).double().flatten()
        actual = torch.load(directory / "output.pt", weights_only=True).double().flatten()
        result["relative_l2"] = ((actual - baseline).norm() / baseline.norm()).item()
        result["cosine"] = torch.nn.functional.cosine_similarity(actual, baseline, dim=0).item()
        if result["relative_l2"] > 0.01 or result["cosine"] < 0.999:
            result["status"] = "numerical_failure"
    (directory / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
