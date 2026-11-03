"""Paired full-video VAE decode, preserving default tiles and comparing every output value."""

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
    p.add_argument("--output", required=True)
    args = p.parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cls = AutoencoderKLHunyuanVideo if args.model == "hunyuan" else AutoencoderKLCogVideoX
    vae = cls.from_pretrained(args.model_path, subfolder="vae", torch_dtype=torch.bfloat16).eval().cuda()
    vae.enable_tiling()
    shape = (1, 16, 33, 90, 160) if args.model == "hunyuan" else (1, 16, 21, 96, 170)
    torch.manual_seed(42)
    latent = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    records = []
    reference = None
    with torch.inference_mode():
        for size in (0, 4, 0, 4):
            if size:
                vae.enable_memory_efficient_decode(size)
            else:
                vae.disable_memory_efficient_decode()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            entry = torch.cuda.memory_allocated()
            start = time.perf_counter()
            output = vae.decode(latent).sample
            torch.cuda.synchronize()
            record = {
                "chunk_size": size,
                "seconds": time.perf_counter() - start,
                "entry_allocated": entry,
                "peak_allocated": torch.cuda.max_memory_allocated(),
                "shape": list(output.shape),
            }
            actual = output.cpu()
            del output
            if reference is None:
                reference = actual
            record["bitwise_equal"] = torch.equal(actual, reference)
            if not record["bitwise_equal"]:
                delta = actual.float() - reference.float()
                record["relative_l2"] = (delta.norm() / reference.float().norm()).item()
                assert record["relative_l2"] <= 0.01
                del delta
            records.append(record)
            del actual
            output_path.write_text(json.dumps(records, indent=2))
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
