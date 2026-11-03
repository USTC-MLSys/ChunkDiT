"""Attribute decoder max allocated memory to nested modules on production-sized tiles."""

import argparse
import json
from pathlib import Path

import torch

from diffusers import AutoencoderKLCogVideoX, AutoencoderKLHunyuanVideo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["hunyuan", "cogvideo"], required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cls = AutoencoderKLHunyuanVideo if args.model == "hunyuan" else AutoencoderKLCogVideoX
    vae = cls.from_pretrained(args.model_path, subfolder="vae", torch_dtype=torch.bfloat16).eval().cuda()
    stack = []
    records = []
    handles = []

    def observe():
        peak = torch.cuda.max_memory_allocated()
        for state in stack:
            state["peak"] = max(state["peak"], peak)
        torch.cuda.reset_peak_memory_stats()

    def pre(name):
        def hook(module, inputs):
            torch.cuda.synchronize()
            observe()
            entry = torch.cuda.memory_allocated()
            stack.append(
                {
                    "name": name,
                    "type": type(module).__name__,
                    "entry": entry,
                    "peak": entry,
                    "shape": list(inputs[0].shape) if inputs and torch.is_tensor(inputs[0]) else None,
                }
            )

        return hook

    def post(module, inputs, output):
        torch.cuda.synchronize()
        observe()
        state = stack.pop()
        state["increment"] = state["peak"] - state["entry"]
        records.append(state)

    for name, module in vae.decoder.named_modules():
        handles.extend([module.register_forward_pre_hook(pre(name)), module.register_forward_hook(post)])
    torch.manual_seed(42)
    with torch.inference_mode():
        if args.model == "hunyuan":
            vae.decoder(torch.randn(1, 16, 5, 32, 32, device="cuda", dtype=torch.bfloat16))
        else:
            _, cache = vae.decoder(torch.randn(1, 16, 3, 30, 45, device="cuda", dtype=torch.bfloat16))
            vae.decoder(torch.randn(1, 16, 2, 30, 45, device="cuda", dtype=torch.bfloat16), conv_cache=cache)
    for handle in handles:
        handle.remove()
    result = sorted(records, key=lambda r: r["increment"], reverse=True)
    output_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result[:18], indent=2))


if __name__ == "__main__":
    main()
