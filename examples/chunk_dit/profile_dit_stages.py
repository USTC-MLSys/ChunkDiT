"""Locate DiT stage peaks in one real pipeline denoising step without VAE decoding."""

import argparse
import json
from pathlib import Path

import torch
from benchmark import verify_model_devices

from diffusers import CogVideoXPipeline, HunyuanVideoPipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["hunyuan", "cogvideo"], required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--chunk-size", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    cls = HunyuanVideoPipeline if args.model == "hunyuan" else CogVideoXPipeline
    pipe = cls.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
    model = pipe.transformer.eval()
    model.enable_chunk_dit(args.chunk_size)
    pipe.vae.enable_tiling()
    pipe.vae.enable_memory_efficient_decode(4)
    pipe.enable_model_cpu_offload()
    result = {"model": args.model, "chunk_size": args.chunk_size, "stages": [], "gaps": []}
    pending = {}
    handles = []

    def root_pre(module, inputs, kwargs):
        inputs, kwargs = module._hf_hook.pre_forward(module, *inputs, **kwargs)
        torch.cuda.synchronize()
        result["devices"] = verify_model_devices(pipe, "transformer")
        result["entry_allocated"] = torch.cuda.memory_allocated()
        result["inputs"] = {k: list(v.shape) for k, v in kwargs.items() if torch.is_tensor(v)}
        torch.cuda.reset_peak_memory_stats()
        return inputs, kwargs

    def pre(name):
        def hook(module, inputs, kwargs):
            torch.cuda.synchronize()
            result["gaps"].append({"before": name, "peak_allocated": torch.cuda.max_memory_allocated()})
            torch.cuda.reset_peak_memory_stats()
            workspace = kwargs.get("chunk_dit_workspace")
            pending[name] = {
                "name": name,
                "entry_allocated": torch.cuda.memory_allocated(),
                "input_shape": list(inputs[0].shape) if inputs and torch.is_tensor(inputs[0]) else None,
                "kv_bytes": sum(
                    x.numel() * x.element_size() for x in (workspace.key, workspace.value) if x is not None
                )
                if workspace is not None
                else None,
            }

        return hook

    def post(name):
        def hook(module, inputs, kwargs, output):
            torch.cuda.synchronize()
            record = pending.pop(name)
            record["peak_allocated"] = torch.cuda.max_memory_allocated()
            record["exit_allocated"] = torch.cuda.memory_allocated()
            result["stages"].append(record)
            torch.cuda.reset_peak_memory_stats()

        return hook

    def root_post(module, inputs, kwargs, output):
        torch.cuda.synchronize()
        result["tail_peak_allocated"] = torch.cuda.max_memory_allocated()

    handles.append(model.register_forward_pre_hook(root_pre, with_kwargs=True))
    handles.append(model.register_forward_hook(root_post, with_kwargs=True))
    for name, module in model.named_children():
        stages = (
            [(f"{name}.{i}", block) for i, block in enumerate(module)]
            if isinstance(module, torch.nn.ModuleList)
            else [(name, module)]
        )
        for stage, child in stages:
            handles.append(child.register_forward_pre_hook(pre(stage), with_kwargs=True))
            handles.append(child.register_forward_hook(post(stage), with_kwargs=True))
    hunyuan = args.model == "hunyuan"
    with torch.inference_mode():
        output = pipe(
            prompt="A cat walks on the grass, realistic style.",
            height=720 if hunyuan else 768,
            width=1280 if hunyuan else 1360,
            num_frames=129 if hunyuan else 81,
            num_inference_steps=1,
            output_type="latent",
            generator=torch.Generator("cuda").manual_seed(42),
        )
    del output
    for handle in handles:
        handle.remove()
    result["max_stage"] = max(result["stages"], key=lambda r: r["peak_allocated"])
    result["peak_allocated"] = max(
        result["tail_peak_allocated"],
        result["max_stage"]["peak_allocated"],
        max(record["peak_allocated"] for record in result["gaps"]),
    )
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["max_stage"]))


if __name__ == "__main__":
    main()
