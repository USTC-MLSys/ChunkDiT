"""Synthetic production-width blocks: no checkpoints, text encoders, scheduler or VAE."""

import argparse
import json
import statistics
import time

import torch

from diffusers.models.chunk_dit_utils import ChunkDiTWorkspace
from diffusers.models.embeddings import get_1d_rotary_pos_embed
from diffusers.models.transformers.cogvideox_transformer_3d import CogVideoXBlock
from diffusers.models.transformers.transformer_hunyuan_video import (
    HunyuanVideoSingleTransformerBlock,
    HunyuanVideoTransformerBlock,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--block", choices=["hunyuan-double", "hunyuan-single", "cogvideo"], required=True)
    parser.add_argument("--chunk-size", type=int, default=0)
    parser.add_argument("--tokens", type=int)
    args = parser.parse_args()
    torch.manual_seed(42)
    cog = args.block == "cogvideo"
    tokens = args.tokens or (44880 if cog else 118800)
    text = 226 if cog else 256
    dim = 3072
    heads, head_dim = (48, 64) if cog else (24, 128)
    if cog:
        block = CogVideoXBlock(dim, heads, head_dim, 512, attention_bias=True)
    elif args.block == "hunyuan-double":
        block = HunyuanVideoTransformerBlock(heads, head_dim, 4)
    else:
        block = HunyuanVideoSingleTransformerBlock(heads, head_dim)
    block = block.eval().to(device="cuda", dtype=torch.bfloat16)
    hidden = torch.randn(1, tokens, dim, device="cuda", dtype=torch.bfloat16)
    encoder = torch.randn(1, text, dim, device="cuda", dtype=torch.bfloat16)
    temb = torch.randn(1, 512 if cog else dim, device="cuda", dtype=torch.bfloat16)
    rotary = tuple(x.cuda() for x in get_1d_rotary_pos_embed(head_dim, tokens, use_real=True))
    kwargs = {"hidden_states": hidden, "encoder_hidden_states": encoder, "temb": temb}
    kwargs["image_rotary_emb" if cog or args.block.endswith("single") else "freqs_cis"] = rotary
    records = []
    with torch.inference_mode():
        for run in range(4):
            if args.chunk_size:
                kwargs["chunk_dit_workspace"] = ChunkDiTWorkspace(args.chunk_size)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            entry = torch.cuda.memory_allocated()
            start = time.perf_counter()
            output = block(**kwargs)
            torch.cuda.synchronize()
            if run:
                records.append(
                    {
                        "seconds": time.perf_counter() - start,
                        "entry": entry,
                        "peak": torch.cuda.max_memory_allocated(),
                        "activation_peak": torch.cuda.max_memory_allocated() - entry,
                    }
                )
            del output
            kwargs.pop("chunk_dit_workspace", None)
    print(
        json.dumps(
            {
                "block": args.block,
                "chunk_size": args.chunk_size,
                "tokens": tokens,
                "synthetic": True,
                "gpu": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "measurements": records,
                "median_seconds": statistics.median(x["seconds"] for x in records),
            }
        )
    )


if __name__ == "__main__":
    main()
