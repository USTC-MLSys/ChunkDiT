# Copyright 2026 The HuggingFace Team. All rights reserved.
# Licensed under the Apache License, Version 2.0.

import torch
import torch.nn.functional as F


class MemoryEfficientDecodeMixin:
    def enable_memory_efficient_decode(self, conv_chunk_size: int = 4):
        """Bound large decoder Conv3d workspace by chunking output time, preserving global normalization."""
        if isinstance(conv_chunk_size, bool) or not isinstance(conv_chunk_size, int) or conv_chunk_size < 1:
            raise ValueError("conv_chunk_size must be a positive integer")
        for module in self.decoder.modules():
            if isinstance(module, torch.nn.Conv3d):
                module._decode_conv_chunk_size = conv_chunk_size

    def disable_memory_efficient_decode(self):
        """Restore the original decoder convolutions without changing weights."""
        for module in self.decoder.modules():
            if isinstance(module, torch.nn.Conv3d):
                module._decode_conv_chunk_size = None


def chunked_conv3d(conv, inputs, chunk_size, min_input_bytes=256 * 1024**2):
    if hasattr(conv, "_hf_hook") or hasattr(conv, "_diffusers_hook"):
        raise ValueError(
            "Decoder convolution chunking requires resident weights; use model CPU offload instead of layer hooks"
        )
    if conv.training or torch.is_grad_enabled():
        raise ValueError("Memory-efficient decode requires eval() and no_grad() or inference_mode()")
    if conv.padding_mode != "zeros" or isinstance(conv.padding, str) or conv.padding[0] != 0:
        raise ValueError("Decoder chunking requires explicit temporal padding and zero Conv3d padding mode")
    effective_kernel = conv.dilation[0] * (conv.kernel_size[0] - 1) + 1
    frames = (inputs.shape[2] - effective_kernel) // conv.stride[0] + 1
    if frames <= chunk_size or conv.kernel_size[0] == 1 or inputs.numel() * inputs.element_size() < min_input_bytes:
        return F.conv3d(inputs, conv.weight, conv.bias, conv.stride, conv.padding, conv.dilation, conv.groups)
    output = None
    for start in range(0, frames, chunk_size):
        end = min(start + chunk_size, frames)
        first = start * conv.stride[0]
        last = (end - 1) * conv.stride[0] + effective_kernel
        chunk = inputs[:, :, first:last].contiguous()
        result = F.conv3d(chunk, conv.weight, conv.bias, conv.stride, conv.padding, conv.dilation, conv.groups)
        if output is None:
            output = result.new_empty((result.shape[0], result.shape[1], frames, *result.shape[3:]))
        output[:, :, start:end].copy_(result)
        del chunk, result
    return output
