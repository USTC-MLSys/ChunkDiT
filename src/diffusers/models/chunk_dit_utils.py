# Copyright 2026 The HuggingFace Team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0

from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from .embeddings import apply_rotary_emb


class ChunkDiTMixin:
    _chunk_dit_size = None

    def enable_chunk_dit(self, chunk_size: int = 2048):
        """Enable inference-only token chunking without changing checkpoint parameters."""
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer")
        self._validate_chunk_dit()
        self._chunk_dit_size = chunk_size

    def disable_chunk_dit(self):
        """Restore the original dense execution path."""
        self._chunk_dit_size = None

    def _validate_chunk_dit(self):
        from .attention_processor import AttnProcessor2_0, CogVideoXAttnProcessor2_0
        from .transformers.transformer_hunyuan_video import HunyuanVideoAttnProcessor2_0

        if getattr(self, "is_cache_enabled", False):
            raise ValueError("Disable caching before using chunkDiT")
        if getattr(self, "peft_config", None):
            raise ValueError("Adapters are not supported with chunkDiT v1")
        if getattr(self, "is_quantized", False):
            raise ValueError("Quantized transformers are not supported with chunkDiT v1")
        for name, module in self.named_modules():
            if getattr(module, "fused_projections", False):
                raise ValueError("Unfuse QKV projections before using chunkDiT")
            if hasattr(module, "processor"):
                processor = module.processor
                expected = (
                    AttnProcessor2_0
                    if name.startswith("context_embedder.")
                    else HunyuanVideoAttnProcessor2_0
                    if hasattr(self, "single_transformer_blocks")
                    else CogVideoXAttnProcessor2_0
                )
                if type(processor) is not expected:
                    raise ValueError("chunkDiT requires the original attention processors")
                if (
                    getattr(processor, "_attention_backend", None) is not None
                    or getattr(processor, "_parallel_config", None) is not None
                ):
                    raise ValueError(
                        "Custom attention backends and sequence parallelism are not supported with chunkDiT"
                    )
        if (
            self.config.in_channels != self.config.out_channels
            or getattr(self.config, "image_condition_type", None) is not None
        ):
            raise ValueError("chunkDiT v1 supports text-to-video models only")

    def _chunk_dit_workspace(self):
        if self._chunk_dit_size is None:
            return None
        self._validate_chunk_dit()
        if self.training or torch.is_grad_enabled():
            raise ValueError("chunkDiT requires model.eval() and torch.no_grad() or torch.inference_mode()")
        return ChunkDiTWorkspace(self._chunk_dit_size)


class ChunkDiTWorkspace:
    def __init__(self, chunk_size):
        self.chunk_size = chunk_size
        self.key = None
        self.value = None

    def chunks(self, states):
        for start in range(0, states.shape[1], self.chunk_size):
            yield start, min(start + self.chunk_size, states.shape[1])

    def project_output(self, states, normalize, projection):
        self.key = self.value = None
        output = None
        for start, end in self.chunks(states):
            projected = projection(normalize(states[:, start:end]))
            if output is None:
                output = projected.new_empty((states.shape[0], states.shape[1], projected.shape[-1]))
            output[:, start:end].copy_(projected)
            del projected
        return output

    def write(self, key, value, start, length):
        if self.key is None:
            shape = (key.shape[0], key.shape[1], length, key.shape[-1])
            self.key = key.new_empty(shape)
            self.value = value.new_empty(shape)
        self.key[:, :, start : start + key.shape[2]].copy_(key)
        self.value[:, :, start : start + value.shape[2]].copy_(value)

    def attention(self, query, mask=None):
        context = (
            sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION])
            if query.is_cuda
            else nullcontext()
        )
        with context:
            return (
                F.scaled_dot_product_attention(query, self.key, self.value, attn_mask=mask, dropout_p=0.0)
                .transpose(1, 2)
                .flatten(2)
            )


def project(projection, norm, states, heads, rotary=None):
    result = projection(states).unflatten(-1, (heads, -1)).transpose(1, 2)
    if norm is not None:
        result = norm(result)
    if rotary is not None:
        result = apply_rotary_emb(result, rotary)
    return result


def rotary_slice(rotary, start, end):
    return None if rotary is None else tuple(component[start:end] for component in rotary)


def hunyuan_chunk_forward(block, hidden_states, encoder_hidden_states, temb, mask, rotary, workspace, single=False):
    video_length = hidden_states.shape[1]
    length = video_length + encoder_hidden_states.shape[1]
    attn = block.attn
    streams = ((hidden_states, 0), (encoder_hidden_states, video_length))
    if single:
        modulation = block.norm.linear(block.norm.silu(temb)).chunk(3, dim=1)
        parameters = (modulation, modulation)
    else:
        parameters = tuple(norm.linear(norm.silu(temb)).chunk(6, dim=1) for norm in (block.norm1, block.norm1_context))

    def normalize(states, stream):
        shift, scale = parameters[stream][:2]
        norm = block.norm.norm if single else (block.norm1.norm if stream == 0 else block.norm1_context.norm)
        return norm(states) * (1 + scale[:, None]) + shift[:, None]

    for stream, (states, offset) in enumerate(streams):
        key_proj = attn.to_k if single or stream == 0 else attn.add_k_proj
        value_proj = attn.to_v if single or stream == 0 else attn.add_v_proj
        key_norm = attn.norm_k if single or stream == 0 else attn.norm_added_k
        for start, end in workspace.chunks(states):
            normalized = normalize(states[:, start:end], stream)
            positions = rotary_slice(rotary, start, end) if stream == 0 else None
            key = project(key_proj, key_norm, normalized, attn.heads, positions)
            value = project(value_proj, None, normalized, attn.heads)
            workspace.write(key, value, offset + start, length)
            del key, value, normalized

    outputs = []
    for stream, (states, offset) in enumerate(streams):
        output = torch.empty_like(states)
        query_proj = attn.to_q if single or stream == 0 else attn.add_q_proj
        query_norm = attn.norm_q if single or stream == 0 else attn.norm_added_q
        for start, end in workspace.chunks(states):
            residual = states[:, start:end]
            normalized = normalize(residual, stream)
            positions = rotary_slice(rotary, start, end) if stream == 0 else None
            query = project(query_proj, query_norm, normalized, attn.heads, positions)
            attended = workspace.attention(query, mask)
            del query
            if single:
                mlp = block.act_mlp(block.proj_mlp(normalized))
                result = residual + parameters[stream][2][:, None] * block.proj_out(torch.cat((attended, mlp), dim=-1))
                del mlp
            else:
                projection = attn.to_out if stream == 0 else None
                attended = (
                    projection[1](projection[0](attended)) if projection is not None else attn.to_add_out(attended)
                )
                _, _, gate, shift_mlp, scale_mlp, gate_mlp = parameters[stream]
                result = residual + attended * gate[:, None]
                norm = block.norm2 if stream == 0 else block.norm2_context
                ff = block.ff if stream == 0 else block.ff_context
                ff_input = norm(result) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
                result = result + gate_mlp[:, None] * ff(ff_input)
                del ff_input
            output[:, start:end].copy_(result)
            del normalized, attended, result
        outputs.append(output)
    return tuple(outputs)


def cogvideox_chunk_forward(block, hidden_states, encoder_hidden_states, temb, rotary, workspace):
    attn = block.attn1
    text_length = encoder_hidden_states.shape[1]
    length = text_length + hidden_states.shape[1]
    streams = ((encoder_hidden_states, 0), (hidden_states, text_length))
    parameters1 = block.norm1.linear(block.norm1.silu(temb)).chunk(6, dim=1)
    parameters2 = block.norm2.linear(block.norm2.silu(temb)).chunk(6, dim=1)

    def normalize(states, stream, norm, parameters):
        shift, scale, gate = parameters[3:] if stream == 0 else parameters[:3]
        return norm.norm(states) * (1 + scale[:, None]) + shift[:, None], gate[:, None]

    for stream, (states, offset) in enumerate(streams):
        for start, end in workspace.chunks(states):
            normalized, _ = normalize(states[:, start:end], stream, block.norm1, parameters1)
            positions = rotary_slice(rotary, start, end) if stream == 1 else None
            key = project(attn.to_k, attn.norm_k, normalized, attn.heads, positions)
            value = project(attn.to_v, None, normalized, attn.heads)
            workspace.write(key, value, offset + start, length)
            del key, value, normalized

    outputs = []
    for stream, (states, offset) in enumerate(streams):
        output = torch.empty_like(states)
        for start, end in workspace.chunks(states):
            residual = states[:, start:end]
            normalized, gate = normalize(residual, stream, block.norm1, parameters1)
            positions = rotary_slice(rotary, start, end) if stream == 1 else None
            query = project(attn.to_q, attn.norm_q, normalized, attn.heads, positions)
            attended = workspace.attention(query)
            del query, normalized
            attended = attn.to_out[1](attn.to_out[0](attended))
            result = residual + gate * attended
            normalized, gate_ff = normalize(result, stream, block.norm2, parameters2)
            output[:, start:end].copy_(result + gate_ff * block.ff(normalized))
            del normalized, attended, result
        outputs.append(output)
    return outputs[1], outputs[0]
