import tempfile
import weakref

import pytest
import torch

from diffusers import CogVideoXTransformer3DModel, HunyuanVideoTransformer3DModel
from diffusers.models.embeddings import get_3d_rotary_pos_embed


def tiny_model(kind, batch=2, device="cpu", dtype=torch.float32):
    torch.manual_seed(7)
    if kind == "hunyuan":
        model = HunyuanVideoTransformer3DModel(
            in_channels=4,
            out_channels=4,
            num_attention_heads=2,
            attention_head_dim=16,
            num_layers=2,
            num_single_layers=2,
            num_refiner_layers=1,
            patch_size=1,
            patch_size_t=1,
            guidance_embeds=True,
            text_embed_dim=16,
            pooled_projection_dim=8,
            rope_axes_dim=(4, 4, 8),
        )
        inputs = {
            "hidden_states": torch.randn(batch, 4, 1, 3, 4),
            "timestep": torch.ones(batch),
            "encoder_hidden_states": torch.randn(batch, 7, 16),
            "encoder_attention_mask": torch.tensor([[1] * 5 + [0] * 2] * batch),
            "pooled_projections": torch.randn(batch, 8),
            "guidance": torch.ones(batch),
        }
    else:
        model = CogVideoXTransformer3DModel(
            num_attention_heads=2,
            attention_head_dim=16,
            in_channels=4,
            out_channels=4,
            time_embed_dim=16,
            text_embed_dim=16,
            num_layers=2,
            sample_width=4,
            sample_height=4,
            sample_frames=5,
            patch_size=2,
            patch_size_t=2,
            temporal_compression_ratio=4,
            max_text_seq_length=7,
            use_rotary_positional_embeddings=True,
        )
        rotary = get_3d_rotary_pos_embed(
            embed_dim=16, crops_coords=None, grid_size=(2, 2), temporal_size=2, grid_type="slice", max_size=(2, 2)
        )
        inputs = {
            "hidden_states": torch.randn(batch, 4, 4, 4, 4),
            "timestep": torch.ones(batch),
            "encoder_hidden_states": torch.randn(batch, 7, 16),
            "image_rotary_emb": rotary,
        }
    if kind == "hunyuan" and batch > 1:
        inputs["encoder_attention_mask"][1, 4] = 0
    inputs = {
        k: tuple(x.to(device) for x in v)
        if isinstance(v, tuple)
        else v.to(device=device, dtype=dtype if v.is_floating_point() else v.dtype)
        for k, v in inputs.items()
    }
    return model.to(device=device, dtype=dtype).eval(), inputs


@pytest.mark.parametrize("kind", ["hunyuan", "cogvideo"])
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("size", [1, 3, 7, 12, 19, 64])
@torch.no_grad()
def test_equivalence(kind, batch, size):
    model, inputs = tiny_model(kind, batch)
    original = {k: v.clone() for k, v in inputs.items() if torch.is_tensor(v)}
    dense = model(**inputs).sample
    weights = {k: v.clone() for k, v in model.state_dict().items()}
    model.enable_chunk_dit(size)
    chunked = model(**inputs).sample
    torch.testing.assert_close(chunked, dense, atol=1e-5, rtol=1e-4)
    for k, v in original.items():
        torch.testing.assert_close(inputs[k], v, atol=0, rtol=0)
    for k, v in weights.items():
        torch.testing.assert_close(model.state_dict()[k], v, atol=0, rtol=0)
    torch.testing.assert_close(model(**inputs).sample, chunked, atol=0, rtol=0)
    model.disable_chunk_dit()
    torch.testing.assert_close(model(**inputs).sample, dense, atol=0, rtol=0)


@pytest.mark.parametrize("kind", ["hunyuan", "cogvideo"])
def test_guards_and_serialization(kind):
    model, inputs = tiny_model(kind)
    for value in (0, -1, 1.5, True):
        with pytest.raises(ValueError):
            model.enable_chunk_dit(value)
    model.enable_chunk_dit(3)
    with pytest.raises(ValueError, match="no_grad"):
        model(**inputs)
    with torch.no_grad():
        model.train()
        with pytest.raises(ValueError, match="eval"):
            model(**inputs)
        model.eval()
        expected = model(**inputs).sample
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            loaded = type(model).from_pretrained(directory).eval()
            assert loaded._chunk_dit_size is None
            torch.testing.assert_close(loaded(**inputs).sample, expected, atol=1e-5, rtol=1e-4)
        model._cache_config = object()
        with pytest.raises(ValueError, match="caching"):
            model(**inputs)
        model._cache_config = None
        torch.testing.assert_close(model(**inputs).sample, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("kind", ["hunyuan", "cogvideo"])
@torch.no_grad()
def test_cuda_bfloat16(kind):
    model, inputs = tiny_model(kind, device="cuda", dtype=torch.bfloat16)
    dense = model(**inputs).sample.float()
    model.enable_chunk_dit(3)
    chunked = model(**inputs).sample.float()
    assert ((chunked - dense).norm() / dense.norm()).item() <= 0.01
    assert torch.nn.functional.cosine_similarity(chunked.flatten(), dense.flatten(), dim=0).item() >= 0.999


@pytest.mark.parametrize("kind", ["hunyuan", "cogvideo"])
@torch.no_grad()
def test_processor_conflict_and_shape_change(kind):
    model, inputs = tiny_model(kind)
    model.enable_chunk_dit(3)
    blocks = model.transformer_blocks
    attention = blocks[0].attn if kind == "hunyuan" else blocks[0].attn1
    original = attention.processor
    attention.set_processor(object())
    with pytest.raises(ValueError, match="original attention"):
        model(**inputs)
    attention.set_processor(original)
    model(**inputs)
    _, smaller = tiny_model(kind, batch=1)
    result = model(**smaller).sample
    model.disable_chunk_dit()
    torch.testing.assert_close(result, model(**smaller).sample, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("kind", ["hunyuan", "cogvideo"])
@torch.no_grad()
def test_output_head_releases_kv_and_bounds_tokens(kind):
    model, inputs = tiny_model(kind)
    expected = model(**inputs, return_dict=False)[0]
    model.enable_chunk_dit(3)
    references = {}
    sizes = {"norm_out": [], "proj_out": []}
    if kind == "cogvideo":
        sizes["norm_final"] = []
    last = model.single_transformer_blocks[-1] if kind == "hunyuan" else model.transformer_blocks[-1]

    def capture_workspace(module, args, kwargs, output):
        workspace = kwargs["chunk_dit_workspace"]
        references["workspace"] = weakref.ref(workspace)
        references["key"] = weakref.ref(workspace.key)
        references["value"] = weakref.ref(workspace.value)

    def check_head(name):
        def check(module, args):
            workspace = references["workspace"]()
            assert workspace.key is None and workspace.value is None
            assert references["key"]() is None and references["value"]() is None
            sizes[name].append(args[0].shape[1])

        return check

    handles = [last.register_forward_hook(capture_workspace, with_kwargs=True)]
    for name in sizes:
        handles.append(getattr(model, name).register_forward_pre_hook(check_head(name)))
    try:
        actual = model(**inputs, return_dict=False)[0]
    finally:
        for handle in handles:
            handle.remove()
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
    assert references["workspace"]() is None
    assert sizes["proj_out"] == sizes["norm_out"]
    assert len(sizes["proj_out"]) > 1 and max(sizes["proj_out"]) <= 3
    if kind == "cogvideo":
        assert sizes["norm_final"] == sizes["norm_out"]
