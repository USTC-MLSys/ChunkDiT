import pytest
import torch

from diffusers.models.autoencoders.memory_efficient_decode import chunked_conv3d


@pytest.mark.parametrize("chunk_size", [1, 2, 4, 16])
@pytest.mark.parametrize("stride,dilation,groups,bias", [(1, 1, 1, True), (2, 1, 2, False), (1, 2, 2, True)])
@torch.no_grad()
def test_temporal_convolution(chunk_size, stride, dilation, groups, bias):
    torch.manual_seed(42)
    conv = torch.nn.Conv3d(
        4, 6, (3, 3, 3), stride=(stride, 1, 1), padding=(0, 1, 1), dilation=(dilation, 1, 1), groups=groups, bias=bias
    ).eval()
    inputs = torch.randn(2, 4, 15, 7, 9)
    expected = conv(inputs)
    actual = chunked_conv3d(conv, inputs, chunk_size, min_input_bytes=0)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("kind", ["hunyuan", "cogvideo"])
def test_vae_runtime_configuration(kind, tmp_path):
    from diffusers import AutoencoderKLCogVideoX, AutoencoderKLHunyuanVideo

    cls = AutoencoderKLHunyuanVideo if kind == "hunyuan" else AutoencoderKLCogVideoX
    model = cls(block_out_channels=(8, 8, 8, 8), latent_channels=4, layers_per_block=1, norm_num_groups=2).eval()
    weights = {k: v.clone() for k, v in model.state_dict().items()}
    for value in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            model.enable_memory_efficient_decode(value)
    model.enable_memory_efficient_decode(4)
    assert all(
        getattr(m, "_decode_conv_chunk_size", None) == 4
        for m in model.decoder.modules()
        if isinstance(m, torch.nn.Conv3d)
    )
    assert not any(getattr(m, "_decode_conv_chunk_size", None) for m in model.encoder.modules())
    for k, v in weights.items():
        torch.testing.assert_close(model.state_dict()[k], v, atol=0, rtol=0)
    model.save_pretrained(tmp_path)
    restored = cls.from_pretrained(tmp_path).eval()
    assert not any(getattr(m, "_decode_conv_chunk_size", None) for m in restored.decoder.modules())
    with pytest.raises(ValueError, match="no_grad"):
        model.decode(torch.randn(1, 4, 3, 2, 2))
    model.disable_memory_efficient_decode()
    assert not any(getattr(m, "_decode_conv_chunk_size", None) for m in model.decoder.modules())


def test_layer_offload_guard():
    conv = torch.nn.Conv3d(2, 2, 3).eval()
    conv._hf_hook = object()
    with torch.no_grad(), pytest.raises(ValueError, match="resident weights"):
        chunked_conv3d(conv, torch.randn(1, 2, 5, 5, 5), 2, min_input_bytes=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("kind", ["hunyuan", "cogvideo"])
@torch.no_grad()
def test_cuda_model_offload(kind):
    from accelerate import cpu_offload_with_hook

    from diffusers import AutoencoderKLCogVideoX, AutoencoderKLHunyuanVideo

    cls = AutoencoderKLHunyuanVideo if kind == "hunyuan" else AutoencoderKLCogVideoX
    model = (
        cls(block_out_channels=(8, 8, 8, 8), latent_channels=4, layers_per_block=1, norm_num_groups=2)
        .eval()
        .to(device="cuda", dtype=torch.bfloat16)
    )
    latent = torch.randn(1, 4, 3, 2, 2, device="cuda", dtype=torch.bfloat16)
    expected = model.decode(latent).sample
    model.enable_memory_efficient_decode(4)
    _, hook = cpu_offload_with_hook(model, execution_device=torch.device("cuda"))
    actual = model.decode(latent).sample
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    hook.offload()
