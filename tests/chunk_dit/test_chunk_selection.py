import json
import math
import sys
import weakref
from pathlib import Path, PureWindowsPath

import pytest
import torch

from examples.chunk_dit.select_chunk_size import (
    aligned_candidates,
    fit_memory_model,
    maximum_chunk,
    predict_peak,
    profile_candidate,
    rank_candidates,
    workload_shape,
)


def test_production_workload_shapes():
    h = workload_shape(
        "hunyuan",
        {"patch_size": 2, "patch_size_t": 1, "num_attention_heads": 24, "attention_head_dim": 128},
        720,
        1280,
        129,
        1,
        256,
    )
    c = workload_shape(
        "cogvideo",
        {"patch_size": 2, "patch_size_t": 2, "num_attention_heads": 48, "attention_head_dim": 64},
        768,
        1360,
        81,
        2,
        226,
    )
    assert h["video_tokens"] == 118800 and h["latent_frames"] == 33
    assert c["video_tokens"] == 44880 and c["latent_frames"] == 22
    assert h["hidden_dim"] == c["hidden_dim"] == 3072
    assert h["total_tokens"] == 119056 and c["total_tokens"] == 45106


@pytest.mark.parametrize(
    "height,width,frames,batch,text",
    [(0, 1280, 129, 1, 256), (721, 1280, 129, 1, 256), (720, 1280, 0, 1, 256), (720, 1280, 129, 0, 256)],
)
def test_invalid_workload(height, width, frames, batch, text):
    with pytest.raises(ValueError):
        workload_shape(
            "hunyuan",
            {"patch_size": 2, "num_attention_heads": 24, "attention_head_dim": 128},
            height,
            width,
            frames,
            batch,
            text,
        )


def test_alignment_and_full_stream_boundary():
    assert aligned_candidates(1100, 2048, 256, 5000) == [1280, 1536, 1792, 2048]
    assert aligned_candidates(128, 2048, 128, 257) == [128, 256, 384]
    assert aligned_candidates(1100, 1200, 256, 800) == []
    with pytest.raises(ValueError):
        aligned_candidates(2048, 1024, 256, 5000)


def test_calibration_includes_fixed_hidden_states():
    weights, kv, fixed, token_bytes, alpha = 1000, 200, 400, 8, 12
    points = [
        {"chunk_size": c, "peak_allocated": weights + kv + fixed + alpha * token_bytes * c} for c in (128, 256, 512)
    ]
    fit = fit_memory_model(points, weights, kv, token_bytes)
    assert fit["fixed_bytes"] == pytest.approx(fixed) and fit["alpha"] == pytest.approx(alpha)
    assert maximum_chunk(fit, 1600 + 96 * 300, 1000) == 300
    assert maximum_chunk(fit, 1599, 1000) == 0
    assert maximum_chunk(fit, 10**9, 512) == 512


def test_fit_covers_noisy_observed_points_and_zero_slope():
    points = [
        {"chunk_size": 128, "peak_allocated": 2000},
        {"chunk_size": 256, "peak_allocated": 2300},
        {"chunk_size": 512, "peak_allocated": 2500},
    ]
    fit = fit_memory_model(points, 1000, 200, 8)
    assert all(predict_peak(fit, p["chunk_size"]) >= p["peak_allocated"] - 1e-9 for p in points)
    flat = fit_memory_model([{"chunk_size": 128, "peak_allocated": 1800}], 1000, 200, 8)
    assert flat["alpha"] == 0 and maximum_chunk(flat, 1800, 128) == 128
    assert maximum_chunk(flat, 1799, 128) == 0
    with pytest.raises(ValueError):
        fit_memory_model([], 1000, 200, 8)


def test_fastest_feasible_is_not_assumed_to_be_largest():
    candidates = [
        {"chunk_size": 1024, "status": "ok", "peak_allocated": 20, "estimated_dit_seconds": 5.0},
        {"chunk_size": 2048, "status": "ok", "peak_allocated": 24, "estimated_dit_seconds": 3.0},
        {"chunk_size": 4096, "status": "ok", "peak_allocated": 30, "estimated_dit_seconds": 4.0},
        {"chunk_size": 8192, "status": "oom"},
        {"chunk_size": 16384, "status": "pruned_by_memory_model"},
    ]
    assert rank_candidates(candidates, 30)[0]["chunk_size"] == 2048
    assert rank_candidates(candidates, 23)[0]["chunk_size"] == 1024
    assert rank_candidates(candidates, 19) == []
    fit = fit_memory_model([{"chunk_size": 1, "peak_allocated": 10}, {"chunk_size": 2, "peak_allocated": 8}], 1, 1, 1)
    assert fit["alpha"] == 0 and math.isfinite(predict_peak(fit, 3))


def test_oom_is_recorded_and_kv_is_released(monkeypatch):
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    references = []

    def exhausted_block(**kwargs):
        workspace = kwargs["chunk_dit_workspace"]
        workspace.key = torch.ones(2)
        workspace.value = torch.ones(2)
        references.extend([weakref.ref(workspace.key), weakref.ref(workspace.value)])
        raise torch.cuda.OutOfMemoryError("Test allocation failure")

    result = profile_candidate([("block", exhausted_block, 1, {})], 2048, 1, 1)
    assert result["status"] == "oom"
    assert result["blocks"]["block"]["status"] == "oom"
    assert all(reference() is None for reference in references)


@pytest.fixture
def selector_cli(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from examples.chunk_dit import select_chunk_size as selector

    class Model:
        loads = 0
        forwards = 0
        enabled = []

        @staticmethod
        def load_config(*args, **kwargs):
            return {"patch_size": 2, "patch_size_t": 1, "num_attention_heads": 24, "attention_head_dim": 128}

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            cls.loads += 1
            return cls()

        def eval(self):
            return self

        def _validate_chunk_dit(self):
            pass

        def parameters(self):
            return [torch.zeros(1, dtype=torch.bfloat16)]

        def buffers(self):
            return []

        def to(self, *args):
            return self

        def enable_chunk_dit(self, chunk):
            type(self).enabled.append(chunk)

        def __call__(self, **kwargs):
            type(self).forwards += 1
            return torch.zeros(1)

    model_dir = tmp_path / "weights"
    (model_dir / "transformer").mkdir(parents=True)
    (model_dir / "transformer" / "config.json").write_text("{}")
    output = tmp_path / "result.json"
    monkeypatch.setattr(selector, "HunyuanVideoTransformer3DModel", Model)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (16 * selector.GIB, 16 * selector.GIB))
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(name="Test GPU", major=9, minor=0, multi_processor_count=132),
    )
    monkeypatch.setattr(torch.cuda, "get_allocator_backend", lambda: "native")
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch, "manual_seed", lambda _: None)
    monkeypatch.setattr(
        selector, "make_inputs", lambda *args: ({}, [("double", None, 20, {}), ("single", None, 40, {})])
    )

    def profile(specs, chunk, *args):
        return {
            "chunk_size": chunk,
            "status": "ok",
            "peak_allocated": 4 * selector.GIB,
            "estimated_dit_seconds": 6.0,
            "blocks": {
                name: {"status": "ok", "peak_allocated": 4 * selector.GIB, "median_seconds": 0.1, "layer_count": count}
                for name, _, count, _ in specs
            },
        }

    def measure(call, warmups, repeats):
        for _ in range(warmups + repeats):
            call()
        return {"status": "ok", "median_seconds": 7.0, "peak_allocated": 4 * selector.GIB}

    monkeypatch.setattr(selector, "profile_candidate", profile)
    monkeypatch.setattr(selector, "measure", measure)

    def run(*extra):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "select_chunk_size.py",
                "--model",
                "hunyuan",
                "--model-path",
                str(model_dir),
                "--memory-budget-gib",
                "10",
                "--min-chunk-size",
                "1024",
                "--max-chunk-size",
                "1024",
                "--output",
                str(output),
                *extra,
            ],
        )
        selector.main()
        return json.loads(output.read_text())

    return run, Model, output


def test_default_selection_and_cache_never_call_complete_model(selector_cli, capsys):
    run, model, _ = selector_cli
    result = run()
    assert model.forwards == 0 and model.loads == 1
    assert result["validations"] == [] and result["full_forward_validated"] is False
    assert result["validation_mode"] == "block_only" and result["estimated_dit_seconds"] == 6.0
    assert "verified_peak_allocated" not in result and "verified_dit_seconds" not in result
    assert "estimated_DiT=" in capsys.readouterr().out
    run()
    assert model.forwards == 0 and model.loads == 1
    assert "Cache hit:" in capsys.readouterr().out


def test_validation_is_opt_in_and_cache_modes_are_separate(selector_cli):
    run, model, _ = selector_cli
    run()
    result = run("--validate-forward")
    assert model.loads == 2 and model.forwards == 2
    assert result["full_forward_validated"] is True and result["validation_mode"] == "full_forward"
    assert result["verified_dit_seconds"] == 7.0 and len(result["validations"]) == 1
    run("--validate-forward")
    assert model.loads == 2 and model.forwards == 2
    run()
    assert model.loads == 3 and model.forwards == 2


def test_removed_latency_option_is_rejected(selector_cli):
    run, model, _ = selector_cli
    with pytest.raises(SystemExit) as error:
        run("--max-dit-seconds", "35")
    assert error.value.code == 2 and model.forwards == 0 and model.loads == 0


def test_optional_validation_enforces_memory_budget(selector_cli, monkeypatch):
    from examples.chunk_dit import select_chunk_size as selector

    run, model, _ = selector_cli

    def oversized(call, warmups, repeats):
        for _ in range(warmups + repeats):
            call()
        return {"status": "ok", "median_seconds": 7.0, "peak_allocated": 10 * selector.GIB}

    monkeypatch.setattr(selector, "measure", oversized)
    with pytest.raises(SystemExit, match="full-forward"):
        run("--validate-forward")
    assert model.forwards == 2


def assert_private_metadata_absent(document, private_values):
    serialized = json.dumps(document)
    assert all(value not in serialized for value in private_values)
    if isinstance(document, dict):
        for value in document.values():
            assert_private_metadata_absent(value, private_values)
    elif isinstance(document, list):
        for value in document:
            assert_private_metadata_absent(value, private_values)
    elif isinstance(document, str):
        assert not Path(document).is_absolute()
        assert not PureWindowsPath(document).is_absolute()


def test_selector_json_is_private_and_environment_changes_invalidate_cache(selector_cli, monkeypatch):
    run, model, output = selector_cli
    private_path = str(output.parent / "private-checkpoint")
    private_setting = "private-allocator-setting"
    config = model.load_config()
    config["local_metadata"] = {"checkpoint": private_path, "setting": private_setting}
    monkeypatch.setattr(model, "load_config", staticmethod(lambda *args, **kwargs: config))
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", private_setting)
    writes = []
    write_text = Path.write_text

    def record_write(path, text, *args, **kwargs):
        if path == output:
            writes.append(json.loads(text))
        return write_text(path, text, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", record_write)
    result = run()
    run()
    assert model.loads == 1
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", "changed-private-setting")
    changed_environment = run()
    assert model.loads == 2 and result["cache_key"] != changed_environment["cache_key"]
    config["local_metadata"]["setting"] = "changed-checkpoint-setting"
    changed_checkpoint = run()
    assert model.loads == 3 and changed_checkpoint["cache_key"] != changed_environment["cache_key"]
    run()
    assert model.loads == 3
    assert any(row["status"] == "running" for row in writes)
    assert any(row["status"] == "passed" for row in writes)
    for document in writes:
        assert_private_metadata_absent(
            document,
            [str(output.parent), private_setting, "changed-private-setting", "changed-checkpoint-setting", "Test GPU"],
        )
        assert set(document["identity"]) == {"model_type", "dtype", "workload", "constraints", "sampling"}


def test_oom_json_does_not_include_backend_error_details(monkeypatch, tmp_path):
    from examples.chunk_dit.select_chunk_size import measure

    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    private_detail = str(tmp_path / "private-runtime.conf")

    def exhausted():
        raise torch.cuda.OutOfMemoryError(private_detail)

    result = measure(exhausted, 0, 1)
    assert result["status"] == "oom"
    assert_private_metadata_absent(result, [private_detail])


def test_benchmark_json_excludes_paths_commands_and_gpu_environment(monkeypatch, tmp_path):
    from examples.chunk_dit import benchmark_chunk_selection as benchmark

    popen = benchmark.subprocess.Popen
    child = """
import json
import sys
from pathlib import Path

output = Path(sys.argv[sys.argv.index('--output') + 1])
if '--force' in sys.argv:
    output.write_text(json.dumps({
        'status': 'passed', 'chunk_size': 1024,
        'identity': {'constraints': {'grid': [1024]}},
        'candidates': [{'status': 'ok', 'blocks': {'double': {'measurements': [{}]}}}],
        'validations': [],
    }))
    print('Calibrating C=1024', flush=True)
    print('Selected chunk_size=1024', flush=True)
else:
    print('Cache hit: chunk_size=1024', flush=True)
"""
    monkeypatch.setattr(
        benchmark.subprocess,
        "Popen",
        lambda command, **kwargs: popen([sys.executable, "-c", child, *command[3:]], **kwargs),
    )
    private_gpu = "private-device-selection"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", private_gpu)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_chunk_selection.py",
            "--model",
            "hunyuan",
            "--model-path",
            str(tmp_path / "private-checkpoint"),
            "--memory-budget-gib",
            "80",
            "--runs",
            "1",
            "--output",
            str(tmp_path / "benchmark"),
        ],
    )
    benchmark.main()
    documents = {
        path.relative_to(tmp_path).as_posix(): json.loads(path.read_text()) for path in tmp_path.rglob("*.json")
    }
    assert len(documents) == 4
    for document in documents.values():
        assert_private_metadata_absent(document, [str(tmp_path), sys.executable, private_gpu])
    summary = documents["benchmark/summary.json"]
    assert summary["status"] == "passed" and summary["runs"][0]["chunk_size"] == 1024
    assert "model_path" not in summary["arguments"] and "output" not in summary["arguments"]
    for name in ("benchmark/run-1/timing.json", "benchmark/cache-hit/timing.json"):
        timing = documents[name]
        assert "command" not in timing and "physical_gpu" not in timing
        assert len(timing["command_fingerprint"]) == 64
