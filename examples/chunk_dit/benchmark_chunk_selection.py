"""Time the selector in fresh processes, separating block search from full-forward validation."""

import argparse
import hashlib
import json
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path


def run_once(command, directory):
    directory.mkdir(parents=True, exist_ok=True)
    events = []
    started = time.perf_counter()
    with (directory / "process.txt").open("w") as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            log.write(line)
            log.flush()
            match = re.search(r"(Calibrating|Profiling|Validating complete Transformer) C=(\d+)", line)
            stage = (
                {
                    "Calibrating": "calibration",
                    "Profiling": "search",
                    "Validating complete Transformer": "full_validation",
                }.get(match[1])
                if match
                else None
            )
            if stage:
                events.append({"stage": stage, "chunk_size": int(match[2]), "seconds": time.perf_counter() - started})
                print(f"{directory.name}: {line.strip()} at {events[-1]['seconds']:.2f}s", flush=True)
            elif line.startswith(("Selected chunk_size=", "Cache hit:")):
                events.append({"stage": "finish", "seconds": time.perf_counter() - started})
                print(f"{directory.name}: {line.strip()}", flush=True)
        code = process.wait()
    total = time.perf_counter() - started
    if code:
        raise RuntimeError(f"Selector exited with {code}; see {directory / 'process.txt'}")
    phases = {
        "startup_and_load": events[0]["seconds"],
        "calibration": 0.0,
        "search": 0.0,
        "full_validation": 0.0,
        "shutdown": total - events[-1]["seconds"],
    }
    for current, following in zip(events, events[1:]):
        phases[current["stage"]] += following["seconds"] - current["seconds"]
    timing = {
        "command_fingerprint": hashlib.sha256(json.dumps(command).encode()).hexdigest(),
        "wall_seconds": total,
        "phase_wall_seconds": phases,
        "progress_events": events,
    }
    (directory / "timing.json").write_text(json.dumps(timing, indent=2))
    return timing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["hunyuan", "cogvideo"], required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--memory-budget-gib", type=float, required=True)
    parser.add_argument("--reserve-gib", type=float, default=1.0)
    parser.add_argument("--min-chunk-size", type=int, default=1024)
    parser.add_argument("--max-chunk-size", type=int, default=8192)
    parser.add_argument("--alignment", type=int, default=256)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--validate-forward", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("runs must be positive")
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    selector = Path(__file__).with_name("select_chunk_size.py")
    base = [
        sys.executable,
        "-u",
        str(selector),
        "--model",
        args.model,
        "--model-path",
        args.model_path,
        "--memory-budget-gib",
        str(args.memory_budget_gib),
        "--reserve-gib",
        str(args.reserve_gib),
        "--min-chunk-size",
        str(args.min_chunk_size),
        "--max-chunk-size",
        str(args.max_chunk_size),
        "--alignment",
        str(args.alignment),
    ]
    if args.validate_forward:
        base.append("--validate-forward")
    summary = {
        "model": args.model,
        "scope": "fresh Python process wall time, local checkpoint; downloads excluded",
        "phase_method": "parent timestamps unbuffered stdout markers; phase intervals include orchestration/cleanup",
        "arguments": {
            key: getattr(args, key)
            for key in (
                "model",
                "memory_budget_gib",
                "reserve_gib",
                "min_chunk_size",
                "max_chunk_size",
                "alignment",
                "runs",
                "validate_forward",
            )
        },
        "runs": [],
        "status": "running",
    }
    for index in range(args.runs):
        directory = root / f"run-{index + 1}"
        result_path = directory / "selection.json"
        timing = run_once([*base, "--output", str(result_path), "--force"], directory)
        selection = json.loads(result_path.read_text())
        assert selection["status"] == "passed"
        measured = [row for row in selection["candidates"] if row["status"] == "ok"]
        timing.update(
            chunk_size=selection["chunk_size"],
            grid_count=len(selection["identity"]["constraints"]["grid"]),
            profiled_candidates=len(measured),
            pruned_candidates=sum(row["status"] == "pruned_by_memory_model" for row in selection["candidates"]),
            block_forward_calls=sum(len(b["measurements"]) for row in measured for b in row["blocks"].values()),
            full_forward_calls=sum(len(v.get("measurements", [])) for v in selection["validations"]),
        )
        summary["runs"].append(timing)
        (root / "summary.json").write_text(json.dumps(summary, indent=2))
    cached = run_once([*base, "--output", str(result_path)], root / "cache-hit")
    assert "Cache hit:" in (root / "cache-hit" / "process.txt").read_text()
    assert not any(e["stage"] != "finish" for e in cached["progress_events"])
    summary.update(
        status="passed",
        cache_hit_wall_seconds=cached["wall_seconds"],
        median_wall_seconds=statistics.median(run["wall_seconds"] for run in summary["runs"]),
        median_phase_wall_seconds={
            name: statistics.median(run["phase_wall_seconds"][name] for run in summary["runs"])
            for name in summary["runs"][0]["phase_wall_seconds"]
        },
    )
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    print(
        json.dumps(
            {
                k: summary[k]
                for k in ("model", "median_wall_seconds", "median_phase_wall_seconds", "cache_hit_wall_seconds")
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
