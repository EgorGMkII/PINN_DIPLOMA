import json
from pathlib import Path
import sys
from types import SimpleNamespace

from diploma_pinn.instrumentation import StepProfiler
from diploma_pinn.integrations import JsonlTracker, build_wandb_tracker


def test_jsonl_tracker_persists_each_record(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    tracker = JsonlTracker(path)
    tracker.log({"loss/total": 1.25}, step=3)
    tracker.finish({"status": "ok"})
    assert json.loads(path.read_text(encoding="utf-8")) == {"step": 3, "loss/total": 1.25}


def test_cpu_profiler_records_named_phase() -> None:
    profiler = StepProfiler()
    with profiler.phase("sampling"):
        sum(range(10))
    summary = profiler.summary()
    assert summary["phases"]["sampling"]["count"] == 1
    assert summary["phases"]["sampling"]["mean_ms"] >= 0


def test_wandb_tracker_initializes_lazily_in_offline_mode(monkeypatch) -> None:
    calls = {}

    class Run:
        def __init__(self) -> None:
            self.summary = {}

        def log(self, values, step) -> None:
            calls["log"] = (values, step)

        def finish(self) -> None:
            calls["finished"] = True

    run = Run()
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(
        init=lambda **kwargs: calls.update(init=kwargs) or run
    ))
    tracker = build_wandb_tracker(
        mode="offline", project="project", group="group", run_name="run", config={"seed": 1}
    )
    tracker.log({"loss": 2.0}, step=4)
    tracker.finish({"evaluation": {"T": 0.9}})
    assert calls["init"]["mode"] == "offline"
    assert calls["log"] == ({"loss": 2.0}, 4)
    assert run.summary["evaluation/T"] == 0.9
    assert calls["finished"] is True
