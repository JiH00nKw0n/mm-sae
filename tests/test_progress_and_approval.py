import json
import time

import pytest

from mm_sae.config import Config
from mm_sae.io import RunStore, atomic_json
from mm_sae.progress import ProgressReporter, TaskProgress, progress_task, read_status, stage_progress


def test_eta_uses_work_since_resume_and_does_not_invent_whole_run_eta(tmp_path, monkeypatch):
    monkeypatch.setattr("mm_sae.progress.time.monotonic", lambda: 20.0)
    task = TaskProgress("resumed download", total=100, unit="bytes", initial=40, completed=60, started=10)
    row = task.snapshot()
    assert row["units_per_second"] == 2
    assert row["eta_seconds"] == 20
    with ProgressReporter(tmp_path, ["prepare", "train"], [], interval=0.01):
        with stage_progress("prepare"):
            with progress_task("read images", total=10) as meter:
                meter.update(3)
                time.sleep(0.04)
                status = read_status(tmp_path)
                assert status["tasks"][0]["completed"] == 3
                assert status["whole_run_eta_seconds"] is None
    assert read_status(tmp_path)["state"] == "paused"
    assert read_status(tmp_path)["stages_completed"] == ["prepare"]


def test_failed_task_is_not_reported_as_completed(tmp_path):
    with pytest.raises(ValueError, match="bad input"):
        with ProgressReporter(tmp_path, ["train"], [], interval=0.01):
            with stage_progress("train"), progress_task("image SAE", 10):
                raise ValueError("bad input")
    status = read_status(tmp_path)
    assert status["state"] == "failed"
    assert status["stages_completed"] == []
    assert status["error"] == "bad input"


def test_approval_is_required_and_bound_to_exact_config_and_review(tmp_path):
    document = tmp_path / "review.md"
    document.write_text("Original experiment definition")
    receipt = tmp_path / "approval.json"
    config = Config.model_validate(
        {
            "output": str(tmp_path / "run"),
            "execution": {
                "require_approval": True,
                "approval_file": str(receipt),
                "review_document": str(document),
            },
        }
    )
    store = RunStore(config)
    with pytest.raises(PermissionError, match="not been approved"):
        store.require_approval()
    # A synthetic approval exists only in this pytest temporary directory.
    atomic_json(receipt, {"decision": "approved", "signature": store.signature})
    store.require_approval()
    config.training.arguments["seed"] = 99
    with pytest.raises(PermissionError, match="does not match"):
        RunStore(config).require_approval()
    config.training.arguments["seed"] = 0
    document.write_text("Changed experiment definition")
    with pytest.raises(PermissionError, match="does not match"):
        RunStore(config).require_approval()
    assert json.loads(receipt.read_text())["signature"] == store.signature


def test_heartbeat_survives_a_partially_imported_torch_module(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace())
    with ProgressReporter(tmp_path, ['probe'], [], interval=.01):
        with stage_progress('probe'), progress_task('fit', 2) as task:
            task.update(1)
            time.sleep(.04)
            status = read_status(tmp_path)
            assert status['tasks'][0]['completed'] == 1
            assert status['gpu_allocator'] is None
    assert read_status(tmp_path)['state'] == 'completed'
