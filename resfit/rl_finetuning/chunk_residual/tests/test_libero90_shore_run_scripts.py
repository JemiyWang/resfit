from pathlib import Path
import os
import subprocess


ROOT = Path(__file__).resolve().parents[4]
RUNNER = ROOT / "run_libero90_shore_task.sh"


def run(*args):
    return subprocess.run(
        ["bash", str(RUNNER), *map(str, args)],
        cwd=ROOT,
        env={**os.environ, "DRY_RUN": "1"},
        text=True,
        capture_output=True,
    )


def test_task57_train_dry_run_uses_task_local_inputs():
    p = run("train", 57, 2)
    assert p.returncode == 0, p.stderr
    assert "--libero_suite libero_90" in p.stdout
    assert "--libero_task_id 57" in p.stdout
    assert "converted/libero_90/task57/meta/stats.json" in p.stdout
    assert "libero90_task57_pi0_feat.npz" in p.stdout
    assert "libero90_task57_pi0_feat_gc_value.pt" in p.stdout
    assert "libero90_task57_pi0_feat_high_actor.pt" in p.stdout
    assert "CUDA_VISIBLE_DEVICES=2" in p.stdout


def test_all_supported_task_mappings():
    for task_id, gpu_id in [(57, 2), (60, 3), (63, 4), (64, 5)]:
        p = run("cache", task_id, gpu_id)
        assert p.returncode == 0, p.stderr
        assert f"task{task_id}" in p.stdout


def test_wrong_gpu_mapping_fails():
    p = run("train", 63, 5)
    assert p.returncode != 0
    assert "expected GPU4" in p.stderr


def test_unknown_mode_fails():
    p = run("unknown", 57, 2)
    assert p.returncode != 0
