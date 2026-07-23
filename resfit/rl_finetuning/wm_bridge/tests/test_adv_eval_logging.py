import json
import os

import numpy as np

from resfit.rl_finetuning.wm_bridge import fake_eval


class _StubAdvScorer:
    """逐帧打分:第 k 集第 t 帧 → 0.1*episode + 0.01*t,便于断言写出的值。"""
    def __init__(self):
        self.ep = 0
    def score_frames(self, frames):
        T = frames.shape[0] if hasattr(frames, "shape") else len(frames)
        vals = np.array([0.1 * self.ep + 0.01 * t for t in range(T)], np.float32)
        self.ep += 1
        return vals


class _StubEvalEnv:
    """Minimal evaluation environment passed through to the rollout helper."""


def _pred_frames():
    return np.zeros((50, 3, 3, 8, 8), np.float32)


def test_no_adv_scorer_is_bitwise_task7(tmp_path, monkeypatch):
    """adv_scorer=None 时行为与 Task 7 一致:只存 checkpoint,不写 jsonl。"""
    monkeypatch.setattr(fake_eval, "save_checkpoint", lambda *a, **k: None)
    ev = fake_eval.make_imagination_evaluator(str(tmp_path), config=None)
    m = ev(env=None, agent=object(), num_episodes=1, device="cpu", global_step=0)
    assert m["eval/success_rate"] == 0.0
    assert not os.path.exists(tmp_path / "imagined_adv_eval.jsonl")


def test_logs_raw_values_per_episode(tmp_path, monkeypatch):
    monkeypatch.setattr(fake_eval, "save_checkpoint", lambda *a, **k: None)
    monkeypatch.setattr(fake_eval, "_rollout_pred_frames",
                        lambda env, agent, max_segments: _pred_frames())
    ev = fake_eval.make_imagination_evaluator(
        str(tmp_path), config=None,
        adv_scorer=_StubAdvScorer(), n_eval_episodes=3)
    ev(env=_StubEvalEnv(), agent=object(), num_episodes=None, device="cpu", global_step=50000)
    rows = [json.loads(l) for l in open(tmp_path / "imagined_adv_eval.jsonl")]
    assert len(rows) == 3
    assert all(r["env_step"] == 50000 for r in rows)
    # 第 1 集 traj = [0.0, 0.01, ...]; final = 0.49, max = 0.49
    assert rows[0]["episode_idx"] == 0
    assert abs(rows[0]["adv_final"] - 0.49) < 1e-4
    assert abs(rows[0]["adv_max"] - 0.49) < 1e-4
    assert abs(rows[0]["adv_mean"] - 0.245) < 1e-4
    assert len(rows[0]["adv_traj"]) == 50


def test_no_success_failure_classification_in_log(tmp_path, monkeypatch):
    """★ 只存原始值,绝不写 success/failure 字段(阈值留给用户离线定)。"""
    monkeypatch.setattr(fake_eval, "save_checkpoint", lambda *a, **k: None)
    monkeypatch.setattr(fake_eval, "_rollout_pred_frames",
                        lambda env, agent, max_segments: _pred_frames())
    ev = fake_eval.make_imagination_evaluator(
        str(tmp_path), config=None,
        adv_scorer=_StubAdvScorer(), n_eval_episodes=1)
    ev(env=_StubEvalEnv(), agent=object(), num_episodes=None, device="cpu", global_step=0)
    row = json.loads(open(tmp_path / "imagined_adv_eval.jsonl").readline())
    assert "success" not in row and "is_success" not in row and "threshold" not in row


def test_appends_across_eval_points(tmp_path, monkeypatch):
    """训练中断不丢已存:每个 eval 点追加,不覆盖。"""
    monkeypatch.setattr(fake_eval, "save_checkpoint", lambda *a, **k: None)
    monkeypatch.setattr(fake_eval, "_rollout_pred_frames",
                        lambda env, agent, max_segments: _pred_frames())
    ev = fake_eval.make_imagination_evaluator(
        str(tmp_path), config=None,
        adv_scorer=_StubAdvScorer(), n_eval_episodes=2)
    ev(env=_StubEvalEnv(), agent=object(), num_episodes=None, device="cpu", global_step=50000)
    ev(env=_StubEvalEnv(), agent=object(), num_episodes=None, device="cpu", global_step=100000)
    rows = [json.loads(l) for l in open(tmp_path / "imagined_adv_eval.jsonl")]
    assert len(rows) == 4
    assert {r["env_step"] for r in rows} == {50000, 100000}
