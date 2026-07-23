import os

import pytest

from resfit.rl_finetuning.wm_bridge import fake_eval


def test_returns_constant_zero_success_rate(tmp_path, monkeypatch):
    monkeypatch.setattr(fake_eval, "save_checkpoint", lambda *a, **k: None)
    ev = fake_eval.make_imagination_evaluator(str(tmp_path), config=None)
    m = ev(env=None, agent=object(), num_episodes=1, device="cpu", global_step=0)
    assert m["eval/success_rate"] == 0.0


def test_saves_checkpoint_every_call(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(fake_eval, "save_checkpoint",
                        lambda agent, path, **k: calls.append((path, k)))
    ev = fake_eval.make_imagination_evaluator(str(tmp_path), config=None)
    ev(env=None, agent=object(), num_episodes=1, device="cpu", global_step=10)
    ev(env=None, agent=object(), num_episodes=1, device="cpu", global_step=20)
    assert len(calls) == 4                       # ★ 每次 eval 存滚动和按步持久 checkpoint
    assert [os.path.basename(path) for path, _ in calls] == [
        "imagination_last.pt",
        "imagination_step_0000010.pt",
        "imagination_last.pt",
        "imagination_step_0000020.pt",
    ]
    assert [kwargs["global_step"] for _, kwargs in calls] == [10, 10, 20, 20]


def test_tolerates_extra_kwargs(tmp_path, monkeypatch):
    monkeypatch.setattr(fake_eval, "save_checkpoint", lambda *a, **k: None)
    ev = fake_eval.make_imagination_evaluator(str(tmp_path), config=None)
    m = ev(env=None, agent=object(), num_episodes=1, device="cpu",
           global_step=0, save_video=False, save_q_plots=False,
           run_name="x", output_dir="y", subgoal=None, base_policy=None)
    assert m["eval/success_rate"] == 0.0


def test_creates_output_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(fake_eval, "save_checkpoint", lambda *a, **k: None)
    target = tmp_path / "nested" / "run"
    ev = fake_eval.make_imagination_evaluator(str(target), config=None)
    ev(env=None, agent=object(), num_episodes=1, device="cpu", global_step=0)
    assert os.path.isdir(str(target))
