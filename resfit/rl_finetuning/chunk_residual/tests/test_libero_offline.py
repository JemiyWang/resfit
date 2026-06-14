import io
import json as _json

import numpy as np
import pytest

from resfit.rl_finetuning.chunk_residual.libero_offline import libero_task_language


def test_libero_task_language_moka():
    pytest.importorskip("libero")   # 仅此测试需真 libero;residual env / 无 PYTHONPATH 时优雅跳过
    lang = libero_task_language("libero_10", 8)
    assert lang == "put both moka pots on the stove"


def _make_stub_root(tmp_path, episodes):
    """episodes: list[(episode_index, language, length)] → 建 meta/episodes.jsonl + 空 parquet 占位。"""
    root = tmp_path / "ds"
    (root / "meta").mkdir(parents=True)
    with open(root / "meta" / "episodes.jsonl", "w") as f:
        for ei, lang, length in episodes:
            f.write(_json.dumps({"episode_index": ei, "tasks": [lang], "length": length}) + "\n")
    for ei, _, _ in episodes:
        d = root / "data" / f"chunk-{ei // 1000:03d}"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"episode_{ei:06d}.parquet").write_text("stub")
    return str(root)


def test_find_demo_episodes_filters_by_language(tmp_path):
    from resfit.rl_finetuning.chunk_residual.libero_offline import find_demo_episodes
    root = _make_stub_root(tmp_path, [(0, "task A", 5), (1, "task B", 7), (1001, "task A", 9)])
    paths = find_demo_episodes(root, "task A")
    assert len(paths) == 2
    assert paths[0].endswith("data/chunk-000/episode_000000.parquet")
    assert paths[1].endswith("data/chunk-001/episode_001001.parquet")


def test_find_demo_episodes_no_match_raises(tmp_path):
    from resfit.rl_finetuning.chunk_residual.libero_offline import find_demo_episodes
    import pytest
    root = _make_stub_root(tmp_path, [(0, "task A", 5)])
    with pytest.raises(ValueError):
        find_demo_episodes(root, "nonexistent task")


def _make_demo_parquet(path, T=3):
    import pandas as pd
    from PIL import Image
    def jpg(rgb):
        buf = io.BytesIO(); Image.fromarray(rgb).save(buf, format="JPEG"); return {"bytes": buf.getvalue(), "path": None}
    rows = []
    for t in range(T):
        av = np.full((256, 256, 3), t * 10, np.uint8)
        wr = np.full((256, 256, 3), t * 5, np.uint8)
        rows.append({"image": jpg(av), "wrist_image": jpg(wr),
                     "state": np.full(8, float(t), np.float32), "actions": np.full(7, float(t) * 0.1, np.float32),
                     "episode_index": 0})
    pd.DataFrame(rows).to_parquet(path)


def test_read_libero_demo_shapes(tmp_path):
    from resfit.rl_finetuning.chunk_residual.libero_offline import read_libero_demo
    p = tmp_path / "episode_000000.parquet"
    _make_demo_parquet(str(p), T=4)
    d = read_libero_demo(str(p))
    assert d["state"].shape == (4, 8) and d["state"].dtype == np.float32
    assert d["action"].shape == (4, 7) and d["action"].dtype == np.float32
    assert d["agentview"].shape == (4, 256, 256, 3) and d["agentview"].dtype == np.uint8
    assert d["wrist"].shape == (4, 256, 256, 3) and d["wrist"].dtype == np.uint8


class _IdScaler:
    def scale(self, x):  # 恒等,便于断言值穿透
        return x

class _IdStd:
    def standardize(self, x):
        return x


def _toy_demo(T=3):
    # agentview 上半亮(255)下半暗(0):用来验"没被上下翻转"
    av = np.zeros((T, 256, 256, 3), np.uint8); av[:, :128] = 255
    wr = np.zeros((T, 256, 256, 3), np.uint8)
    return {"state": np.arange(T * 8, dtype=np.float32).reshape(T, 8),
            "action": np.arange(T * 7, dtype=np.float32).reshape(T, 7) * 0.1,
            "agentview": av, "wrist": wr}


def test_demo_to_transitions_schema_reward_noflip():
    import torch
    from resfit.rl_finetuning.chunk_residual.libero_offline import _demo_to_transitions, AGENTVIEW_KEY, WRIST_KEY
    demo = _toy_demo(T=3)
    tds = _demo_to_transitions(demo, action_scaler=_IdScaler(), state_standardizer=_IdStd(),
                               base_actions=None, image_size=84)
    assert len(tds) == 2                         # T-1 个 transition
    td = tds[0]
    obs = td["obs"]
    assert set(obs.keys()) >= {"observation.state", "observation.base_action",
                               "observation.stage_id", AGENTVIEW_KEY, WRIST_KEY}
    assert obs["observation.state"].shape == (8,)
    assert obs["observation.base_action"].shape == (7,)
    assert obs[AGENTVIEW_KEY].shape == (3, 84, 84) and obs[AGENTVIEW_KEY].dtype == torch.uint8
    assert obs["observation.stage_id"].item() == 0.0
    # gt 模式(base_actions=None):base_action == action
    assert torch.allclose(obs["observation.base_action"], td["action"])
    # reward/done:仅末帧 transition 的 next 给 reward=1/done=True
    assert tds[-1]["next"]["reward"].item() == 1.0 and bool(tds[-1]["next"]["done"]) is True
    assert tds[0]["next"]["reward"].item() == 0.0 and bool(tds[0]["next"]["done"]) is False
    assert td["max_stage"].item() == 0.0 and td["_priority"].item() == 10.0
    # 命门①:没被上下翻转 → 上半(行<42)应远亮于下半
    img = obs[AGENTVIEW_KEY].float()
    assert img[:, :42, :].mean() > img[:, 42:, :].mean() + 50


def test_demo_to_transitions_single_frame_returns_empty():
    from resfit.rl_finetuning.chunk_residual.libero_offline import _demo_to_transitions
    demo = _toy_demo(T=1)   # T<2 → 无 transition
    assert _demo_to_transitions(demo, action_scaler=_IdScaler(), state_standardizer=_IdStd(),
                                base_actions=None, image_size=84) == []


class _StubBase:
    """记录每次 select_action 的 raw_obs;返回固定原始动作。"""
    def __init__(self):
        self.seen = []
        self.reset_calls = 0
    def reset(self):
        self.reset_calls += 1
    def select_action(self, raw_obs):
        import torch
        self.seen.append({k: (v.clone() if hasattr(v, "clone") else v) for k, v in raw_obs.items()})
        return torch.full((1, 7), 0.5)


def test_libero_demo_base_actions_feeds_raw_state_and_noflip_84():
    import torch
    from resfit.rl_finetuning.chunk_residual.libero_offline import _libero_demo_base_actions, AGENTVIEW_KEY, WRIST_KEY
    demo = _toy_demo(T=3)
    base = _StubBase()
    out = _libero_demo_base_actions(demo, base, _IdScaler(), image_size=84, device="cpu")
    assert out.shape == (3, 7)
    assert torch.allclose(out, torch.full((3, 7), 0.5))      # 恒等 scaler 穿透
    assert base.reset_calls == 1                              # 逐 demo reset 一次
    first = base.seen[0]
    # 命门③(base 侧):喂 raw(未标准化)state,即 demo["state"][0]
    assert torch.allclose(first["observation.state"].reshape(-1), torch.as_tensor(demo["state"][0]))
    # 喂 84 CHW float[0,1] 图、且没翻转(上半亮);agentview 与 wrist 都要在
    img = first[AGENTVIEW_KEY].reshape(3, 84, 84)
    assert img.max() <= 1.0 + 1e-6
    assert img[:, :42, :].mean() > img[:, 42:, :].mean() + 0.2
    assert WRIST_KEY in first and first[WRIST_KEY].reshape(3, 84, 84).max() <= 1.0 + 1e-6
