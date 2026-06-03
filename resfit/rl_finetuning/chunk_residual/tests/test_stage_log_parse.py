"""stage_log_parse 纯函数:purity 行解析 / reach sidecar 读取 / 单步 stage 折叠。"""
import json

from resfit.rl_finetuning.chunk_residual.stage_log_parse import (
    parse_stage_purity_line,
    load_reach_sidecar,
    fold_episode_max_stage,
)


# --- parse_stage_purity_line ---------------------------------------------
def test_parse_purity_real_line():
    line = ("[stage-purity] regress 3547/9980=35.5%  stage0:0/2404=0%  "
            "stage1:413/1963=21%  stage2:7/1238=1%  stage3:3127/4375=71%\n")
    out = parse_stage_purity_line(line)
    # 未回退率 = 1 - regress/total
    assert out[0] == 1.0          # 0/2404
    assert abs(out[1] - (1 - 413 / 1963)) < 1e-9
    assert abs(out[3] - (1 - 3127 / 4375)) < 1e-9


def test_parse_purity_skips_zero_total():
    # b==0 的桶不应出现在结果里(避免除零)
    out = parse_stage_purity_line("[stage-purity] regress 0/0=0%  stage2:0/0=0%")
    assert 2 not in out


def test_parse_purity_non_purity_line_returns_empty():
    assert parse_stage_purity_line("[env_steps 10000] eval success_rate=0.200") == {}


# --- load_reach_sidecar ---------------------------------------------------
def test_load_reach_sidecar_roundtrip(tmp_path):
    p = tmp_path / "run_reach.json"
    p.write_text(json.dumps({"step": 42, "n_episodes": 50,
                             "reach": {"1": 1.0, "2": 0.8, "3": 0.2}}))
    got = load_reach_sidecar(str(p))
    assert got["step"] == 42
    assert got["reach"] == {1: 1.0, 2: 0.8, 3: 0.2}   # key 转 int


def test_load_reach_sidecar_missing_returns_none(tmp_path):
    assert load_reach_sidecar(str(tmp_path / "nope.json")) is None


# --- fold_episode_max_stage ----------------------------------------------
def test_fold_takes_max():
    assert fold_episode_max_stage(2, 3, reward=0.0, top_stage=4) == 3
    assert fold_episode_max_stage(3, 1, reward=0.0, top_stage=4) == 3


def test_fold_success_jumps_to_top():
    assert fold_episode_max_stage(1, 1, reward=1.0, top_stage=4) == 4


def test_fold_no_success_no_jump():
    assert fold_episode_max_stage(0, 0, reward=0.5, top_stage=4) == 0
