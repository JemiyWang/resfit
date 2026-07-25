import json
from pathlib import Path
import shlex
import subprocess
import types

import numpy as np
import pandas as pd
import pytest

from resfit.rl_finetuning.wm_bridge import builder
from resfit.rl_finetuning.wm_bridge import contract
from resfit.rl_finetuning.wm_bridge import launch_imagination
from resfit.rl_finetuning.wm_bridge.builder import (
    format_offline_build_stats,
    format_offline_startup_banner,
    make_offline_factories,
    parse_bridge_args,
    prepare_offline_runtime,
    write_bridge_cache_meta,
    write_bridge_run_config,
)
from resfit.rl_finetuning.wm_bridge.block_offline_cache import EndpointRecord
from resfit.rl_finetuning.wm_bridge.block_offline_chunk import BuildStats
from resfit.rl_finetuning.wm_bridge.contract import ContractError
from resfit.rl_finetuning.wm_bridge.wm_driver import CAMERA_KEYS


def _production_launch_argv():
    tokens = shlex.split(
        Path("launch_block_imagination.sh").read_text(),
        comments=True,
        posix=True,
    )
    module = "resfit.rl_finetuning.wm_bridge.launch_imagination"
    start = tokens.index(module) + 1
    end = tokens.index("2>&1", start)
    return [token for token in tokens[start:end] if token != "\n"]


def test_production_launch_enables_exact_mixed_configuration():
    text = Path("launch_block_imagination.sh").read_text()
    tokens = shlex.split(text, comments=True, posix=True)
    required = {
        "--offline_chunk_dataset": "${BLK}/block_success",
        "--offline_chunk_cache_root":
            "/mnt/mnt/data/resfit/cache/block_mixed_replay",
        "--offline_fraction": "0.5",
        "--batch_size": "256",
        "--actor": "raw",
        "--action_scale": "0.2",
        "--min_range_per_dim": "0.1",
        "--demo_bc_coef": "0.1",
        "--critic_warmup_steps": "10000",
        "--learning_starts": "10000",
    }
    for flag, value in required.items():
        assert tokens.count(flag) == 1
        assert tokens[tokens.index(flag) + 1] == value
    assert "--bc_coef_final" not in tokens
    assert tokens.count("--no_stage_balanced") == 1
    assert "--offline_base_mode" not in tokens
    assert "--offline_dataset_path" not in tokens
    assert "--offline_buffer_cache" not in tokens
    assert "OUT=/mnt/mnt/data/resfit/outputs_imagination/" \
        "block_shore_mixed50_seed${SEED}" in tokens
    assert tokens[tokens.index("--wandb_name") + 1] == \
        "block_shore_mixed50_seed${SEED}"
    init_indices = [
        index for index, token in enumerate(tokens)
        if token == "--init_state_dataset"
    ]
    assert [tokens[index + 1] for index in init_indices] == [
        "${BLK}/block_success", "${BLK}/block_fail"]


def test_cup_launcher_is_aligned():
    text = Path("launch_cup_imagination.sh").read_text()
    tokens = shlex.split(text, comments=True, posix=True)
    required = {
        "--task_profile cup",
        "--offline_fraction 0.5",
        "--batch_size 256",
        "--chunk_length 50",
        "--demo_bc_coef 0.1",
        "--critic_warmup_steps 10000",
        "--learning_starts 10000",
        "--utd 4",
        "--actor_lr 0.000001",
        "--critic_lr 0.0001",
        "--n_step 1",
        "--gamma 0.995",
        "--pi0_prompt \"pick cup\"",
        "--dataset cup_success",
        "cup_value_pi0feat.pt",
        "cup_shore_mixed50_seed",
    }
    for item in required:
        assert item in text
    assert "--bc_coef_final" not in text
    assert tokens.index("mkdir") < tokens.index("tee")


def test_cup_preparation_uses_three_disjoint_shards():
    tokens = shlex.split(
        Path("prepare_cup_pi0feat.sh").read_text(),
        comments=True, posix=True,
    )
    assert "for shard in 0 1 2; do" in " ".join(tokens)
    assert tokens.count("${PORTS[$shard]}") == 1
    num_shards = [
        tokens[index + 1] for index, token in enumerate(tokens)
        if token == "--num_shards"
    ]
    shard_indices = [
        tokens[index + 1] for index, token in enumerate(tokens)
        if token == "--shard_index"
    ]
    assert num_shards == ["3", "3"]
    assert shard_indices == ["${shard}", "${shard}"]
    assert "${OUT}/success_shard${shard}.npz" in tokens
    assert "${OUT}/fail_shard${shard}.npz" in tokens


def test_cup_value_script_argv_matches_parser():
    from resfit.rl_finetuning.chunk_residual.train_hiql_value import build_parser

    tokens = [
        token for token in shlex.split(
            Path("train_cup_pi0feat_value.sh").read_text(),
            comments=True, posix=True,
        ) if token != "\n"
    ]
    module = "resfit.rl_finetuning.chunk_residual.train_hiql_value"
    args = build_parser().parse_args(tokens[tokens.index(module) + 1:])
    assert args.pi0_image_keys == ["top_head", "hand_left", "hand_right"]
    assert len(args.success_dataset) == 3
    assert len(args.failure_dataset) == 3


def test_cup_launcher_actual_argv_matches_bridge_and_trainer_parsers(
        tmp_path, monkeypatch):
    capture = tmp_path / "capture-argv.sh"
    capture.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "{\n"
        "  printf 'GPU=%s\\n' \"${CUDA_VISIBLE_DEVICES}\"\n"
        "  printf 'ROOT=%s\\n' \"${HF_LEROBOT_HOME}\"\n"
        "  printf 'ARG=%s\\n' \"$@\"\n"
        "} > \"${CAPTURE_ARGV:?}\"\n",
        encoding="utf-8",
    )
    capture.chmod(0o755)
    captured = tmp_path / "argv.txt"
    monkeypatch.setenv("CAPTURE_ARGV", str(captured))

    text = Path("launch_cup_imagination.sh").read_text()
    command = (
        "/mnt/mnt/data/envs/residual/bin/python -m "
        "resfit.rl_finetuning.wm_bridge.launch_imagination"
    )
    assert text.count(command) == 1
    text = text.replace(command, str(capture))
    text = text.replace(
        "OUT=/mnt/mnt/data/resfit/outputs_imagination/"
        "cup_shore_mixed50_seed${SEED}",
        f"OUT={tmp_path}/cup_shore_mixed50_seed${{SEED}}",
    )
    launcher = tmp_path / "launch-cup-captured.sh"
    launcher.write_text(text, encoding="utf-8")
    launcher.chmod(0o755)
    subprocess.run(
        ["bash", str(launcher), "5", "7"],
        cwd=Path.cwd(),
        check=True,
    )

    lines = captured.read_text(encoding="utf-8").splitlines()
    assert lines[:2] == [
        "GPU=5",
        "ROOT=/mnt/mnt/data/domains_rise/cup",
    ]
    argv = [line.removeprefix("ARG=") for line in lines[2:]]
    duplicate_flags = {
        flag for flag in argv
        if flag.startswith("--") and argv.count(flag) > 1
    }
    assert duplicate_flags == {"--init_state_dataset"}

    bridge_args, passthrough = parse_bridge_args(argv)
    from resfit.rl_finetuning.chunk_residual.train_chunk_residual import (
        build_parser as build_trainer_parser,
    )
    trainer_args = build_trainer_parser().parse_args(passthrough)
    assert bridge_args.task_profile == "cup"
    assert bridge_args.init_state_dataset == [
        "/mnt/mnt/data/domains_rise/cup/cup_success",
        "/mnt/mnt/data/domains_rise/cup/cup_fail",
    ]
    assert trainer_args.seed == 7
    assert trainer_args.output_dir == \
        f"{tmp_path}/cup_shore_mixed50_seed7"


def test_teleavatar_public_wording_is_task_neutral():
    forbidden = {
        "resfit/rl_finetuning/wm_bridge/teleavatar_start_sampler.py": {
            "从 block 真实 episode",
            "block_success/block_fail 的完整路径列表",
            "至少一个 block 数据集路径",
        },
        "resfit/rl_finetuning/wm_bridge/builder.py": {
            "block kai0 chunk replay",
            "block V (pi0_feat, block_value_pi0feat.pt)",
            "想象起点数据集(block_success/fail 完整路径)",
            "block 动作 min/max JSON",
            "same block_success tree",
        },
        "resfit/rl_finetuning/wm_bridge/contract.py": {
            "block mixed replay is enabled",
            "trainer's lazy block-offline injection points",
        },
        "resfit/rl_finetuning/chunk_residual/"
        "build_pi0_feat_cache_via_serve.py": {
            "block/teleavatar 的 kai0 serve obs",
            "teleavatar/block 数据源",
            "teleavatar(block 三相机)",
            "如 block_success",
        },
    }
    for path, phrases in forbidden.items():
        text = Path(path).read_text()
        for phrase in phrases:
            assert phrase not in text


def test_production_batch_is_exactly_128_plus_128():
    batch_size, fraction = 256, 0.5
    assert int(batch_size * (1 - fraction)) == 128
    assert int(batch_size * fraction) == 128


def test_live_smoke_plan_keeps_production_batch_contract():
    text = Path(
        "docs/superpowers/plans/"
        "2026-07-23-rise-resfit-mixed-online-offline.md"
    ).read_text()
    section = text.split(
        "**Step 6: Run opt-in two-episode live cache smoke**", 1
    )[1].split("**Step 7: Inspect production dry-run metadata**", 1)[0]
    assert (
        "--offline_num_demos 2 --offline_fraction 0.5 --batch_size 256"
        in section
    )
    assert "online_batch_size=128 offline_batch_size=128" in section
    assert "--batch_size 2 \\" not in section
    assert "online_batch_size=1 offline_batch_size=1" not in section


class RuntimeStub:
    dataset_root = "/data/block_success"
    endpoint_store = object()
    replay_cache_dir = "/cache/replay"

    def selected_episodes(self, num_demos=None):
        episodes = (
            types.SimpleNamespace(num_frames=51),
            types.SimpleNamespace(num_frames=101),
        )
        return episodes if num_demos is None else episodes[:num_demos]


def _complete_build_stats():
    return BuildStats(
        episodes=2,
        skipped_short_episodes=0,
        transitions=3,
        endpoint_hits=1,
        endpoint_misses=1,
        reward_mean=1.25,
        reward_std=0.5,
        potential_delta_mean=2.0,
        potential_delta_std=0.25,
        expert_norm_mean=11.0,
        base_norm_mean=7.0,
        residual_norm_mean=4.0,
        expert_saturation_fraction=0.1,
        base_saturation_fraction=0.2,
    )


def _write_success_dataset(parent, name="block_success"):
    root = parent / name
    parquet = root / "data/chunk-000/episode_000000.parquet"
    parquet.parent.mkdir(parents=True)
    pd.DataFrame({"frame_index": range(51)}).to_parquet(parquet)
    for key in CAMERA_KEYS:
        video = root / f"videos/chunk-000/{key}/episode_000000.mp4"
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes(f"video-{key}".encode())
    stats = root / "meta/stats.json"
    stats.parent.mkdir(parents=True)
    (root / "meta/info.json").write_text(
        '{"codebase_version": "v2.1"}', encoding="utf-8")
    stats.write_text('{"action": {}, "state": {}}', encoding="utf-8")
    (root / "meta/episodes_stats.jsonl").write_text(
        '{"episode_index": 0, "stats": {}}\n', encoding="utf-8")
    return root


def _mixed_passthrough(output_dir):
    return [
        "--offline_fraction", "0.5",
        "--offline_num_demos", "1",
        "--batch_size", "256",
        "--base_policy_type", "pi05",
        "--base_action_mode", "replan",
        "--chunk_length", "50",
        "--n_step", "1",
        "--gamma", "0.995",
        "--actor", "raw",
        "--action_scale", "0.2",
        "--min_range_per_dim", "0.1",
        "--no_stage_balanced",
        "--output_dir", str(output_dir),
    ]


def test_parse_bridge_offline_args_are_removed_from_passthrough():
    bridge, rest = parse_bridge_args([
        "--value_ckpt", "/tmp/value.pt",
        "--offline_chunk_dataset", "/data/block_success",
        "--offline_chunk_cache_root", "/cache/block",
        "--offline_rebuild",
        "--batch_size", "256",
    ])
    assert bridge.offline_chunk_dataset == "/data/block_success"
    assert bridge.offline_chunk_cache_root == "/cache/block"
    assert bridge.offline_rebuild is True
    assert "--offline_chunk_dataset" not in rest
    assert rest == ["--batch_size", "256"]


def test_parse_bridge_args_defaults_to_block_profile():
    args, _ = builder.parse_bridge_args([
        "--value_ckpt", "/tmp/value.pt",
    ])
    assert args.task_profile == "block"


def test_prepare_cup_runtime_translates_task_fields(tmp_path, monkeypatch):
    dataset = _write_success_dataset(tmp_path, name="cup_success")
    value_ckpt = tmp_path / "cup_value.pt"
    value_ckpt.write_bytes(b"value-weights")
    monkeypatch.setenv("HF_LEROBOT_HOME", str(tmp_path))
    bridge, _ = parse_bridge_args([
        "--value_ckpt", str(value_ckpt),
        "--task_profile", "cup",
        "--offline_chunk_dataset", str(dataset),
        "--offline_chunk_cache_root", str(tmp_path / "cache"),
        "--pi0_serve_ckpt_id", "pi05_pick_cup_awbc_49999",
    ])
    runtime, translated = builder.prepare_offline_runtime(
        bridge, _mixed_passthrough(tmp_path / "output"))
    assert translated[translated.index("--pi0_prompt") + 1] == "pick cup"
    assert translated[translated.index("--dataset") + 1] == "cup_success"
    assert runtime.bridge_meta["offline_source"] == "cup_success"
    assert runtime.bridge_meta["pi0_prompt"] == "pick cup"


def test_paper_metadata_records_physical_clock_and_awbc_provenance(
    tmp_path,
    monkeypatch,
):
    dataset = _write_success_dataset(tmp_path, name="paper_success")
    value_ckpt = tmp_path / "paper_value_pi0feat.pt"
    value_ckpt.write_bytes(b"value-weights")
    checkpoint_dir = tmp_path / "checkpoints/paper/19999"
    norm_stats = (
        checkpoint_dir
        / "assets/pick_paper_all_merged/norm_stats.json"
    )
    norm_stats.parent.mkdir(parents=True)
    norm_stats.write_text('{"norm": "paper"}', encoding="utf-8")
    adv_ckpt = tmp_path / "value_paper/model.safetensors"
    adv_ckpt.parent.mkdir()
    adv_ckpt.write_bytes(b"adv")
    monkeypatch.setenv("HF_LEROBOT_HOME", str(tmp_path))
    bridge, _ = parse_bridge_args([
        "--value_ckpt", str(value_ckpt),
        "--task_profile", "paper",
        "--offline_chunk_dataset", str(dataset),
        "--offline_chunk_cache_root", str(tmp_path / "cache"),
        "--pi0_serve_ckpt_id", "pi05_paper_awbc_19999",
        "--pi0_serve_ckpt_dir", str(checkpoint_dir),
        "--pi0_asset_id", "pick_paper_all_merged",
        "--pi0_pooling", "mean",
        "--source_wandb_run", "eegyfzmz",
        "--adv_ckpt", str(adv_ckpt),
        "--adv_config", "value_paper",
        "--init_state_dataset", str(dataset),
    ])

    runtime, _ = builder.prepare_offline_runtime(
        bridge,
        _mixed_passthrough(tmp_path / "output"),
    )
    meta = runtime.bridge_meta
    assert meta["task_profile"] == "paper"
    assert meta["dataset_fps"] == 20.0
    assert meta["physical_chunk_seconds"] == 2.5
    assert meta["pi0_policy_state_dim"] == 14
    assert meta["pi0_serve_ckpt_dir"] == str(checkpoint_dir.resolve())
    assert meta["pi0_asset_id"] == "pick_paper_all_merged"
    assert meta["pi0_pooling"] == "mean"
    assert meta["source_wandb_run"] == "eegyfzmz"
    assert meta["adv_ckpt"] == str(adv_ckpt)
    assert meta["adv_config"] == "value_paper"
    assert meta["init_state_datasets"] == [str(dataset)]
    assert len(meta["pi0_norm_stats_sha256"]) == 64


def test_write_success_dataset_helper_can_create_cup_source(tmp_path):
    dataset = _write_success_dataset(tmp_path, name="cup_success")
    assert dataset.name == "cup_success"
    assert (dataset / "meta/episodes_stats.jsonl").is_file()


def _capture_imagination_factory_task_fields(monkeypatch, adv_prompt=None):
    captured = {}
    scorer = object()

    monkeypatch.setattr(
        "resfit.rl_finetuning.wm_bridge.scorers.Kai0HiqlScorer."
        "from_value_ckpt",
        lambda *args, **kwargs: scorer,
    )
    monkeypatch.setattr(contract, "check_scorer", lambda *args, **kw: None)
    monkeypatch.setattr(
        contract, "check_mixed_scorer", lambda *args, **kw: None)
    monkeypatch.setattr(
        contract, "check_psi_samesource", lambda *args, **kw: None)

    class CapturingAdvClient:
        def __init__(self, *, host, port, prompt):
            captured["adv_prompt"] = prompt

    class CapturingSampler:
        def __init__(self, datasets, caption="build block"):
            captured["datasets"] = datasets
            captured["caption"] = caption

    monkeypatch.setattr(
        "resfit.rl_finetuning.wm_bridge.adv_client.AdvServeClient",
        CapturingAdvClient,
    )
    monkeypatch.setattr(
        "resfit.rl_finetuning.wm_bridge.teleavatar_start_sampler."
        "TeleavatarStartSampler",
        CapturingSampler,
    )
    monkeypatch.setattr(
        "resfit.rl_finetuning.wm_bridge.base_bridge.Kai0ImaginationBase",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        "resfit.rl_finetuning.wm_bridge.imagination_env.ImaginationVecEnv",
        lambda **kwargs: types.SimpleNamespace(**kwargs),
    )
    monkeypatch.setattr(
        "resfit.rl_finetuning.wm_bridge.wm_client.WmServeClient",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        "openpi_client.websocket_client_policy.WebsocketClientPolicy",
        lambda **kwargs: object(),
    )

    argv = [
        "--value_ckpt", "/tmp/cup_value.pt",
        "--task_profile", "cup",
        "--init_state_dataset", "/tmp/cup_success",
        "--adv_host", "127.0.0.1",
    ]
    if adv_prompt is not None:
        argv.extend(["--adv_prompt", adv_prompt])
    bridge_args, _ = parse_bridge_args(argv)
    factories = builder.build_imagination_factories(bridge_args)
    factories["load_pi05_base_policy"](
        types.SimpleNamespace(
            host="127.0.0.1",
            port=8000,
            prompt="pick cup",
            action_dim=16,
        ),
        "cpu",
    )
    factories["create_vectorized_env"]()
    return captured


def test_cup_factory_uses_profile_prompt_for_sampler_and_adv_fallback(
    monkeypatch,
):
    captured = _capture_imagination_factory_task_fields(monkeypatch)
    assert captured["caption"] == "pick cup"
    assert captured["adv_prompt"] == "pick cup"


def test_explicit_adv_prompt_overrides_cup_profile(monkeypatch):
    captured = _capture_imagination_factory_task_fields(
        monkeypatch, adv_prompt="custom cup monitor")
    assert captured["adv_prompt"] == "custom cup monitor"


def test_cup_runtime_missing_cache_root_uses_task_neutral_error(
    tmp_path, monkeypatch,
):
    dataset = _write_success_dataset(tmp_path, name="cup_success")
    value_ckpt = tmp_path / "cup_value.pt"
    value_ckpt.write_bytes(b"value-weights")
    monkeypatch.setenv("HF_LEROBOT_HOME", str(tmp_path))
    bridge, _ = parse_bridge_args([
        "--value_ckpt", str(value_ckpt),
        "--task_profile", "cup",
        "--offline_chunk_dataset", str(dataset),
        "--pi0_serve_ckpt_id", "pi05_pick_cup_awbc_49999",
    ])
    with pytest.raises(ContractError, match="required for mixed replay"):
        prepare_offline_runtime(
            bridge, _mixed_passthrough(tmp_path / "output"))


def test_offline_factories_count_and_forward_shared_objects(
    tmp_path, monkeypatch,
):
    runtime = RuntimeStub()
    runtime.replay_cache_dir = str(tmp_path / "cache")
    runtime.bridge_meta = {"output_dir": str(tmp_path / "output")}
    shared_base, shared_scorer = object(), object()
    state = {"base": shared_base}
    captured = {}

    def fake_build(rb, path, **kwargs):
        captured.update(kwargs)
        return _complete_build_stats()

    monkeypatch.setattr(builder, "build_block_offline_buffer", fake_build)
    factories = make_offline_factories(runtime, state, shared_scorer)
    assert factories["count_offline_transitions"](
        runtime.dataset_root, num_demos=2) == 3
    with pytest.raises(ContractError, match="dataset_path"):
        factories["count_offline_transitions"](
            "/another/root", num_demos=2)

    factories["build_offline_buffer"](
        object(), runtime.dataset_root, num_demos=2,
        base_policy="generic", base_mode="gt", gamma=0.995,
        bonus=100.0, mode="staged", potential="generic")
    assert captured["base_policy"] is shared_base
    assert captured["scorer"] is shared_scorer
    assert captured["endpoint_store"] is runtime.endpoint_store
    assert len(captured["episodes"]) == 2
    assert "base_mode" not in captured
    assert "bonus" not in captured
    assert "mode" not in captured
    assert "potential" not in captured


def test_offline_factory_rejects_non_gt_compatibility_mode():
    factories = make_offline_factories(
        RuntimeStub(), {"base": object()}, object())

    with pytest.raises(ContractError, match="base_mode"):
        factories["build_offline_buffer"](
            object(), RuntimeStub.dataset_root, base_mode="base_policy")


def test_offline_factory_requires_explicit_gt_compatibility_mode(monkeypatch):
    monkeypatch.setattr(
        builder,
        "build_block_offline_buffer",
        lambda *args, **kwargs: types.SimpleNamespace(transitions=0),
    )
    factories = make_offline_factories(
        RuntimeStub(), {"base": object()}, object())

    with pytest.raises(ContractError, match="base_mode"):
        factories["build_offline_buffer"](
            object(), RuntimeStub.dataset_root, gamma=0.995)


def test_first_build_prints_and_atomically_persists_complete_stats(
    tmp_path, monkeypatch, capsys,
):
    runtime = RuntimeStub()
    runtime.replay_cache_dir = str(tmp_path / "cache-generation")
    runtime.bridge_meta = {
        "output_dir": str(tmp_path / "output"),
        "offline_source": "block_success",
    }
    stats = _complete_build_stats()
    monkeypatch.setattr(
        builder, "build_block_offline_buffer", lambda *a, **k: stats)
    factories = make_offline_factories(
        runtime, {"base": object()}, object())

    factories["build_offline_buffer"](
        object(), runtime.dataset_root, num_demos=2, base_mode="gt")

    output = capsys.readouterr().out
    for token in (
        "reward_mean=1.25", "reward_std=0.5",
        "potential_delta_mean=2", "potential_delta_std=0.25",
        "expert_norm_mean=11", "base_norm_mean=7",
        "residual_norm_mean=4", "expert_saturation_fraction=0.1",
        "base_saturation_fraction=0.2", "endpoint_hits=1",
        "endpoint_misses=1",
    ):
        assert token in output
    expected = runtime.bridge_meta["build_stats"]
    assert json.loads(
        (tmp_path / "output/bridge_run_config.json").read_text()
    )["build_stats"] == expected
    assert json.loads(
        (tmp_path / "cache-generation/bridge_meta.json").read_text()
    )["build_stats"] == expected
    assert not list(tmp_path.rglob("*.tmp"))


def test_replace_option_rejects_bare_authoritative_flag():
    with pytest.raises(ContractError, match="offline_dataset_path"):
        builder._replace_option(
            ["--offline_dataset_path", "--batch_size", "256"],
            "--offline_dataset_path",
            "/data/block_success",
        )


def test_prepare_runtime_fingerprints_and_replaces_trainer_options(
        tmp_path, monkeypatch):
    dataset = _write_success_dataset(tmp_path)
    value_ckpt = tmp_path / "value.pt"
    value_ckpt.write_bytes(b"value-weights")
    cache_root = tmp_path / "cache"
    monkeypatch.setenv("HF_LEROBOT_HOME", str(tmp_path))
    bridge_args, _ = parse_bridge_args([
        "--value_ckpt", str(value_ckpt),
        "--offline_chunk_dataset", str(dataset),
        "--offline_chunk_cache_root", str(cache_root),
        "--pi0_serve_ckpt_id", "pi05_block_awbc_49999",
    ])
    passthrough = _mixed_passthrough(tmp_path / "output") + [
        "--offline_dataset_path", "/old/data",
        "--offline_buffer_cache=/old/cache",
        "--offline_base_mode", "base_policy",
        "--pi0_prompt", "assemble the three pieces",
        "--pi0_action_dim=14",
        "--dataset", "wrong_dataset",
        "--data_source", "lerobot",
    ]

    runtime, translated = prepare_offline_runtime(bridge_args, passthrough)

    assert runtime is not None
    assert runtime.dataset_root == str(dataset.resolve())
    assert len(runtime.all_episodes) == 1
    assert len(runtime.selected_episodes()) == 1
    assert len(runtime.endpoint_fingerprint) == 64
    assert len(runtime.replay_fingerprint) == 64
    assert Path(runtime.replay_cache_dir).is_dir()
    assert translated.count("--offline_dataset_path") == 1
    assert translated.count("--offline_buffer_cache") == 1
    assert translated.count("--offline_base_mode") == 1
    assert translated.count("--pi0_prompt") == 1
    assert translated.count("--pi0_action_dim") == 1
    assert translated.count("--dataset") == 1
    assert translated.count("--data_source") == 1
    assert translated[translated.index("--offline_dataset_path") + 1] == str(
        dataset.resolve())
    assert translated[translated.index("--offline_buffer_cache") + 1] == (
        runtime.replay_cache_dir)
    assert translated[translated.index("--offline_base_mode") + 1] == "gt"
    assert translated[translated.index("--pi0_prompt") + 1] == "build block"
    assert translated[translated.index("--pi0_action_dim") + 1] == "16"
    assert translated[translated.index("--dataset") + 1] == "block_success"
    assert translated[translated.index("--data_source") + 1] == "hdf5"

    meta = runtime.bridge_meta
    assert meta["trainer_compat_mode"] == "gt"
    assert meta["actual_base_mode"] == "kai0_chunk"
    assert meta["offline_reward"] == "gamma_phi_next_minus_phi"
    assert meta["online_batch_size"] == 128
    assert meta["offline_batch_size"] == 128
    assert meta["pi0_serve_ckpt_id"] == "pi05_block_awbc_49999"
    assert meta["pi0_action_dim"] == 16
    assert meta["normalization_dataset_root"] == str(dataset.resolve())
    assert meta["normalization_stats_path"] == str(
        (dataset / "meta/episodes_stats.jsonl").resolve())
    assert meta["endpoint_cache_status"] == "miss"
    assert meta["replay_cache_status"] == "miss"


def test_trainer_signature_uses_actual_trainer_parser_defaults(
    tmp_path, monkeypatch,
):
    from resfit.rl_finetuning.chunk_residual import train_chunk_residual

    dataset = _write_success_dataset(tmp_path)
    value_ckpt = tmp_path / "value.pt"
    value_ckpt.write_bytes(b"value-weights")
    monkeypatch.setenv("HF_LEROBOT_HOME", str(tmp_path))
    real_build_parser = train_chunk_residual.build_parser

    def drifted_build_parser():
        parser = real_build_parser()
        parser.set_defaults(stage_reward_bonus=7.0)
        return parser

    monkeypatch.setattr(
        train_chunk_residual, "build_parser", drifted_build_parser)
    bridge_args, _ = parse_bridge_args([
        "--value_ckpt", str(value_ckpt),
        "--offline_chunk_dataset", str(dataset),
        "--offline_chunk_cache_root", str(tmp_path / "cache"),
        "--pi0_serve_ckpt_id", "pi05_block_awbc_49999",
    ])

    runtime, _ = prepare_offline_runtime(
        bridge_args, _mixed_passthrough(tmp_path / "output"))

    assert runtime.bridge_meta["trainer_cache_signature"][
        "stage_reward_bonus"
    ] == 7.0


def test_cache_hit_preserves_stats_and_banner_is_authoritative(
    tmp_path, monkeypatch,
):
    dataset = _write_success_dataset(tmp_path)
    short_parquet = dataset / "data/chunk-000/episode_000001.parquet"
    pd.DataFrame({"frame_index": range(50)}).to_parquet(short_parquet)
    for key in CAMERA_KEYS:
        video = dataset / f"videos/chunk-000/{key}/episode_000001.mp4"
        video.write_bytes(f"short-{key}".encode())
    value_ckpt = tmp_path / "value.pt"
    value_ckpt.write_bytes(b"value-weights")
    monkeypatch.setenv("HF_LEROBOT_HOME", str(tmp_path))
    bridge_args, _ = parse_bridge_args([
        "--value_ckpt", str(value_ckpt),
        "--offline_chunk_dataset", str(dataset),
        "--offline_chunk_cache_root", str(tmp_path / "cache"),
        "--pi0_serve_ckpt_id", "pi05_block_awbc_49999",
    ])
    passthrough = builder._replace_option(
        _mixed_passthrough(tmp_path / "output"),
        "--offline_num_demos",
        2,
    )
    first, _ = prepare_offline_runtime(bridge_args, passthrough)
    record = EndpointRecord(
        frame_indices=np.array([0], dtype="int64"),
        base_actions=np.zeros((1, 50, 16), dtype="float32"),
        prefix_features=np.zeros((1, 2), dtype="float32"),
        proprio=np.zeros((1, 16), dtype="float32"),
    )
    first.endpoint_store.save_episode(0, record)
    stats = _complete_build_stats()
    cached_meta = dict(first.bridge_meta, build_stats=stats.__dict__)
    write_bridge_cache_meta(first.replay_cache_dir, cached_meta)
    Path(first.replay_cache_dir, "buffer_meta.json").write_text(json.dumps({
        "signature": first.bridge_meta["trainer_cache_signature"],
        "n_transitions": 1,
    }))
    Path(first.replay_cache_dir, "storage").mkdir()

    incomplete, _ = prepare_offline_runtime(bridge_args, passthrough)

    assert incomplete.bridge_meta["endpoint_cache_status"] == "miss"
    complete_record = EndpointRecord(
        frame_indices=np.array([0, 50], dtype="int64"),
        base_actions=np.zeros((2, 50, 16), dtype="float32"),
        prefix_features=np.zeros((2, 2), dtype="float32"),
        proprio=np.zeros((2, 16), dtype="float32"),
    )
    first.endpoint_store.save_episode(0, complete_record)
    cached, _ = prepare_offline_runtime(bridge_args, passthrough)

    assert cached.replay_cache_dir == first.replay_cache_dir
    assert cached.bridge_meta["replay_cache_status"] == "hit"
    assert cached.bridge_meta["endpoint_cache_status"] == "hit"
    assert cached.bridge_meta["replay_cache"] == "hit"
    assert cached.bridge_meta["endpoint_cache"] == "hit"
    assert cached.bridge_meta["build_stats"] == stats.__dict__
    banner = format_offline_startup_banner(cached)
    for token in (
        "offline_source=block_success", "offline_episodes=2",
        "offline_transitions=1", "batch=128+128",
        "online_batch_size=128", "offline_batch_size=128",
        "trainer_compat_mode=gt", "actual_offline_base=kai0_chunk",
        "offline_reward=gamma_phi_next_minus_phi",
        "endpoint_cache=hit", "replay_cache=hit",
    ):
        assert token in banner
    assert "reward_mean=1.25" in format_offline_build_stats(
        cached.bridge_meta["build_stats"], cached)


@pytest.mark.parametrize(
    ("has_storage", "stats_kind", "matching_signature", "expected_hit"),
    [
        (False, "complete", True, False),
        (True, "missing", True, False),
        (True, "malformed", True, False),
        (True, "complete", False, False),
        (True, "complete", True, True),
    ],
)
def test_replay_hit_matches_trainer_and_requires_build_stats(
    tmp_path,
    monkeypatch,
    has_storage,
    stats_kind,
    matching_signature,
    expected_hit,
):
    dataset = _write_success_dataset(tmp_path)
    value_ckpt = tmp_path / "value.pt"
    value_ckpt.write_bytes(b"value-weights")
    monkeypatch.setenv("HF_LEROBOT_HOME", str(tmp_path))
    bridge_args, _ = parse_bridge_args([
        "--value_ckpt", str(value_ckpt),
        "--offline_chunk_dataset", str(dataset),
        "--offline_chunk_cache_root", str(tmp_path / "cache"),
        "--pi0_serve_ckpt_id", "pi05_block_awbc_49999",
    ])
    passthrough = _mixed_passthrough(tmp_path / "output")
    first, _ = prepare_offline_runtime(bridge_args, passthrough)
    signature = first.bridge_meta["trainer_cache_signature"]
    if not matching_signature:
        signature = dict(signature, gamma=0.0)
    Path(first.replay_cache_dir, "buffer_meta.json").write_text(
        json.dumps({"signature": signature, "n_transitions": 1}))
    if has_storage:
        Path(first.replay_cache_dir, "storage").mkdir()
    metadata = dict(first.bridge_meta)
    if stats_kind != "missing":
        metadata["build_stats"] = _complete_build_stats().__dict__
    if stats_kind == "malformed":
        metadata["build_stats"]["reward_mean"] = "not-a-number"
    write_bridge_cache_meta(first.replay_cache_dir, metadata)

    resolved, _ = prepare_offline_runtime(bridge_args, passthrough)

    assert (resolved.replay_cache_dir == first.replay_cache_dir) is expected_hit
    assert resolved.bridge_meta["replay_cache"] == (
        "hit" if expected_hit else "miss")
    assert ("build_stats" in resolved.bridge_meta) is expected_hit


def test_production_launch_tokens_dry_run_publish_exact_metadata(
        tmp_path, monkeypatch):
    dataset = _write_success_dataset(tmp_path)
    value_ckpt = tmp_path / "value.pt"
    value_ckpt.write_bytes(b"value-weights")
    cache_root = tmp_path / "cache"
    output_dir = tmp_path / "output"
    monkeypatch.setenv("HF_LEROBOT_HOME", str(tmp_path))

    argv = _production_launch_argv()
    for flag, value in (
        ("--value_ckpt", value_ckpt),
        ("--offline_chunk_dataset", dataset),
        ("--offline_chunk_cache_root", cache_root),
        ("--output_dir", output_dir),
        ("--seed", 0),
    ):
        argv = builder._replace_option(argv, flag, value)

    bridge_args, passthrough = parse_bridge_args(argv)
    runtime, translated = prepare_offline_runtime(bridge_args, passthrough)
    write_bridge_run_config(str(output_dir), runtime.bridge_meta)
    published = json.loads(
        (output_dir / "bridge_run_config.json").read_text())

    assert {
        key: published[key]
        for key in (
            "online_batch_size",
            "offline_batch_size",
            "trainer_compat_mode",
            "actual_base_mode",
            "offline_reward",
        )
    } == {
        "online_batch_size": 128,
        "offline_batch_size": 128,
        "trainer_compat_mode": "gt",
        "actual_base_mode": "kai0_chunk",
        "offline_reward": "gamma_phi_next_minus_phi",
    }
    assert published["trainer_cache_signature"] == {
        "dataset": "block_success",
        "offline_dataset_path": str(dataset.resolve()),
        "num_demos": None,
        "offline_cap": 1,
        "action_scale": 0.2,
        "min_range_per_dim": 0.1,
        "reward_shaping": "none",
        "stage_reward_bonus": 1.0,
        "gamma": 0.995,
        "n_step": 1,
        "image_keys": sorted(CAMERA_KEYS),
        "task": "TwoArmBoxCleanup",
        "stage_cache": None,
    }
    from resfit.rl_finetuning.chunk_residual.chunk_env_wrapper import (
        resolve_shaping_mode,
    )
    from resfit.rl_finetuning.chunk_residual.train_chunk_residual import (
        _offline_buffer_signature,
        build_parser,
    )

    trainer_args = build_parser().parse_args(translated)
    trainer_signature = _offline_buffer_signature(
        trainer_args,
        CAMERA_KEYS,
        runtime.bridge_meta["offline_transitions"],
        resolve_shaping_mode(
            trainer_args.reward_shaping,
            trainer_args.staged_reward,
        ),
    )
    assert published["trainer_cache_signature"] == trainer_signature


def test_prepare_runtime_requires_trainer_metadata_to_resolve_to_offline_dataset(
        tmp_path, monkeypatch):
    dataset = _write_success_dataset(tmp_path / "offline")
    value_ckpt = tmp_path / "value.pt"
    value_ckpt.write_bytes(b"value-weights")
    monkeypatch.setenv("HF_LEROBOT_HOME", str(tmp_path / "other"))
    bridge_args, _ = parse_bridge_args([
        "--value_ckpt", str(value_ckpt),
        "--offline_chunk_dataset", str(dataset),
        "--offline_chunk_cache_root", str(tmp_path / "cache"),
        "--pi0_serve_ckpt_id", "pi05_block_awbc_49999",
    ])

    with pytest.raises(ContractError, match="HF_LEROBOT_HOME"):
        prepare_offline_runtime(
            bridge_args, _mixed_passthrough(tmp_path / "output"))


def test_metadata_writers_publish_atomic_json(tmp_path):
    metadata = {"actual_base_mode": "kai0_chunk", "online_batch_size": 128}
    output = tmp_path / "output"
    cache = tmp_path / "cache-generation"

    write_bridge_run_config(str(output), metadata)
    write_bridge_cache_meta(str(cache), metadata)

    assert json.loads((output / "bridge_run_config.json").read_text()) == metadata
    assert json.loads((cache / "bridge_meta.json").read_text()) == metadata


def test_prepare_pure_online_is_a_noop(tmp_path, monkeypatch):
    value_ckpt = tmp_path / "value.pt"
    value_ckpt.write_bytes(b"unused")
    bridge_args, _ = parse_bridge_args([
        "--value_ckpt", str(value_ckpt),
    ])
    passthrough = ["--offline_fraction", "0.0", "--batch_size", "17"]

    runtime, translated = prepare_offline_runtime(bridge_args, passthrough)

    assert runtime is None
    assert translated == passthrough
    assert set(tmp_path.iterdir()) == {value_ckpt}
    assert not {
        "--offline_dataset_path",
        "--offline_buffer_cache",
        "--offline_base_mode",
    }.intersection(translated)

    scorer = object()
    monkeypatch.setattr(
        "resfit.rl_finetuning.wm_bridge.scorers.Kai0HiqlScorer."
        "from_value_ckpt",
        lambda *args, **kwargs: scorer,
    )
    monkeypatch.setattr(contract, "check_scorer", lambda *args: None)
    monkeypatch.setattr(contract, "check_mixed_scorer", lambda *args, **kw: None)
    monkeypatch.setattr(contract, "check_psi_samesource", lambda *args: None)
    factories = builder.build_imagination_factories(bridge_args)
    assert set(factories) == {
        "create_vectorized_env",
        "run_dexmg_evaluation",
        "load_pi05_base_policy",
    }


def test_launcher_writes_metadata_before_installing_fakes(monkeypatch):
    events = []
    bridge_args = types.SimpleNamespace(imagination_gamma=0.995)
    runtime = types.SimpleNamespace(
        bridge_meta={
            "offline_source": "block_success",
            "offline_episodes": 2,
            "offline_transitions": 3,
            "online_batch_size": 128,
            "offline_batch_size": 128,
            "trainer_compat_mode": "gt",
            "actual_base_mode": "kai0_chunk",
            "actual_offline_base": "kai0_chunk",
            "offline_reward": "gamma_phi_next_minus_phi",
            "endpoint_cache_status": "miss",
            "replay_cache_status": "miss",
            "endpoint_cache": "miss",
            "replay_cache": "miss",
        },
        replay_cache_dir="/cache/generation",
    )
    monkeypatch.setattr(contract, "check_upstream_symbols", lambda: None)
    monkeypatch.setattr(contract, "check_agent_image_size", lambda: None)
    monkeypatch.setattr(contract, "check_wrapper_step_loop", lambda: None)
    monkeypatch.setattr(
        contract,
        "check_offline_hook_points",
        lambda: events.append("hook_points"),
    )
    monkeypatch.setattr(
        contract, "check_passthrough_runtime_args", lambda *a, **k: None)
    monkeypatch.setattr(
        contract,
        "parse_mixed_passthrough",
        lambda argv: types.SimpleNamespace(output_dir="/output"),
    )
    monkeypatch.setattr(
        builder, "parse_bridge_args", lambda argv: (bridge_args, ["trainer"]))

    def prepare(args, passthrough):
        events.append("runtime")
        return runtime, ["translated"]

    def factories(args, offline_runtime=None):
        assert offline_runtime is runtime
        events.append("factories")
        return {}

    monkeypatch.setattr(builder, "prepare_offline_runtime", prepare)
    monkeypatch.setattr(builder, "build_imagination_factories", factories)
    monkeypatch.setattr(
        builder,
        "write_bridge_run_config",
        lambda output, metadata: events.append("run_config"),
    )
    monkeypatch.setattr(
        builder,
        "write_bridge_cache_meta",
        lambda output, metadata: events.append("cache_meta"),
    )
    monkeypatch.setattr(
        launch_imagination,
        "install_fakes",
        lambda value: events.append("install"),
    )
    monkeypatch.setattr(
        launch_imagination.runpy,
        "run_module",
        lambda *a, **k: events.append("run"),
    )

    launch_imagination.main([])

    assert events == [
        "hook_points", "runtime", "factories", "run_config", "cache_meta",
        "install", "run"
    ]
