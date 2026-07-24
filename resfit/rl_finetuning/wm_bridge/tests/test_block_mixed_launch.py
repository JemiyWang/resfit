import json
from pathlib import Path
import shlex
import types

import pandas as pd
import pytest

from resfit.rl_finetuning.wm_bridge import builder
from resfit.rl_finetuning.wm_bridge import contract
from resfit.rl_finetuning.wm_bridge import launch_imagination
from resfit.rl_finetuning.wm_bridge.builder import (
    make_offline_factories,
    parse_bridge_args,
    prepare_offline_runtime,
    write_bridge_cache_meta,
    write_bridge_run_config,
)
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
        "--bc_coef_final": "0.01",
        "--critic_warmup_steps": "10000",
        "--learning_starts": "10000",
    }
    for flag, value in required.items():
        assert tokens.count(flag) == 1
        assert tokens[tokens.index(flag) + 1] == value
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


def test_production_batch_is_exactly_128_plus_128():
    batch_size, fraction = 256, 0.5
    assert int(batch_size * (1 - fraction)) == 128
    assert int(batch_size * fraction) == 128


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


def _write_success_dataset(parent):
    root = parent / "block_success"
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


def test_offline_factories_count_and_forward_shared_objects(monkeypatch):
    runtime = RuntimeStub()
    shared_base, shared_scorer = object(), object()
    state = {"base": shared_base}
    captured = {}

    def fake_build(rb, path, **kwargs):
        captured.update(kwargs)
        return types.SimpleNamespace(transitions=3)

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
    ):
        argv = builder._replace_option(argv, flag, value)

    bridge_args, passthrough = parse_bridge_args(argv)
    runtime, _ = prepare_offline_runtime(bridge_args, passthrough)
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
        bridge_meta={"actual_base_mode": "kai0_chunk"},
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
