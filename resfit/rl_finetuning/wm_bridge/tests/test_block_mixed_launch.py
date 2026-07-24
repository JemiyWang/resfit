import json
from pathlib import Path
import types

import pandas as pd
import pytest

from resfit.rl_finetuning.wm_bridge import builder
from resfit.rl_finetuning.wm_bridge import contract
from resfit.rl_finetuning.wm_bridge import launch_imagination
from resfit.rl_finetuning.wm_bridge.builder import (
    parse_bridge_args,
    prepare_offline_runtime,
    write_bridge_cache_meta,
    write_bridge_run_config,
)
from resfit.rl_finetuning.wm_bridge.contract import ContractError
from resfit.rl_finetuning.wm_bridge.wm_driver import CAMERA_KEYS


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
    stats.write_text('{"action": {}, "state": {}}', encoding="utf-8")
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
        "--pi0_prompt", "build block",
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


def test_replace_option_rejects_bare_authoritative_flag():
    with pytest.raises(ContractError, match="offline_dataset_path"):
        builder._replace_option(
            ["--offline_dataset_path", "--batch_size", "256"],
            "--offline_dataset_path",
            "/data/block_success",
        )


def test_prepare_runtime_fingerprints_and_replaces_trainer_options(tmp_path):
    dataset = _write_success_dataset(tmp_path)
    value_ckpt = tmp_path / "value.pt"
    value_ckpt.write_bytes(b"value-weights")
    cache_root = tmp_path / "cache"
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
    assert translated[translated.index("--offline_dataset_path") + 1] == str(
        dataset.resolve())
    assert translated[translated.index("--offline_buffer_cache") + 1] == (
        runtime.replay_cache_dir)
    assert translated[translated.index("--offline_base_mode") + 1] == "gt"

    meta = runtime.bridge_meta
    assert meta["trainer_compat_mode"] == "gt"
    assert meta["actual_base_mode"] == "kai0_chunk"
    assert meta["offline_reward"] == "gamma_phi_next_minus_phi"
    assert meta["online_batch_size"] == 128
    assert meta["offline_batch_size"] == 128
    assert meta["pi0_serve_ckpt_id"] == "pi05_block_awbc_49999"


def test_metadata_writers_publish_atomic_json(tmp_path):
    metadata = {"actual_base_mode": "kai0_chunk", "online_batch_size": 128}
    output = tmp_path / "output"
    cache = tmp_path / "cache-generation"

    write_bridge_run_config(str(output), metadata)
    write_bridge_cache_meta(str(cache), metadata)

    assert json.loads((output / "bridge_run_config.json").read_text()) == metadata
    assert json.loads((cache / "bridge_meta.json").read_text()) == metadata


def test_prepare_pure_online_is_a_noop(tmp_path):
    value_ckpt = tmp_path / "value.pt"
    value_ckpt.write_bytes(b"unused")
    bridge_args, _ = parse_bridge_args([
        "--value_ckpt", str(value_ckpt),
    ])
    passthrough = ["--offline_fraction", "0.0", "--batch_size", "17"]

    runtime, translated = prepare_offline_runtime(bridge_args, passthrough)

    assert runtime is None
    assert translated == passthrough


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
        "runtime", "factories", "run_config", "cache_meta", "install", "run"
    ]
