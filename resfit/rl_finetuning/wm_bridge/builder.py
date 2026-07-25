"""bridge 参数解析 + 组装 launcher 的 5 个假符号。

parse_bridge_args 摘走 bridge 专属参数(--value_ckpt/--wm_host/... ),其余原样透传给 trainer
(trainer 的 --pi0_host/--pi0_port/--task/--chunk_length 等不认识 bridge 参数,不摘会报错)。

5 个假符号(见 launch_imagination 的 _TARGETS):
  create_vectorized_env → ImaginationVecEnv(wm=D-serve客户端, base=kai0共享实例, scorer=V, sampler)
  run_dexmg_evaluation  → make_imagination_evaluator(存 checkpoint;可选 adv-logging)
  load_pi05_base_policy → Kai0ImaginationBase(kai0 serve 客户端);与 env 共享同一实例→1 serve/chunk
  count_offline_transitions / build_offline_buffer → Teleavatar kai0 chunk replay

★ 共享实例:trainer 先 build_base_policy(→ load_pi05_base_policy 假物,建 Kai0ImaginationBase 存 state),
   再 create_vectorized_env(用 state["base"] 建 env)。两处同一 base,窗口缓存跨两个调用方生效。
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, is_dataclass
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from resfit.rl_finetuning.wm_bridge.block_offline_cache import (
    EndpointStore,
    dataset_manifest,
    endpoint_fingerprint,
    file_sha256,
    replay_fingerprint,
    resolve_replay_generation,
)
from resfit.rl_finetuning.wm_bridge.block_offline_chunk import (
    ENDPOINT_CACHE_READ_ERRORS,
    EpisodeRef,
    build_block_offline_buffer,
    catalog_episodes,
    endpoint_record_covers_slices,
    plan_episode_chunks,
)
from resfit.rl_finetuning.wm_bridge.wm_driver import CAMERA_KEYS


_BUILD_STATS_FIELDS = (
    "episodes",
    "skipped_short_episodes",
    "transitions",
    "endpoint_hits",
    "endpoint_misses",
    "reward_mean",
    "reward_std",
    "potential_delta_mean",
    "potential_delta_std",
    "expert_norm_mean",
    "base_norm_mean",
    "residual_norm_mean",
    "expert_saturation_fraction",
    "base_saturation_fraction",
)


def parse_bridge_args(argv):
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--value_ckpt", required=True, help="Teleavatar V (pi0_feat value checkpoint)")
    p.add_argument("--wm_host", default="127.0.0.1", help="D-serve host(d_serve.py)")
    p.add_argument("--wm_port", type=int, default=9000, help="D-serve port")
    p.add_argument("--init_state_dataset", action="append", default=[],
                   help="想象起点 Teleavatar 数据集完整路径,可重复。成功/失败都传")
    p.add_argument("--pi0_serve_ckpt_id", default=None,
                   help="在线 kai0 serve 标签,与 value.pt 的 pi0_feat_signature.serve_ckpt_id 比对")
    p.add_argument("--imagination_gamma", type=float, default=0.995)
    p.add_argument("--max_segments", type=int, default=2)
    p.add_argument("--num_denois_steps", type=int, default=10)
    p.add_argument("--action_norm_json", default=None, help="Teleavatar 动作 min/max JSON;缺省 ±1")
    p.add_argument("--allow_dummy_scorer", action="store_true")
    p.add_argument("--offline_chunk_dataset", default=None)
    p.add_argument("--offline_chunk_cache_root", default=None)
    p.add_argument("--offline_rebuild", action="store_true")
    # 优势估计器 proxy 监控(spec §6.7):给了 --adv_host 才开;每 eval rollout N 集逐帧打分写 jsonl
    p.add_argument("--adv_host", default=None,
                   help="优势估计器 serve host(adv_serve.py);给了才开 adv proxy 监控")
    p.add_argument("--adv_port", type=int, default=8002)
    p.add_argument(
        "--task_profile",
        choices=("block", "cup", "paper"),
        default="block",
        help="teleavatar task profile controlling prompt and success source",
    )
    p.add_argument("--adv_prompt", default=None)
    p.add_argument("--n_eval_episodes", type=int, default=10,
                   help="每 eval 点 adv rollout 集数(spec §6.7 默认 10)")
    bridge_args, rest = p.parse_known_args(argv)
    return bridge_args, rest


@dataclass(frozen=True)
class OfflineRuntime:
    dataset_root: str
    all_episodes: tuple[EpisodeRef, ...]
    num_demos: int | None
    endpoint_fingerprint: str
    replay_fingerprint: str
    endpoint_store: EndpointStore
    replay_cache_dir: str
    bridge_meta: dict

    def selected_episodes(
        self,
        override: int | None = None,
    ) -> tuple[EpisodeRef, ...]:
        limit = self.num_demos if override is None else override
        if limit is not None and limit < 0:
            raise ValueError("num_demos must be non-negative")
        return self.all_episodes if limit is None else self.all_episodes[:limit]


def _replace_option(argv, name, value):
    from resfit.rl_finetuning.wm_bridge.contract import ContractError

    output, index = [], 0
    while index < len(argv):
        token = argv[index]
        if token == name:
            if index + 1 >= len(argv) or argv[index + 1].startswith("--"):
                raise ContractError(f"{name} requires exactly one value")
            index += 2
            continue
        if token.startswith(name + "="):
            index += 1
            continue
        output.append(token)
        index += 1
    output.extend([name, str(value)])
    return output


def _write_json_atomic(directory, filename, metadata):
    target_dir = Path(directory)
    target_dir.mkdir(parents=True, exist_ok=True)
    final_path = target_dir / filename
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target_dir,
            prefix=f".{filename}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temp_path = Path(stream.name)
            json.dump(metadata, stream, indent=2, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, final_path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def write_bridge_run_config(output_dir, metadata):
    _write_json_atomic(output_dir, "bridge_run_config.json", metadata)


def write_bridge_cache_meta(replay_cache_dir, metadata):
    _write_json_atomic(replay_cache_dir, "bridge_meta.json", metadata)


def _resolve_trainer_normalization_source(dataset_root):
    """Resolve the files LeRobotDatasetMetadata will actually use.

    The generic trainer runs its dexmg/HDF5 path with ``root=None``, so
    LeRobot resolves metadata as ``HF_LEROBOT_HOME / repo_id``.  Make that
    implicit process-wide lookup explicit and fail before cache generation if
    it is not the same dataset tree used by the offline bridge.
    """
    from resfit.rl_finetuning.wm_bridge.contract import ContractError

    lerobot_home = os.environ.get("HF_LEROBOT_HOME")
    if not lerobot_home:
        raise ContractError(
            "HF_LEROBOT_HOME is required for mixed replay so trainer "
            "normalization can be tied to offline_chunk_dataset")
    dataset_name = os.path.basename(dataset_root)
    trainer_root = os.path.realpath(
        os.path.join(os.path.expanduser(lerobot_home), dataset_name))
    if trainer_root != dataset_root:
        raise ContractError(
            "HF_LEROBOT_HOME/--dataset must resolve to offline_chunk_dataset: "
            f"trainer={trainer_root!r}, offline={dataset_root!r}")

    meta_dir = Path(dataset_root) / "meta"
    info_path = meta_dir / "info.json"
    if not info_path.is_file():
        raise ContractError(
            f"trainer normalization metadata is missing {info_path}")
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(
            f"cannot read trainer normalization metadata {info_path}: {exc}") from exc
    version = str(info.get("codebase_version", ""))
    # LeRobotDatasetMetadata aggregates v2.1+ stats from episodes_stats.jsonl;
    # older datasets use stats.json directly.
    try:
        version_numbers = tuple(
            int(part) for part in version.removeprefix("v").split(".")[:2])
    except ValueError as exc:
        raise ContractError(
            f"invalid codebase_version {version!r} in {info_path}") from exc
    if len(version_numbers) != 2:
        raise ContractError(
            f"invalid codebase_version {version!r} in {info_path}")
    stats_name = (
        "episodes_stats.jsonl" if version_numbers >= (2, 1) else "stats.json")
    stats_path = meta_dir / stats_name
    if not stats_path.is_file():
        raise ContractError(
            f"trainer normalization statistics are missing {stats_path}")
    return trainer_root, str(stats_path.resolve())


def _endpoint_cache_status(endpoint_store, episodes):
    eligible = []
    for episode in episodes:
        slices = plan_episode_chunks(episode.num_frames)
        if slices:
            eligible.append((episode, slices))
    if not eligible:
        return "miss"

    complete_hits = 0
    for episode, slices in eligible:
        try:
            record = endpoint_store.load_episode(episode.episode_id)
        except ENDPOINT_CACHE_READ_ERRORS:
            record = None
        if record is None:
            continue
        if endpoint_record_covers_slices(record, slices):
            complete_hits += 1
    if complete_hits == len(eligible):
        return "hit"
    if complete_hits:
        return "partial"
    return "miss"


def _read_complete_build_stats(replay_path):
    try:
        metadata = json.loads(
            (Path(replay_path) / "bridge_meta.json").read_text(
                encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    stats = metadata.get("build_stats")
    if not isinstance(stats, dict) or not set(_BUILD_STATS_FIELDS).issubset(stats):
        return None
    if not all(
        isinstance(stats[field], (int, float))
        and not isinstance(stats[field], bool)
        and np.isfinite(float(stats[field]))
        for field in _BUILD_STATS_FIELDS
    ):
        return None
    return stats


def prepare_offline_runtime(bridge_args, passthrough):
    from resfit.rl_finetuning.wm_bridge import contract
    from resfit.rl_finetuning.wm_bridge.task_profiles import get_task_profile

    profile = get_task_profile(bridge_args.task_profile)

    if bridge_args.offline_chunk_dataset is None:
        return None, list(passthrough)

    dataset_root = os.path.realpath(bridge_args.offline_chunk_dataset)
    dataset_name = os.path.basename(dataset_root)
    translated = list(passthrough)
    translated = _replace_option(
        translated, "--pi0_prompt", profile.prompt)
    translated = _replace_option(
        translated, "--pi0_action_dim", profile.action_dim)
    translated = _replace_option(translated, "--dataset", dataset_name)
    translated = _replace_option(translated, "--data_source", "hdf5")
    parsed = contract.parse_mixed_passthrough(translated)
    contract.check_mixed_replay_args(
        parsed,
        dataset_root,
        expected_prompt=profile.prompt,
        expected_source=profile.success_dataset,
    )

    if not bridge_args.offline_chunk_cache_root:
        raise contract.ContractError(
            "offline_chunk_cache_root is required for mixed replay")
    if not bridge_args.pi0_serve_ckpt_id:
        raise contract.ContractError(
            "pi0_serve_ckpt_id is required for endpoint fingerprinting")
    if parsed.offline_num_demos is not None and parsed.offline_num_demos <= 0:
        raise contract.ContractError("offline_num_demos must be positive")

    cache_root = os.path.realpath(bridge_args.offline_chunk_cache_root)
    normalization_root, stats_path = _resolve_trainer_normalization_source(
        dataset_root)
    all_episodes = catalog_episodes(dataset_root)
    selected = (
        all_episodes
        if parsed.offline_num_demos is None
        else all_episodes[:parsed.offline_num_demos]
    )
    if not selected:
        raise contract.ContractError("offline_num_demos selected no episodes")

    manifest = dataset_manifest(all_episodes)
    value_sha256 = file_sha256(bridge_args.value_ckpt)
    stats_sha256 = file_sha256(stats_path)
    endpoint_fp = endpoint_fingerprint(
        manifest=manifest,
        pi0_serve_ckpt_id=bridge_args.pi0_serve_ckpt_id,
        prompt=parsed.pi0_prompt,
        camera_keys=CAMERA_KEYS,
        chunk_length=50,
        stride=50,
    )
    replay_fp = replay_fingerprint(
        endpoint_fp=endpoint_fp,
        value_sha256=value_sha256,
        gamma=parsed.gamma,
        num_demos=len(selected),
        stats_sha256=stats_sha256,
        action_scale=parsed.action_scale,
        min_range_per_dim=parsed.min_range_per_dim,
        image_size=84,
        image_keys=CAMERA_KEYS,
        n_step=parsed.n_step,
    )
    offline_transitions = sum(
        len(plan_episode_chunks(episode.num_frames))
        for episode in selected
    )
    from resfit.rl_finetuning.chunk_residual.chunk_env_wrapper import (
        resolve_shaping_mode,
    )
    from resfit.rl_finetuning.chunk_residual.train_chunk_residual import (
        _offline_buffer_signature,
        _offline_cache_valid,
        build_parser,
    )

    signature_argv = _replace_option(
        translated, "--offline_dataset_path", dataset_root)
    signature_argv = _replace_option(
        signature_argv, "--offline_base_mode", "gt")
    signature_args = build_parser().parse_args(signature_argv)
    trainer_cache_signature = _offline_buffer_signature(
        signature_args,
        CAMERA_KEYS,
        offline_transitions,
        resolve_shaping_mode(
            signature_args.reward_shaping,
            signature_args.staged_reward,
        ),
    )

    def replay_generation_valid(path):
        return (
            _offline_cache_valid(str(path), trainer_cache_signature)
            and _read_complete_build_stats(path) is not None
        )

    endpoint_store = EndpointStore(
        cache_root,
        endpoint_fp,
        force_rebuild=bridge_args.offline_rebuild,
    )
    replay_cache_dir = resolve_replay_generation(
        cache_root,
        replay_fp,
        force_rebuild=bridge_args.offline_rebuild,
        is_valid=replay_generation_valid,
    )
    endpoint_cache_status = _endpoint_cache_status(endpoint_store, selected)
    replay_path = Path(replay_cache_dir)
    cached_build_stats = (
        _read_complete_build_stats(replay_path)
        if replay_generation_valid(replay_path)
        else None
    )
    replay_cache_status = "hit" if cached_build_stats is not None else "miss"
    online_batch_size = int(
        parsed.batch_size * (1.0 - parsed.offline_fraction))
    offline_batch_size = int(
        parsed.batch_size * parsed.offline_fraction)
    metadata = {
        "schema": 1,
        "task_profile": profile.name,
        "offline_source": dataset_name,
        "dataset_root": dataset_root,
        "offline_episodes": len(selected),
        "offline_transitions": offline_transitions,
        "endpoint_fingerprint": endpoint_fp,
        "replay_fingerprint": replay_fp,
        "endpoint_cache_dir": str(endpoint_store.directory),
        "replay_cache_dir": replay_cache_dir,
        "value_sha256": value_sha256,
        "stats_sha256": stats_sha256,
        "normalization_dataset_root": normalization_root,
        "normalization_stats_path": stats_path,
        "gamma": float(parsed.gamma),
        "num_demos": len(selected),
        "action_scale": float(parsed.action_scale),
        "min_range_per_dim": float(parsed.min_range_per_dim),
        "chunk_length": 50,
        "n_step": int(parsed.n_step),
        "image_size": 84,
        "image_keys": list(CAMERA_KEYS),
        "pi0_prompt": parsed.pi0_prompt,
        "pi0_action_dim": int(parsed.pi0_action_dim),
        "pi0_serve_ckpt_id": bridge_args.pi0_serve_ckpt_id,
        "trainer_cache_signature": trainer_cache_signature,
        "trainer_compat_mode": "gt",
        "actual_base_mode": "kai0_chunk",
        "actual_offline_base": "kai0_chunk",
        "offline_reward": "gamma_phi_next_minus_phi",
        "online_batch_size": online_batch_size,
        "offline_batch_size": offline_batch_size,
        "endpoint_cache_status": endpoint_cache_status,
        "replay_cache_status": replay_cache_status,
        "endpoint_cache": endpoint_cache_status,
        "replay_cache": replay_cache_status,
        "offline_rebuild": bool(bridge_args.offline_rebuild),
        "output_dir": parsed.output_dir,
    }
    if cached_build_stats is not None:
        metadata["build_stats"] = cached_build_stats
    runtime = OfflineRuntime(
        dataset_root=dataset_root,
        all_episodes=all_episodes,
        num_demos=parsed.offline_num_demos,
        endpoint_fingerprint=endpoint_fp,
        replay_fingerprint=replay_fp,
        endpoint_store=endpoint_store,
        replay_cache_dir=replay_cache_dir,
        bridge_meta=metadata,
    )
    translated = _replace_option(
        translated, "--offline_dataset_path", dataset_root)
    translated = _replace_option(
        translated, "--offline_buffer_cache", replay_cache_dir)
    translated = _replace_option(translated, "--offline_base_mode", "gt")
    return runtime, translated


def _normalizer(path):
    from resfit.rl_finetuning.wm_bridge.wm_driver import ACTION_DIM, ActionNormalizer
    if path is None:
        return ActionNormalizer(-np.ones(ACTION_DIM, np.float32), np.ones(ACTION_DIM, np.float32))
    import json
    d = json.load(open(path))
    return ActionNormalizer(np.asarray(d["min"], np.float32), np.asarray(d["max"], np.float32))


def _check_offline_source(dataset_path, runtime):
    from resfit.rl_finetuning.wm_bridge.contract import ContractError

    if os.path.realpath(dataset_path) != os.path.realpath(runtime.dataset_root):
        raise ContractError(
            f"dataset_path {dataset_path!r} does not match "
            f"{runtime.dataset_root!r}")


def format_offline_startup_banner(runtime):
    meta = runtime.bridge_meta
    return (
        f"offline_source={meta['offline_source']} "
        f"offline_episodes={meta['offline_episodes']} "
        f"offline_transitions={meta['offline_transitions']} "
        f"batch={meta['online_batch_size']}+{meta['offline_batch_size']} "
        f"online_batch_size={meta['online_batch_size']} "
        f"offline_batch_size={meta['offline_batch_size']} "
        f"trainer_compat_mode={meta['trainer_compat_mode']} "
        f"actual_offline_base={meta['actual_offline_base']} "
        f"offline_reward={meta['offline_reward']} "
        f"endpoint_cache={meta['endpoint_cache']} "
        f"replay_cache={meta['replay_cache']}"
    )


def _build_stats_dict(stats):
    if is_dataclass(stats):
        return asdict(stats)
    return dict(stats)


def format_offline_build_stats(stats, runtime):
    values = _build_stats_dict(stats)
    details = " ".join(
        f"{field}={values[field]:g}" for field in _BUILD_STATS_FIELDS)
    return f"offline_build_stats {details} replay_cache={runtime.replay_cache_dir}"


def make_offline_factories(offline_runtime, state, scorer) -> dict:
    from resfit.rl_finetuning.wm_bridge.contract import ContractError

    def fake_count_offline_transitions(dataset_path, num_demos=None):
        _check_offline_source(dataset_path, offline_runtime)
        episodes = offline_runtime.selected_episodes(num_demos)
        return sum(
            len(plan_episode_chunks(episode.num_frames))
            for episode in episodes
        )

    def fake_build_offline_buffer(offline_rb, dataset_path, **kwargs):
        _check_offline_source(dataset_path, offline_runtime)
        if state["base"] is None:
            raise RuntimeError("kai0 base must be created before offline replay")

        kwargs = dict(kwargs)
        kwargs.pop("base_policy", None)
        base_mode = kwargs.pop("base_mode", None)
        if base_mode != "gt":
            raise ContractError(
                "offline base_mode must be explicitly 'gt' as a trainer "
                "compatibility marker")
        for key in (
            "bonus",
            "mode",
            "stage_cache",
            "potential",
            "subgoal",
            "way_steps",
            "act_feat_seqs",
        ):
            kwargs.pop(key, None)
        kwargs.pop("scorer", None)
        kwargs.pop("endpoint_store", None)
        kwargs.pop("episodes", None)

        num_demos = kwargs.get("num_demos")
        kwargs["num_demos"] = num_demos
        stats = build_block_offline_buffer(
            offline_rb,
            dataset_path,
            base_policy=state["base"],
            scorer=scorer,
            endpoint_store=offline_runtime.endpoint_store,
            episodes=offline_runtime.selected_episodes(num_demos),
            **kwargs,
        )
        offline_runtime.bridge_meta["build_stats"] = _build_stats_dict(stats)
        write_bridge_run_config(
            offline_runtime.bridge_meta["output_dir"],
            offline_runtime.bridge_meta,
        )
        write_bridge_cache_meta(
            offline_runtime.replay_cache_dir,
            offline_runtime.bridge_meta,
        )
        print(format_offline_build_stats(stats, offline_runtime))

    return {
        "count_offline_transitions": fake_count_offline_transitions,
        "build_offline_buffer": fake_build_offline_buffer,
    }


def build_imagination_factories(bridge_args, offline_runtime=None) -> dict:
    from resfit.rl_finetuning.wm_bridge import contract
    from resfit.rl_finetuning.wm_bridge.base_bridge import Kai0ImaginationBase
    from resfit.rl_finetuning.wm_bridge.fake_eval import make_imagination_evaluator
    from resfit.rl_finetuning.wm_bridge.imagination_env import ImaginationVecEnv
    from resfit.rl_finetuning.wm_bridge.scorers import Kai0HiqlScorer
    from resfit.rl_finetuning.wm_bridge.task_profiles import get_task_profile
    from resfit.rl_finetuning.wm_bridge.teleavatar_start_sampler import TeleavatarStartSampler
    from resfit.rl_finetuning.wm_bridge.wm_client import WmServeClient

    profile = get_task_profile(bridge_args.task_profile)
    scorer = Kai0HiqlScorer.from_value_ckpt(bridge_args.value_ckpt)
    contract.check_scorer(scorer, bridge_args.allow_dummy_scorer)
    mixed_enabled = offline_runtime is not None
    contract.check_mixed_scorer(
        scorer,
        enabled=mixed_enabled,
        serve_ckpt_id=bridge_args.pi0_serve_ckpt_id,
    )
    if not mixed_enabled and not bridge_args.allow_dummy_scorer:
        contract.check_psi_samesource(scorer, bridge_args.pi0_serve_ckpt_id)
    normalizer = _normalizer(bridge_args.action_norm_json)

    # 优势估计器打分器(adv proxy 监控):给了 --adv_host 才建;否则 eval 只存 checkpoint。
    adv_scorer = None
    if bridge_args.adv_host:
        from resfit.rl_finetuning.wm_bridge.adv_client import AdvServeClient
        adv_prompt = bridge_args.adv_prompt or profile.prompt
        adv_scorer = AdvServeClient(
            host=bridge_args.adv_host,
            port=bridge_args.adv_port,
            prompt=adv_prompt,
        )

    state = {"base": None, "env": None, "output_dir": None, "config": None}

    def fake_load_pi05_base_policy(cfg, device, schema="dexmg"):
        from openpi_client.websocket_client_policy import WebsocketClientPolicy
        # queue 模式 eval 会建第二个 base 实例;想象路只用一个(env+trainer 共享),复用 state["base"]
        if state["base"] is None:
            client = WebsocketClientPolicy(host=cfg.host, port=cfg.port)
            state["base"] = Kai0ImaginationBase(
                client, prompt=getattr(cfg, "prompt", "build block"),
                action_dim=getattr(cfg, "action_dim", 16),
                policy_state_dim=profile.policy_state_dim,
            )
        return state["base"]

    def fake_create_vectorized_env(*, env_name=None, num_envs=1, device="cpu",
                                   state_mode=None, **kw):
        if state["env"] is not None:
            return state["env"]                       # eval_vec 复用同一 env(不另起想象流)
        assert state["base"] is not None, \
            "load_pi05_base_policy(建 base)须在 create_vectorized_env 之前调"
        state["env"] = ImaginationVecEnv(
            wm=WmServeClient(host=bridge_args.wm_host, port=bridge_args.wm_port),
            base=state["base"], scorer=scorer,
            sampler=TeleavatarStartSampler(
                bridge_args.init_state_dataset,
                caption=profile.prompt,
            ),
            normalizer=normalizer, gamma=bridge_args.imagination_gamma,
            max_segments=bridge_args.max_segments,
            num_denois_steps=bridge_args.num_denois_steps, device=device)
        return state["env"]

    def fake_run_dexmg_evaluation(**kw):
        out_dir = kw.get("output_dir") or state["output_dir"] or "outputs_imagination"
        return make_imagination_evaluator(
            out_dir, state["config"], adv_scorer=adv_scorer,
            n_eval_episodes=bridge_args.n_eval_episodes)(**kw)

    factories = {
        "create_vectorized_env": fake_create_vectorized_env,
        "run_dexmg_evaluation": fake_run_dexmg_evaluation,
        "load_pi05_base_policy": fake_load_pi05_base_policy,
    }
    if offline_runtime is not None:
        factories.update(make_offline_factories(
            offline_runtime, state, scorer))
    return factories
