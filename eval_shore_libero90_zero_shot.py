#!/usr/bin/env python3
"""Frozen task-8 SHORE evaluation on four LIBERO-90 tasks.

Heavy LIBERO, OpenPI, Torch, and plotting imports are intentionally lazy so
help, offline tests, plotting, and launcher dry-runs do not require a GPU.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shlex
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from unittest import mock


TASK_IDS = (3, 8, 21, 63)
DEFAULT_PORTS = (8000, 8001, 8002, 8003)
TASK_LANGUAGES = {
    3: "put the butter at the back in the top drawer of the cabinet and close it",
    8: "open the top drawer of the cabinet and put the bowl in it",
    21: "turn on the stove and put the frying pan on it",
    63: "stack the left bowl on the right bowl and place them in the tray",
}
TASK_TITLES = {
    3: "Butter in top drawer",
    8: "Bowl in top drawer",
    21: "Pan on lit stove",
    63: "Stack bowls in tray",
}

REPO_ROOT = Path("/mnt/mnt/data/resfit")
OPENPI_ROOT = Path("/mnt/mnt/data/chj/openpi")
EVAL_PYTHON = Path("/mnt/mnt/data/envs/resfit-libero/bin/python")
OPENPI_PYTHON = OPENPI_ROOT / ".venv/bin/python"
BASE_CHECKPOINT = (
    OPENPI_ROOT
    / "checkpoints/pi0_libero/"
    "pi0_libero_is-dceq77jzghdxvjj2-devmachine-0_20260523_220232/29999"
)
RESIDUAL_CHECKPOINT = (
    REPO_ROOT / "outputs_chunk/libero10_task8_pi0feat_bp_bc01_h10/best.pt"
)
GC_VALUE_CHECKPOINT = (
    REPO_ROOT / "outputs_chunk/libero10_t8moka_pi0_feat_gc_value.pt"
)
HIGH_ACTOR_CHECKPOINT = (
    REPO_ROOT / "outputs_chunk/libero10_t8moka_pi0_feat_high_actor.pt"
)
PI0_FEATURE_CACHE = REPO_ROOT / "outputs_chunk/libero10_t8moka_pi0_feat.npz"
PI0_STATS = Path(
    "/mnt/mnt/data/chj/lerobot_cache/physical-intelligence/libero/meta/stats.json"
)
DEFAULT_JSON_OUT = REPO_ROOT / "paper/data/fig_shore_libero90_zero_shot.json"
DEFAULT_NPZ_OUT = REPO_ROOT / "paper/data/fig_shore_libero90_zero_shot.npz"
DEFAULT_PDF_OUT = REPO_ROOT / "paper/figure/fig_shore_libero90_zero_shot.pdf"

ROW_FIELDS = {
    "task_id",
    "task_language",
    "episode_index",
    "init_state_id",
    "success",
    "return",
    "length",
    "elapsed_seconds",
    "seed",
    "source_checkpoint_sha256",
}
RESUME_KEYS = ("suite", "task_id", "seed", "source_checkpoint_sha256")
SIGNATURE_KEYS = ("serve_ckpt_id", "image_keys", "proprio_key", "pooling", "prompt")


def rolling_success(successes: Sequence[float], window: int = 10):
    """Return a trailing success mean with partial leading windows."""
    import numpy as np

    if window < 1:
        raise ValueError("window must be at least one")
    values = np.asarray(successes, dtype=np.float64)
    return np.asarray(
        [
            values[max(0, index - window + 1) : index + 1].mean()
            for index in range(len(values))
        ],
        dtype=np.float64,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_episode_coverage(
    rows: Sequence[Mapping[str, Any]], expected_ids: Iterable[int]
) -> None:
    expected = list(expected_ids)
    actual = [int(row["init_state_id"]) for row in rows]
    duplicates = sorted(key for key, count in Counter(actual).items() if count > 1)
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    if duplicates or missing or unexpected or len(actual) != len(expected):
        raise ValueError(
            "invalid init-state coverage: "
            f"duplicate={duplicates}, missing={missing}, unexpected={unexpected}"
        )


def _validate_row(row: Mapping[str, Any], *, line_number: int | None = None) -> None:
    missing = sorted(ROW_FIELDS - set(row))
    where = f" on line {line_number}" if line_number is not None else ""
    if missing:
        raise ValueError(f"episode row{where} is missing fields: {missing}")
    if not isinstance(row["success"], bool):
        raise ValueError(f"episode success{where} must be boolean")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(f"blank JSONL row on line {line_number}: {path}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSONL on line {line_number}: {path}: {error}"
                ) from error
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row on line {line_number} is not an object")
            _validate_row(row, line_number=line_number)
            rows.append(row)
    return rows


class EpisodeJsonlWriter:
    """Append complete JSONL rows durably, one episode at a time."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.stream = None

    def __enter__(self) -> "EpisodeJsonlWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("a", encoding="utf-8")
        return self

    def append(self, row: Mapping[str, Any]) -> None:
        if self.stream is None:
            raise RuntimeError("EpisodeJsonlWriter must be used as a context manager")
        _validate_row(row)
        self.stream.write(json.dumps(dict(row), sort_keys=True) + "\n")
        self.stream.flush()
        os.fsync(self.stream.fileno())

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self.stream is not None:
            self.stream.close()
            self.stream = None


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_text_atomic(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def validate_resume_config(path: Path, expected: Mapping[str, Any]) -> None:
    path = Path(path)
    if not path.exists():
        return
    try:
        actual = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read existing resume config {path}: {error}") from error
    keys = list(RESUME_KEYS)
    for optional in ("source_paths", "source_hashes"):
        if optional in expected:
            keys.append(optional)
    differences = [key for key in keys if actual.get(key) != expected.get(key)]
    if differences:
        details = ", ".join(
            f"{key}: existing={actual.get(key)!r}, requested={expected.get(key)!r}"
            for key in differences
        )
        raise ValueError(f"incompatible resume config: {details}")


def missing_init_ids(rows: Sequence[Mapping[str, Any]], num_episodes: int) -> list[int]:
    if num_episodes < 1 or num_episodes > 50:
        raise ValueError("num_episodes must be in [1, 50]")
    actual = [int(row["init_state_id"]) for row in rows]
    duplicates = sorted(key for key, count in Counter(actual).items() if count > 1)
    unexpected = sorted(set(actual) - set(range(num_episodes)))
    if duplicates or unexpected:
        raise ValueError(
            f"invalid resume rows: duplicate={duplicates}, unexpected={unexpected}"
        )
    completed = set(actual)
    return [init_id for init_id in range(num_episodes) if init_id not in completed]


def validate_pi0_feature_signature(
    cache_signature: Mapping[str, Any],
    gc_signature: Mapping[str, Any] | None,
) -> None:
    if not gc_signature:
        raise ValueError("GC value checkpoint has no pi0_feat_signature")
    mismatches = []
    for key in SIGNATURE_KEYS:
        expected = gc_signature.get(key)
        if expected is not None and cache_signature.get(key) != expected:
            mismatches.append(
                f"{key}: cache={cache_signature.get(key)!r}, gc={expected!r}"
            )
    if mismatches:
        raise ValueError("PI0 feature signature mismatch: " + "; ".join(mismatches))


def freeze_and_validate(module, name: str) -> None:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    trainable = [
        parameter_name
        for parameter_name, parameter in module.named_parameters()
        if parameter.requires_grad
    ]
    if trainable:
        raise RuntimeError(f"{name} still has trainable parameters: {trainable}")


def resolve_server_script() -> Path:
    candidates = (
        OPENPI_ROOT / "pi0_serve/serve_with_feat.py",
        REPO_ROOT / "pi0_serve/serve_with_feat.py",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    matches = sorted(OPENPI_ROOT.rglob("serve_with_feat.py"))
    if matches:
        return matches[0].resolve()
    raise FileNotFoundError(
        "serve_with_feat.py not found in OpenPI or the ResFiT integration tree"
    )


def _server_argv(port: int) -> list[str]:
    return [
        str(OPENPI_PYTHON),
        str(resolve_server_script()),
        "--config",
        "pi0_libero",
        "--dir",
        str(BASE_CHECKPOINT),
        "--port",
        str(port),
        "--pooling",
        "last",
    ]


def _worker_argv(
    *, task_id: int, port: int, task_dir: Path, episodes: int
) -> list[str]:
    return [
        str(EVAL_PYTHON),
        str(Path(__file__).resolve()),
        "worker",
        "--suite",
        "libero_90",
        "--task-id",
        str(task_id),
        "--seed",
        "1",
        "--num-episodes",
        str(episodes),
        "--device",
        "cuda",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--run-dir",
        str(task_dir),
    ]


def build_launch_groups(
    *,
    gpu_ids: Sequence[int],
    ports: Sequence[int],
    run_id: str,
    episodes: int,
) -> list[dict[str, Any]]:
    if len(gpu_ids) != 4:
        raise ValueError("gpu_ids must contain exactly four IDs")
    if len(ports) != 4:
        raise ValueError("ports must contain exactly four integers")
    if len(set(int(port) for port in ports)) != 4:
        raise ValueError("ports must be distinct")
    if not run_id or Path(run_id).name != run_id:
        raise ValueError("run_id must be one non-empty path component")
    if episodes < 1 or episodes > 50:
        raise ValueError("episodes must be in [1, 50]")
    run_root = REPO_ROOT / "outputs_eval/shore_libero90_zero_shot" / run_id
    groups = []
    for task_id, gpu_id, port in zip(TASK_IDS, gpu_ids, ports):
        task_dir = run_root / f"task{task_id:02d}"
        server_argv = _server_argv(int(port))
        worker_argv = _worker_argv(
            task_id=task_id,
            port=int(port),
            task_dir=task_dir,
            episodes=episodes,
        )
        groups.append(
            {
                "task_id": task_id,
                "gpu_id": int(gpu_id),
                "port": int(port),
                "task_dir": task_dir,
                "server_argv": server_argv,
                "worker_argv": worker_argv,
                "server_command": shlex.join(server_argv),
                "worker_command": shlex.join(worker_argv),
            }
        )
    return groups


def _command_with_environment(environment: Mapping[str, str], argv: Sequence[str]) -> str:
    assignments = " ".join(
        f"{key}={shlex.quote(value)}" for key, value in environment.items()
    )
    return f"{assignments} {shlex.join(argv)}"


def _group_environments(gpu_id: int) -> tuple[dict[str, str], dict[str, str]]:
    python_path = os.pathsep.join(
        [
            str(REPO_ROOT),
            str(OPENPI_ROOT / "src"),
            str(OPENPI_ROOT / "packages/openpi-client/src"),
        ]
    )
    common = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": str(gpu_id),
        "PYTHONPATH": python_path
        + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else ""),
    }
    server = {**common, "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.65"}
    worker = {**common, "MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl"}
    return server, worker


def _wait_for_port(
    host: str,
    port: int,
    server: subprocess.Popen,
    timeout_seconds: float = 180.0,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(
                f"feature server for port {port} exited early with {server.returncode}"
            )
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.25)
    raise TimeoutError(f"feature server did not open {host}:{port} within timeout")


def _run_process_group(group: Mapping[str, Any]) -> tuple[int, str]:
    task_dir = Path(group["task_dir"])
    task_dir.mkdir(parents=True, exist_ok=True)
    server_env, worker_env = _group_environments(int(group["gpu_id"]))
    server = None
    with ExitStack() as stack:
        server_log = stack.enter_context(
            (task_dir / "server.log").open("a", encoding="utf-8", buffering=1)
        )
        stdout_log = stack.enter_context(
            (task_dir / "stdout.log").open("a", encoding="utf-8", buffering=1)
        )
        try:
            server = subprocess.Popen(
                group["server_argv"],
                cwd=REPO_ROOT,
                env=server_env,
                stdout=server_log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            write_text_atomic(task_dir / "server.pid", f"{server.pid}\n")
            _wait_for_port("127.0.0.1", int(group["port"]), server)
            worker = subprocess.Popen(
                group["worker_argv"],
                cwd=REPO_ROOT,
                env=worker_env,
                stdout=stdout_log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            write_text_atomic(task_dir / "worker.pid", f"{worker.pid}\n")
            return_code = worker.wait()
            return return_code, (
                f"task {group['task_id']} worker exited with {return_code}"
            )
        except Exception as error:
            return 1, f"task {group['task_id']} launch failed: {error}"
        finally:
            if server is not None and server.poll() is None:
                server.terminate()
                try:
                    server.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()


def launch_all(args: argparse.Namespace) -> int:
    groups = build_launch_groups(
        gpu_ids=args.gpu_ids,
        ports=args.ports,
        run_id=args.run_id,
        episodes=args.episodes,
    )
    if args.dry_run:
        for group in groups:
            server_env, worker_env = _group_environments(group["gpu_id"])
            server_display = {
                key: server_env[key]
                for key in ("CUDA_VISIBLE_DEVICES", "XLA_PYTHON_CLIENT_MEM_FRACTION")
            }
            worker_display = {
                key: worker_env[key]
                for key in ("CUDA_VISIBLE_DEVICES", "MUJOCO_GL")
            }
            print(
                f"[task {group['task_id']:02d} | GPU {group['gpu_id']} | "
                f"port {group['port']}]"
            )
            print("server:", _command_with_environment(server_display, group["server_argv"]))
            print("worker:", _command_with_environment(worker_display, group["worker_argv"]))
        return 0

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(_run_process_group, groups))
    for _, message in results:
        print(message, file=sys.stderr)
    return 1 if any(return_code != 0 for return_code, _ in results) else 0


def plot_results(
    *,
    run_root: Path,
    expected_episodes: int,
    json_out: Path,
    npz_out: Path,
    pdf_out: Path,
) -> None:
    import numpy as np

    run_root = Path(run_root)
    if expected_episodes < 1:
        raise ValueError("expected_episodes must be positive")
    payload_tasks = []
    common_hash = None
    successes_matrix = []
    returns_matrix = []
    lengths_matrix = []
    rolling_matrix = []
    for task_id in TASK_IDS:
        task_dir = run_root / f"task{task_id:02d}"
        if not (task_dir / "COMPLETED").is_file():
            raise ValueError(f"task {task_id} has no COMPLETED marker")
        rows = load_jsonl(task_dir / "episodes.jsonl")
        validate_episode_coverage(rows, range(expected_episodes))
        rows = sorted(rows, key=lambda row: int(row["init_state_id"]))
        if any(int(row["task_id"]) != task_id for row in rows):
            raise ValueError(f"task {task_id} directory contains rows for another task")
        config = json.loads((task_dir / "config.json").read_text(encoding="utf-8"))
        row_hashes = {str(row["source_checkpoint_sha256"]) for row in rows}
        if len(row_hashes) != 1:
            raise ValueError(f"task {task_id} contains multiple residual hashes")
        task_hash = next(iter(row_hashes))
        if config.get("source_checkpoint_sha256") != task_hash:
            raise ValueError(f"task {task_id} config and rows use different residual hashes")
        if common_hash is None:
            common_hash = task_hash
        elif task_hash != common_hash:
            raise ValueError("tasks do not share one residual checkpoint hash")

        successes = [int(row["success"]) for row in rows]
        returns = [float(row["return"]) for row in rows]
        lengths = [int(row["length"]) for row in rows]
        rolling = rolling_success(successes, window=10)
        successes_matrix.append(successes)
        returns_matrix.append(returns)
        lengths_matrix.append(lengths)
        rolling_matrix.append(rolling)
        payload_tasks.append(
            {
                "task_id": task_id,
                "task_language": rows[0]["task_language"],
                "init_state_ids": [int(row["init_state_id"]) for row in rows],
                "successes": successes,
                "rolling_success": rolling.tolist(),
                "returns": returns,
                "lengths": lengths,
            }
        )

    payload = {
        "experiment": "frozen task8 SHORE zero-shot LIBERO-90 evaluation",
        "suite": "libero_90",
        "source_checkpoint_sha256": common_hash,
        "rolling_window": 10,
        "expected_episodes": expected_episodes,
        "tasks": payload_tasks,
    }
    write_json_atomic(Path(json_out), payload)

    npz_out = Path(npz_out)
    npz_out.parent.mkdir(parents=True, exist_ok=True)
    npz_tmp = npz_out.with_name(f".{npz_out.name}.{os.getpid()}.tmp")
    try:
        with npz_tmp.open("wb") as stream:
            np.savez_compressed(
                stream,
                task_ids=np.asarray(TASK_IDS, dtype=np.int64),
                init_state_ids=np.tile(
                    np.arange(expected_episodes, dtype=np.int64), (4, 1)
                ),
                successes=np.asarray(successes_matrix, dtype=np.int8),
                rolling_success=np.asarray(rolling_matrix, dtype=np.float64),
                returns=np.asarray(returns_matrix, dtype=np.float64),
                lengths=np.asarray(lengths_matrix, dtype=np.int64),
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(npz_tmp, npz_out)
    finally:
        if npz_tmp.exists():
            npz_tmp.unlink()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(8.0, 5.8), sharex=True, sharey=True)
    x_values = np.arange(1, expected_episodes + 1)
    for axis, task in zip(axes.flat, payload_tasks):
        axis.plot(
            x_values,
            task["rolling_success"],
            color="#1677b3",
            linewidth=1.8,
            label="SHORE",
        )
        axis.set_title(f"Task {task['task_id']}: {TASK_TITLES[task['task_id']]}")
        axis.set_ylim(0.0, 1.0)
        axis.set_xlim(1, max(2, expected_episodes))
        axis.grid(alpha=0.25, linewidth=0.6)
        axis.set_xlabel("Evaluation episode")
        axis.set_ylabel("10-episode rolling success rate")
        axis.text(
            0.98,
            0.05,
            f"{sum(task['successes'])} / {expected_episodes}",
            ha="right",
            va="bottom",
            transform=axis.transAxes,
        )
    figure.suptitle(
        "Frozen task8 SHORE weights evaluated zero-shot; no target-task training",
        fontsize=10.5,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    pdf_out = Path(pdf_out)
    pdf_out.parent.mkdir(parents=True, exist_ok=True)
    pdf_tmp = pdf_out.with_name(f".{pdf_out.name}.{os.getpid()}.tmp")
    try:
        figure.savefig(pdf_tmp, format="pdf", bbox_inches="tight")
        os.replace(pdf_tmp, pdf_out)
    finally:
        plt.close(figure)
        if pdf_tmp.exists():
            pdf_tmp.unlink()


def _source_paths(args: argparse.Namespace) -> dict[str, str]:
    return {
        "base_checkpoint": str(Path(args.base_checkpoint).resolve()),
        "residual_checkpoint": str(Path(args.residual_checkpoint).resolve()),
        "gc_value_checkpoint": str(Path(args.gc_value_checkpoint).resolve()),
        "high_actor_checkpoint": str(Path(args.high_actor_checkpoint).resolve()),
        "pi0_feature_cache": str(Path(args.pi0_feature_cache).resolve()),
        "pi0_stats": str(Path(args.pi0_stats).resolve()),
    }


def _source_hashes(args: argparse.Namespace) -> dict[str, str]:
    return {
        "residual_checkpoint": sha256_file(Path(args.residual_checkpoint)),
        "gc_value_checkpoint": sha256_file(Path(args.gc_value_checkpoint)),
        "high_actor_checkpoint": sha256_file(Path(args.high_actor_checkpoint)),
        "pi0_feature_cache": sha256_file(Path(args.pi0_feature_cache)),
        "pi0_stats": sha256_file(Path(args.pi0_stats)),
    }


def _set_init_state_id(chunk_env, init_state_id: int) -> None:
    vector_wrapper = chunk_env.vec_env
    sync_vector = vector_wrapper.vec_env
    if not hasattr(sync_vector, "envs") or len(sync_vector.envs) != 1:
        raise RuntimeError("worker requires one synchronous LIBERO environment")
    raw_env = sync_vector.envs[0]
    while not hasattr(raw_env, "init_idx") and hasattr(raw_env, "env"):
        raw_env = raw_env.env
    if not hasattr(raw_env, "init_idx"):
        raise RuntimeError("cannot reach LiberoGymWrapper.init_idx")
    raw_env.init_idx = int(init_state_id)


def _build_frozen_components(args: argparse.Namespace):
    import torch

    from resfit.rl_finetuning.chunk_residual.chunk_env_wrapper import (
        ChunkResidualEnvWrapper,
    )
    from resfit.rl_finetuning.chunk_residual.hiql_gc_value import load_gc_value
    from resfit.rl_finetuning.chunk_residual.hiql_subgoal import (
        HiqlSubgoal,
        representative_goal,
    )
    from resfit.rl_finetuning.chunk_residual.libero_env import (
        create_libero_vectorized_env,
    )
    from resfit.rl_finetuning.chunk_residual.pi0_feat_cache import (
        load_pi0_feat_cache,
    )
    from resfit.rl_finetuning.chunk_residual.train_chunk_residual import (
        build_base_policy,
        build_libero_scalers,
    )
    from resfit.rl_finetuning.config.residual_td3 import ResidualTD3BoxCleanConfig
    from resfit.rl_finetuning.off_policy.rl.q_agent import QAgent

    bundle = torch.load(
        args.residual_checkpoint,
        map_location="cpu",
        mmap=True,
        weights_only=False,
    )
    if "config" not in bundle or "agent_state_dict" not in bundle:
        raise ValueError("residual checkpoint lacks config or agent_state_dict")
    source_args = copy.copy(bundle["config"])
    if getattr(source_args, "env_family", None) != "libero":
        raise ValueError("source residual checkpoint is not a LIBERO run")
    if getattr(source_args, "subgoal_conditioned", None) is not True:
        raise ValueError("source residual checkpoint is not subgoal-conditioned")
    if getattr(source_args, "actor", None) != "raw":
        raise ValueError("source residual checkpoint does not use the raw actor")
    if int(getattr(source_args, "chunk_length", -1)) != 1:
        raise ValueError("frozen evaluator requires the source chunk_length=1")

    runtime_args = copy.copy(source_args)
    runtime_args.env_family = "libero"
    runtime_args.libero_suite = args.suite
    runtime_args.libero_task_id = args.task_id
    runtime_args.libero_stats_json = str(args.pi0_stats)
    runtime_args.pi0_host = args.host
    runtime_args.pi0_port = args.port
    runtime_args.pi0_prompt = TASK_LANGUAGES[args.task_id]
    runtime_args.device = args.device
    runtime_args.wandb_mode = "disabled"
    runtime_args.smoke = False
    runtime_args.online_finetune_value = False
    runtime_args.online_finetune_high_actor = False

    action_scale = float(source_args.action_scale)
    min_range = float(
        getattr(
            source_args,
            "min_range_per_dim",
            getattr(source_args, "min_range", 0.1),
        )
    )
    action_scaler, state_standardizer = build_libero_scalers(
        args.pi0_stats,
        args.device,
        action_scale=action_scale,
        min_range_per_dim=min_range,
    )
    base_policy = build_base_policy(runtime_args, args.device)
    if getattr(base_policy, "prompt", None) != TASK_LANGUAGES[args.task_id]:
        raise RuntimeError(
            "target PI0 prompt does not match the fixed LIBERO-90 task language"
        )
    image_keys = list(base_policy.config.image_features.keys())

    _, gc_info = load_gc_value(args.gc_value_checkpoint, map_location="cpu")
    if gc_info.get("state_mode") != "pi0_feat":
        raise ValueError("GC value checkpoint state_mode is not pi0_feat")
    sequences, _, cache_signature = load_pi0_feat_cache(args.pi0_feature_cache)
    if not sequences:
        raise ValueError("PI0 feature cache contains no source sequences")
    validate_pi0_feature_signature(
        cache_signature,
        gc_info.get("pi0_feat_signature"),
    )
    goal = representative_goal(sequences)
    if int(goal.shape[0]) != int(gc_info["mean"].shape[0]):
        raise ValueError("representative PI0 goal dimension does not match GC value")
    subgoal = HiqlSubgoal.from_ckpts(
        args.gc_value_checkpoint,
        args.high_actor_checkpoint,
        goal=goal,
        device=args.device,
        renorm_subgoal=True,
        base_policy=None,
    )

    config = ResidualTD3BoxCleanConfig()
    config.agent.actor_lr = float(source_args.actor_lr)
    config.agent.critic_lr = float(source_args.critic_lr)
    config.agent.actor.action_scale = action_scale
    config.agent.bc_loss_coef = float(source_args.demo_bc_coef)
    config.agent.bc_loss_dynamic = 0
    config.agent.device = args.device
    agent = QAgent(
        obs_shape=(3, 84, 84),
        prop_shape=(8,),
        action_dim=7 * int(source_args.chunk_length),
        rl_cameras=image_keys,
        cfg=config.agent,
        residual_actor=True,
        stage_conditioned=False,
        num_stages=1,
        stage_budget=None,
        subgoal_conditioned=True,
        subgoal_dim=subgoal.rep_dim,
    )
    incompatible = agent.load_state_dict(bundle["agent_state_dict"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"strict residual load failed: {incompatible}")
    freeze_and_validate(agent, "residual_agent")
    freeze_and_validate(subgoal.vf, "gc_value")
    freeze_and_validate(subgoal.ha, "high_actor")

    vector_env = create_libero_vectorized_env(
        args.suite,
        args.task_id,
        num_envs=1,
        device=args.device,
        debug=True,
    )
    env = ChunkResidualEnvWrapper(
        vector_env,
        base_policy,
        action_scaler,
        state_standardizer,
        chunk_length=int(source_args.chunk_length),
        reward_shaping_mode="none",
        base_action_mode=str(source_args.base_action_mode),
    )
    return env, agent, subgoal, base_policy, int(source_args.chunk_length)


def _rollout_one_episode(
    *,
    env,
    agent,
    subgoal,
    base_policy,
    init_state_id: int,
    chunk_length: int,
    horizon: int,
) -> tuple[bool, float, int, float]:
    import torch

    _set_init_state_id(env, init_state_id)
    started = time.monotonic()
    obs, _ = env.reset()
    episode_return = 0.0
    chunks = 0
    with torch.inference_mode():
        while True:
            prefix_feat = base_policy.last_prefix_feat()
            if prefix_feat is None:
                raise RuntimeError(
                    "PI0 server did not provide prefix_feat before residual action"
                )
            obs["observation.subgoal"] = subgoal.subgoal_online(
                obs,
                prefix_feat=prefix_feat,
            ).to(obs["observation.state"].device)
            residual = agent.act(
                obs,
                eval_mode=True,
                stddev=0.0,
                cpu=False,
            )
            next_obs, reward, terminated, truncated, _ = env.step(residual)
            reward_value = float(reward[0].item())
            episode_return += reward_value
            chunks += 1
            done = bool((terminated | truncated)[0].item())
            if done:
                success = bool(reward_value == 1.0)
                length = min(horizon, chunks * chunk_length)
                return (
                    success,
                    episode_return,
                    length,
                    time.monotonic() - started,
                )
            obs = next_obs


def run_worker(args: argparse.Namespace) -> int:
    import numpy as np
    import torch

    if args.suite != "libero_90":
        raise ValueError("this evaluator is frozen to suite libero_90")
    if args.task_id not in TASK_IDS:
        raise ValueError(f"task_id must be one of {TASK_IDS}")
    if args.num_episodes < 1 or args.num_episodes > 50:
        raise ValueError("num_episodes must be in [1, 50]")
    if not str(args.device).startswith("cuda"):
        raise ValueError("formal SHORE workers require a CUDA device")

    source_paths = _source_paths(args)
    for name, path_string in source_paths.items():
        path = Path(path_string)
        if name == "base_checkpoint":
            if not path.is_dir():
                raise FileNotFoundError(f"{name} directory not found: {path}")
        elif not path.is_file():
            raise FileNotFoundError(f"{name} file not found: {path}")
    source_hashes = _source_hashes(args)
    residual_hash = source_hashes["residual_checkpoint"]

    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "stdout.log").touch(exist_ok=True)
    (run_dir / "server.log").touch(exist_ok=True)
    config = {
        "suite": args.suite,
        "task_id": args.task_id,
        "task_language": TASK_LANGUAGES[args.task_id],
        "seed": args.seed,
        "num_episodes": args.num_episodes,
        "host": args.host,
        "port": args.port,
        "device": args.device,
        "source_checkpoint_sha256": residual_hash,
        "source_paths": source_paths,
        "source_hashes": source_hashes,
        "python_version": sys.version,
        "torch_version": torch.__version__,
    }
    config_path = run_dir / "config.json"
    validate_resume_config(config_path, config)
    if not config_path.exists():
        write_json_atomic(config_path, config)

    episodes_path = run_dir / "episodes.jsonl"
    existing_rows = load_jsonl(episodes_path)
    for row in existing_rows:
        mismatches = []
        if int(row["task_id"]) != args.task_id:
            mismatches.append("task_id")
        if int(row["seed"]) != args.seed:
            mismatches.append("seed")
        if str(row["source_checkpoint_sha256"]) != residual_hash:
            mismatches.append("source_checkpoint_sha256")
        if mismatches:
            raise ValueError(f"incompatible existing episode row: {mismatches}")
    remaining = missing_init_ids(existing_rows, args.num_episodes)
    if not remaining:
        validate_episode_coverage(existing_rows, range(args.num_episodes))
        if not (run_dir / "COMPLETED").exists():
            write_text_atomic(run_dir / "COMPLETED", "complete\n")
        print(f"task {args.task_id}: all {args.num_episodes} init states already complete")
        return 0

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env = None
    try:
        env, agent, subgoal, base_policy, chunk_length = _build_frozen_components(args)
        horizon = 400
        with EpisodeJsonlWriter(episodes_path) as writer:
            for episode_index, init_state_id in enumerate(
                remaining,
                start=len(existing_rows),
            ):
                success, episode_return, length, elapsed = _rollout_one_episode(
                    env=env,
                    agent=agent,
                    subgoal=subgoal,
                    base_policy=base_policy,
                    init_state_id=init_state_id,
                    chunk_length=chunk_length,
                    horizon=horizon,
                )
                row = {
                    "task_id": args.task_id,
                    "task_language": TASK_LANGUAGES[args.task_id],
                    "episode_index": episode_index,
                    "init_state_id": init_state_id,
                    "success": success,
                    "return": episode_return,
                    "length": length,
                    "elapsed_seconds": elapsed,
                    "seed": args.seed,
                    "source_checkpoint_sha256": residual_hash,
                }
                writer.append(row)
                mark = "success" if success else "failure"
                print(
                    f"task {args.task_id} init {init_state_id:02d}: {mark}, "
                    f"return={episode_return:.3f}, length={length}",
                    flush=True,
                )
    finally:
        if env is not None:
            env.close()

    rows = load_jsonl(episodes_path)
    validate_episode_coverage(rows, range(args.num_episodes))
    rows = sorted(rows, key=lambda row: int(row["init_state_id"]))
    summary = {
        "suite": args.suite,
        "task_id": args.task_id,
        "task_language": TASK_LANGUAGES[args.task_id],
        "episodes": len(rows),
        "successes": sum(bool(row["success"]) for row in rows),
        "success_rate": sum(bool(row["success"]) for row in rows) / len(rows),
        "mean_return": sum(float(row["return"]) for row in rows) / len(rows),
        "mean_length": sum(int(row["length"]) for row in rows) / len(rows),
        "seed": args.seed,
        "source_checkpoint_sha256": residual_hash,
    }
    write_json_atomic(run_dir / "summary.json", summary)
    write_text_atomic(run_dir / "COMPLETED", "complete\n")
    return 0


def _fixture_row(task_id: int, init_state_id: int, success: bool) -> dict:
    return {
        "task_id": task_id,
        "task_language": f"task {task_id}",
        "episode_index": init_state_id,
        "init_state_id": init_state_id,
        "success": success,
        "return": float(success),
        "length": 10 + init_state_id,
        "elapsed_seconds": 0.1,
        "seed": 1,
        "source_checkpoint_sha256": "same-hash",
    }


class ShoreSelfTests(unittest.TestCase):
    def test_rolling_success_uses_partial_leading_windows(self) -> None:
        self.assertEqual(
            rolling_success([1, 0, 1, 1], window=3).tolist(),
            [1.0, 0.5, 2.0 / 3.0, 2.0 / 3.0],
        )

    def test_missing_and_duplicate_init_states_are_rejected(self) -> None:
        rows = [_fixture_row(3, index, bool(index % 2)) for index in range(4)]
        validate_episode_coverage(rows, range(4))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_episode_coverage(rows + [rows[-1]], range(4))
        with self.assertRaisesRegex(ValueError, "missing"):
            validate_episode_coverage(rows[:-1], range(4))

    def test_resume_selects_missing_ids_and_rejects_incompatible_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.json"
            expected = {
                "suite": "libero_90",
                "task_id": 21,
                "seed": 1,
                "source_checkpoint_sha256": "same-hash",
            }
            write_json_atomic(config_path, expected)
            validate_resume_config(config_path, expected)
            rows = [
                _fixture_row(21, 0, False),
                _fixture_row(21, 2, True),
            ]
            self.assertEqual(missing_init_ids(rows, 4), [1, 3])
            with self.assertRaisesRegex(ValueError, "seed"):
                validate_resume_config(config_path, dict(expected, seed=2))

    def test_jsonl_writer_flushes_fsyncs_and_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "episodes.jsonl"
            row = _fixture_row(8, 0, True)
            with mock.patch("os.fsync", wraps=os.fsync) as fsync:
                with EpisodeJsonlWriter(path) as writer:
                    writer.append(row)
            self.assertEqual(fsync.call_count, 1)
            self.assertEqual(load_jsonl(path), [row])

    def test_launch_mapping_defaults_and_port_override(self) -> None:
        groups = build_launch_groups(
            gpu_ids=[4, 5, 6, 7],
            ports=DEFAULT_PORTS,
            run_id="test-run",
            episodes=2,
        )
        self.assertEqual(
            [(group["task_id"], group["gpu_id"], group["port"]) for group in groups],
            [(3, 4, 8000), (8, 5, 8001), (21, 6, 8002), (63, 7, 8003)],
        )
        overridden = build_launch_groups(
            gpu_ids=[2, 3, 4, 5],
            ports=[8100, 8101, 8102, 8103],
            run_id="formal",
            episodes=50,
        )
        self.assertEqual(
            [group["port"] for group in overridden],
            [8100, 8101, 8102, 8103],
        )
        for group in overridden:
            self.assertIn(str(group["port"]), group["server_command"])
            self.assertIn(str(group["port"]), group["worker_command"])

    def test_signature_validation_ignores_target_prompt_only(self) -> None:
        source = {
            "serve_ckpt_id": "base",
            "image_keys": ["a", "b"],
            "proprio_key": "state",
            "pooling": "last",
            "prompt": "source task8",
        }
        validate_pi0_feature_signature(source, dict(source))
        changed = dict(source, pooling="mean")
        with self.assertRaisesRegex(ValueError, "pooling"):
            validate_pi0_feature_signature(changed, source)

    def test_plot_writes_figure_and_machine_readable_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_root = root / "run"
            for task_id in TASK_IDS:
                task_dir = run_root / f"task{task_id:02d}"
                task_dir.mkdir(parents=True)
                rows = [
                    _fixture_row(task_id, index, (index + task_id) % 3 == 0)
                    for index in range(4)
                ]
                with (task_dir / "episodes.jsonl").open("w", encoding="utf-8") as stream:
                    for row in rows:
                        stream.write(json.dumps(row) + "\n")
                write_json_atomic(
                    task_dir / "config.json",
                    {
                        "suite": "libero_90",
                        "task_id": task_id,
                        "seed": 1,
                        "source_checkpoint_sha256": "same-hash",
                    },
                )
                (task_dir / "COMPLETED").write_text("complete\n", encoding="utf-8")

            json_out = root / "data" / "shore.json"
            npz_out = root / "data" / "shore.npz"
            pdf_out = root / "figure" / "shore.pdf"
            plot_results(
                run_root=run_root,
                expected_episodes=4,
                json_out=json_out,
                npz_out=npz_out,
                pdf_out=pdf_out,
            )
            self.assertTrue(json_out.is_file())
            self.assertTrue(npz_out.is_file())
            self.assertTrue(pdf_out.is_file())
            payload = json.loads(json_out.read_text(encoding="utf-8"))
            self.assertEqual(len(payload["tasks"]), 4)
            self.assertEqual(
                sum(len(task["successes"]) for task in payload["tasks"]),
                16,
            )
            import numpy as np

            with np.load(npz_out) as data:
                self.assertEqual(data["successes"].shape, (4, 4))
                self.assertEqual(data["rolling_success"].shape, (4, 4))


def run_self_tests() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ShoreSelfTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate frozen task-8 SHORE weights zero-shot on LIBERO-90, "
            "launch the four-GPU fleet, or plot validated results."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("self-test", help="run offline embedded unit tests")

    worker = subparsers.add_parser(
        "worker",
        help="evaluate one fixed LIBERO-90 task, one init state per episode",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    worker.add_argument("--suite", default="libero_90")
    worker.add_argument("--task-id", type=int, choices=TASK_IDS, required=True)
    worker.add_argument("--seed", type=int, default=1)
    worker.add_argument("--num-episodes", type=int, default=50)
    worker.add_argument("--device", default="cuda")
    worker.add_argument("--host", default="127.0.0.1")
    worker.add_argument("--port", type=int, required=True)
    worker.add_argument("--run-dir", type=Path, required=True)
    worker.add_argument("--base-checkpoint", type=Path, default=BASE_CHECKPOINT)
    worker.add_argument(
        "--residual-checkpoint",
        type=Path,
        default=RESIDUAL_CHECKPOINT,
    )
    worker.add_argument(
        "--gc-value-checkpoint",
        type=Path,
        default=GC_VALUE_CHECKPOINT,
    )
    worker.add_argument(
        "--high-actor-checkpoint",
        type=Path,
        default=HIGH_ACTOR_CHECKPOINT,
    )
    worker.add_argument(
        "--pi0-feature-cache",
        type=Path,
        default=PI0_FEATURE_CACHE,
    )
    worker.add_argument("--pi0-stats", type=Path, default=PI0_STATS)

    launcher = subparsers.add_parser(
        "launch-all",
        help="launch four feature-server/worker groups concurrently",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    launcher.add_argument(
        "--gpu-ids",
        type=int,
        nargs=4,
        required=True,
        metavar=("GPU0", "GPU1", "GPU2", "GPU3"),
        help="exactly four physical GPU IDs for tasks 3, 8, 21, and 63",
    )
    launcher.add_argument(
        "--ports",
        type=int,
        nargs=4,
        default=DEFAULT_PORTS,
        metavar=("PORT0", "PORT1", "PORT2", "PORT3"),
        help="exactly four distinct server ports in task order",
    )
    launcher.add_argument("--run-id", required=True)
    launcher.add_argument("--episodes", type=int, default=50)
    launcher.add_argument(
        "--dry-run",
        action="store_true",
        help="print resolved commands without creating processes",
    )

    plot = subparsers.add_parser(
        "plot",
        help="validate four completed tasks and write JSON, NPZ, and a 2x2 PDF",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    plot.add_argument("--run-root", type=Path, required=True)
    plot.add_argument("--expected-episodes", type=int, default=50)
    plot.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    plot.add_argument("--npz-out", type=Path, default=DEFAULT_NPZ_OUT)
    plot.add_argument("--pdf-out", type=Path, default=DEFAULT_PDF_OUT)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "self-test":
        return run_self_tests()
    if args.command == "worker":
        return run_worker(args)
    if args.command == "launch-all":
        return launch_all(args)
    if args.command == "plot":
        plot_results(
            run_root=args.run_root,
            expected_episodes=args.expected_episodes,
            json_out=args.json_out,
            npz_out=args.npz_out,
            pdf_out=args.pdf_out,
        )
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
