#!/usr/bin/env bash
# 复用克隆现成 LIBERO env(py3.8,robosuite1.4.x + bddl + robomimic + libero + torch2.4.1)
# → libero-residual,补 resfit 残差 RL 依赖。2026-06-12 实测共存通过。
# 用法:bash resfit/scripts/setup_libero_residual_env.sh
set -euo pipefail
SRC="${SRC:-/mnt/mnt/data/envs/libero}"
DST="${DST:-/mnt/mnt/data/envs/libero-residual}"
OPENPI_CLIENT="${OPENPI_CLIENT:-/mnt/mnt/data/chj/openpi/packages/openpi-client}"

# 1) 克隆(同盘 hardlink,快;不动原 env,免影响别人用 libero)
conda create -y --clone "$SRC" -p "$DST"
# 2) 补 RL 栈(py3.8 上限:gymnasium1.1.1 / torchrl0.5.0 / tensordict0.5.0)
conda run -p "$DST" pip install "gymnasium>=1.0" torchrl
# 3) openpi_client(pi05 websocket base 用;editable)
conda run -p "$DST" pip install -e "$OPENPI_CLIENT"

# 共存终核
MUJOCO_GL=egl conda run -p "$DST" python - <<'PY'
import torch, torchrl, tensordict, gymnasium, numpy, PIL
import robosuite, mujoco, bddl, robomimic
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from gymnasium.vector import AsyncVectorEnv, AutoresetMode
from openpi_client import websocket_client_policy, image_tools
print(f"[verify] OK py-deps coexist | robosuite {robosuite.__version__} torch {torch.__version__} "
      f"torchrl {torchrl.__version__} gymnasium {gymnasium.__version__}")
PY

cat <<'NOTE'
[setup] libero-residual ready.
注意(两处 py3.8 不兼容,代码侧已绕过):
  - lerobot 不装(py3.8 不支持,要 >=3.10):LIBERO 路直读 stats.json 取 norm(见 libero_obs.load_libero_norm_stats)。
  - 不依赖 kai0 resfit_pi05(py3.10-only,dataclass slots):LiberoPi05Adapter 自包含,直接基于 openpi_client。
NOTE
