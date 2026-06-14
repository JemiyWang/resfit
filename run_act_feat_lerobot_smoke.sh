#!/usr/bin/env bash
# act_feat × LeRobot 数据源 可跑性 smoke(真 pouring 数据 + 真 ACT base,GPU 2)。
# 目的:验证 LeRobot 数据源全链能跑通(gc_value → high_actor → train_chunk_residual
# offline buffer + 在线 rollout/eval),非训出好模型。
set -euo pipefail

export PYTHONPATH=/mnt/mnt/data/resfit_lerobot_wt
export CUDA_VISIBLE_DEVICES=2
export MUJOCO_EGL_DEVICE_ID=0          # 单卡掩码下用逻辑号 0(不是物理 2)
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export HF_HUB_OFFLINE=1

ROOT=/mnt/mnt/data/resfit/resfit/dataset/ankile/dexmg-two-arm-pouring
DS=ankile/dexmg-two-arm-pouring
BASE=/mnt/mnt/data/wjm/residual/residual-offpolicy-rl/artifacts/run_anw5pphu_best:v2
DUMMY_HDF5=/tmp/UNUSED_dummy.hdf5      # gc_value/high_actor 的 --hdf5 是 required 但 lerobot+act_feat 路不读
CACHE=/tmp/pouring_act_feat_cache_n2.npz
GC=/tmp/pouring_gc_value_smoke.pt
HA=/tmp/pouring_high_actor_smoke.pt
N=2                                    # demo 数(冒烟少量)

RUN="conda run -n residual --no-capture-output python -u"

# --- 0) 先建一份小 act_feat 缓存(num_demos!=None 时主脚本不落盘,故单独建)---
#     chunk_residual 的 act_feat 子目标强制要 --act_feat_cache 已存在,故必须先有缓存。
ROOT="$ROOT" DS="$DS" BASE="$BASE" CACHE="$CACHE" N="$N" $RUN - <<'PY'
import os
os.environ.setdefault("HF_HUB_OFFLINE","1")
from resfit.rl_finetuning.chunk_residual.train_hiql_value import read_per_demo_states, setup_act_feat
from resfit.rl_finetuning.chunk_residual.act_feature import act_feat_signature
from resfit.rl_finetuning.chunk_residual.act_feat_cache import save_act_feat_cache
ROOT=os.environ["ROOT"]; DS=os.environ["DS"]; BASE=os.environ["BASE"]
CACHE=os.environ["CACHE"]; N=int(os.environ["N"])
class A: pass
a=A(); a.act_feat_cache=None; a.act_base_ckpt=BASE; a.act_image_keys=None
a.act_proprio_key="observation.state"; a.pooling="mean"; a.state_mode="act_feat"
ext, ckpt, keys, _ = setup_act_feat(a)
seqs,_,(mean,std)=read_per_demo_states("/tmp/UNUSED_dummy.hdf5", DS, "act_feat", num_demos=N,
    act_feat_cache=None, act_extractor=ext, act_image_keys=keys, act_ckpt_id=ckpt,
    act_proprio_key="observation.state", pooling="mean", data_source="lerobot", lerobot_root=ROOT)
sig=dict(act_feat_signature(ckpt, keys, "observation.state", "mean"), dataset_id=str(DS), num_demos=N)
save_act_feat_cache(CACHE, seqs, (mean,std), signature=sig)
print(f"[build_smoke_cache] wrote {CACHE}: demos={len(seqs)} state_dim={seqs[0].shape[1]}")
PY

# --- 1) gc_value(act_feat × lerobot),命中上一步缓存 ---
$RUN -m resfit.rl_finetuning.chunk_residual.train_hiql_gc_value \
  --hdf5 "$DUMMY_HDF5" \
  --state_mode act_feat --data_source lerobot --lerobot_root "$ROOT" \
  --act_base_ckpt "$BASE" --dataset "$DS" \
  --num_demos "$N" --steps 50 \
  --act_feat_cache "$CACHE" --output "$GC"

# --- 2) high_actor,命中缓存 + 吃 gc_value.pt ---
$RUN -m resfit.rl_finetuning.chunk_residual.train_hiql_high_actor \
  --hdf5 "$DUMMY_HDF5" \
  --state_mode act_feat --data_source lerobot --lerobot_root "$ROOT" \
  --act_base_ckpt "$BASE" --dataset "$DS" \
  --num_demos "$N" --steps 50 \
  --act_feat_cache "$CACHE" --gc_value_ckpt "$GC" --output "$HA"

# --- 3) 主训:offline buffer(lerobot/no-stage)+ 在线 rollout/eval ---
#     残差 base + 在线 act_feat 提特征都用同一 ACT(--base_wandb_id 接受本地目录)。
$RUN -m resfit.rl_finetuning.chunk_residual.train_chunk_residual \
  --task TwoArmPouring \
  --base_wandb_id "$BASE" --dataset "$DS" \
  --data_source lerobot --lerobot_root "$ROOT" \
  --subgoal_conditioned --reward_shaping none --offline_base_mode gt \
  --gc_value_ckpt "$GC" --high_actor_ckpt "$HA" --act_feat_cache "$CACHE" \
  --offline_num_demos "$N" --offline_fraction 0.5 \
  --total_env_steps 30 --learning_starts 10 --eval_every_env_steps 20 \
  --eval_num_envs 2 --eval_num_episodes 2 --utd 1 \
  --wandb_mode disabled \
  --output_dir /tmp/chunk_residual_smoke_out2

echo "[smoke] DONE"
