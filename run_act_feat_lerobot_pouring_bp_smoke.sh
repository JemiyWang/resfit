#!/usr/bin/env bash
# pouring 小规模验证:lerobot 数据源 + offline_base_mode=base_policy(对齐 three_piece,未验证过的组合)。
# 超参对齐 three_piece_actfeat_bp_bc01,但 no-stage、小集快跑。目的=验证全链能跑通,非训好模型。
set -uo pipefail
export PYTHONPATH=/mnt/mnt/data/resfit
export CUDA_VISIBLE_DEVICES=2
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
cd /mnt/mnt/data/resfit

ROOT=/mnt/mnt/data/resfit/resfit/dataset/ankile/dexmg-two-arm-pouring
DS=ankile/dexmg-two-arm-pouring
BASE=/mnt/mnt/data/wjm/residual/residual-offpolicy-rl/artifacts/run_anw5pphu_best:v2
DUMMY=/tmp/UNUSED_dummy.hdf5
CACHE=/tmp/pouring_actfeat_bp_n4.npz
GC=/tmp/pouring_gc_bp.pt
HA=/tmp/pouring_high_bp.pt
N=4
RUN="conda run -n residual --no-capture-output python -u"

echo "=========== [0] 建小 act_feat 缓存(num_demos!=None 不自动落盘,需脚手架) ==========="
ROOT="$ROOT" DS="$DS" BASE="$BASE" CACHE="$CACHE" N="$N" $RUN - <<'PY'
import os
os.environ.setdefault("HF_HUB_OFFLINE","1")
from resfit.rl_finetuning.chunk_residual.train_hiql_value import read_per_demo_states, setup_act_feat
from resfit.rl_finetuning.chunk_residual.act_feature import act_feat_signature
from resfit.rl_finetuning.chunk_residual.act_feat_cache import save_act_feat_cache
ROOT=os.environ["ROOT"]; DS=os.environ["DS"]; BASE=os.environ["BASE"]; CACHE=os.environ["CACHE"]; N=int(os.environ["N"])
class A: pass
a=A(); a.act_feat_cache=None; a.act_base_ckpt=BASE; a.act_image_keys=None
a.act_proprio_key="observation.state"; a.pooling="mean"; a.state_mode="act_feat"
ext, ckpt, keys, _ = setup_act_feat(a)
seqs,_,(mean,std)=read_per_demo_states("/tmp/UNUSED_dummy.hdf5", DS, "act_feat", num_demos=N,
    act_feat_cache=None, act_extractor=ext, act_image_keys=keys, act_ckpt_id=ckpt,
    act_proprio_key="observation.state", pooling="mean", data_source="lerobot", lerobot_root=ROOT)
sig=dict(act_feat_signature(ckpt, keys, "observation.state", "mean"), dataset_id=str(DS), num_demos=N)
save_act_feat_cache(CACHE, seqs, (mean,std), signature=sig)
print(f"[cache] wrote {CACHE}: demos={len(seqs)} state_dim={seqs[0].shape[1]}")
PY
[ -s "$CACHE" ] || { echo "FATAL: act_feat 缓存建失败"; exit 1; }

echo "=========== [1] gc_value(lerobot, act_feat) ==========="
$RUN -m resfit.rl_finetuning.chunk_residual.train_hiql_gc_value \
  --hdf5 "$DUMMY" --state_mode act_feat --data_source lerobot --lerobot_root "$ROOT" \
  --act_base_ckpt "$BASE" --dataset "$DS" --num_demos "$N" --steps 80 \
  --act_feat_cache "$CACHE" --output "$GC"
[ -s "$GC" ] || { echo "FATAL: gc_value 失败"; exit 1; }

echo "=========== [2] high_actor ==========="
$RUN -m resfit.rl_finetuning.chunk_residual.train_hiql_high_actor \
  --hdf5 "$DUMMY" --state_mode act_feat --data_source lerobot --lerobot_root "$ROOT" \
  --act_base_ckpt "$BASE" --dataset "$DS" --num_demos "$N" --steps 80 \
  --act_feat_cache "$CACHE" --gc_value_ckpt "$GC" --output "$HA"
[ -s "$HA" ] || { echo "FATAL: high_actor 失败"; exit 1; }

echo "=========== [3] 主训 smoke: base_policy(验证点) + 对齐 three_piece 超参 + no-stage 小集 ==========="
$RUN -m resfit.rl_finetuning.chunk_residual.train_chunk_residual \
  --task TwoArmPouring --base_wandb_id "$BASE" --dataset "$DS" \
  --data_source lerobot --lerobot_root "$ROOT" \
  --subgoal_conditioned --reward_shaping none \
  --offline_base_mode base_policy \
  --gc_value_ckpt "$GC" --high_actor_ckpt "$HA" --act_feat_cache "$CACHE" \
  --offline_num_demos "$N" --offline_fraction 0.5 --demo_bc_coef 0.1 \
  --action_scale 0.05 --actor_lr 1e-6 --actor raw \
  --chunk_length 1 --base_action_mode queue --base_n_action_steps 10 \
  --total_env_steps 40 --learning_starts 10 --eval_every_env_steps 30 \
  --eval_num_envs 2 --eval_num_episodes 2 --utd 1 \
  --wandb_mode disabled --output_dir /tmp/pouring_smoke_bp_out
echo "[smoke] pouring lerobot+base_policy 全链 DONE"
