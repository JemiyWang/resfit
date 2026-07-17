#!/usr/bin/env bash
# pouring ablation seed runner.
# Baseline (full HiRes-RL, verified against wandb run b4yerjy2 = ..._staged_joint_seed2):
#   staged reward + stage-balanced + BC0.1 + subgoal(+joint), action_scale 0.05, base_policy anchor.
# Modes (each = single-variable ablation off that baseline; everything else locked):
#   nosubgoal          : subgoal OFF (joint bundled-off); staged reward + BC0.1 kept.
#   nostage            : staged reward shaping OFF (all stage mechanisms off); BC0.1 + subgoal(+joint) kept.
#   nostage_nosubgoal  : staged OFF AND subgoal OFF (joint off); BC0.1 kept -> flat demo-anchored residual.
# subgoal OFF also drops --online_finetune_value/high_actor (assert in online_hiql_finetune.py:63)
#   and gc_value/high_actor/act_feat_cache (residual uses base-dim eef state; _needs_act_feat_cache->False).
# OFFCACHE reuse (offline_buffer_signature = task/dataset/action_scale/reward_shaping/subgoal/base,
#   NOT seed and NOT bc), verified against the stored buffer_meta.json signatures:
#   nostage           -> reuse COMPLETE ..._subgoal_joint_nobc_nopot_offcache  (reward none + subgoal on)
#   nostage_nosubgoal -> reuse COMPLETE ...nosubgoal_nojoint_nopot_offcache    (reward none + subgoal off)
#   nosubgoal         -> NEW staged+subgoal-off offcache, must build (~41G, ~80min base_policy CPU forward)
# Usage: bash run_pouring_ablation_seed.sh <nosubgoal|nostage|nostage_nosubgoal> <gpu> <seed>
set -euo pipefail
MODE=${1:?Usage: bash run_pouring_ablation_seed.sh <nosubgoal|nostage|nostage_nosubgoal> <gpu> <seed>}
GPU=${2:?需要 gpu}
SEED=${3:?需要 seed}

export PATH="/root/miniconda3/bin:$PATH"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export CUDA_VISIBLE_DEVICES="$GPU"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_OFFLINE=1
export PYTHONPATH=/mnt/mnt/data/resfit

# wandb 身份。entity/email 非机密,写死;API key 绝不写进本文件——本脚本被 git 跟踪且
# remote 在 GitHub,提交 key 会被 secret scanning 自动吊销,在跑的 run 全部断线。
# key 三选一(每台机器做一次):`wandb login <key>` 写 ~/.netrc / 外部 export WANDB_API_KEY
# / 放 ~/.wandb_env(不在仓库内,不会被 clone 走)。
[ -f "$HOME/.wandb_env" ] && . "$HOME/.wandb_env"
export WANDB_ENTITY="${WANDB_ENTITY:-674575221-beijing-institute-of-technology}"
export WANDB_USER_EMAIL="${WANDB_USER_EMAIL:-674575221@qq.com}"
if [ -z "${WANDB_API_KEY:-}" ] && ! grep -q api.wandb.ai "$HOME/.netrc" 2>/dev/null; then
  echo "[pouring_${MODE}_seed${SEED}] FATAL 无 wandb 凭证: 先 'wandb login <key>' 或 export WANDB_API_KEY" >&2
  exit 4
fi

cd /mnt/mnt/data/resfit || exit 3
RUN="conda run -n residual --no-capture-output python -u"

DS=ankile/dexmg-two-arm-pouring
HDF5=resfit/dataset/two_arm_pouring.hdf5
STAGES=outputs_chunk/two_arm_pouring_stages.npz
CACHE=outputs_chunk/pouring_act_feat_hdf5.npz
BASE="/mnt/mnt/data/wjm/residual/residual-offpolicy-rl/artifacts/run_anw5pphu_best:v2"
GCV=outputs_chunk/pouring_gc_value_actfeat_hdf5_hiqlv512.pt
HIA=outputs_chunk/pouring_high_actor_actfeat_hdf5_hiqlv512.pt
WAY=15

# Optional eval-env override (env var EVAL_NUM_ENVS). Fewer eval envs => less GPU memory
# + less CPU/rendering during eval, but it is NOT metric-neutral: never compare arms that
# ran with different EVAL_NUM_ENVS. evaluate_dexmg.py stops the instant the 50th episode
# completes (`while done_episodes < num_episodes` :194, break :289), discarding the up to
# num_envs-1 episodes still in flight -- and those skew long, i.e. toward failures, which
# inflates the reported rate as num_envs grows. Measured on TwoArmPouring step-0, where the
# residual is exactly 0 so every run evaluates the identical frozen base: envs=8 -> .786
# (n=29, [.70,.86]), envs=4 -> .547 (n=9, [.48,.60]), envs=1 -> .598 (n=8, [.52,.66]).
# Disjoint ranges, one base ckpt, exec=10 throughout, and June/July envs=8 runs agree
# (.791/.782) -> the knob, not seed noise or code drift. TwoArmLiftTray shows no such split.
# Unset => code default 8; leave it unset unless every arm in the comparison overrides it too.
EVAL_ENVS_FLAGS=()
[ -n "${EVAL_NUM_ENVS:-}" ] && EVAL_ENVS_FLAGS=(--eval_num_envs "$EVAL_NUM_ENVS")

# Subgoal(+joint) flag set, shared by any subgoal-ON mode.
SUBGOAL_ON_FLAGS=(--subgoal_conditioned --gc_value_ckpt "$GCV" --high_actor_ckpt "$HIA" \
  --act_feat_cache "$CACHE" --subgoal_way_steps "$WAY" \
  --online_finetune_value --online_finetune_high_actor)

case "$MODE" in
  nosubgoal)
    DEMO_BC="0.1"
    SHAPING_FLAGS=(--stage_balanced --reward_shaping staged --stage_reward_bonus 1.0)
    STAGE_FLAGS=(--offline_stage_cache "$STAGES")
    SUBGOAL_FLAGS=()          # subgoal off -> joint off (coupling enforced in code)
    USE_SUBGOAL=0
    OFFCACHE=outputs_chunk/pouring_actfeat_hdf5_bp_bc01_hiqlv512_sg15_as005_staged_nosubgoal_offcache
    NAME=pouring_actfeat_hdf5_bp_bc01_hiqlv512_sg15_as005_staged_nosubgoal_seed${SEED}
    ;;
  nostage)
    DEMO_BC="0.1"
    SHAPING_FLAGS=(--no_stage_balanced --reward_shaping none)
    STAGE_FLAGS=()
    SUBGOAL_FLAGS=("${SUBGOAL_ON_FLAGS[@]}")
    USE_SUBGOAL=1
    OFFCACHE=outputs_chunk/pouring_actfeat_hdf5_bp_hiqlv512_sg15_as005_subgoal_joint_nobc_nopot_offcache
    NAME=pouring_actfeat_hdf5_bp_bc01_hiqlv512_sg15_as005_nostage_joint_seed${SEED}
    ;;
  nostage_nosubgoal)
    DEMO_BC="0.1"
    SHAPING_FLAGS=(--no_stage_balanced --reward_shaping none)
    STAGE_FLAGS=()
    SUBGOAL_FLAGS=()          # subgoal off -> joint off
    USE_SUBGOAL=0
    OFFCACHE=outputs_chunk/pouring_hdf5_bp_bc01_as005_nosubgoal_nojoint_nopot_offcache
    NAME=pouring_actfeat_hdf5_bp_bc01_hiqlv512_sg15_as005_nostage_nosubgoal_seed${SEED}
    ;;
  *)
    echo "Usage: bash run_pouring_ablation_seed.sh <nosubgoal|nostage|nostage_nosubgoal> <gpu> <seed>" >&2
    exit 2
    ;;
esac

gate(){ [ -s "$1" ] || { echo "[pouring_${MODE}_seed${SEED}] FATAL required asset missing: $1"; exit 4; }; }

echo "[pouring_${MODE}_seed${SEED}] gate reusable assets(base always; gc_value/high_actor/act_feat only when subgoal on)..."
gate "$BASE/policy/model.safetensors"
if [ "$USE_SUBGOAL" = "1" ]; then gate "$CACHE"; gate "$GCV"; gate "$HIA"; fi
[ -e "$OFFCACHE/buffer_meta.json" ] && echo "[pouring_${MODE}_seed${SEED}] offcache exists (reuse/read): $OFFCACHE" \
  || echo "[pouring_${MODE}_seed${SEED}] offcache missing -> this run may build it: $OFFCACHE"
if [ ${#STAGE_FLAGS[@]} -gt 0 ]; then
  [ -e "$STAGES" ] && echo "[pouring_${MODE}_seed${SEED}] stage cache exists: $STAGES" \
    || echo "[pouring_${MODE}_seed${SEED}] stage cache missing -> training will replay and create it"
fi

echo "[pouring_${MODE}_seed${SEED}] START GPU=$CUDA_VISIBLE_DEVICES seed=$SEED mode=$MODE $(date '+%F %T')"
$RUN -m resfit.rl_finetuning.chunk_residual.train_chunk_residual \
  --task TwoArmPouring --base_wandb_id "$BASE" --dataset "$DS" \
  --offline_dataset_path "$HDF5" \
  --chunk_length 1 --base_action_mode queue --base_n_action_steps 10 \
  --action_scale 0.05 --actor_lr 1e-6 --actor raw \
  "${SHAPING_FLAGS[@]}" \
  --offline_fraction 0.5 "${STAGE_FLAGS[@]}" \
  --offline_base_mode base_policy --demo_bc_coef "$DEMO_BC" \
  --offline_buffer_cache "$OFFCACHE" \
  "${SUBGOAL_FLAGS[@]}" \
  "${EVAL_ENVS_FLAGS[@]}" \
  --seed "$SEED" \
  --total_env_steps 500000 \
  --wandb_project dexmg-chunk-residual --wandb_name "$NAME" \
  --output_dir "outputs_chunk/$NAME"
echo "[pouring_${MODE}_seed${SEED}] ALL DONE $(date '+%F %T') out=outputs_chunk/$NAME"
