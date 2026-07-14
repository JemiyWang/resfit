#!/usr/bin/env bash
# three_piece ablation seed runner.
# Baseline (full HiRes-RL): three_piece_actfeat_bp_bc01_hiqlv512_sg15_staged_joint_seed<seed>
#   = staged reward + stage-balanced replay + BC0.1 + subgoal(+joint).
# Modes (each = single-variable ablation off that baseline; everything else locked):
#   nosubgoal          : subgoal OFF (joint bundled-off); staged reward + BC0.1 kept.
#   nostage            : staged reward shaping OFF (all stage mechanisms off); BC0.1 + subgoal(+joint) kept.
#   nostage_nosubgoal  : staged OFF AND subgoal OFF (joint off); BC0.1 kept -> flat demo-anchored residual.
# NOTE: subgoal OFF must also drop --online_finetune_value/high_actor -- the code asserts
#       "subgoal is not None" when joint is on (online_hiql_finetune.py:63). With subgoal off
#       the residual actor/critic use base-dim (eef) state, so act_feat/gc_value/high_actor are
#       not needed (_needs_act_feat_cache -> False, train_chunk_residual.py:507).
# Usage:
#   bash run_three_piece_ablation_seed.sh <nosubgoal|nostage|nostage_nosubgoal> <gpu> <seed>
set -euo pipefail
MODE=${1:?Usage: bash run_three_piece_ablation_seed.sh <nosubgoal|nostage|nostage_nosubgoal> <gpu> <seed>}
GPU=${2:?需要 gpu}
SEED=${3:?需要 seed}

export PATH="/root/miniconda3/bin:$PATH"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export CUDA_VISIBLE_DEVICES="$GPU"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_OFFLINE=1
export PYTHONPATH=/mnt/mnt/data/resfit
cd /mnt/mnt/data/resfit || exit 3
RUN="conda run -n residual --no-capture-output python -u"

HDF5=resfit/dataset/two_arm_three_piece_assembly.hdf5
DS=ankile/dexmg-two-arm-three-piece-assembly
STAGES=outputs_chunk/three_piece_stages_4stage.npz
CACHE=outputs_chunk/three_piece_act_feat.npz
BASE=resfit/out/piecce/best
GCV=outputs_chunk/three_piece_gc_value_actfeat_hiqlv512.pt
HIA=outputs_chunk/three_piece_high_actor_actfeat_hiqlv512.pt
WAY=15

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
    OFFCACHE=outputs_chunk/three_piece_actfeat_bp_bc01_hiqlv512_sg15_staged_nosubgoal_offcache
    NAME=three_piece_actfeat_bp_bc01_hiqlv512_sg15_staged_nosubgoal_seed${SEED}
    ;;
  nostage)
    DEMO_BC="0.1"
    SHAPING_FLAGS=(--no_stage_balanced --reward_shaping none)
    STAGE_FLAGS=()
    SUBGOAL_FLAGS=("${SUBGOAL_ON_FLAGS[@]}")
    USE_SUBGOAL=1
    OFFCACHE=outputs_chunk/three_piece_actfeat_bp_bc01_hiqlv512_sg15_nostage_offcache
    NAME=three_piece_actfeat_bp_bc01_hiqlv512_sg15_nostage_joint_seed${SEED}
    ;;
  nostage_nosubgoal)
    DEMO_BC="0.1"
    SHAPING_FLAGS=(--no_stage_balanced --reward_shaping none)
    STAGE_FLAGS=()
    SUBGOAL_FLAGS=()          # subgoal off -> joint off
    USE_SUBGOAL=0
    OFFCACHE=outputs_chunk/three_piece_actfeat_bp_bc01_hiqlv512_sg15_nostage_nosubgoal_offcache
    NAME=three_piece_actfeat_bp_bc01_hiqlv512_sg15_nostage_nosubgoal_seed${SEED}
    ;;
  *)
    echo "Usage: bash run_three_piece_ablation_seed.sh <nosubgoal|nostage|nostage_nosubgoal> <gpu> <seed>" >&2
    exit 2
    ;;
esac

gate(){ [ -s "$1" ] || { echo "[piece_${MODE}_seed${SEED}] FATAL required asset missing: $1"; exit 4; }; }

echo "[piece_${MODE}_seed${SEED}] gate reusable assets(base always; gc_value/high_actor/act_feat only when subgoal on)..."
gate "$BASE/policy/model.safetensors"
if [ "$USE_SUBGOAL" = "1" ]; then gate "$CACHE"; gate "$GCV"; gate "$HIA"; fi
[ -e "$OFFCACHE/buffer_meta.json" ] && echo "[piece_${MODE}_seed${SEED}] offcache exists: $OFFCACHE" \
  || echo "[piece_${MODE}_seed${SEED}] offcache missing -> this run may build it: $OFFCACHE"
if [ ${#STAGE_FLAGS[@]} -gt 0 ]; then
  [ -e "$STAGES" ] && echo "[piece_${MODE}_seed${SEED}] stage cache exists: $STAGES" \
    || echo "[piece_${MODE}_seed${SEED}] stage cache missing -> training will replay and create it"
fi

echo "[piece_${MODE}_seed${SEED}] START GPU=$CUDA_VISIBLE_DEVICES seed=$SEED mode=$MODE $(date '+%F %T')"
$RUN -m resfit.rl_finetuning.chunk_residual.train_chunk_residual \
  --task TwoArmThreePieceAssembly --base_wandb_id "$BASE" --dataset "$DS" \
  --chunk_length 1 --base_action_mode queue --base_n_action_steps 10 \
  --action_scale 0.05 --actor_lr 1e-6 --actor raw \
  "${SHAPING_FLAGS[@]}" \
  --offline_dataset_path "$HDF5" \
  --offline_fraction 0.5 "${STAGE_FLAGS[@]}" \
  --offline_base_mode base_policy --demo_bc_coef "$DEMO_BC" \
  --offline_buffer_cache "$OFFCACHE" \
  "${SUBGOAL_FLAGS[@]}" \
  --seed "$SEED" \
  --total_env_steps 500000 \
  --wandb_project dexmg-chunk-residual --wandb_name "$NAME" \
  --output_dir "outputs_chunk/$NAME"
echo "[piece_${MODE}_seed${SEED}] ALL DONE $(date '+%F %T') out=outputs_chunk/$NAME"
