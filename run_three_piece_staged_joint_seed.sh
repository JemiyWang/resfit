#!/usr/bin/env bash
# three_piece staged+joint seed 变体:逐字复现 staged_joint,唯一变量 = --seed。
#   参数取自 run_three_piece_staged_joint.sh,仅 output_dir/wandb_name 加 _seed<SEED> + 显式 --seed。
#   ⚠ 用 4 段 three_piece_stages_4stage.npz(已存在,直读;绝不碰旧 5 段)。
#   offcache 只读复用 three_piece_actfeat_bp_hiqlv512_sg15_staged_offcache
#   (seed 不进签名,两个 seed 并发只读同一份;省重建 29G)。
# 用法: setsid bash run_three_piece_staged_joint_seed.sh <gpu> <seed> > three_piece_..._seed<seed>.log 2>&1 < /dev/null &
set -u
GPU=${1:?需要 gpu}; SEED=${2:?需要 seed}
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
STAGES=outputs_chunk/three_piece_stages_4stage.npz                        # 4段(已存在,直读)
CACHE=outputs_chunk/three_piece_act_feat.npz                              # 复用
BASE=resfit/out/piecce/best                                              # build_base_policy 自动取 policy/ 子目录
GCV=outputs_chunk/three_piece_gc_value_actfeat_hiqlv512.pt                # 复用
HIA=outputs_chunk/three_piece_high_actor_actfeat_hiqlv512.pt              # 复用(way_steps15)
OFFCACHE=outputs_chunk/three_piece_actfeat_bp_hiqlv512_sg15_staged_offcache  # 只读复用(已存在)
NAME=three_piece_actfeat_bp_bc01_hiqlv512_sg15_staged_joint_seed${SEED}
WAY=15
gate(){ [ -s "$1" ] || { echo "[piece_seed$SEED] FATAL 复用产物缺失: $1"; exit 4; }; }

echo "[piece_seed$SEED] gate 复用产物(act_feat/stages/gc_value/high_actor/base/offcache)..."
gate "$CACHE"; gate "$STAGES"; gate "$GCV"; gate "$HIA"; gate "$BASE/policy/model.safetensors"; gate "$OFFCACHE/buffer_meta.json"
echo "[piece_seed$SEED] START piece staged+joint  GPU=$CUDA_VISIBLE_DEVICES seed=$SEED  $(date '+%F %T')  (offcache/stages 复用)"

$RUN -m resfit.rl_finetuning.chunk_residual.train_chunk_residual \
  --task TwoArmThreePieceAssembly --base_wandb_id "$BASE" --dataset "$DS" \
  --chunk_length 1 --base_action_mode queue --base_n_action_steps 10 \
  --action_scale 0.05 --actor_lr 1e-6 --actor raw --stage_balanced \
  --reward_shaping staged --stage_reward_bonus 1.0 \
  --offline_dataset_path "$HDF5" \
  --offline_fraction 0.5 --offline_stage_cache "$STAGES" \
  --offline_base_mode base_policy --demo_bc_coef 0.1 \
  --offline_buffer_cache "$OFFCACHE" \
  --subgoal_conditioned --gc_value_ckpt "$GCV" --high_actor_ckpt "$HIA" --act_feat_cache "$CACHE" \
  --subgoal_way_steps "$WAY" \
  --online_finetune_value --online_finetune_high_actor \
  --seed "$SEED" \
  --total_env_steps 500000 \
  --wandb_project dexmg-chunk-residual --wandb_name "$NAME" \
  --output_dir "outputs_chunk/$NAME"
echo "[piece_seed$SEED] ALL DONE piece staged+joint $(date '+%F %T')  out=outputs_chunk/$NAME"
