#!/usr/bin/env bash
# three_piece staged + 联合训练(online joint finetune)。
#   参照 three_piece_actfeat_bp_bc01_hiqlv512_sg15_pothiql,只改两处:
#     ① reward: potential(object-aware hiql V) → staged(stage 整数净加 bonus;不需要 value.pt)
#     ② 打开联合训练: --online_finetune_value --online_finetune_high_actor
#   其余完全照搬 piece pothiql:actor raw / chunk1 / queue / base_n10 / action_scale0.05 /
#     actor_lr1e-6 / stage_balanced / offline_fraction0.5 / base_policy锚 + bc0.1 /
#     subgoal sg15(gc_value hiqlv512 + high_actor)/ 500k。
#
#   ⚠ 关键(piece 特有):piece 2026-06-29 已改 4 段。staged reward 直接吃离散 stage,
#     绝不能复用 22h run 在用的旧 5 段 three_piece_stages.npz(会"在线4段/离线5段"矛盾)。
#     → 用新路径 three_piece_stages_4stage.npz;不存在 → 主训自动用当前 4 段检测器
#       sim-replay 重生成并保存(offline_stage_replay.py:230/278/370)。旧 5 段 cache 不碰。
#   gc_value/high_actor/act_feat stage-agnostic(geometric/clamp_to_goal,不读 stage)→ 复用免重训。
# 用法: setsid bash run_three_piece_staged_joint.sh <gpu> > three_piece_staged_joint.log 2>&1 < /dev/null &
set -u
GPU=${1:-5}
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
STAGES=outputs_chunk/three_piece_stages_4stage.npz                        # 新4段(首次 sim-replay 重生;不碰旧5段)
CACHE=outputs_chunk/three_piece_act_feat.npz                              # 复用
BASE=resfit/out/piecce/best                                              # build_base_policy 自动取 policy/ 子目录
GCV=outputs_chunk/three_piece_gc_value_actfeat_hiqlv512.pt                # 复用(joint 在线微调,初值取它)
HIA=outputs_chunk/three_piece_high_actor_actfeat_hiqlv512.pt              # 复用(way_steps15;joint 在线微调初值)
OFFCACHE=outputs_chunk/three_piece_actfeat_bp_hiqlv512_sg15_staged_offcache  # 新(staged+4段签名,重建)
OUT=outputs_chunk/three_piece_actfeat_bp_bc01_hiqlv512_sg15_staged_joint  # 新
WAY=15
gate(){ [ -s "$1" ] || { echo "[piece_staged_joint] FATAL 复用产物缺失: $1"; exit 4; }; }

echo "[piece_staged_joint] gate 复用产物(act_feat/gc_value/high_actor/base)..."
gate "$CACHE"; gate "$GCV"; gate "$HIA"; gate "$BASE/policy/model.safetensors"
[ -e "$STAGES" ] && echo "[piece_staged_joint] 注意:$STAGES 已存在,将直接读(确认是4段);若想重生成请先删它" \
                  || echo "[piece_staged_joint] $STAGES 不存在 → 首次将 sim-replay 重生成 4 段(较慢,一次性)"
echo "[piece_staged_joint] START piece staged+joint  GPU=$CUDA_VISIBLE_DEVICES  $(date '+%F %T')"

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
  --total_env_steps 500000 \
  --wandb_project dexmg-chunk-residual --wandb_name three_piece_actfeat_bp_bc01_hiqlv512_sg15_staged_joint \
  --output_dir "$OUT"
echo "[piece_staged_joint] ALL DONE piece staged+joint $(date '+%F %T')  out=$OUT"
