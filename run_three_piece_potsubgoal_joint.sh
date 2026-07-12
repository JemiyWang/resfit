#!/usr/bin/env bash
# three_piece potsubgoal(A2) + 联合训练(online joint finetune)。
#   与 three_piece_actfeat_bp_bc01_hiqlv512_sg15_staged_joint 逐字对齐,只改 reward 一处:
#     staged(--reward_shaping staged --stage_reward_bonus 1.0)
#       → A2 goal-cond potential(--reward_shaping potential --potential_source hiql_subgoal)
#     即 Φ=V(s,z)*scale(gc_value hiqlv512),在线+离线两侧都用它做 PBS shaping。
#   其余完全照搬 staged_joint:actor raw / chunk1 / queue / base_n10 / action_scale0.05 /
#     actor_lr1e-6 / stage_balanced / offline_fraction0.5 / base_policy锚 + bc0.1 /
#     subgoal sg15(gc_value + high_actor)/ joint(online_finetune_value/high_actor)/ 500k。
#
#   A2 关键(代码已自动处理,无需额外 flag):
#     · train_env_shaping_mode(hiql_subgoal,…)=none → 训练 wrapper 零 stage shaping,
#       主循环独占全部 gc shaping(避免 stage 任务的 double-shaping Critical 坑)。
#     · 离线 build_offline_buffer(mode=potential, potential=gc_potential):reward=稀疏base
#       + V(s,z) 差分,不叠 stage bonus;stage_cache 仅供 stage_balanced 采样贴标签(正交)。
#     · 前置校验 validate_hiql_subgoal_args:reward_shaping=potential / subgoal_conditioned /
#       renorm_subgoal(默认 True)/ gc_value_ckpt —— 均满足。
#
#   ⚠ piece 4 段(three_piece_stages_4stage.npz),绝不复用旧 5 段;stage 仅用于采样,非 reward。
#   offcache 新签名(potential_source=hiql_subgoal)+ 新路径,必重建(约 29G,~80min base_policy 建缓存)。
# 用法: setsid bash run_three_piece_potsubgoal_joint.sh <gpu> > three_piece_potsubgoal_joint.log 2>&1 < /dev/null &
set -u
GPU=${1:-2}
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
STAGES=outputs_chunk/three_piece_stages_4stage.npz                              # 复用 4 段(仅采样用)
CACHE=outputs_chunk/three_piece_act_feat.npz                                    # 复用
BASE=resfit/out/piecce/best                                                    # build_base_policy 自动取 policy/ 子目录
GCV=outputs_chunk/three_piece_gc_value_actfeat_hiqlv512.pt                      # 复用(V(s,z);potential + joint 初值)
HIA=outputs_chunk/three_piece_high_actor_actfeat_hiqlv512.pt                    # 复用(way_steps15;joint 初值)
OFFCACHE=outputs_chunk/three_piece_actfeat_bp_hiqlv512_sg15_potsubgoal_joint_offcache  # 新(potsubgoal 签名,重建)
OUT=outputs_chunk/three_piece_actfeat_bp_bc01_hiqlv512_sg15_potsubgoal_joint    # 新
WAY=15
gate(){ [ -s "$1" ] || { echo "[piece_potsubgoal_joint] FATAL 复用产物缺失: $1"; exit 4; }; }

echo "[piece_potsubgoal_joint] gate 复用产物(act_feat/gc_value/high_actor/base)..."
gate "$CACHE"; gate "$GCV"; gate "$HIA"; gate "$BASE/policy/model.safetensors"
[ -e "$STAGES" ] && echo "[piece_potsubgoal_joint] 注意:$STAGES 已存在,将直接读(确认是4段)" \
                  || echo "[piece_potsubgoal_joint] $STAGES 不存在 → 首次将 sim-replay 重生成 4 段(较慢,一次性)"
echo "[piece_potsubgoal_joint] START piece potsubgoal+joint  GPU=$CUDA_VISIBLE_DEVICES  $(date '+%F %T')"

$RUN -m resfit.rl_finetuning.chunk_residual.train_chunk_residual \
  --task TwoArmThreePieceAssembly --base_wandb_id "$BASE" --dataset "$DS" \
  --chunk_length 1 --base_action_mode queue --base_n_action_steps 10 \
  --action_scale 0.05 --actor_lr 1e-6 --actor raw --stage_balanced \
  --reward_shaping potential --potential_source hiql_subgoal \
  --offline_dataset_path "$HDF5" \
  --offline_fraction 0.5 --offline_stage_cache "$STAGES" \
  --offline_base_mode base_policy --demo_bc_coef 0.1 \
  --offline_buffer_cache "$OFFCACHE" \
  --subgoal_conditioned --gc_value_ckpt "$GCV" --high_actor_ckpt "$HIA" --act_feat_cache "$CACHE" \
  --subgoal_way_steps "$WAY" \
  --online_finetune_value --online_finetune_high_actor \
  --total_env_steps 500000 \
  --wandb_project dexmg-chunk-residual --wandb_name three_piece_actfeat_bp_bc01_hiqlv512_sg15_potsubgoal_joint \
  --output_dir "$OUT"
echo "[piece_potsubgoal_joint] ALL DONE piece potsubgoal+joint $(date '+%F %T')  out=$OUT"
