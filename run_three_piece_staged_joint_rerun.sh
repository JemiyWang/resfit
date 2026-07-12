#!/bin/bash
# three_piece staged_joint 重跑(seed 0)。原 run `three_piece_actfeat_bp_bc01_hiqlv512_sg15_staged_joint`
# 卡在 env_steps 440000/500000 没跑完,这里逐字对齐它重跑一个完整的:
#   - 所有超参与原 run 一字不差(seed 沿用默认 0,不传 --seed);
#   - 仅 wandb_name / output_dir 加 _rerun 后缀,避免覆盖原没跑完的 best.pt;
#   - offcache(29G)/gc_value/high_actor/act_feat/stage_cache 全部复用现成产物,秒级加载不重建。
# 用法: setsid bash run_three_piece_staged_joint_rerun.sh <gpu> > three_piece_staged_joint_rerun.log 2>&1 < /dev/null &
set -euo pipefail
GPU=${1:-3}
export CUDA_VISIBLE_DEVICES="$GPU"
export MUJOCO_GL=egl
export HF_HUB_OFFLINE=1
export PYTHONPATH=/mnt/mnt/data/resfit
export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64
cd /mnt/mnt/data/resfit
PY=/mnt/mnt/data/envs/residual/bin/python
echo "[three_piece_staged_rerun] START GPU=$GPU seed=0(默认)  $(date '+%F %T')"
exec $PY -u -m resfit.rl_finetuning.chunk_residual.train_chunk_residual \
  --task TwoArmThreePieceAssembly \
  --base_wandb_id resfit/out/piecce/best \
  --dataset ankile/dexmg-two-arm-three-piece-assembly \
  --chunk_length 1 --base_action_mode queue --base_n_action_steps 10 \
  --action_scale 0.05 --actor_lr 1e-6 --actor raw \
  --stage_balanced --reward_shaping staged --stage_reward_bonus 1.0 \
  --offline_dataset_path resfit/dataset/two_arm_three_piece_assembly.hdf5 \
  --offline_fraction 0.5 --offline_stage_cache outputs_chunk/three_piece_stages_4stage.npz \
  --offline_base_mode base_policy --demo_bc_coef 0.1 \
  --offline_buffer_cache outputs_chunk/three_piece_actfeat_bp_hiqlv512_sg15_staged_offcache \
  --subgoal_conditioned \
  --gc_value_ckpt outputs_chunk/three_piece_gc_value_actfeat_hiqlv512.pt \
  --high_actor_ckpt outputs_chunk/three_piece_high_actor_actfeat_hiqlv512.pt \
  --act_feat_cache outputs_chunk/three_piece_act_feat.npz \
  --subgoal_way_steps 15 \
  --online_finetune_value --online_finetune_high_actor \
  --total_env_steps 500000 \
  --wandb_project dexmg-chunk-residual \
  --wandb_name three_piece_actfeat_bp_bc01_hiqlv512_sg15_staged_joint_rerun \
  --output_dir outputs_chunk/three_piece_actfeat_bp_bc01_hiqlv512_sg15_staged_joint_rerun
