#!/usr/bin/env bash
# libero10 task8 残差 RL —— 复刻 aligned_bp_bc01_h10 配方,但走 pi0_feat 子目标(--subgoal_conditioned)。
# 与基线 libero10_task8_aligned_bp_bc01_h10 唯一区别:开 pi0_feat 子目标(gc_value/high_actor/cache=task8 自己的)
# + 独立 offcache(加 subgoal 字段签名变,需新建一次)+ 独立 output/wandb_name。
# 训练内参全沿用金标准默认(actor_lr1e-6/critic_lr1e-4/batch256/stddev0.05/buffer20万/utd4/stage_balanced默认True)。
# serve 须在 127.0.0.1:8000(pi0_libero/pooling last);本机训练用卡 5(空),serve 在卡 4。
set -euo pipefail
cd /mnt/mnt/data/resfit

export CUDA_VISIBLE_DEVICES=5
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=5
# libero 在 resfit-libero 里 editable .pth 已失效;源码在此,显式上 PYTHONPATH。
export PYTHONPATH=/mnt/mnt/data/wam-b1k/third_party/LIBERO:/mnt/mnt/data/resfit${PYTHONPATH:+:$PYTHONPATH}
# 共享机满载,默认抓满核 BLAS/cv2 线程会 thrash(实测 offcache build 慢 ~55x)。锁单线程治本。
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

RESF=/mnt/mnt/data/resfit
PY=/mnt/mnt/data/envs/resfit-libero/bin/python

"$PY" -u -m resfit.rl_finetuning.chunk_residual.train_chunk_residual \
  --env_family libero --libero_suite libero_10 --libero_task_id 8 \
  --libero_stats_json /mnt/mnt/data/chj/lerobot_cache/physical-intelligence/libero/meta/stats.json \
  --actor raw --chunk_length 1 \
  --action_scale 0.05 --min_range_per_dim 0.1 --gamma 0.99 --n_step 3 \
  --offline_base_mode base_policy --base_policy_type pi05 --base_action_mode queue \
  --pi0_host 127.0.0.1 --pi0_port 8000 --pi0_action_dim 7 --pi0_execute_horizon 10 \
  --offline_fraction 0.5 --demo_bc_coef 0.1 \
  --offline_buffer_cache outputs_chunk/libero10_task8_pi0feat_bp_bc01_h10_offcache \
  --subgoal_conditioned \
  --gc_value_ckpt "$RESF/outputs_chunk/libero10_t8moka_pi0_feat_gc_value.pt" \
  --high_actor_ckpt "$RESF/outputs_chunk/libero10_t8moka_pi0_feat_high_actor.pt" \
  --pi0_feat_cache "$RESF/outputs_chunk/libero10_t8moka_pi0_feat.npz" \
  --subgoal_way_steps 25 --renorm_subgoal \
  --eval_num_envs 8 --eval_num_episodes 10 \
  --total_env_steps 500000 --learning_starts 10000 --utd 4 \
  --device cuda --seed 0 \
  --output_dir outputs_chunk/libero10_task8_pi0feat_bp_bc01_h10 \
  --wandb_mode online --wandb_project dexmg-chunk-residual \
  --wandb_name libero10_task8_pi0feat_bp_bc01_h10
