#!/usr/bin/env bash
# LIBERO task3 pi0_feat 残差 --smoke:验证 eval 在线子目标注入(不再 KeyError)。
# 复用现成 offcache outputs_chunk/libero_task3_pi0feat_offcache(签名须严丝合缝命中,否则 ~80min 重建)。
# serve 须在 127.0.0.1:8000(pi0_libero/pooling last)。本机用卡 5(空),serve 在卡 4。
set -euo pipefail
cd /mnt/mnt/data/resfit

export CUDA_VISIBLE_DEVICES=5
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=5
# libero 在 resfit-libero 里 editable 安装的 .pth 已失效;源码实在这,显式上 PYTHONPATH。
export PYTHONPATH=/mnt/mnt/data/wam-b1k/third_party/LIBERO:/mnt/mnt/data/resfit${PYTHONPATH:+:$PYTHONPATH}
# 共享机满载,默认抓满核 BLAS/cv2 线程会 thrash(实测 offcache build 慢 ~55x)。锁单线程。
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

RESF=/mnt/mnt/data/resfit
PY=/mnt/mnt/data/envs/resfit-libero/bin/python

"$PY" -u -m resfit.rl_finetuning.chunk_residual.train_chunk_residual \
  --env_family libero --libero_suite libero_10 --libero_task_id 3 \
  --libero_stats_json /mnt/mnt/data/chj/lerobot_cache/physical-intelligence/libero/meta/stats.json \
  --actor raw --chunk_length 1 \
  --action_scale 0.05 --min_range_per_dim 0.1 --gamma 0.99 --n_step 3 \
  --offline_base_mode base_policy --base_policy_type pi05 --base_action_mode queue \
  --pi0_host 127.0.0.1 --pi0_port 8000 --pi0_action_dim 7 --pi0_execute_horizon 10 \
  --offline_fraction 0.5 --demo_bc_coef 0.1 \
  --offline_buffer_cache outputs_chunk/libero_task3_pi0feat_offcache \
  --subgoal_conditioned \
  --gc_value_ckpt "$RESF/outputs_chunk/libero10_task3_pi0_feat_gc_value.pt" \
  --high_actor_ckpt "$RESF/outputs_chunk/libero10_task3_pi0_feat_high_actor.pt" \
  --pi0_feat_cache "$RESF/outputs_chunk/libero10_task3_pi0_feat.npz" \
  --subgoal_way_steps 25 --renorm_subgoal \
  --eval_num_episodes 3 \
  --device cuda --seed 0 \
  --output_dir outputs_chunk/libero_task3_pi0feat_residual_smoke \
  --wandb_mode disabled --smoke
