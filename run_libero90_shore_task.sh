#!/usr/bin/env bash
set -euo pipefail

MODE=${1:-}
TASK_ID=${2:-}
GPU_ID=${3:-}
ROOT=/mnt/mnt/data/resfit
DATA_BASE=/mnt/mnt/data/chj/lerobot_cache/physical-intelligence/libero/converted/libero_90
OUT="$ROOT/outputs_chunk"
PY_RES=/mnt/mnt/data/envs/residual/bin/python
PY_LIB=/mnt/mnt/data/envs/resfit-libero/bin/python
DRY_RUN=${DRY_RUN:-0}

case "$TASK_ID" in
  57) EXPECTED_GPU=2; TASK_LANG="pick up the cream cheese and put it in the tray" ;;
  60) EXPECTED_GPU=3; TASK_LANG="pick up the black bowl on the left and put it in the tray" ;;
  63) EXPECTED_GPU=4; TASK_LANG="stack the left bowl on the right bowl and place them in the tray" ;;
  64) EXPECTED_GPU=5; TASK_LANG="stack the right bowl on the left bowl and place them in the tray" ;;
  *) echo "unsupported task: $TASK_ID" >&2; exit 2 ;;
esac

if [[ "$GPU_ID" != "$EXPECTED_GPU" ]]; then
  echo "Task $TASK_ID expected GPU$EXPECTED_GPU, got GPU$GPU_ID" >&2
  exit 2
fi

DATA="$DATA_BASE/task$TASK_ID"
STATS="$DATA/meta/stats.json"
CACHE="$OUT/libero90_task${TASK_ID}_pi0_feat.npz"
GCV="$OUT/libero90_task${TASK_ID}_pi0_feat_gc_value.pt"
HA="$OUT/libero90_task${TASK_ID}_pi0_feat_high_actor.pt"
DUMMY="/tmp/libero90_task${TASK_ID}_pi0feat.h5"
OFFCACHE="$OUT/libero90_task${TASK_ID}_pi0feat_bp_bc01_h10_offcache"
ONLINE_OUT="$OUT/libero90_task${TASK_ID}_pi0feat_bp_bc01_h10"
IMGK=observation.images.agentview,observation.images.robot0_eye_in_hand

run() {
  if [[ "$DRY_RUN" == 1 ]]; then
    printf 'RUN'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

if [[ "$DRY_RUN" != 1 ]]; then
  if [[ ! -f "$STATS" ]]; then
    echo "missing task data: $STATS" >&2
    exit 2
  fi
  mkdir -p "$OUT"
fi

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID="$GPU_ID"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH="/mnt/mnt/data/wam-b1k/third_party/LIBERO:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

case "$MODE" in
  cache)
    run env CUDA_VISIBLE_DEVICES="$GPU_ID" "$PY_RES" -u -m \
      resfit.rl_finetuning.chunk_residual.build_pi0_feat_cache_via_serve \
      --data_source libero --host 127.0.0.1 --port 8000 \
      --lerobot_root "$DATA" --language "$TASK_LANG" \
      --serve_ckpt_id pi0_libero --pooling last --out_cache "$CACHE"
    ;;
  hierarchy)
    if [[ "$DRY_RUN" != 1 ]]; then
      if [[ ! -f "$CACHE" ]]; then
        echo "missing cache: $CACHE" >&2
        exit 2
      fi
      "$PY_RES" -c "import h5py; h5py.File('$DUMMY', 'w').close()"
    fi
    run env CUDA_VISIBLE_DEVICES="$GPU_ID" "$PY_RES" -u -m \
      resfit.rl_finetuning.chunk_residual.train_hiql_gc_value \
      --state_mode pi0_feat --pi0_feat_cache "$CACHE" \
      --pi0_serve_ckpt_id pi0_libero --pi0_image_keys "$IMGK" \
      --pi0_proprio_key observation.state --pi0_pooling last \
      --pi0_prompt "$TASK_LANG" --hdf5 "$DUMMY" \
      --dataset "physical-intelligence/libero90-task$TASK_ID" \
      --device cuda --output "$GCV"
    run env CUDA_VISIBLE_DEVICES="$GPU_ID" "$PY_RES" -u -m \
      resfit.rl_finetuning.chunk_residual.train_hiql_high_actor \
      --state_mode pi0_feat --pi0_feat_cache "$CACHE" \
      --pi0_serve_ckpt_id pi0_libero --pi0_image_keys "$IMGK" \
      --pi0_proprio_key observation.state --pi0_pooling last \
      --pi0_prompt "$TASK_LANG" --gc_value_ckpt "$GCV" \
      --hdf5 "$DUMMY" --dataset "physical-intelligence/libero90-task$TASK_ID" \
      --device cuda --output "$HA"
    ;;
  train)
    if [[ "$DRY_RUN" != 1 ]]; then
      for asset in "$CACHE" "$GCV" "$HA"; do
        if [[ ! -f "$asset" ]]; then
          echo "missing asset: $asset" >&2
          exit 2
        fi
      done
    fi
    run env CUDA_VISIBLE_DEVICES="$GPU_ID" "$PY_LIB" -u -m \
      resfit.rl_finetuning.chunk_residual.train_chunk_residual \
      --env_family libero --libero_suite libero_90 --libero_task_id "$TASK_ID" \
      --libero_stats_json "$STATS" --actor raw --chunk_length 1 \
      --action_scale 0.05 --min_range_per_dim 0.1 --gamma 0.99 --n_step 3 \
      --offline_base_mode base_policy --base_policy_type pi05 --base_action_mode queue \
      --pi0_host 127.0.0.1 --pi0_port 8000 --pi0_action_dim 7 \
      --pi0_execute_horizon 10 --offline_fraction 0.5 --demo_bc_coef 0.1 \
      --offline_buffer_cache "$OFFCACHE" --subgoal_conditioned \
      --gc_value_ckpt "$GCV" --high_actor_ckpt "$HA" --pi0_feat_cache "$CACHE" \
      --subgoal_way_steps 25 --renorm_subgoal \
      --eval_num_envs 8 --eval_num_episodes 10 \
      --total_env_steps 500000 --learning_starts 10000 --utd 4 \
      --device cuda --seed 0 --output_dir "$ONLINE_OUT" \
      --wandb_mode online --wandb_project dexmg-chunk-residual \
      --wandb_name "libero90_task${TASK_ID}_pi0feat_bp_bc01_h10"
    ;;
  *)
    echo "usage: $0 <cache|hierarchy|train> <task_id> <gpu_id>" >&2
    exit 2
    ;;
esac
