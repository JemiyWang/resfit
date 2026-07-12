#!/usr/bin/env bash
# pouring bcdecay01to001 @ as005:与 run_pouring_hiqlv512_sg15_bcfixed01_as005.sh 对齐,仅一处改:
#   bc loss 从"固定 0.1"改为"0.1 线性衰减到 0.01"(--bc_coef_final 0.01)。action_scale 仍 0.05。
# offcache 复用 as005 的:bc_coef 不进 offcache 签名、action_scale 与 as005 相同 → 内容逐字节相同,
#   签名匹配会命中 valid 分支秒读(只读 memmap,与 as005 无写冲突)。故不另建,省 41G + 40min,
#   也避免双进程同写撞坏 + 抢 CPU 拖慢 as005 的构建(offline buffer 无文件锁/无原子写)。
# 同步:as005 此刻可能仍在建该 offcache;本脚本先等其 buffer_meta.json(_save_offline_buffer 最后写=
#   完成哨兵)出现再启 python。等不到(as005 挂了)则 FATAL 退出,绝不在共享路径自建以免碰撞。
# 用法: setsid bash run_pouring_hiqlv512_sg15_bcdecay01to001_as005.sh <gpu> > pouring_hiqlv512_sg15_bcdecay01to001_as005.log 2>&1 < /dev/null &
set -u
GPU=${1:-1}
export PATH="/root/miniconda3/bin:$PATH"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export CUDA_VISIBLE_DEVICES="$GPU"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_OFFLINE=1
export PYTHONPATH=/mnt/mnt/data/resfit
cd /mnt/mnt/data/resfit || exit 3
RUN="conda run -n residual --no-capture-output python -u"

TASK=TwoArmPouring
DS=ankile/dexmg-two-arm-pouring
HDF5=resfit/dataset/two_arm_pouring.hdf5
BASE=/mnt/mnt/data/wjm/residual/residual-offpolicy-rl/artifacts/run_anw5pphu_best:v2
CACHE=outputs_chunk/pouring_act_feat_hdf5.npz                                     # 复用(与 action_scale 无关)
GC=outputs_chunk/pouring_gc_value_actfeat_hdf5_hiqlv512.pt                        # 复用(与 action_scale 无关)
HA=outputs_chunk/pouring_high_actor_actfeat_hdf5_hiqlv512.pt                      # 复用(与 action_scale 无关)
OFFCACHE=outputs_chunk/pouring_actfeat_hdf5_bp_hiqlv512_sg15_as005_offcache       # 复用 as005 的(签名相同→命中)
OUT=outputs_chunk/pouring_actfeat_hdf5_bp_bcdecay01to001_hiqlv512_sg15_as005      # 新(bc 0.1→0.01 衰减)
ACTION_SCALE=0.05
META="$OFFCACHE/buffer_meta.json"
gate(){ [ -s "$1" ] || { echo "[bcd] FATAL 复用产物缺失: $1"; exit 4; }; }

echo "[bcd] gate 复用产物(gc_value/high_actor/act_feat)..."
gate "$CACHE"; gate "$GC"; gate "$HA"

# === 等 as005 把共享 offcache 建完(buffer_meta.json 最后写=完成哨兵)===
echo "[bcd] 等待共享 offcache 就绪(as005 在建): $META  $(date '+%F %T')"
WAITED=0; MAXWAIT=7200
while [ ! -f "$META" ]; do
  sleep 30; WAITED=$((WAITED+30))
  if [ "$WAITED" -ge "$MAXWAIT" ]; then
    echo "[bcd] FATAL 等 ${MAXWAIT}s 仍无 $META(as005 可能已挂);为避免共享路径双写碰撞,拒绝自建,退出。"
    echo "[bcd]   如确需独立跑:把 OFFCACHE 改成新路径(如 ..._bcdecay001_offcache)重启,会自建一份。"
    exit 5
  fi
  if [ $((WAITED % 300)) -eq 0 ]; then echo "[bcd] ...已等 ${WAITED}s,$META 尚未出现"; fi
done
echo "[bcd] 共享 offcache 已就绪 → 启动主训(将命中 valid 缓存秒读) $(date '+%F %T')"
echo "[bcd] START pouring ($TASK) hiqlv512 sg15 bc:0.1→0.01衰减 action_scale=$ACTION_SCALE  GPU=$CUDA_VISIBLE_DEVICES"

# === 主训(与 as005 逐字一致,仅加 --bc_coef_final 0.01 / wandb_name / output_dir 改)===
$RUN -m resfit.rl_finetuning.chunk_residual.train_chunk_residual \
  --task "$TASK" --base_wandb_id "$BASE" --dataset "$DS" \
  --offline_dataset_path "$HDF5" \
  --chunk_length 1 --base_action_mode queue --base_n_action_steps 10 \
  --action_scale "$ACTION_SCALE" --actor_lr 1e-6 --actor raw \
  --reward_shaping none \
  --offline_fraction 0.5 --offline_base_mode base_policy --demo_bc_coef 0.1 --bc_coef_final 0.01 \
  --offline_buffer_cache "$OFFCACHE" \
  --subgoal_conditioned --gc_value_ckpt "$GC" --high_actor_ckpt "$HA" --act_feat_cache "$CACHE" \
  --subgoal_way_steps 15 \
  --total_env_steps 500000 \
  --wandb_project dexmg-chunk-residual --wandb_name pouring_actfeat_hdf5_bp_bcdecay01to001_hiqlv512_sg15_as005 \
  --output_dir "$OUT"
echo "[bcd] ALL DONE pouring hiqlv512 sg15 bc0.1→0.01 as005 $(date '+%F %T')  out=$OUT"
