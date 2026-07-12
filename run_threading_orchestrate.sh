#!/usr/bin/env bash
# 编排器:等共享 gc_value(hiqlv512)训完后,自动并行起两条 threading 实验。
#   sg15 -> GPU3(gc_value 训完即释放),sg8 bcdecay -> GPU5。
# 用法: setsid bash run_threading_orchestrate.sh > threading_orchestrate.log 2>&1 < /dev/null &
set -u
cd /mnt/mnt/data/resfit || exit 3
GCV=outputs_chunk/two_arm_threading_gc_value_actfeat_hiqlv512.pt

echo "[orch] 等待 gc_value 训练结束... $(date '+%F %T')"
while pgrep -f 'train_hiql_gc_value.*two_arm_threading' >/dev/null 2>&1; do sleep 20; done

if [ ! -s "$GCV" ]; then
  echo "[orch] FATAL gc_value 训练未产出 $GCV(可能失败),不启动两条。$(date '+%F %T')"
  exit 4
fi
echo "[orch] gc_value 就绪($(du -h "$GCV"|cut -f1)),启动 sg15(GPU3)+ sg8(GPU5) $(date '+%F %T')"

setsid bash run_threading_hiqlv512_sg15.sh 3 > threading_hiqlv512_sg15.log 2>&1 < /dev/null &
echo "[orch] sg15 launched on GPU3"
setsid bash run_threading_hiqlv512_sg8_bcdecay.sh 5 > threading_hiqlv512_sg8_bcdecay.log 2>&1 < /dev/null &
echo "[orch] sg8 bcdecay launched on GPU5"
echo "[orch] DONE $(date '+%F %T')"
