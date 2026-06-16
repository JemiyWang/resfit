#!/usr/bin/env bash
# Source this before any training that decodes LeRobot videos with the torchcodec backend.
# torchcodec is ~8x faster than pyav for random-frame decode, but on this machine it needs:
#   (1) nvidia-npp-cu12 installed in the residual env (provides libnppicc.so.12 etc.)
#   (2) the env's newer libstdc++ (GLIBCXX_3.4.29+) and the nvidia pip libs on LD_LIBRARY_PATH,
#       because torchcodec's bundled FFmpeg7 links libvpl/libicuuc which need a newer libstdc++
#       than the system /usr/lib one.
# Usage:  source resfit/lerobot/shell/torchcodec_env.sh   (then run python under conda env residual)
_RESIDUAL_ENV=/mnt/mnt/data/envs/residual
_NV="$_RESIDUAL_ENV/lib/python3.10/site-packages/nvidia"
export LD_LIBRARY_PATH="$_RESIDUAL_ENV/lib:$(ls -d "$_NV"/*/lib 2>/dev/null | tr '\n' ':')${LD_LIBRARY_PATH:-}"
