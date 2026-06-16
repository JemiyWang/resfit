"""端到端落盘实测:真实 LeRobot(torchcodec)build → 真 memmap 落盘 → 量 storage 字节,
坐实图改 uint8 后 offcache 落盘大小按 1/4 字节算(而非 float32)。
num_demos=8、base_mode=gt、只吃 CPU;落盘到 /tmp,跑完自删。
"""
import os
import shutil
import sys

import torch
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer

from resfit.rl_finetuning.chunk_residual.lerobot_demo_source import open_lerobot
from resfit.rl_finetuning.chunk_residual.offline_stage_replay import build_offline_buffer


class _Identity:
    def scale(self, x):
        return x

    def standardize(self, x):
        return x


IMAGE_KEYS = [
    "observation.images.agentview",
    "observation.images.robot0_eye_in_hand",
    "observation.images.robot1_eye_in_hand",
]
REPO = "ankile/dexmg-two-arm-three-piece-assembly"
ROOT = "/root/.cache/huggingface/lerobot/ankile/dexmg-two-arm-three-piece-assembly"
N_DEMOS = 8
OUT = "/tmp/test_lerobot_offcache_uint8"

print(f"[size] open lerobot + build {N_DEMOS} demos ...", flush=True)
ds = open_lerobot(REPO, ROOT)
rb = TensorDictReplayBuffer(
    storage=LazyTensorStorage(max_size=30000, device="cpu"), batch_size=4)
n = build_offline_buffer(
    rb, None, action_scaler=_Identity(), state_standardizer=_Identity(),
    image_keys=IMAGE_KEYS, bonus=1.0, mode="staged", gamma=0.99, num_demos=N_DEMOS,
    data_source="lerobot", lerobot_repo_id=REPO, lerobot_root=ROOT,
    _lerobot_ds=ds, base_mode="gt")
img = rb[0]["obs"][IMAGE_KEYS[0]]
print(f"[size] built {n} transitions; img dtype={img.dtype} "
      f"element_size={img.element_size()}B", flush=True)

# 真 memmap 落盘(复刻 _save_offline_buffer 的落盘动作)
if os.path.exists(OUT):
    shutil.rmtree(OUT)
data = rb[:len(rb)]
for k in ("index", "_weight"):
    if k in data.keys():
        data = data.exclude(k)
data.memmap(os.path.join(OUT, "storage"))


def dir_bytes(p):
    t = 0
    for root, _, files in os.walk(p):
        for f in files:
            t += os.path.getsize(os.path.join(root, f))
    return t


sb = dir_bytes(os.path.join(OUT, "storage"))
FULL = 238819                      # piece 全集 transition 数(实测)
ratio = FULL / n
print(f"[size] 落盘 storage = {sb/1e9:.2f} GB ({n} 条)")
print(f"[size] 外推全集(uint8,本次改动) ≈ {sb*ratio/1e9:.0f} GB")
print(f"[size] 若仍 float32(×4)全集应 ≈ {sb*ratio*4/1e9:.0f} GB  (= 改前实测 113G)")
shutil.rmtree(OUT, ignore_errors=True)
print("[size] 临时落盘已清理")
sys.exit(0)
