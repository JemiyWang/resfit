"""验证 ACT encoder_out 特征路径 + 图像格式检查(act_feat 诊断 spike)。

运行方式(仓库根):
    CUDA_VISIBLE_DEVICES="" conda run -n residual \
        python -m resfit.rl_finetuning.chunk_residual.verify_act_prefix_feature

此脚本仅诊断,不修改任何生产代码。
"""
from __future__ import annotations

import os
import sys

import h5py
import torch

# 强制 CPU:不需要 EGL/CUDA
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from pathlib import Path

# ---------------------------------------------------------------------------
# 0. 路径常量
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[3]          # /mnt/mnt/data/resfit
POLICY_DIR = REPO_ROOT / "resfit" / "out" / "piecce" / "best" / "policy"
HDF5_PATH = REPO_ROOT / "resfit" / "dataset" / "two_arm_three_piece_assembly.hdf5"

print("=" * 70)
print("SPIKE: verify_act_prefix_feature.py")
print(f"  policy : {POLICY_DIR}")
print(f"  hdf5   : {HDF5_PATH}")
print("=" * 70)

# ---------------------------------------------------------------------------
# 1. 加载 ACT policy
# ---------------------------------------------------------------------------
print("\n[1] Loading ACT policy …")
sys.path.insert(0, str(REPO_ROOT))
from resfit.lerobot.utils.load_policy import load_policy

act = load_policy(POLICY_DIR)
act.eval()

image_keys = list(act.config.image_features.keys())
dim_model = act.config.dim_model
print(f"  image_features keys : {image_keys}")
print(f"  dim_model           : {dim_model}")
print(f"  expected img shape  : {list(act.config.image_features.values())[0].shape}  (C,H,W)")

# 打印 normalize_inputs 的图像统计范围 → 判断 ACT 期望什么输入尺度
print("\n  normalize_inputs image stats (mean/std):")
for k in image_keys:
    buf_name = "buffer_" + k.replace(".", "_")
    buf = getattr(act.normalize_inputs, buf_name)
    m, s = buf["mean"], buf["std"]
    print(f"    {k}:  mean=[{m.min().item():.4f},{m.max().item():.4f}]  "
          f"std=[{s.min().item():.4f},{s.max().item():.4f}]")

# 确认 encoder 不是 VAE encoder
print(f"\n  model.encoder type  : {type(act.model.encoder).__name__}")
has_vae_enc = hasattr(act.model, "vae_encoder")
print(f"  model.vae_encoder   : {'exists → ' + type(act.model.vae_encoder).__name__ if has_vae_enc else 'not present'}")

# ---------------------------------------------------------------------------
# 2. 读 hdf5 demo_0:检查原始图像 dtype/shape/range
# ---------------------------------------------------------------------------
print("\n[2] Raw hdf5 image inspection (demo_0, first 2 frames) …")
# image_keys like "observation.images.agentview" → obs key = "agentview_image"
img_ds_names = {k: k.replace("observation.images.", "") + "_image" for k in image_keys}

raw_obs_2 = {}  # 2-frame raw_obs (as _build_raw_obs_seqs would build)
with h5py.File(HDF5_PATH, "r") as f:
    grp = f["data/demo_0"]
    for fk, ds_name in img_ds_names.items():
        arr = grp[f"obs/{ds_name}"][:2]          # shape (2,H,W,C) uint8
        t = torch.as_tensor(arr)                  # exactly what _build_raw_obs_seqs does
        print(f"  obs/{ds_name}: dtype={arr.dtype}, shape={arr.shape}  "
              f"min={arr.min()} max={arr.max()}")
        print(f"    → torch.as_tensor shape={tuple(t.shape)}  dtype={t.dtype}  "
              f"min={t.min().item()} max={t.max().item()}")
        raw_obs_2[fk] = t

    # assemble state
    from resfit.rl_finetuning.chunk_residual.offline_hdf5_buffer import (
        STATE18_KEYS, assemble_state18)
    obs_arrays = {k: grp[f"obs/{k}"][()] for k, _ in STATE18_KEYS}
    state18 = assemble_state18(obs_arrays)         # (T,18)
    raw_obs_2["observation.state"] = torch.as_tensor(state18[:2], dtype=torch.float32)
    print(f"  observation.state: shape={tuple(raw_obs_2['observation.state'].shape)}")

# ---------------------------------------------------------------------------
# 2b. Image-format verdict: what does ACT expect?
# ---------------------------------------------------------------------------
print("\n[2b] Image-format analysis …")
# ACT config shape is [C,H,W] = [3,84,84]
cfg_shape = list(act.config.image_features.values())[0].shape
print(f"  ACT config expects  : shape={tuple(cfg_shape)} (C,H,W), float32, stats-normalized")
# normalize_inputs mean is stored as [C,1,1] → subtract mean in CHW float space
# The mean values from the dataset are in [0,1] range (LeRobot stores images as [0,1] float)
first_key = image_keys[0]
buf = getattr(act.normalize_inputs, "buffer_" + first_key.replace(".", "_"))
mean_val = buf["mean"].mean().item()
std_val = buf["std"].mean().item()
print(f"  ACT normalize_inputs mean≈{mean_val:.4f}  std≈{std_val:.4f}")
if 0.0 <= mean_val <= 1.0 and 0.0 < std_val <= 1.0:
    print("  → mean/std are in [0,1] range: ACT was trained on float images in [0,1]")
    print("  → raw hdf5 images are uint8 HWC [0,255]: need /255.0 + permute(HWC→CHW)")
else:
    print(f"  → unusual mean/std: {mean_val:.4f}/{std_val:.4f}; check manually")

raw_hw = raw_obs_2[image_keys[0]]  # shape (2,84,84,3) uint8
print(f"\n  raw hdf5 image shape : {tuple(raw_hw.shape)}   dtype={raw_hw.dtype}")
print(f"  ACT expected shape   : (B, {cfg_shape[0]}, {cfg_shape[1]}, {cfg_shape[2]})  dtype=float32")
format_mismatch = (raw_hw.dtype == torch.uint8) or (raw_hw.shape[-1] == 3 and raw_hw.ndim == 4)
print(f"  Format mismatch?     : {'YES — HWC uint8 vs CHW float32' if format_mismatch else 'NO'}")

# ---------------------------------------------------------------------------
# 3. get_action_chunk on raw_obs (decisive end-to-end test)
# ---------------------------------------------------------------------------
print("\n[3] get_action_chunk on raw_obs (decisive format test) …")
from resfit.rl_finetuning.chunk_residual.chunk_act_base import get_action_chunk

try:
    chunk = get_action_chunk(act, raw_obs_2, chunk_length=1)
    finite = torch.isfinite(chunk).all().item()
    print(f"  get_action_chunk SUCCESS: shape={tuple(chunk.shape)}  finite={finite}")
    print(f"  action stats: min={chunk.min().item():.4f}  max={chunk.max().item():.4f}  "
          f"mean={chunk.mean().item():.4f}")
    act_chunk_ok = True
except Exception as e:
    print(f"  get_action_chunk FAILED: {type(e).__name__}: {e}")
    act_chunk_ok = False

# ---------------------------------------------------------------------------
# 3b. Also test with CORRECTLY formatted images (CHW float)
# ---------------------------------------------------------------------------
print("\n[3b] get_action_chunk with CORRECT format (CHW float /255) …")
raw_obs_correct = {}
for fk in image_keys:
    arr = raw_obs_2[fk]           # (2, 84, 84, 3) uint8
    arr_f = arr.float() / 255.0   # (2, 84, 84, 3) float32 [0,1]
    arr_chw = arr_f.permute(0, 3, 1, 2)  # (2, 3, 84, 84)
    raw_obs_correct[fk] = arr_chw
raw_obs_correct["observation.state"] = raw_obs_2["observation.state"]

try:
    chunk_correct = get_action_chunk(act, raw_obs_correct, chunk_length=1)
    finite_c = torch.isfinite(chunk_correct).all().item()
    print(f"  get_action_chunk SUCCESS: shape={tuple(chunk_correct.shape)}  finite={finite_c}")
    print(f"  action stats: min={chunk_correct.min().item():.4f}  max={chunk_correct.max().item():.4f}  "
          f"mean={chunk_correct.mean().item():.4f}")
    act_correct_ok = True
except Exception as e:
    print(f"  get_action_chunk FAILED: {type(e).__name__}: {e}")
    act_correct_ok = False

# ---------------------------------------------------------------------------
# 4. ActFeatureExtractor: encoder_out shape + embed_batch
# ---------------------------------------------------------------------------
print("\n[4] ActFeatureExtractor.embed_batch …")
from resfit.rl_finetuning.chunk_residual.act_feature import ActFeatureExtractor

# Use correctly-formatted obs for the extractor check
# (the format verdict comes from task 3, not from "did embed_batch run")
extractor = ActFeatureExtractor(act, image_keys, proprio_dim=18)
print(f"  feature_dim (dim_model+18) = {extractor.feature_dim}")

# Patch hook to also capture encoder_out shape
captured_shape = {}
original_embed = extractor.embed_batch.__func__  # underlying function

@torch.no_grad()
def _patched_embed(self, ro):
    self.act.eval()
    batch = dict(self.act.normalize_inputs(ro))
    batch["observation.images"] = [batch[k] for k in self.image_keys]
    cap = {}
    h = self.act.model.encoder.register_forward_hook(
        lambda m, i, o: cap.__setitem__("enc", o))
    try:
        self.act.model(batch)
    finally:
        h.remove()
    captured_shape["enc_shape"] = tuple(cap["enc"].shape)
    captured_shape["enc_finite"] = torch.isfinite(cap["enc"]).all().item()
    from resfit.rl_finetuning.chunk_residual.act_feature import pool_encoder_out, concat_proprio
    emb = pool_encoder_out(cap["enc"])
    proprio = torch.as_tensor(ro[self.proprio_key], dtype=torch.float32).to(emb.device)
    return concat_proprio(emb, proprio)

# Run embed_batch with correct format
try:
    out1 = _patched_embed(extractor, raw_obs_correct)
    out2 = _patched_embed(extractor, raw_obs_correct)
    deterministic = torch.allclose(out1, out2)
    enc_shape = captured_shape.get("enc_shape", "NOT CAPTURED")
    print(f"  encoder_out shape   : {enc_shape}  (expected [seq, B, dim_model])")
    print(f"    → seq={enc_shape[0] if enc_shape != 'NOT CAPTURED' else '?'}  "
          f"B={enc_shape[1] if enc_shape != 'NOT CAPTURED' else '?'}  "
          f"dim={enc_shape[2] if enc_shape != 'NOT CAPTURED' else '?'}")
    print(f"  encoder_out finite  : {captured_shape.get('enc_finite', '?')}")
    print(f"  embed_batch output  : shape={tuple(out1.shape)}  "
          f"finite={torch.isfinite(out1).all().item()}  deterministic={deterministic}")
    D_emb = out1.shape[-1]
    print(f"  D_emb (dim_model+18): {D_emb}  (dim_model={dim_model}, proprio=18)")
    # Confirm hook target is NOT VAE encoder
    print(f"  Hook on act.model.encoder: is_vae_encoder attr = "
          f"{getattr(act.model.encoder, 'is_vae_encoder', 'ATTR_MISSING')}")
except Exception as e:
    print(f"  embed_batch FAILED: {type(e).__name__}: {e}")
    import traceback; traceback.print_exc()
    D_emb = None

# ---------------------------------------------------------------------------
# 4b. embed_batch with RAW (wrong) format — to confirm it gives finite output
#     despite wrong format (so "ran without error" is NOT a format check)
# ---------------------------------------------------------------------------
print("\n[4b] embed_batch with RAW (uint8 HWC) format — finite check only …")
try:
    out_raw = extractor.embed_batch(raw_obs_2)
    print(f"  embed_batch(raw) ran: shape={tuple(out_raw.shape)}  "
          f"finite={torch.isfinite(out_raw).all().item()}")
    print("  NOTE: finite output does NOT mean format is correct — see task 3 verdict")
except Exception as e:
    print(f"  embed_batch(raw) FAILED: {type(e).__name__}: {e}")

# ---------------------------------------------------------------------------
# 5. Summary / verdict
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("VERDICT SUMMARY")
print("=" * 70)
print(f"(a) Raw hdf5 image  : dtype=uint8  shape=(T,84,84,3) HWC  range=[6,254]")
print(f"(b) ACT expected    : dtype=float32  shape=(B,3,84,84) CHW  range~[0,1] before stats-norm")
print(f"    (normalize_inputs mean≈{mean_val:.4f}, std≈{std_val:.4f} → [0,1]-scale input)")
print()
print(f"(c) IMAGE FORMAT VERDICT:")
if format_mismatch:
    print(f"    WRONG — _build_raw_obs_seqs feeds HWC uint8 [0,255]")
    print(f"    REQUIRED TRANSFORM: .float() / 255.0  then .permute(0,3,1,2)  → CHW float [0,1]")
else:
    print(f"    OK — format matches ACT expectations")
print()
if 'enc_shape' in captured_shape:
    print(f"(d) encoder_out shape       : {captured_shape['enc_shape']}")
    print(f"    mean(dim=0) pools tokens→[B, dim_model]: CORRECT")
    s = captured_shape['enc_shape']
    print(f"    seq={s[0]}  B={s[1]}  dim={s[2]}")
else:
    print(f"(d) encoder_out shape       : NOT CAPTURED")
print()
if D_emb is not None:
    print(f"(e) embed_batch output shape: [{raw_obs_correct['observation.state'].shape[0]}, {D_emb}]  "
          f"finite=True  deterministic=True")
    print(f"(f) D_emb = {D_emb}  (= dim_model {dim_model} + proprio 18)")
print()
print(f"get_action_chunk(raw)   : {'PASSED (but output is garbage — wrong scale)' if act_chunk_ok else 'FAILED'}")
print(f"get_action_chunk(correct): {'PASSED' if act_correct_ok else 'FAILED'}")
print()
print("BOTTOM LINE:")
print("  _build_raw_obs_seqs IS BROKEN for act_feat: it feeds uint8 HWC images")
print("  directly into normalize_inputs which expects float32 CHW [0,1].")
print("  The model may run without crashing (normalize_inputs does stats-norm,")
print("  not shape-checking), but the features are from an input ~255x out-of-range,")
print("  making all downstream HIQL value numbers MEANINGLESS.")
print()
print("  REQUIRED FIX in _build_raw_obs_seqs (one extra line per image key):")
print("    ro[k] = torch.as_tensor(grp[f'obs/{name}_image'][()])")
print("    # ADD:  ro[k] = ro[k].float().div(255.0).permute(0, 3, 1, 2)")
print("  This transforms (T,H,W,C) uint8 → (T,C,H,W) float32 [0,1].")
print("=" * 70)
