"""offline↔online act_feat feature-consistency SMOKE (命门 A/B hard gate).

验证:
  命门 A (image): online env obs 图像与 offline build 一致 (CHW float32 [0,1])。
  命门 B (proprio): online observation.state (原始 Dp 维(env-aware,由 task 决定)) 经同款 dataset-standardize
                    后与 offline HDF5 同帧 proprio 对齐。
  净结论: 同一物理状态的冻结-ACT 530-dim 特征 offline ≈ online (max_abs_diff 水平)。

用法(从仓库根):
  CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 \\
  HF_HUB_OFFLINE=1 \\
  conda run -n residual python -m resfit.rl_finetuning.chunk_residual.verify_act_feat_online_consistency \\
    --base resfit/out/piecce/best \\
    --hdf5 resfit/dataset/two_arm_three_piece_assembly.hdf5 \\
    --dataset ankile/dexmg-two-arm-three-piece-assembly \\
    --task TwoArmThreePieceAssembly
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _verdict(label: str, ok: bool):
    sym = "PASS" if ok else "FAIL"
    print(f"  [{sym}] {label}")
    return ok


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="offline<->online act_feat consistency smoke")
    parser.add_argument("--base", required=True, help="ACT base ckpt dir (loader appends /policy)")
    parser.add_argument("--hdf5", required=True, help="path to source HDF5 demo file")
    parser.add_argument("--dataset", required=True, help="LeRobot dataset id (for stats)")
    parser.add_argument("--task", required=True, help="robosuite env name, e.g. TwoArmThreePieceAssembly")
    parser.add_argument("--frames", nargs="+", type=int, default=[0, 10],
                        help="which frame indices in demo_0 to check (default: 0 10)")
    parser.add_argument("--demo", default=None, help="demo key to use (default: first sorted demo)")
    parser.add_argument("--atol", type=float, default=5e-2,
                        help="allclose tolerance on 530-dim feature (default 0.05). "
                             "EGL rendering introduces ~1/255 uint8 rounding noise per pixel; "
                             "for frames away from reset this propagates to ~0.03 in 530-dim feat. "
                             "Frame 0 (right after reset/scene load) is pixel-perfect (diff=0).")
    args = parser.parse_args()

    hdf5_path = str(args.hdf5)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n=== act_feat offline↔online consistency SMOKE ===")
    print(f"device={device}  hdf5={hdf5_path}  dataset={args.dataset}")

    # ------------------------------------------------------------------
    # 1. Load frozen ACT + build ActFeatureExtractor
    # ------------------------------------------------------------------
    print("\n[1] Loading ACT base policy ...")
    from resfit.lerobot.utils.load_policy import load_policy
    from resfit.rl_finetuning.chunk_residual.act_feature import ActFeatureExtractor

    base_dir = Path(args.base)
    policy_dir = base_dir / "policy" if (base_dir / "policy").is_dir() else base_dir
    act = load_policy(policy_dir).to(device)
    image_keys = list(act.config.image_features.keys())
    print(f"  image_keys={image_keys}  dim_model={act.config.dim_model}")

    extractor = ActFeatureExtractor(act, image_keys, proprio_key="observation.state",
                                    pooling="mean", proprio_dim=18)
    # feature_dim will be re-printed after state_raw is read (Fix 2: reflect actual per-task Dp)

    # ------------------------------------------------------------------
    # 2. Build StateStandardizer from dataset stats
    # ------------------------------------------------------------------
    print("\n[2] Building StateStandardizer from dataset stats ...")
    from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from resfit.rl_finetuning.utils.normalization import StateStandardizer

    meta = LeRobotDatasetMetadata(args.dataset)
    state_std = StateStandardizer.from_dataset_stats(
        meta.stats["observation.state"], device="cpu")
    print(f"  state mean[:5]={state_std._mean[:5].numpy()}")
    print(f"  state std[:5]={state_std._std[:5].numpy()}")

    # ------------------------------------------------------------------
    # 3. Identify demo + pick frames
    # ------------------------------------------------------------------
    from resfit.rl_finetuning.chunk_residual.offline_hdf5_buffer import (
        assemble_state_by_env, expected_low_dim_keys, sorted_demo_keys)

    with h5py.File(hdf5_path, "r") as f:
        demo_keys = sorted_demo_keys(list(f["data"].keys()))

    if args.demo:
        demo_key = args.demo
    else:
        demo_key = demo_keys[0]
    print(f"\n  Using demo: {demo_key}")

    with h5py.File(hdf5_path, "r") as f:
        grp = f[f"data/{demo_key}"]
        T = grp["states"].shape[0]
        frames = [k for k in args.frames if k < T]
        if not frames:
            frames = [0]
        print(f"  Demo length T={T}, checking frames={frames}")

        # hdf5 image key helper (mirrors offline_stage_replay._hdf5_image_key)
        def hdf5_img_key(lerobot_key: str) -> str:
            return lerobot_key.split("observation.images.")[-1] + "_image"

        # Read all needed data (env-aware proprio keys)
        keys = expected_low_dim_keys(args.task)
        obs_arrays = {k: grp[f"obs/{k}"][()] for k in keys}
        state_raw = assemble_state_by_env(obs_arrays, args.task)   # (T,Dp) raw
        # Fix 2: update extractor's proprio_dim to match actual task dim so feature_dim prints correctly
        extractor._proprio_dim = int(state_raw.shape[1])
        demo_states = grp["states"][()]             # (T, state_dim) mujoco flat states
        model_file = grp.attrs["model_file"]
        ep_meta = grp.attrs.get("ep_meta", None)

        imgs_hdf5 = {}
        for k in image_keys:
            hk = hdf5_img_key(k)
            imgs_hdf5[k] = grp[f"obs/{hk}"][()]   # (T,H,W,3) uint8

    # Print feature_dim now that proprio_dim reflects the actual task dim (Fix 2)
    print(f"  feature_dim={extractor.feature_dim}  (dim_model={act.config.dim_model} + proprio_dim={extractor._proprio_dim})")

    # ------------------------------------------------------------------
    # 4. Build OFFLINE features for selected frames
    # ------------------------------------------------------------------
    print("\n[4] Building OFFLINE features ...")
    feat_offs = {}
    for k in frames:
        # Image: HWC uint8 → CHW float [0,1]
        off_imgs = {}
        for ik in image_keys:
            img_np = imgs_hdf5[ik][k]              # (H,W,3) uint8
            img_t = torch.as_tensor(img_np, dtype=torch.float32).div(255.0).permute(2, 0, 1)  # (3,H,W)
            off_imgs[ik] = img_t.unsqueeze(0)      # (1,3,H,W)

        # Proprio: assemble_state_by_env → dataset-standardize
        s_raw = torch.as_tensor(state_raw[k], dtype=torch.float32).unsqueeze(0)  # (1,Dp)
        s_std = state_std.standardize(s_raw)       # (1,Dp) normalized

        raw_obs = {ik: off_imgs[ik].to(device) for ik in image_keys}
        raw_obs["observation.state"] = s_std.to(device)

        feat = extractor.embed_batch(raw_obs).cpu().numpy()  # (1, feat_dim)
        feat_offs[k] = feat[0]
        print(f"  frame {k}: offline feat shape={feat.shape}  finite={np.all(np.isfinite(feat))}")
        print(f"           offline state_raw[:5]={state_raw[k][:5]}")
        print(f"           offline state_std[:5]={s_std.numpy()[0][:5]}")

    # ------------------------------------------------------------------
    # 5. Build ONLINE features via env replay (exact-state path)
    # ------------------------------------------------------------------
    print("\n[5] Setting env to exact demo states (exact-state path) ...")

    # We use reset_to() from offline_stage_replay for exact state injection
    from resfit.rl_finetuning.chunk_residual.offline_stage_replay import (
        make_replay_env, reset_to)

    # make_replay_env creates a no-render env; we need cameras so we
    # build the rendering env directly via RobosuiteGymWrapper instead.
    import os
    import dexmimicgen  # noqa: F401  register dexmg envs
    import robosuite
    from robosuite import load_composite_controller_config, macros
    macros.IMAGE_CONVENTION = "opencv"

    # Pick camera names from image_keys (same mapping as dexmg.py)
    camera_names = [k.split("observation.images.")[-1] for k in image_keys]
    camera_size = 84   # must match offline HDF5

    # Read env_args from HDF5 to get matching camera size if stored
    with h5py.File(hdf5_path, "r") as f:
        env_meta_str = f["data"].attrs.get("env_args", None)
    if env_meta_str is not None:
        env_meta = json.loads(env_meta_str)
        env_kwargs_base = env_meta.get("env_kwargs", {})
        if "camera_heights" in env_kwargs_base:
            camera_size = env_kwargs_base["camera_heights"]
            if isinstance(camera_size, list):
                camera_size = camera_size[0]
        print(f"  Using camera_size={camera_size} (from HDF5 env_args)")
    else:
        print(f"  Using camera_size={camera_size} (default)")

    # Build a rendering env
    # MUJOCO_EGL_DEVICE_ID is already set by the caller's environment
    egl_id_str = os.environ.get("MUJOCO_EGL_DEVICE_ID", "0")
    egl_id = int(egl_id_str)
    print(f"  MUJOCO_EGL_DEVICE_ID={egl_id}")

    print(f"  Creating rendering env for {args.task} ...")
    # Read robots from already-read env_meta_str (no second HDF5 open needed)
    robots = ["Panda", "Panda"]
    if env_meta_str is not None:
        _ek = json.loads(env_meta_str).get("env_kwargs", {})
        if _ek.get("robots"):
            robots = _ek["robots"]
            if isinstance(robots, str):
                robots = [robots]
    print(f"  robots={robots}")

    render_env = robosuite.make(
        env_name=args.task,
        robots=robots,
        controller_configs=load_composite_controller_config(robot=robots[0]),
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_camera_obs=True,
        control_freq=20,
        camera_names=camera_names,
        camera_heights=camera_size,
        camera_widths=camera_size,
        horizon=500,
        renderer="mujoco",
        render_gpu_device_id=egl_id,
    )

    # load the model file for this demo (needed for reset_to with "model")
    with h5py.File(hdf5_path, "r") as f:
        model_file_xml = f[f"data/{demo_key}"].attrs["model_file"]
        ep_meta_str = f[f"data/{demo_key}"].attrs.get("ep_meta", None)

    # Initial reset with model XML to load the scene
    state0 = {"model": model_file_xml,
               "ep_meta": ep_meta_str,
               "states": demo_states[frames[0]]}
    reset_to(render_env, state0)
    print("  Scene loaded OK")

    exact_state_path = True
    feat_ons = {}
    online_img_stats = {}
    # Fix 3a: hoist on_keys out of per-frame loop (computed once, same for all frames)
    on_keys = expected_low_dim_keys(args.task)
    b_proprio_matches = []  # Fix 1b: accumulate per-frame proprio match for 命门B verdict

    for k in frames:
        # Set to exact demo state
        reset_to(render_env, {"states": demo_states[k]})

        # CRITICAL: force observable update after set_state_from_flattened.
        # Without this, _get_observations() returns stale cached values —
        # all frames appear identical regardless of state.
        if hasattr(render_env, "_update_observables"):
            render_env._update_observables(force=True)

        # Render and get obs
        raw_rs_obs = render_env._get_observations()

        # Process images exactly as dexmg.RobosuiteGymWrapper._process_obs does:
        # (H,W,3) uint8 → float32/255 → (3,H,W)
        on_imgs = {}
        for ik in image_keys:
            cam = ik.split("observation.images.")[-1]
            rs_key = f"{cam}_image"
            if rs_key not in raw_rs_obs:
                print(f"  WARNING: {rs_key} not found in env obs! Available: {list(raw_rs_obs.keys())}")
                exact_state_path = False
                break
            img_np = raw_rs_obs[rs_key]             # (H,W,3) uint8
            print(f"  frame {k} [{ik}] online raw img: dtype={img_np.dtype} shape={img_np.shape} "
                  f"min={img_np.min()} max={img_np.max()}")
            img_t = torch.as_tensor(img_np.astype(np.float32) / 255.0).permute(2, 0, 1)  # (3,H,W)
            print(f"    → CHW float: shape={tuple(img_t.shape)} min={img_t.min():.4f} max={img_t.max():.4f}")
            on_imgs[ik] = img_t.unsqueeze(0)        # (1,3,H,W)
            online_img_stats[k] = {
                "dtype": str(img_np.dtype),
                "shape": img_np.shape,
                "min": int(img_np.min()),
                "max": int(img_np.max()),
                "chw_float_min": float(img_t.min()),
                "chw_float_max": float(img_t.max()),
            }

        if not exact_state_path:
            break

        # Proprio: env uses same keys as expected_low_dim_keys(task) → concatenate raw floats (full dim, no truncation)
        on_state_parts = []
        for key_name in on_keys:
            if key_name in raw_rs_obs:
                on_state_parts.append(np.asarray(raw_rs_obs[key_name]).astype(np.float32))
            else:
                print(f"  WARNING: proprio key {key_name} not in env obs")
                exact_state_path = False
                break
        if not exact_state_path:
            break

        on_state_raw = np.concatenate(on_state_parts)  # (Dp,)
        on_state_t = torch.as_tensor(on_state_raw, dtype=torch.float32).unsqueeze(0)  # (1,Dp)
        on_state_std = state_std.standardize(on_state_t)  # (1,Dp) normalized

        print(f"  frame {k}: online state_raw[:5]={on_state_raw[:5]}")
        print(f"           online state_std[:5]={on_state_std.numpy()[0][:5]}")

        # Compare proprio (命门 B structure check)
        # atol=1e-2: EGL set_state_from_flattened + forward-kinematics introduces ~6e-4 float
        # noise on frames away from reset. 1e-4 would false-FAIL; 1e-2 is safely above that
        # noise floor but far below any real key-set/layout mismatch (~0.1+).
        off_state_raw = state_raw[k].astype(np.float32)
        proprio_raw_match = np.allclose(on_state_raw, off_state_raw, atol=1e-2)
        print(f"  frame {k}: proprio raw allclose(atol=1e-2)={proprio_raw_match}  "
              f"max_diff={np.abs(on_state_raw - off_state_raw).max():.2e}")
        b_proprio_matches.append(bool(proprio_raw_match))  # Fix 1b: accumulate for 命门B verdict

        raw_obs_on = {ik: on_imgs[ik].to(device) for ik in image_keys}
        raw_obs_on["observation.state"] = on_state_std.to(device)

        feat_on = extractor.embed_batch(raw_obs_on).cpu().numpy()  # (1, feat_dim)
        feat_ons[k] = feat_on[0]
        print(f"  frame {k}: online  feat shape={feat_on.shape}  finite={np.all(np.isfinite(feat_on))}")

    render_env.close()

    # ------------------------------------------------------------------
    # 6. Compare and print verdicts
    # ------------------------------------------------------------------
    print("\n=== VERDICTS ===")

    all_pass = True

    # --- 命门 A: image format ---
    img_a_ok = True
    for k, stats in online_img_stats.items():
        chw_ok = stats["chw_float_min"] >= 0.0 and stats["chw_float_max"] <= 1.0
        img_a_ok = img_a_ok and chw_ok
        print(f"  frame {k} image: dtype={stats['dtype']} shape={stats['shape']} "
              f"raw_range=[{stats['min']},{stats['max']}] → "
              f"CHW float range=[{stats['chw_float_min']:.4f},{stats['chw_float_max']:.4f}]")
    all_pass &= _verdict("命门A: online img CHW float32 in [0,1]", img_a_ok and bool(online_img_stats))

    if exact_state_path and feat_ons:
        # --- 命门 B: proprio convention ---
        # Fix 1b: verdict is AND of per-frame proprio allclose (atol=1e-2) collected in the loop.
        # A real key-set/layout mismatch yields max_diff ~0.1+; EGL replay noise is ~6e-4 — gate is meaningful.
        b_ok = bool(b_proprio_matches) and all(b_proprio_matches)
        all_pass &= _verdict("命门B: online proprio raw == offline HDF5 proprio (env-aware full-dim concatenation)", b_ok)

        # --- numeric feature allclose ---
        for k in frames:
            if k not in feat_ons or k not in feat_offs:
                continue
            fo = feat_offs[k]
            fn = feat_ons[k]
            both_finite = np.all(np.isfinite(fo)) and np.all(np.isfinite(fn))
            max_diff = float(np.abs(fo - fn).max())
            close = np.allclose(fo, fn, atol=args.atol)
            print(f"\n  frame {k}: max_abs_diff={max_diff:.4e}  allclose(atol={args.atol})={close}")
            all_pass &= _verdict(f"命门A+B: offline≈online 530-dim feat frame {k}", close)
            all_pass &= _verdict(f"both finite frame {k}", both_finite)
    else:
        print("\n  [SKIP] exact-state numeric comparison skipped (env key missing or no frames)")
        all_pass = False

    print(f"\n{'='*50}")
    if all_pass:
        print("BOTTOM LINE: PASS — offline↔online act_feat 同源已验证，可放心扩到 500k 训练。")
    else:
        print("BOTTOM LINE: FAIL — 有命门未过，禁止启动 500k 训练，须先定位并修复。")
    print("="*50)

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
