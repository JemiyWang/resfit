"""真 base policy 在真 demo 上跑 _demo_base_actions,验 obs 格式 + base_action 合理。

跑法(本机,controller 跑):
  cd /mnt/mnt/data/resfit && CUDA_VISIBLE_DEVICES=2 MUJOCO_GL=egl WANDB_MODE=offline HF_HUB_OFFLINE=1 \
  conda run -n residual --no-capture-output python -m \
  resfit.rl_finetuning.chunk_residual.verify_base_as_base
"""
import h5py
import torch

from resfit.rl_finetuning.chunk_residual.train_chunk_residual import build_parser, build_base_policy
from resfit.rl_finetuning.chunk_residual import offline_stage_replay as osr
from resfit.rl_finetuning.chunk_residual.offline_hdf5_buffer import sorted_demo_keys

HDF5 = "resfit/dataset/two_arm_three_piece_assembly.hdf5"


def main():
    args = build_parser().parse_args([
        "--task", "TwoArmThreePieceAssembly",
        "--base_wandb_id", "/mnt/mnt/data/resfit/resfit/out/piecce/best",
        "--dataset", "ankile/dexmg-two-arm-three-piece-assembly",
        "--base_action_mode", "queue", "--chunk_length", "1", "--base_n_action_steps", "10",
        "--action_scale", "0.05",
        "--offline_dataset_path", HDF5, "--device", "cuda",
    ])
    base_policy = build_base_policy(args, device="cuda")
    base_policy.config.n_action_steps = 10
    image_keys = list(base_policy.config.image_features.keys())
    # 与训练同款 action_scaler(train_chunk_residual.py:363-365)
    from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from resfit.rl_finetuning.utils.normalization import ActionScaler
    meta = LeRobotDatasetMetadata(args.dataset)
    action_scaler = ActionScaler.from_dataset_stats(
        meta.stats["action"], action_scale=args.action_scale,
        min_range_per_dim=args.min_range_per_dim, device="cuda")

    with h5py.File(HDF5, "r") as f:
        demos = sorted_demo_keys(list(f["data"].keys()))[:3]
        for ep in demos:
            grp = f[f"data/{ep}"]
            base_n = osr._demo_base_actions(base_policy, grp, image_keys, action_scaler, "cuda")
            gt = action_scaler.scale(torch.as_tensor(grp["actions"][()], dtype=torch.float32))
            bc = (gt - base_n)                          # 隐含残差目标 = GT - base
            assert torch.isfinite(base_n).all(), f"{ep}: base_n 非有限"
            assert base_n.shape == gt.shape, f"{ep}: shape {base_n.shape} vs GT {gt.shape}"
            print(f"{ep} T={len(base_n)} base|mean|={base_n.abs().mean():.4f} "
                  f"bc_target|mean|={bc.abs().mean():.4f} bc_target|max|={bc.abs().max():.4f}")
            assert bc.abs().mean() > 1e-4, f"{ep}: bc_target ~0(base≈GT?格式可能错或 base 完美)"
    print("VERIFY OK: 真 obs 格式吃得下,base_action 有限、bc_target=GT-base 非零有界")


if __name__ == "__main__":
    main()
