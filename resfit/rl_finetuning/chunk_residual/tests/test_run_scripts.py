from pathlib import Path


def test_lerobot_hiql_aligned_script_has_coffee_case_and_v_flags():
    script = Path("run_act_feat_lerobot_hiql_aligned_bp_bc01.sh")
    text = script.read_text()
    assert "coffee)" in text
    assert "TASK=TwoArmCoffee" in text
    assert "SUB=dexmg-two-arm-coffee" in text
    assert "ROOT=/mnt/mnt/data/resfit/resfit/dataset/ankile/$SUB" in text
    assert "COFFEE_ACT_BASE" in text
    assert "/mnt/mnt/data/resfit/resfit/out/coffee/best" in text
    assert '--data_source lerobot --lerobot_root "$ROOT"' in text
    for flag in (
        "--rep_dim 10",
        "--value_rep_mode concat",
        "--value_hidden 512",
        "--value_layers 3",
        "--use_layer_norm 1",
        "--value_loss_mode hiql",
        "--value_mask_mode hiql",
        "--goal_future_mode geometric",
    ):
        assert flag in text
    assert '${KEY}_actfeat_lerobot_hiql_aligned_bp_bc01' in text


def test_lifttray_staged_joint_script_pothiql_to_staged():
    text = Path("run_lifttray_staged_joint.sh").read_text()
    # staged 特征齐全
    assert "--reward_shaping staged" in text
    assert "--stage_reward_bonus 1.0" in text
    assert "--stage_balanced" in text
    assert "--offline_stage_cache outputs_chunk/two_arm_lift_tray_stages.npz" in text
    # pothiql 残留必须清除
    assert "--reward_shaping potential" not in text
    assert "--potential_source" not in text
    assert "--hiql_value_ckpt" not in text
    # 对齐参照实验的关键不变量
    for flag in ("--task TwoArmLiftTray", "--action_scale 0.05", "--actor_lr 1e-6",
                 "--offline_fraction 0.5", "--offline_base_mode base_policy",
                 "--demo_bc_coef 0.1", "--subgoal_conditioned", "--subgoal_way_steps 15",
                 "--online_finetune_value", "--online_finetune_high_actor",
                 "--total_env_steps 500000",
                 "--base_action_mode queue", "--base_n_action_steps 10"):
        assert flag in text
    # 新 offcache / 输出名(staged 签名,勿复用 pothiql)
    assert "lifttray_actfeat_hdf5_bp_hiqlv512_sg15_staged_offcache" in text
    assert "lifttray_actfeat_hdf5_bp_bc01_hiqlv512_sg15_staged_joint" in text


def test_pouring_staged_joint_script_aligned_with_rerun0703():
    text = Path("run_pouring_staged_joint.sh").read_text()
    # —— staged 差异 ——
    assert "--reward_shaping staged" in text
    assert "--stage_reward_bonus 1.0" in text
    assert "--stage_balanced" in text
    assert "two_arm_pouring_stages.npz" in text
    assert "--potential_source" not in text        # 删掉(staged 无势函数)
    assert "--hiql_value_ckpt" not in text          # staged 不需 value.pt
    assert "_staged_joint_offcache" in text         # 新 offcache(重建)
    assert "pouring_actfeat_hdf5_bp_bc01_hiqlv512_sg15_as005_staged_joint" in text
    # —— 对齐参照的关键不变量 ——
    for flag in (
        "--task TwoArmPouring",
        "--dataset ankile/dexmg-two-arm-pouring",
        "--offline_dataset_path resfit/dataset/two_arm_pouring.hdf5",
        "--chunk_length 1 --base_action_mode queue --base_n_action_steps 10",
        "--action_scale 0.05 --actor_lr 1e-6 --actor raw",
        "--offline_fraction 0.5 --offline_base_mode base_policy --demo_bc_coef 0.1",
        "--subgoal_conditioned",
        "--gc_value_ckpt outputs_chunk/pouring_gc_value_actfeat_hdf5_hiqlv512.pt",
        "--high_actor_ckpt outputs_chunk/pouring_high_actor_actfeat_hdf5_hiqlv512.pt",
        "--act_feat_cache outputs_chunk/pouring_act_feat_hdf5.npz",
        "--subgoal_way_steps 15 --online_finetune_value --online_finetune_high_actor",
        "--total_env_steps 500000",
        "run_anw5pphu_best:v2",
    ):
        assert flag in text, flag
