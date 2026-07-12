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


def test_pouring_pothiql_joint_as01_single_var_change_from_rerun0703():
    """as01 变体:相对 rerun0703 唯一变量 = action_scale 0.05→0.1,offcache 必重建、run 名换 as01。"""
    text = Path("run_pouring_pothiql_joint_as01.sh").read_text()
    # —— 唯一变量:action_scale 0.1 ——
    assert "--action_scale 0.1 --actor_lr 1e-6 --actor raw" in text
    assert "--action_scale 0.05 --actor_lr" not in text   # 命令行 0.05 残留必须清除(注释提及不算)
    # —— 新 offcache(as01 签名,勿复用/勿 gate as005 的 buffer_meta)——
    assert "pouring_actfeat_hdf5_bp_hiqlv512_sg15_as01_pothiql_joint_offcache" in text
    assert "as005_pothiql_joint_offcache" not in text
    assert 'gate "$OFFCACHE/buffer_meta.json"' not in text   # 首建,不 gate 完成哨兵
    # —— 新 run 名/输出目录(as01)——
    assert "pouring_actfeat_hdf5_bp_bc01_hiqlv512_sg15_as01_pothiql_joint_rerun0703" in text
    assert "as005_pothiql_joint_rerun0703" not in text
    # —— 对齐 rerun0703 的关键不变量(除 action_scale 外逐字一致)——
    for flag in (
        "--task TwoArmPouring",
        "--dataset ankile/dexmg-two-arm-pouring",
        "--offline_dataset_path resfit/dataset/two_arm_pouring.hdf5",
        "--chunk_length 1 --base_action_mode queue --base_n_action_steps 10",
        "--reward_shaping potential --potential_source hiql",
        "--hiql_value_ckpt outputs_chunk/pouring_value_hdf5.pt",
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


def test_lifttray_pothiql_joint_as01_single_var_change_from_rerun0629():
    """lifttray as01 变体:相对 rerun0629 唯一变量 = action_scale 0.05→0.1,offcache 必重建、名插入 as01。"""
    text = Path("run_lifttray_pothiql_joint_as01.sh").read_text()
    # —— 唯一变量:action_scale 0.1(每参一行格式)——
    assert "--action_scale \\\n  0.1 \\" in text
    assert "\n  0.05 \\" not in text                 # 0.05 残留必须清除(原脚本仅 action_scale 用 0.05)
    # —— 新 offcache(as01 签名,勿复用旧 as005)——
    assert "lifttray_actfeat_hdf5_bp_hiqlv512_sg15_as01_pothiql_joint_offcache" in text
    assert "hiqlv512_sg15_pothiql_joint_offcache" not in text
    # —— 新 run 名/输出目录(插入 as01)——
    assert "lifttray_actfeat_hdf5_bp_bcfixed01_hiqlv512_sg15_as01_pothiql_joint_rerun0629" in text
    assert "hiqlv512_sg15_pothiql_joint_rerun0629" not in text
    # —— 对齐 rerun0629 的关键不变量 ——
    for tok in (
        "TwoArmLiftTray",
        "ankile/dexmg-two-arm-lift-tray",
        "resfit/dataset/two_arm_lift_tray.hdf5",
        "run_e0o0sckj_best:v4",
        "outputs_chunk/lifttray_value_hdf5.pt",
        "outputs_chunk/lifttray_gc_value_actfeat_hdf5_hiqlv512.pt",
        "outputs_chunk/lifttray_high_actor_actfeat_hdf5_hiqlv512.pt",
        "outputs_chunk/lifttray_act_feat_hdf5.npz",
        "--subgoal_conditioned",
        "--online_finetune_value",
        "--online_finetune_high_actor",
        "--reward_shaping",
        "--potential_source",
    ):
        assert tok in text, tok


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
