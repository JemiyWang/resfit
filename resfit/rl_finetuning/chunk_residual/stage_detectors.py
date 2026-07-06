"""dexmg 任务的特权 stage 检测器(训练时用，策略不可见)。

约定:返回 int ∈ [0, NUM_STAGES-1]，0=起步。检测器直接复用 env 自带的
分阶段谓词(dexmimicgen 写 _check_success 时已拆出)，故极薄。
单调闩锁(max-so-far)由调用方(env wrapper)负责，这里只判瞬时阶段。

⚠ unwrap_base_env 仅在 env 与主进程同进程(SyncVectorEnv / debug=True)时有效;
默认训练路径用 AsyncVectorEnv(spawn)，robosuite env 在子进程，必须改走
vec_env.call(...) 或在 worker 内 wrapper 里出 info。见 verify_unwrap_base_env.py。
"""
from __future__ import annotations


def _grasped(env, obj) -> bool:
    """piece 是否被任一夹爪握住。

    注意:robosuite 双臂 env 的 robot.gripper 是 dict({'right':obj,'left':obj}),
    直接 `for g in robot.gripper` 会迭代出**键(str)**,传给 _check_grasp 恒 False
    (env 自带 _check_first_piece_is_assembled 也踩了这个坑)。这里对 dict 取 values()。
    """
    try:
        for robot in env.robots:
            grippers = robot.gripper
            grippers = grippers.values() if hasattr(grippers, "values") else grippers
            for gripper in grippers:
                if env._check_grasp(gripper=gripper,
                                    object_geoms=[g for g in obj.contact_geoms]):
                    return True
    except Exception:
        return False
    return False


def threepiece_stage(env) -> int:
    """TwoArmThreePieceAssembly 4 段:
    0 起步 / 1 抓起 piece1 和 piece2(两件都被握)/ 2 放好 piece1 / 3 放好 piece2(成功)。

    用户口径:抓取里程碑要求 piece1 和 piece2 **都**被握住才算 1(与 threading
    "针+脚架都抓起"同套路);装配里程碑分两段(piece1 装好即 2,不要求先释放;
    piece2 装好即成功 3)。高阶段短路优先,闩锁(max-so-far)由 wrapper 负责,
    这里只判瞬时阶段。
    """
    if env._check_second_piece_is_assembled():    # == _check_success
        return 3
    if env._check_first_piece_is_assembled():
        return 2
    if _grasped(env, env.piece_1) and _grasped(env, env.piece_2):
        return 1
    return 0


def threading_stage(env) -> int:
    """TwoArmThreading 3 段:0 起步 / 1 脚架与针都被抓起 / 2 成功(针穿入环)。

    threading 无 env 内置子阶段谓词(只有 _check_success),故中间段用双物体 grasp 里程碑。
    闩锁(max-so-far)由 wrapper 负责,这里只判瞬时阶段。
    """
    if env._check_success():
        return 2
    if _grasped(env, env.needle) and _grasped(env, env.tripod):
        return 1
    return 0


def lifttray_stage(env) -> int:
    """TwoArmLiftTray 4 段:0 起步 / 1 一个方块搬上盘 / 2 两个方块都上盘 / 3 抬盘成功。

    里程碑用"方块与盘底 pot_base 接触"(持久态,复用 _check_success 同款谓词),按已上盘
    方块数计数(顺序无关);高段短路优先,闩锁(max-so-far)由 wrapper 负责,这里只判瞬时。
    注:不用"离地/抓起"——那是瞬时信号,两次搬运不重叠,经 wrapper max 闩锁会被压成同一段。
    """
    if env._check_success():                              # 3 抬盘成功
        return 3
    n = int(env.check_contact("pot_base", env.obj0)) + \
        int(env.check_contact("pot_base", env.obj1))
    if n >= 2:                                            # 两块都在盘上
        return 2
    if n >= 1:                                            # 一块在盘上
        return 1
    return 0


STAGE_DETECTORS = {
    "TwoArmThreePieceAssembly": threepiece_stage,
    "TwoArmThreading": threading_stage,
    "TwoArmLiftTray": lifttray_stage,
}
NUM_STAGES = {
    "TwoArmThreePieceAssembly": 4,
    "TwoArmThreading": 3,
    "TwoArmLiftTray": 4,
}


def get_stage_detector(task: str):
    """无检测器的任务返回 None(stage_id 恒 0，退化成无 stage)。"""
    return STAGE_DETECTORS.get(task)

# 注:不要在主进程 unwrap robosuite env——默认 AsyncVectorEnv(spawn) 下它在子进程，
# 且 RobosuiteGymWrapper.get_wrapper_attr 拒绝转发。stage 在 worker 的
# RobosuiteGymWrapper.step 内算好、经 info["stage_id"] 透出(见 dexmg.py)。
