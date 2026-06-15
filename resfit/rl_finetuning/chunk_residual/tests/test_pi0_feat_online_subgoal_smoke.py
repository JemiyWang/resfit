"""opt-in 在线子目标 smoke(默认 skip):验证 pi0_feat 完整分层在线出 z 维度/有限。

前置(手动):用 pi0_feat 全量缓存训出 gc_value.pt + high_actor.pt;serve 起着(若跑真 rollout)。
本 smoke 只验"HiqlSubgoal.from_ckpts(pi0_feat) → subgoal_online(prefix_feat) → z 有限",
不起真 env/serve(那一步由用户手动起残差 rollout 验几步不塌)。

跑(opt-in):
  PI0_FEAT_HIER_SMOKE=1 PI0_FEAT_CACHE=<full.npz> PI0_GC_PT=<gc.pt> PI0_HA_PT=<ha.pt> \
    conda run -n residual python -m pytest <this file> -v -s
"""
import os

import numpy as np
import pytest

RUN = os.environ.get("PI0_FEAT_HIER_SMOKE") == "1"
pytestmark = pytest.mark.skipif(
    not RUN, reason="opt-in:需 pi0_feat 全量缓存 + 预训 gc/ha,设 PI0_FEAT_HIER_SMOKE=1")


def test_online_subgoal_z_injected_and_finite():
    """from_ckpts(pi0_feat) → subgoal_online(prefix_feat) → z 维=rep_dim、有限。
    并打印在线 proprio 维(命门②:须与离线 demo state 同维)。"""
    from resfit.rl_finetuning.chunk_residual.hiql_subgoal import HiqlSubgoal, representative_goal
    from resfit.rl_finetuning.chunk_residual.pi0_feat_cache import load_pi0_feat_cache

    seqs, _stats, _sig = load_pi0_feat_cache(os.environ["PI0_FEAT_CACHE"])
    goal = representative_goal(seqs)
    sg = HiqlSubgoal.from_ckpts(os.environ["PI0_GC_PT"], os.environ["PI0_HA_PT"],
                                goal=goal, device="cpu")
    assert sg.state_mode == "pi0_feat"

    proprio_dim = seqs[0].shape[1] - 2048           # 2056 - 2048(prefix) = 8(LIBERO proprio)
    obs = {"observation.state": np.zeros(proprio_dim, np.float32)}
    z = sg.subgoal_online(obs, prefix_feat=np.ones(2048, np.float32))

    assert z.shape[-1] == sg.rep_dim, f"z 维 {z.shape[-1]} != rep_dim {sg.rep_dim}"
    assert np.isfinite(z.detach().cpu().numpy()).all(), "z 含 nan/inf"
    print(f"[smoke] z dim={z.shape[-1]} rep_dim={sg.rep_dim} proprio_dim={proprio_dim} "
          f"(命门②:在线 obs.state 须 == 离线 demo state 这个维度)")
