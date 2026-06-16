"""run_libero_evaluation 单测:env-无关 rollout,用 stub vec env(torch tensor reward/term/trunc)
+ stub agent。不 import libero/robosuite/QAgent,任何装了 torch 的环境可跑。"""
import pytest
import torch

from resfit.rl_finetuning.chunk_residual.libero_eval import run_libero_evaluation


class _ScriptedVecEnv:
    """num_envs 个独立 env;每个 episode 跑 ep_len 步后 done(SAME_STEP 自动重置)。
    末步 reward 从 terminal_rewards 队列按 done 发生顺序消费;reward==1.0→terminated,否则 truncated。
    非末步 reward=step_reward。obs 只需契约键,值无所谓。"""

    def __init__(self, *, num_envs, ep_len, terminal_rewards, step_reward=0.0):
        self.num_envs = num_envs
        self.ep_len = ep_len
        self.step_reward = step_reward
        self._terminal = list(terminal_rewards)
        self._t = [0] * num_envs

    def _obs(self):
        return {"observation.state": torch.zeros(self.num_envs, 8),
                "observation.images.agentview": torch.zeros(self.num_envs, 3, 84, 84)}

    def reset(self, **kwargs):
        self._t = [0] * self.num_envs
        return self._obs(), {}

    def step(self, action):
        rewards = torch.zeros(self.num_envs)
        term = torch.zeros(self.num_envs, dtype=torch.bool)
        trunc = torch.zeros(self.num_envs, dtype=torch.bool)
        for i in range(self.num_envs):
            self._t[i] += 1
            if self._t[i] >= self.ep_len:
                r = self._terminal.pop(0) if self._terminal else 0.0
                rewards[i] = r
                if r == 1.0:
                    term[i] = True
                else:
                    trunc[i] = True
                self._t[i] = 0  # autoreset(SAME_STEP)
            else:
                rewards[i] = self.step_reward
        return self._obs(), rewards, term, trunc, {}


class _StubAgent:
    def __init__(self):
        self.mode = "train"
        self.last_eval_mode = None

    def eval(self):
        self.mode = "eval"

    def train(self, training=True):
        self.mode = "train" if training else "eval"

    def act(self, obs, *, eval_mode=False, stddev=0.0, cpu=True):
        self.last_eval_mode = eval_mode
        b = obs["observation.state"].shape[0]
        return torch.zeros(b, 7)


def test_success_rate_and_returns_single_env_multistep():
    # ep_len=3,step_reward=0.1;末步 reward 序列 [1,0,1,0] → returns [1.2,0.2,1.2,0.2]
    env = _ScriptedVecEnv(num_envs=1, ep_len=3, step_reward=0.1,
                          terminal_rewards=[1.0, 0.0, 1.0, 0.0])
    m = run_libero_evaluation(env=env, agent=_StubAgent(), num_episodes=4, device="cpu")
    assert abs(m["eval/success_rate"] - 0.5) < 1e-6
    assert abs(m["eval/mean_return"] - 0.7) < 1e-6          # (1.2+0.2+1.2+0.2)/4
    assert abs(m["eval/mean_successful_episode_length"] - 3.0) < 1e-6


def test_uses_eval_mode_and_restores_train_mode():
    agent = _StubAgent()
    env = _ScriptedVecEnv(num_envs=1, ep_len=1, terminal_rewards=[1.0, 1.0])
    run_libero_evaluation(env=env, agent=agent, num_episodes=2, device="cpu")
    assert agent.last_eval_mode is True   # rollout 用 eval_mode=True(确定性)
    assert agent.mode == "train"          # 结束 restore 训练模式


def test_stops_exactly_at_num_episodes_multi_env():
    # ep_len=1 → 每步全部 env done;消费顺序 [1,1,0, 1,...],num_episodes=4 → 取前 4 个
    env = _ScriptedVecEnv(num_envs=3, ep_len=1,
                          terminal_rewards=[1.0, 1.0, 0.0, 1.0, 1.0, 1.0])
    m = run_libero_evaluation(env=env, agent=_StubAgent(), num_episodes=4, device="cpu")
    assert abs(m["eval/success_rate"] - 0.75) < 1e-6   # [T,T,F,T];多计了就不会是 0.75


class _SeqSingleEnv:
    """单 env:按 episodes=[(length, terminal_reward), ...] 顺序播放,每跑到 length 步 done
    (reward==1.0→terminated 否则 truncated),autoreset 进入下一项。用于区分成功/失败 episode 的长度。"""
    num_envs = 1

    def __init__(self, episodes):
        self._eps = list(episodes)
        self._i = 0
        self._t = 0

    def _obs(self):
        return {"observation.state": torch.zeros(1, 8)}

    def reset(self, **kwargs):
        self._i = 0
        self._t = 0
        return self._obs(), {}

    def step(self, action):
        length, term_r = self._eps[self._i]
        self._t += 1
        rewards = torch.zeros(1)
        term = torch.zeros(1, dtype=torch.bool)
        trunc = torch.zeros(1, dtype=torch.bool)
        if self._t >= length:
            rewards[0] = term_r
            (term if term_r == 1.0 else trunc)[0] = True
            self._t = 0
            self._i = min(self._i + 1, len(self._eps) - 1)   # autoreset 到下一 episode
        return self._obs(), rewards, term, trunc, {}


def test_mean_successful_length_excludes_failed_episodes():
    # 成功 episode 长 2、失败 episode 长 4 → succ-only 均值必须=2.0(若误把失败也算进去会得 3.0)
    env = _SeqSingleEnv([(2, 1.0), (4, 0.0)])
    m = run_libero_evaluation(env=env, agent=_StubAgent(), num_episodes=2, device="cpu")
    assert abs(m["eval/success_rate"] - 0.5) < 1e-6
    assert abs(m["eval/mean_successful_episode_length"] - 2.0) < 1e-6


class _ThrowingAgent(_StubAgent):
    def act(self, obs, *, eval_mode=False, stddev=0.0, cpu=True):
        raise RuntimeError("boom")


def test_restores_train_mode_on_exception():
    agent = _ThrowingAgent()
    env = _SeqSingleEnv([(1, 1.0)])
    with pytest.raises(RuntimeError):
        run_libero_evaluation(env=env, agent=agent, num_episodes=1, device="cpu")
    assert agent.mode == "train"   # finally 即便异常也还原训练模式


# --- subgoal 在线注入(pi0_feat) ---

class _RecordingAgent(_StubAgent):
    """记录每次 act 收到的 obs 里 observation.subgoal 的形状(无则记 None)。"""
    def __init__(self):
        super().__init__()
        self.seen_subgoal = []

    def act(self, obs, *, eval_mode=False, stddev=0.0, cpu=True):
        sg = obs.get("observation.subgoal")
        self.seen_subgoal.append(None if sg is None else tuple(sg.shape))
        return super().act(obs, eval_mode=eval_mode, stddev=stddev, cpu=cpu)


class _StubSubgoalPi0Feat:
    state_mode = "pi0_feat"
    rep_dim = 10

    def subgoal_online(self, obs, rel_raw=None, prefix_feat=None):
        assert prefix_feat is not None          # 注入必须经 base_policy.last_prefix_feat()
        b = obs["observation.state"].shape[0]
        return torch.zeros(b, self.rep_dim)


class _StubSubgoalEefPiece:
    state_mode = "eef_piece"                     # 非 pi0_feat → 应 fail-fast


class _StubBasePolicy:
    def last_prefix_feat(self):
        return torch.zeros(2048)


def test_injects_subgoal_each_act_when_conditioned():
    agent = _RecordingAgent()
    env = _ScriptedVecEnv(num_envs=1, ep_len=1, terminal_rewards=[1.0, 1.0])
    run_libero_evaluation(env=env, agent=agent, num_episodes=2, device="cpu",
                          subgoal=_StubSubgoalPi0Feat(), base_policy=_StubBasePolicy())
    assert len(agent.seen_subgoal) == 2
    assert all(s == (1, 10) for s in agent.seen_subgoal)   # 每次 act 都注入了 (B, rep_dim)


def test_no_subgoal_omits_key():
    agent = _RecordingAgent()
    env = _ScriptedVecEnv(num_envs=1, ep_len=1, terminal_rewards=[1.0, 1.0])
    run_libero_evaluation(env=env, agent=agent, num_episodes=2, device="cpu")   # subgoal 默认 None
    assert len(agent.seen_subgoal) == 2
    assert all(s is None for s in agent.seen_subgoal)      # 旧路径不注入,零回归


def test_subgoal_requires_base_policy():
    env = _ScriptedVecEnv(num_envs=1, ep_len=1, terminal_rewards=[1.0])
    with pytest.raises(ValueError):
        run_libero_evaluation(env=env, agent=_StubAgent(), num_episodes=1, device="cpu",
                              subgoal=_StubSubgoalPi0Feat(), base_policy=None)


def test_subgoal_rejects_non_pi0_feat():
    env = _ScriptedVecEnv(num_envs=1, ep_len=1, terminal_rewards=[1.0])
    with pytest.raises(NotImplementedError):
        run_libero_evaluation(env=env, agent=_StubAgent(), num_episodes=1, device="cpu",
                              subgoal=_StubSubgoalEefPiece(), base_policy=_StubBasePolicy())


class _StubBasePolicyNoFeat:
    def last_prefix_feat(self):
        return None


def test_subgoal_none_prefix_feat_raises():
    env = _ScriptedVecEnv(num_envs=1, ep_len=1, terminal_rewards=[1.0])
    with pytest.raises(ValueError):
        run_libero_evaluation(env=env, agent=_StubAgent(), num_episodes=1, device="cpu",
                              subgoal=_StubSubgoalPi0Feat(), base_policy=_StubBasePolicyNoFeat())
