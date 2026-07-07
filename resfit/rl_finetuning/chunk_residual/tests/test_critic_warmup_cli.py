from resfit.rl_finetuning.chunk_residual.train_chunk_residual import build_parser


def _base_args():
    # Minimal args build_parser needs; --task/--dataset are the common required-ish flags.
    return ["--task", "TwoArmPouring", "--dataset", "d"]


def test_critic_warmup_steps_default_zero():
    a = build_parser().parse_args(_base_args())
    assert a.critic_warmup_steps == 0


def test_critic_warmup_steps_parses_value():
    a = build_parser().parse_args(_base_args() + ["--critic_warmup_steps", "10000"])
    assert a.critic_warmup_steps == 10000
