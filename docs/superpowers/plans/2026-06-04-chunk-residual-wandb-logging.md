# chunk_residual wandb 上报 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给 `train_chunk_residual.py` 补上 wandb 上报(训练动力学 + eval + stage 诊断),镜像原始脚本的指标集,只加不删。

**Architecture:** 抽一个 helper 模块 `chunk_residual/wandb_logging.py`,把 log_dict 组装做成近乎纯函数(可 TDD 单测),`init_wandb` 是 `wandb.init` 薄封装;训练脚本里加 5 个 argparse 参数,并在循环的训练分支(每 log_freq 步)与 eval 分支调用它们 + `wandb.log`。

**Tech Stack:** Python, PyTorch, wandb, pytest。

**Spec:** `docs/superpowers/specs/2026-06-04-chunk-residual-wandb-logging-design.md`

---

## 文件结构

- Create: `resfit/rl_finetuning/chunk_residual/wandb_logging.py`
  - 职责:wandb 接线。`init_wandb(args)`、`build_train_log_dict(...)`、`build_eval_log_dict(...)`、内部 `_parse_purity(...)`。
- Create: `resfit/rl_finetuning/chunk_residual/tests/test_wandb_logging.py`
  - 职责:helper 的单测(不依赖 wandb/GPU,用 stub hist_fn + monkeypatch wandb.init)。
- Modify: `resfit/rl_finetuning/chunk_residual/train_chunk_residual.py`
  - 职责:把 argparse 抽成 `build_parser()` + 加 5 个 wandb 参数;在 `main()` 循环里接 init/train-log/eval-log/finish。

关键事实(实现时直接用,无需再查):
- `agent.update()` 返回 dict,含 `train/*` 键(critic_loss / critic_qt / actor_loss_base / actor_loss_total / *_grad_norm / actor_l2_penalty 等)与内部 `_actions` `_target_q` `_combined_actions` `_td_errors`。`_actions`/`_target_q` 已是 `.detach().cpu()`(q_agent.py:411,508)。
- `agent.actor_opt` / `agent.critic_opt` / `agent.encoder_opt` 都存在(q_agent.py:120-122),`.param_groups[0]["lr"]` 取学习率。
- `run_dexmg_evaluation(...)` 返回的 dict 含 `eval/success_rate`(键带 `eval/` 前缀)。
- `flatten_stage_diagnostics(...)` 输出扁平 dict,键形如 `diag/count/stage0`、`diag/residual_norm/stage0`、`diag/target_q/stage0`(stage_diag.py:38-46)。训练循环里这个值存在 `last_diag`(可能为 None)。
- `env.stage_purity_summary()` 返回字符串:`"regress {A}/{B}={P:.1%}  stage{L}:{r}/{n}={P:.0%}  ..."`(空格 join),特例 `"no stage steps"`(chunk_env_wrapper.py:181-191)。
- `train_chunk_residual.py` 顶部已 `import os`、`import argparse`。argparse 当前内联在 `main()`(L123 起,`p = argparse.ArgumentParser()`)。训练循环 L284-346:`env_steps`(L286)、`next_eval=0`(L287)、`last_diag`(L289)、训练分支 `if env_steps >= args.learning_starts and len(online_rb) > online_batch_size:`(L302)内有 `for i in range(args.utd):`(L303)与 `m_upd = agent.update(...)`(L315);eval 分支 `if env_steps >= next_eval:`(L324),内有 eval(L325-330)、`sr=m["eval/success_rate"]`(L331)、三个 print(L338-341)、`next_eval += args.eval_every_env_steps`(L342);结尾 `print("done...")`(L346)。

---

### Task 1: build_train_log_dict(训练指标组装)

**Files:**
- Create: `resfit/rl_finetuning/chunk_residual/wandb_logging.py`
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_wandb_logging.py`

- [ ] **Step 1: 写失败测试**

写入 `resfit/rl_finetuning/chunk_residual/tests/test_wandb_logging.py`:

```python
import torch
import pytest
from resfit.rl_finetuning.chunk_residual.wandb_logging import build_train_log_dict


def _fake_m_upd():
    return {
        "train/critic_loss": 0.5,
        "train/actor_loss_total": -1.2,
        "train/critic_grad_norm": 3.0,
        "_actions": torch.zeros(4, 12),
        "_target_q": torch.ones(10, 4),
        "_td_errors": torch.zeros(4),
    }


_LRS = {"actor": 1e-6, "critic": 1e-4, "encoder": 2e-4}
_BUF = {"online": 100, "offline": 50}
_STUB = lambda a: ("H", len(a))


def test_train_log_keeps_train_keys_and_drops_underscore():
    out = build_train_log_dict(_fake_m_upd(), _LRS, _BUF, hist_fn=_STUB)
    assert out["train/critic_loss"] == 0.5
    assert out["train/actor_loss_total"] == -1.2
    assert out["train/critic_grad_norm"] == 3.0
    assert "_actions" not in out
    assert "_target_q" not in out
    assert "_td_errors" not in out


def test_train_log_adds_lr_and_buffer():
    out = build_train_log_dict(_fake_m_upd(), _LRS, _BUF, hist_fn=_STUB)
    assert out["lr/actor"] == 1e-6
    assert out["lr/critic"] == 1e-4
    assert out["lr/encoder"] == 2e-4
    assert out["buffer/online_size"] == 100
    assert out["buffer/offline_size"] == 50


def test_train_log_no_histograms_when_disabled():
    out = build_train_log_dict(_fake_m_upd(), _LRS, _BUF, with_histograms=False)
    assert "histograms/actions" not in out
    assert "histograms/critic_qt" not in out


def test_train_log_histograms_use_hist_fn():
    out = build_train_log_dict(_fake_m_upd(), _LRS, _BUF, hist_fn=_STUB)
    assert out["histograms/actions"] == ("H", 48)    # 4*12
    assert out["histograms/critic_qt"] == ("H", 40)  # 10*4
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd /data2/RL/residual-offpolicy-rl && python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_wandb_logging.py -v`
Expected: FAIL — `ModuleNotFoundError`/`ImportError`(wandb_logging 不存在或无 build_train_log_dict)。

- [ ] **Step 3: 写最小实现**

写入 `resfit/rl_finetuning/chunk_residual/wandb_logging.py`:

```python
"""chunk_residual 训练的 wandb 上报 helper。

train_chunk_residual.py 原本只 print 到 stdout、不接 wandb。这里把指标组装抽成
近乎纯函数(build_train_log_dict / build_eval_log_dict)便于单测;init_wandb 是
wandb.init 的薄封装。详见
docs/superpowers/specs/2026-06-04-chunk-residual-wandb-logging-design.md。
"""
import os

import wandb


def build_train_log_dict(m_upd, lrs, buf_sizes, with_histograms=True,
                         hist_fn=wandb.Histogram):
    """组装训练指标 log_dict。

    m_upd: agent.update() 返回的 dict(含 train/* 与 _ 前缀内部键)。
    lrs: {"actor": float, "critic": float, "encoder": float}。
    buf_sizes: {"online": int, "offline": int}。
    hist_fn: 可注入,单测传 stub 即可绕开 wandb。
    """
    log_dict = {k: v for k, v in m_upd.items() if not k.startswith("_")}
    log_dict["lr/actor"] = lrs["actor"]
    log_dict["lr/critic"] = lrs["critic"]
    log_dict["lr/encoder"] = lrs["encoder"]
    log_dict["buffer/online_size"] = buf_sizes["online"]
    log_dict["buffer/offline_size"] = buf_sizes["offline"]
    if with_histograms:
        if "_actions" in m_upd:
            log_dict["histograms/actions"] = hist_fn(m_upd["_actions"].numpy().reshape(-1))
        if "_target_q" in m_upd:
            log_dict["histograms/critic_qt"] = hist_fn(m_upd["_target_q"].numpy().reshape(-1))
    return log_dict
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd /data2/RL/residual-offpolicy-rl && python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_wandb_logging.py -v`
Expected: PASS(4 个 test 全绿)。

- [ ] **Step 5: 提交**

```bash
cd /data2/RL/residual-offpolicy-rl
git add resfit/rl_finetuning/chunk_residual/wandb_logging.py resfit/rl_finetuning/chunk_residual/tests/test_wandb_logging.py
git commit -m "feat: chunk_residual build_train_log_dict + 单测

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: _parse_purity + build_eval_log_dict(eval/stage 诊断组装)

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/wandb_logging.py`
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_wandb_logging.py`(追加)

- [ ] **Step 1: 写失败测试**

追加到 `resfit/rl_finetuning/chunk_residual/tests/test_wandb_logging.py` 末尾:

```python
from resfit.rl_finetuning.chunk_residual.wandb_logging import (
    _parse_purity, build_eval_log_dict,
)


def test_parse_purity_normal():
    s = "regress 11410/29939=38.1%  stage0:0/9289=0%  stage1:2184/5916=37%"
    out = _parse_purity(s)
    assert abs(out["purity/regress_frac"] - 11410 / 29939) < 1e-9
    assert out["purity/stage0"] == 0.0
    assert abs(out["purity/stage1"] - 2184 / 5916) < 1e-9


def test_parse_purity_no_stage_steps():
    assert _parse_purity("no stage steps") == {"purity/raw": "no stage steps"}


def test_parse_purity_unparseable_falls_back():
    assert _parse_purity("garbage data here") == {"purity/raw": "garbage data here"}


def test_eval_log_with_diag():
    em = {"eval/success_rate": 0.12, "eval/other": 1.0}
    diag = {"diag/residual_norm/stage0": 0.13, "diag/target_q/stage0": 0.8}
    out = build_eval_log_dict(em, diag, "regress 1/2=50%  stage0:1/2=50%")
    assert out["eval/success_rate"] == 0.12
    assert out["eval/other"] == 1.0
    assert out["diag/residual_norm/stage0"] == 0.13
    assert out["diag/target_q/stage0"] == 0.8
    assert out["purity/regress_frac"] == 0.5
    assert out["purity/stage0"] == 0.5


def test_eval_log_diag_none():
    out = build_eval_log_dict({"eval/success_rate": 0.3}, None, "no stage steps")
    assert out["eval/success_rate"] == 0.3
    assert not any(k.startswith("diag/") for k in out)
    assert out["purity/raw"] == "no stage steps"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd /data2/RL/residual-offpolicy-rl && python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_wandb_logging.py -v -k "purity or eval_log"`
Expected: FAIL — `ImportError`(`_parse_purity` / `build_eval_log_dict` 不存在)。

- [ ] **Step 3: 写最小实现**

追加到 `resfit/rl_finetuning/chunk_residual/wandb_logging.py` 末尾:

```python
def _parse_purity(summary):
    """把 stage_purity_summary() 字符串解析成 {purity/regress_frac, purity/stageN}。
    解析不了就回落 {purity/raw: 原串}(不抛异常)。"""
    if not summary or summary == "no stage steps":
        return {"purity/raw": summary}
    out = {}
    try:
        tokens = summary.split()
        for i, tok in enumerate(tokens):
            if tok == "regress":
                a, b = tokens[i + 1].split("=")[0].split("/")
                out["purity/regress_frac"] = float(a) / float(b)
            elif tok.startswith("stage") and ":" in tok:
                label, rest = tok.split(":", 1)
                stage_idx = int(label[len("stage"):])
                r, n = rest.split("=")[0].split("/")
                out[f"purity/stage{stage_idx}"] = float(r) / float(n)
    except Exception:
        return {"purity/raw": summary}
    if not out:
        return {"purity/raw": summary}
    return out


def build_eval_log_dict(eval_metrics, last_diag, purity_summary):
    """组装 eval + stage 诊断 log_dict。

    eval_metrics: run_dexmg_evaluation 返回的 dict(含 eval/* 键)。
    last_diag: flatten_stage_diagnostics 的扁平输出(键已带 diag/ 前缀),可能为 None。
    purity_summary: env.stage_purity_summary() 字符串。
    """
    log_dict = {k: v for k, v in eval_metrics.items() if k.startswith("eval/")}
    if last_diag:
        log_dict.update(last_diag)
    log_dict.update(_parse_purity(purity_summary))
    return log_dict
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd /data2/RL/residual-offpolicy-rl && python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_wandb_logging.py -v`
Expected: PASS(全部 9 个 test 绿)。

- [ ] **Step 5: 提交**

```bash
cd /data2/RL/residual-offpolicy-rl
git add resfit/rl_finetuning/chunk_residual/wandb_logging.py resfit/rl_finetuning/chunk_residual/tests/test_wandb_logging.py
git commit -m "feat: chunk_residual _parse_purity + build_eval_log_dict + 单测

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: init_wandb(wandb.init 薄封装)

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/wandb_logging.py`
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_wandb_logging.py`(追加)

- [ ] **Step 1: 写失败测试**

追加到 `resfit/rl_finetuning/chunk_residual/tests/test_wandb_logging.py` 末尾:

```python
import argparse
from resfit.rl_finetuning.chunk_residual.wandb_logging import init_wandb


def test_init_wandb_smoke_forces_disabled(monkeypatch):
    captured = {}

    def fake_init(**kw):
        captured.update(kw)
        return "RUN"

    monkeypatch.setattr(
        "resfit.rl_finetuning.chunk_residual.wandb_logging.wandb.init", fake_init)
    args = argparse.Namespace(
        smoke=True, wandb_mode="online", wandb_project="P",
        wandb_entity=None, wandb_name="N", output_dir="outputs_chunk/x")
    run = init_wandb(args)
    assert run == "RUN"
    assert captured["mode"] == "disabled"
    assert captured["project"] == "P"
    assert captured["name"] == "N"


def test_init_wandb_name_falls_back_to_output_dir(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "resfit.rl_finetuning.chunk_residual.wandb_logging.wandb.init",
        lambda **kw: captured.update(kw))
    args = argparse.Namespace(
        smoke=False, wandb_mode="disabled", wandb_project="P",
        wandb_entity=None, wandb_name=None, output_dir="outputs_chunk/cl1_stageON/")
    init_wandb(args)
    assert captured["name"] == "cl1_stageON"
    assert captured["mode"] == "disabled"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd /data2/RL/residual-offpolicy-rl && python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_wandb_logging.py -v -k init_wandb`
Expected: FAIL — `ImportError`(`init_wandb` 不存在)。

- [ ] **Step 3: 写最小实现**

追加到 `resfit/rl_finetuning/chunk_residual/wandb_logging.py` 末尾:

```python
def init_wandb(args):
    """按 args 启 wandb run。--smoke 时强制 mode='disabled'(不产生真 run)。

    name 缺省回落为 output_dir 的 basename。返回 wandb run(disabled 下为 no-op run,
    调用方无需判空)。
    """
    name = args.wandb_name or os.path.basename(args.output_dir.rstrip("/"))
    mode = "disabled" if args.smoke else args.wandb_mode
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=name,
        mode=mode,
        config=vars(args),
    )
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd /data2/RL/residual-offpolicy-rl && python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_wandb_logging.py -v`
Expected: PASS(全部 11 个 test 绿)。

- [ ] **Step 5: 提交**

```bash
cd /data2/RL/residual-offpolicy-rl
git add resfit/rl_finetuning/chunk_residual/wandb_logging.py resfit/rl_finetuning/chunk_residual/tests/test_wandb_logging.py
git commit -m "feat: chunk_residual init_wandb 薄封装 + 单测

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: 抽 build_parser() + 加 5 个 wandb 参数

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/train_chunk_residual.py`(`main()` 内联 argparse -> `build_parser()`)
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_wandb_logging.py`(追加)

- [ ] **Step 1: 写失败测试**

追加到 `resfit/rl_finetuning/chunk_residual/tests/test_wandb_logging.py` 末尾:

```python
from resfit.rl_finetuning.chunk_residual.train_chunk_residual import build_parser


def test_parser_has_wandb_defaults():
    args = build_parser().parse_args(["--task", "TwoArmThreePieceAssembly"])
    assert args.wandb_project == "dexmg-chunk-residual"
    assert args.wandb_entity is None
    assert args.wandb_name is None
    assert args.wandb_mode == "online"
    assert args.log_freq == 100


def test_parser_wandb_overrides():
    args = build_parser().parse_args(
        ["--wandb_mode", "disabled", "--wandb_project", "P", "--log_freq", "50"])
    assert args.wandb_mode == "disabled"
    assert args.wandb_project == "P"
    assert args.log_freq == 50
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd /data2/RL/residual-offpolicy-rl && python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_wandb_logging.py -v -k parser`
Expected: FAIL — `ImportError: cannot import name 'build_parser'`(当前 argparse 内联在 main())。

- [ ] **Step 3: 重构 + 加参数**

在 `train_chunk_residual.py` 中,把 `def main():` 里从 `p = argparse.ArgumentParser()`(L123)到 `args = p.parse_args()` 这一整段 argparse 定义**整体移出**成一个新函数 `build_parser()`,放在 `main()` 之前;`main()` 改为 `args = build_parser().parse_args()` 开头。即:

```python
def build_parser():
    p = argparse.ArgumentParser()
    # ... 原 main() 里所有现有的 p.add_argument(...) 原样照搬,顺序不变 ...
    # 在 return 之前新增 5 个 wandb 参数:
    p.add_argument("--wandb_project", default="dexmg-chunk-residual",
                   help="wandb project 名")
    p.add_argument("--wandb_entity", default=None, help="wandb entity(默认用账号默认)")
    p.add_argument("--wandb_name", default=None,
                   help="wandb run 名;缺省回落为 output_dir 的 basename")
    p.add_argument("--wandb_mode", choices=["online", "offline", "disabled"],
                   default="online", help="wandb 模式;--smoke 时自动 disabled")
    p.add_argument("--log_freq", type=int, default=100,
                   help="训练指标上报间隔(env_steps)")
    return p


def main():
    args = build_parser().parse_args()
    # ... main() 的其余逻辑原样保留(从原来 args = p.parse_args() 之后那一行开始)...
```

注意:只移动 argparse 定义段;`main()` 中 `args = p.parse_args()` 之后的所有代码保持不动。不要漏掉任何一个现有的 `p.add_argument`。

- [ ] **Step 4: 跑测试确认通过**

Run: `cd /data2/RL/residual-offpolicy-rl && python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_wandb_logging.py -v`
Expected: PASS(全部 13 个 test 绿)。

并确认模块仍可导入:
Run: `cd /data2/RL/residual-offpolicy-rl && python -c "import resfit.rl_finetuning.chunk_residual.train_chunk_residual as t; print(t.build_parser().parse_args(['--smoke']).wandb_mode)"`
Expected: 打印 `online`(无 import 错误)。

- [ ] **Step 5: 提交**

```bash
cd /data2/RL/residual-offpolicy-rl
git add resfit/rl_finetuning/chunk_residual/train_chunk_residual.py resfit/rl_finetuning/chunk_residual/tests/test_wandb_logging.py
git commit -m "refactor: 抽 build_parser() 并加 5 个 wandb 参数 + 单测

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: 在 main() 循环里接 init/train-log/eval-log/finish

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/train_chunk_residual.py`

本任务是把 helper 接进训练循环。纯接线、依赖完整 GPU+env,无法做自动单测;验证用"导入检查 + 现有单测全绿 + 可选 GPU 烟雾"。

- [ ] **Step 1: 顶部加导入**

在 `train_chunk_residual.py` 顶部 import 区(`import argparse` 附近)加:

```python
import wandb

from resfit.rl_finetuning.chunk_residual.wandb_logging import (
    init_wandb, build_train_log_dict, build_eval_log_dict,
)
```

- [ ] **Step 2: 进 while 前启 wandb,并初始化 next_log**

在 `main()` 里,offline buffer 设置之后、`while env_steps <= total:`(L291)之前,加一行启动 wandb(放在 `obs, _ = env.reset()`(L285)与计数器初始化附近)。把 `next_eval = 0`(L287)那一组初始化改成同时初始化 `next_log`:

```python
    run = init_wandb(args)

    obs, _ = env.reset()
    env_steps = 0
    next_eval = 0
    next_log = args.learning_starts
    best_sr = 0.0
    last_diag = None
```

(即:在已有的计数器初始化块中加入 `run = init_wandb(args)` 与 `next_log = args.learning_starts`;其余行不动。)

- [ ] **Step 3: 训练分支加 train-log**

在 `if env_steps >= args.learning_starts and len(online_rb) > online_batch_size:`(L302)块内、`for i in range(args.utd):` 循环**结束之后**(即与 `for` 同缩进、紧跟其后),加 train-log。此处 `m_upd` 已绑定到最后一次 update 的返回:

```python
            if env_steps >= next_log:
                lrs = {"actor": agent.actor_opt.param_groups[0]["lr"],
                       "critic": agent.critic_opt.param_groups[0]["lr"],
                       "encoder": agent.encoder_opt.param_groups[0]["lr"]}
                buf_sizes = {"online": len(online_rb),
                             "offline": len(offline_rb) if offline_rb else 0}
                wandb.log(build_train_log_dict(m_upd, lrs, buf_sizes), step=env_steps)
                next_log += args.log_freq
```

- [ ] **Step 4: eval 分支加 eval-log**

在 `if env_steps >= next_eval:`(L324)块内,`print("[stage-purity] " + env.stage_purity_summary())`(L341)之后、`next_eval += args.eval_every_env_steps`(L342)之前,加:

```python
            wandb.log(build_eval_log_dict(m, last_diag, env.stage_purity_summary()),
                      step=env_steps)
```

- [ ] **Step 5: 结尾 finish**

在 `print(f"done. best success_rate={best_sr:.3f}")`(L346)之后加:

```python
    wandb.finish()
```

- [ ] **Step 6: 验证(导入 + 现有单测)**

Run: `cd /data2/RL/residual-offpolicy-rl && python -c "import resfit.rl_finetuning.chunk_residual.train_chunk_residual"`
Expected: 无报错(import 成功)。

Run: `cd /data2/RL/residual-offpolicy-rl && python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_wandb_logging.py -v`
Expected: PASS(全部 13 个 test 绿)。

- [ ] **Step 7: 可选 GPU 烟雾(有空闲 GPU 时跑;当前 GPU2 被在跑实验占用)**

用 disabled 模式跑通代码路径(不产生真 run、不碰网络),确认接线不崩:

```bash
cd /data2/RL/residual-offpolicy-rl
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=<空闲GPU> conda run -n residual python -m resfit.rl_finetuning.chunk_residual.train_chunk_residual \
  --task TwoArmThreePieceAssembly --smoke --wandb_mode disabled \
  --base_wandb_id <三-piece 基座 ckpt 目录> --output_dir outputs_chunk/_smoke_wandb
```
Expected: 跑到 `done. best success_rate=...` 且无 wandb 相关报错。
(若无空闲 GPU,此步交给用户在 GPU 空出后执行;自动门槛为 Step 6。)

- [ ] **Step 8: 提交**

```bash
cd /data2/RL/residual-offpolicy-rl
git add resfit/rl_finetuning/chunk_residual/train_chunk_residual.py
git commit -m "feat: 在 train_chunk_residual 循环里接 wandb init/log/finish

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:**
- helper 模块 init_wandb / build_train_log_dict / build_eval_log_dict -> Task 1/2/3。✓
- log_dict 键(train/* + lr/* + buffer/* + histograms/*;eval/* + diag/* + purity/*)-> Task 1/2 测试断言覆盖。✓
- 训练脚本接线(init / 每 log_freq train-log / eval 点 eval-log / finish)-> Task 5。✓
- 新增 5 args + 默认值 -> Task 4。✓
- 频率 gating(next_log / learning_starts;eval 沿用 next_eval)-> Task 5 Step 2/3/4。✓
- 错误处理(smoke->disabled;last_diag None;purity 解析回落)-> Task 3 / Task 2 测试覆盖。✓
- 测试策略 8 点 -> Task 1-4 共 13 个单测覆盖;wiring 用导入+烟雾。✓
- 只加不删、不动在跑实验 -> 全程新增,Task 5 不删现有 print。✓

**Placeholder scan:** 无 TBD/TODO;每个代码步骤都有完整代码;命令均给出预期输出。Step 7 的 `<空闲GPU>`/`<基座 ckpt 目录>` 是运行时占位(无法预知),已注明为可选手动步骤。

**Type consistency:** `build_train_log_dict(m_upd, lrs, buf_sizes, with_histograms, hist_fn)` / `build_eval_log_dict(eval_metrics, last_diag, purity_summary)` / `_parse_purity(summary)` / `init_wandb(args)` / `build_parser()` 在测试与实现、Task 间签名一致。lrs 键 actor/critic/encoder、buf_sizes 键 online/offline 全文一致。purity 键 `purity/regress_frac`、`purity/stageN`、`purity/raw` 一致。
