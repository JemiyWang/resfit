# chunk_residual wandb 上报设计

**日期:** 2026-06-04
**分支:** chunk-residual-validation
**作者:** brainstorming (Claude + 用户)

## 问题

`resfit/rl_finetuning/chunk_residual/train_chunk_residual.py` 是新写的精简训练循环,
**从头就没接 wandb**:文件里所有 `wandb` 字样只是为了从 artifact 拉基座 policy
(`download_policy_from_wandb`),没有 `wandb.init`,也没有 `wandb.log`。所有指标只
`print` 到 stdout(`[env_steps N] eval success_rate`、`[stage-diag]`、`[stage-purity]`)。

对比原始脚本 `train_rlpd_dexmg.py` / `train_residual_td3.py`:两者都
`wandb.init(...)` + 循环里 `wandb.log(log_dict, step=global_step)`,记录 critic_loss /
actor_loss / grad_norm / lr / buffer / action & Q 的 histogram 等完整训练动力学。

后果:chunk_residual 的实验拿不到 wandb 曲线,训练动力学(loss/Q/grad 的连续走势)
在纯 stdout 日志里是黑的 —— 而这恰恰是诊断早期失稳(如 env_steps=20000 那次 success
0.30->0.04 的塌)、判断 lr 能否收敛最需要的维度。

## 目标

给 `train_chunk_residual.py` 补上 wandb 上报,镜像原始脚本的指标集,并顺带把
chunk_residual 独有的 stage 诊断(stage-diag / stage-purity)也推上 wandb。
只加不删:现有 stdout print 全部保留作为 tee 日志后备。

**非目标:**
- 不改正在跑的 stageON 实验(它跑的是旧代码,补丁只对下一个实验生效;不做 resume/回填)。
- 不引入原始脚本的 Hydra/OmegaConf config 系统(chunk_residual 用 argparse,保持现状)。
- 不动训练算法、不改 QAgent。

## 方案

抽一个小 helper 模块,训练脚本调它。理由:训练循环保持精简;log_dict 组装逻辑成为
近乎纯函数,可 TDD 单测(这次 bug 的本质就是"键没拼/没上报",纯函数正好测得到);
契合项目既有 pattern(`stage_utils.py` / `offline_hdf5_buffer.py` 都是独立 helper 模块)。
`wandb.init` / `wandb.log` 本身只是薄 I/O 封装。

## 架构与接口

### 新文件 `resfit/rl_finetuning/chunk_residual/wandb_logging.py`

```
init_wandb(args) -> wandb run
    薄封装 wandb.init。
    - project = args.wandb_project
    - entity  = args.wandb_entity
    - name    = args.wandb_name
    - mode    = "disabled" if args.smoke else args.wandb_mode
    - config  = vars(args)
    disabled 模式下 wandb.init 仍返回一个 no-op run,调用方无需判空。

build_train_log_dict(m_upd, lrs, buf_sizes, with_histograms=True, hist_fn=wandb.Histogram) -> dict
    纯函数,组装训练指标 log_dict。
    - m_upd: agent.update() 返回的 metrics dict
    - lrs:   {"actor": float, "critic": float, "encoder": float}
    - buf_sizes: {"online": int, "offline": int}
    - 行为:
        1. 收 m_upd 里所有不以 "_" 开头的键(即 train/*),原样放入。
        2. 加 lr/actor、lr/critic、lr/encoder。
        3. 加 buffer/online_size、buffer/offline_size。
        4. with_histograms 且 m_upd 含 "_actions": 加 histograms/actions = hist_fn(_actions.numpy().reshape(-1))。
        5. with_histograms 且 m_upd 含 "_target_q": 加 histograms/critic_qt = hist_fn(_target_q.numpy().reshape(-1))。
        (.numpy().reshape(-1) 与原始 train_rlpd_dexmg.py:755,763 对齐)
    - hist_fn 可注入,单测传 stub 即可绕开 wandb 依赖。
    - _actions / _target_q 已是 detach().cpu()(见 q_agent.py:411,508),无额外 GPU->CPU 拷贝。

build_eval_log_dict(eval_metrics, last_diag, purity_summary) -> dict
    纯函数,组装 eval + stage 诊断 log_dict。
    - eval_metrics: run_dexmg_evaluation 返回的 dict(含 eval/success_rate 等 eval/* 键)。
    - last_diag: flatten_stage_diagnostics 的输出(形如 {"diag/residual_norm/stage0": float, ...}),
                 可能为 None(warmup 期未发生 update_actor)。为 None 时跳过 diag 部分。
    - purity_summary: env.stage_purity_summary() 的字符串。解析成 purity/regress_frac 与
                      purity/stageN 数值键。解析失败则只放原始字符串到 purity/raw(不抛异常)。
    - 行为:收所有 eval/* 键;若 last_diag 非空,原样并入(键已是 diag/...);并入解析后的 purity/*。
```

### 训练脚本接线 `train_chunk_residual.py`(只加不删)

- 启动:agent 构好后、进 `while` 循环前 -> `run = init_wandb(args)`。
- 训练分支(在 `if env_steps >= learning_starts` 块内、UTD 循环之后):
  维护 `next_log`(类比已有的 `next_eval`,初值 = learning_starts)。当 `env_steps >= next_log`
  且本轮有 `m_upd`(即发生过 update)时:
  ```
  lrs = {"actor": agent.actor_opt.param_groups[0]["lr"],
         "critic": agent.critic_opt.param_groups[0]["lr"],
         "encoder": agent.encoder_opt.param_groups[0]["lr"]}
  buf_sizes = {"online": len(online_rb), "offline": len(offline_rb) if offline_rb else 0}
  wandb.log(build_train_log_dict(m_upd, lrs, buf_sizes), step=env_steps)
  next_log += args.log_freq
  ```
- eval 分支(现有 print 之后):
  ```
  wandb.log(build_eval_log_dict(m, last_diag, env.stage_purity_summary()), step=env_steps)
  ```
- 结尾(`print("done...")` 之后):`wandb.finish()`。

### 数据流与 x 轴

所有 `wandb.log` 用 `step=env_steps`(单调递增)。训练指标落在 `next_log` 阈值步,eval
指标落在 eval 点步;两者偶尔同一 env_steps,wandb 自动 merge 同 step 的多次 log,不冲突。

### 新增 args(argparse)

```
--wandb_project  default="dexmg-chunk-residual"
--wandb_entity   default=None
--wandb_name     default=None  -> 运行时回落为 os.path.basename(args.output_dir.rstrip("/"))
--wandb_mode     default="online"  (可 "offline" / "disabled")
--log_freq       type=int default=100
```

## 频率 gating

- 训练指标 + histogram:每 `args.log_freq`(默认 100)env_steps 一次,仅在
  `env_steps >= learning_starts` 之后(此前无 m_upd)。
- eval + stage 诊断:沿用 eval 点(每 `eval_every_env_steps`)。

histogram 同频(每 100 步)。因 `_actions`/`_target_q` 已在 CPU 且张量小,均摊到每步
开销 << 当前 ~0.42 s/step 的万分之一,落在测量噪声内。

## 错误处理

- `init_wandb`:`--smoke` 强制 mode="disabled",烟雾/单测不产生真 run。
- `build_eval_log_dict`:`last_diag` 为 None 时跳过 diag 部分(warmup 期正常);
  purity_summary 解析失败时回落到 `purity/raw`(放原始字符串),不抛异常打断训练。
- helper 模块顶部 `import wandb`(wandb 已是依赖,原始脚本在用)。

## 测试策略(TDD)

新建 `resfit/rl_finetuning/chunk_residual/tests/test_wandb_logging.py`:

1. `build_train_log_dict` 收齐 train/* 键、过滤掉 `_` 前缀内部键(`_actions`/`_target_q`
   不泄漏到 log_dict)。
2. `build_train_log_dict` 加了 lr/actor、lr/critic、lr/encoder、buffer/online_size、
   buffer/offline_size。
3. `build_train_log_dict(with_histograms=False)` 不含 histograms/* 键。
4. `build_train_log_dict` 传 stub `hist_fn` 时,histograms/actions、histograms/critic_qt
   等于 stub 在 `.numpy().reshape(-1)` 后的输出(不依赖真 wandb)。
5. `build_eval_log_dict` 收齐 eval/success_rate、并入非空 last_diag 的 diag/* 键、
   解析出 purity/regress_frac 与 purity/stageN。
6. `build_eval_log_dict(last_diag=None)` 不含 diag/* 键、不抛异常。
7. `build_eval_log_dict` 在 purity_summary 不可解析时回落 purity/raw、不抛异常。
8. `init_wandb`:`args.smoke=True` 时以 mode="disabled" 调 wandb.init(monkeypatch
   wandb.init 验传参)。

现有默认测试套件保持全绿(改动是纯新增,不碰既有路径)。

## 验收

- 跑一次带 `--wandb_mode online` 的短 smoke(或 disabled 下走通代码路径)不报错。
- wandb run 里能看到 train/critic_loss、train/actor_loss_total、lr/*、buffer/*、
  histograms/actions、histograms/critic_qt 的按 env_steps 曲线,以及 eval/success_rate、
  diag/*、purity/* 在 eval 点的值。
- 单测全绿;现有套件不回归。
