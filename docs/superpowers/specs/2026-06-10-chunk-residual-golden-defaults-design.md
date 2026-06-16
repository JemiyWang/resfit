# chunk_residual 训练默认值对齐金标准 — 设计文档

- 日期: 2026-06-10
- 分支: `chunk-residual-validation`
- 文件: `resfit/rl_finetuning/chunk_residual/train_chunk_residual.py`(`build_parser`)

## 1. 背景与动机

`train_chunk_residual.py` 是所有 chunk residual 实验共用的主脚本(argparse 默认历史上是给 boxcleanup 配的)。但核查近期 wandb run 发现:**近期所有 three_piece/piece 实验(`subgoal_*`、`nas10_*`、`piece_*`)都手动传同一套超参**,旧 argparse 默认已过时、没人在用。本任务把这套"金标准"设为 argparse 默认,省去每次手传一长串 flag、减少配错风险(如本会话就因漏传这些 flag 起错过 run)。

证据(近期 run 实际 argv):`chunk_length 1` / `actor_lr 1e-6` / `base_action_mode queue` / `base_n_action_steps 10` / `stage_balanced` 全部一致;`base_policy` 模式由 baseanchor 实验验证(best success_rate=0.74)。

## 2. 决策(用户确认)

- **全局改 argparse 默认**(不做 task-conditional 分支 / preset)。
- **一套到底**:含 `offline_base_mode` `gt`→`base_policy`(修 `gt_as_base` bug,见 [[project_resfit_gt_as_base_bug]]/[[project_resfit_base_policy_as_base]];baseanchor 实证好)。
- 旧值全部保留可显式回退。
- `--wandb_project` 默认本就是 `dexmg-chunk-residual`(已=金标准,**不改**)。

## 3. 改动清单(6 个默认值,只改 `build_parser` 里的 `default=`)

| flag | 旧默认 | 新默认 |
|---|---|---|
| `--chunk_length` | 20 | **1** |
| `--actor_lr` | 5e-6 | **1e-6** |
| `--base_action_mode` | replan | **queue** |
| `--base_n_action_steps` | None | **10** |
| `--stage_balanced` | False(`store_true`) | **True**(`store_true,default=True` + 新增 `--no_stage_balanced`) |
| `--offline_base_mode` | gt | **base_policy** |

带值的 4 个(`chunk_length`/`actor_lr`/`base_action_mode`/`base_n_action_steps`)直接改 `default=`,旧值可显式传回退。`offline_base_mode` 同理(choices 保留 `gt`)。

**⚠️ code review 更新(2026-06-10):`--base_n_action_steps` 撤回、不改默认(留 None)。** 改 10 会 break pi05 base:`train_chunk_residual.py:452-454` 在 `base_n_action_steps is not None` 时 assert `base_policy_type=="act"`,pi05 base 不传该参数就吃到默认 10 → assert 崩,而 argparse 传不了 None 回退 → pi05 线被硬 break。act 金标准命令本就显式传 `--base_n_action_steps 10`(零实际影响)。**故实际改 5 个参数**(chunk_length/actor_lr/base_action_mode/stage_balanced/offline_base_mode)。

## 4. `stage_balanced` 的 `--no_` 写法(为什么需要)

`store_true` 的 flag 一旦默认 True 就**无法在命令行关闭**(只能往 True 设)。为保住"回退到旧行为 False"的能力,改为:
```python
p.add_argument("--stage_balanced", dest="stage_balanced", action="store_true", default=True, help=...)
p.add_argument("--no_stage_balanced", dest="stage_balanced", action="store_false", help="关闭 stage-balanced 采样(回退旧行为)")
```
行为(已由 renorm 同款写法验证):不传=True / `--no_stage_balanced`=False / `--stage_balanced`=True。

## 5. `offline_base_mode=base_policy` 的耦合约束(重要)

`base_policy` 模式有 assert(`_validate_offline_base_mode`):需 `base_action_mode=="queue" && chunk_length==1`。
- **新默认组合天然满足**(chunk_length 1 + queue + base_policy)→ 不带任何 flag 跑即合法。
- **耦合坑**:若单独显式回退 `--chunk_length 20`(却不回退 `--offline_base_mode gt`),会触发该 assert **fail-fast 报错**。这是预期的防误配行为,但是改默认引入的新耦合。回退 chunk_length 时须同时 `--offline_base_mode gt`。

## 6. 连带改动

- **测试**(新增 + 核查):
  - 新增 parser 默认断言:不带 flag 时 6 个值 = 金标准;每个旧值可显式回退(`--chunk_length 20 --offline_base_mode gt` 一起回退、`--no_stage_balanced` 等)。
  - 覆盖耦合:默认组合 parse 通过;`--chunk_length 20`(单独,不回退 base_mode)→ `_validate_offline_base_mode` fail-fast;`--chunk_length 20 --offline_base_mode gt`(一起)→ 通过。
  - 核查现有 `build_parser().parse_args([])` 断言(`test_stage_budget`、`test_relabel`、`test_demo_bc`、`test_hiql_potential`、`test_wandb_logging` 等)不被新默认破坏;若有断言旧默认的,改成显式传旧值。
- **不改底层函数签名默认**(只动 argparse 层),与上次 HIQL 默认对齐一致。

## 7. 不影响进行中的实验

正在跑的 aligned A/B(run1/run2)**显式传**全部金标准 flag,不依赖默认,故改默认不影响它们。已启动进程 argv 固定。

## 8. 风险

- **`offline_base_mode=base_policy` 设默认**:多数历史实验其实用 `gt`,base_policy vs gt 无严格同配置 A/B 定论(baseanchor 是 base_policy+staged+bc0.1,无对应 gt 版);且 base_policy 建 offline cache 慢(对 ~23.8 万 transition 跑 base forward)。缓解:`--offline_base_mode gt` 可一键回退;`gt_as_base` bug 已确证、base_policy 是其修法、baseanchor 实证 0.74。
- **全局默认影响 boxcleanup/threading 重跑**(近期未跑,无法确认它们是否也该用这套)。缓解:用户已接受(这套是事实标准);旧值全部可显式回退。
- **耦合**(§5):回退 chunk_length 须同时回退 offline_base_mode,否则 fail-fast。已用测试覆盖+本 spec 记录。
