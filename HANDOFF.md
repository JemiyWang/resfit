# ResFiT 复现 — Handoff (2026-05-30 晚, Phase 2 RL 进行中)

## 0. 一句话现状
**Phase 1（BC）已全部完成/手动停止,Phase 2（残差 TD3 RL）已对 4 个任务启动并在训练。**
- 在跑:Can(GPU5)、Square(GPU0)、CanSort(GPU2)、BoxCleanup(GPU1),各自已存出第一个基座 `best.pt`。
- 还差:**Coffee 的 RL 未启动**(Coffee BC 还在 GPU4 跑)。
- 训练脚本已修两个 bug(见第 5 节),并加了统一启动 wrapper `run_rl.sh`。

---

## 1. 项目目标 & 两阶段
复现 ResFiT 论文(arXiv 2509.19301;讲解 `/data2/kai0/relative-work/relative_paper/ResFiT_2509.19301_讲解.md`)的 5 个仿真任务。
- **Phase 1 — BC**:ACT 在演示上训冻结基座,推 wandb artifact `run_<id>_best`。
- **Phase 2 — 残差 RL**:off-policy TD3(RLPD recipe),加载冻结 BC 基座,只学残差 a=a_base+a_res,把成功率往上抬。只支持 ACT base。

---

## 2. 环境(已搭好,沿用)
- conda env **`residual`**(Python 3.10),仓库 `/data2/RL/residual-offpolicy-rl`。
- **torch 2.7.1 + cu12x**;eval 离屏渲染必须 **`MUJOCO_GL=egl PYOPENGL_PLATFORM=egl`**;所有命令**从仓库根目录跑**(代码靠 cwd 导入)。
- wandb 账号:**674575221-beijing-institute-of-technology**(BIT),已登录。
- 详见项目 memory `project_resfit_repro.md` 与 `project_resfit_rl_dumps_fix.md`。

---

## 3. Phase 2 RL 当前状态(2026-05-30 ~22:00)
| 任务 | GPU | tmux session | 日志 | wandb 项目 | 总步数 | best.pt |
|---|---|---|---|---|---|---|
| Can | 5 | can_rl | can_rl_train.log | robomimic-can-final | 300k | ✅ |
| Square | 0 | square_rl | square_rl_train.log | robomimic-square-final | 300k | ✅ |
| CanSort | 2 | cansorting_rl | cansorting_rl_train.log | dexmg-cansort-final | 500k | ✅ |
| BoxCleanup | 1 | boxcleanup_rl | boxcleanup_rl_train.log | dexmg-box-clean-final | 500k | ✅ |
| **Coffee** | — | (未起) | — | dexmg-coffee-final | 500k | ❌ 还没起 RL |

- **eval 间隔:5 任务都是每 10000 环境步一次**(从 step 0 评基座)。单臂 31 次 eval,双臂 51 次。
- 基座来源:Can/Square BC 训完(单臂 ≈0.8/0.9);**双臂 BC 被手动提前停**(基座够用即停)——BoxCleanup 基座 step-40000、CanSort 基座 step-25000(欠训练但够起 RL,双臂基座本来就弱 ~0.1)。

---

## 4. BC 基座 wandb_id(已全部填进 RL config)
`resfit/rl_finetuning/config/residual_td3.py` 各任务 `base_policy.wandb_id`(格式 project/run_id,RL 会自动拉 `run_<id>_best:latest`):
- Can → `robomimic-can-bc/dvg09ifx`
- Square → `robomimic-square-bc/4dw5df0t`
- BoxCleanup → `dexmg-boxcleanup-bc/d59wny58`
- CanSort → `dexmg-cansorting-bc/0lxbiap5`
- Coffee → `dexmg-coffee-bc/gbiv6udg`
> 这些 run_id 也可在 `wandb/run-<时间戳>-<id>/` 目录名、或各 BC 日志 `View run at` 行核对。

---

## 5. 已做的两处代码修复(关键,非显然)
1. **offline buffer 缓存 dumps 崩溃**(`resfit/rl_finetuning/utils/hugging_face.py`):torchrl 0.8 `PrioritizedSampler.dumps()` 把 `_max_priority`(tensor 元组)直接 json.dump → 崩。已在 `optimized_replay_buffer_dumps` 加 `_sanitize_prioritized_sampler_for_dump`(dump 前转标量、finally 恢复)。崩溃重启前需 `rm -rf offline_buffer_cache/<hash>`(残缺缓存会导致再崩)。
2. **RL 不保存训练结果**(`resfit/rl_finetuning/scripts/train_residual_td3.py`):原本 eval 出新最佳只 print 不存、结尾还 `shutil.rmtree` 删整个 run 目录。已接上 `utils/checkpoint.py::save_checkpoint`,**新最佳时存 `<run_dir>/models/best.pt`**(agent.state_dict+优化器+config+success_rate),并删掉结尾的 rmtree。`best_eval_success_rate` 初值 0,故 step-0 评基座必触发首存。
> 两处都在共享代码里,5 任务通用。详见 memory `project_resfit_rl_dumps_fix.md`。

---

## 6. 怎么启动一个任务的 RL(统一 wrapper)
```bash
cd /data2/RL/residual-offpolicy-rl
bash resfit/rl_finetuning/shell/run_rl.sh <task> <gpu>
# 例: bash resfit/rl_finetuning/shell/run_rl.sh coffee 4
```
wrapper 自动:解析对应官方脚本 `paper_runs/<task>/1_*_residual_rl.sh` → 套 `CUDA_VISIBLE_DEVICES`+`MUJOCO_GL=egl`+`conda run -n residual` → 从仓库根目录跑 → tee 到 `<task>_rl_train.log` → 起 detached tmux `<task>_rl`。**同名 session 已存在时会拒绝启动**(防双进程抢卡)。
- task 取值:can / square / boxcleanup / cansorting / coffee。
- ⚠️ 不要直接 `bash 1_*_residual_rl.sh`(裸脚本无 env,eval 渲染会崩、可能抢 GPU0)。
- 注:Can 原本缺 `1_can_residual_rl.sh`,已补;双臂脚本(cansorting/coffee/boxcleanup)有命令行超参覆盖(n_step=5/buffer=300k/action_scale=0.2 等),但**不覆盖 total_timesteps 与 eval 间隔**。

**起 RL 前置检查**(wrapper 不替你做):
1. 该任务 BC 已推出 `_best`(BC 日志有 `New best success-rate`)。
2. config 里该任务 `wandb_id` 填对(本次已全部填好)。
3. 选空闲 GPU(`nvidia-smi`)。

---

## 7. 下一步 = Coffee(目前被阻塞)
⚠️ **Coffee 暂时起不了 RL**:Coffee BC 还在 GPU4 跑(~step 80000/200000),但**从没出过一次 new best**(本地无 `best/`、无 `best_step_*`,故 wandb 上也没有 `run_gbiv6udg_best`)。RL 会去拉 `run_gbiv6udg_best:latest`,现在**拉不到会直接失败**。

原因已查明:eval 正常在跑(已 16 次),但 **Coffee 基座 16 次 eval 全部 0% 成功**(step 80000 时 `eval success-rate: 0.0%`)。保存条件是 `success_rate > 0`(初值 0),全 0 就永远不触发 → 没有 best。不是 eval 的 bug,是双臂咖啡太难、基座还没学会。

**起 Coffee RL 前必须先有非 0 的基座:**
1. 让 Coffee BC 继续训(现 ~80k/200k,还有一多半),看能否出现 >0% 成功率;Coffee 是论文里最难的任务,可能要接近 200k 才有非 0 base。eval 曲线看 wandb `dexmg-coffee-bc/gbiv6udg`。
2. 若训到 200k 仍 0% → 需排查 Coffee 数据/任务设置(0% 偏低)。
3. 一旦出现 >0% 的 best(`run_gbiv6udg_best` 生成),再:
```bash
cd /data2/RL/residual-offpolicy-rl
bash resfit/rl_finetuning/shell/run_rl.sh coffee 3   # 用空闲卡 3/6/7
```

---

## 8. 监控 & 产物位置
- 看某任务:`tmux attach -t <task>_rl`(Ctrl-b d 退出),或 `tail -f <task>_rl_train.log`。
- GPU:`nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader`。
- **训练好的残差策略**:`run_<时间戳>_residual-rl*<task>*/models/best.pt`(每创新高覆盖;torch.load 后重建 QAgent + load_state_dict 即可加载,QAgent 是 nn.Module)。
- eval 视频 / Q 图:同 run 目录的 `outputs/`。
- **eval 成功率曲线**:wandb 各 `*-final` 项目的 `eval/success_rate`(按 step;`evaluate_dexmg.py` 每次 eval 都 wandb.log)。也可 `grep "eval/success" <task>_rl_train.log`。
- buffer 缓存:`offline_buffer_cache/<hash>/`、`online_buffer_cache/<hash>/`(按任务/配置 hash,重启会加载以加速)。

---

## 9. 画论文图 5(学习曲线)
图 5 = 成功率 vs RL 环境步数。**必须用中间过程的 eval 成功率**(不是 best 那一个点)——这些已在 wandb 各 `*-final` 项目记录。取数:
```python
import wandb
h = wandb.Api().run("674575221-beijing-institute-of-technology/<project>/<run_id>").history(keys=["eval/success_rate"])
```
论文图 5 通常还对比基线(RLPD 等);完整复现可能要再跑各任务的 `2_*_rlpd.sh`。建议先读 PDF 确认图 5 具体几条曲线/哪些方法。

---

## 10. 文件位置速查
- RL 训练脚本:`resfit/rl_finetuning/scripts/train_residual_td3.py`
- RL 任务配置(wandb_id 在此):`resfit/rl_finetuning/config/residual_td3.py`
- RL 启动 wrapper:`resfit/rl_finetuning/shell/run_rl.sh`
- 官方 RL 脚本:`resfit/rl_finetuning/shell/paper_runs/<task>/1_*_residual_rl.sh`(残差)/`2_*_rlpd.sh`(RLPD 基线)
- buffer 缓存修复:`resfit/rl_finetuning/utils/hugging_face.py`;checkpoint 保存:`resfit/rl_finetuning/utils/checkpoint.py`
- RL 日志:仓库根目录 `<task>_rl_train.log`
- RL 产物:仓库根目录 `run_<时间戳>_residual-rl*/`(models/best.pt + outputs/)
- 项目 memory:`/home/ubuntu/.claude/projects/-data2/memory/project_resfit_repro.md` 和 `project_resfit_rl_dumps_fix.md`
