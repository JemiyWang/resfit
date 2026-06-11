# 消除重复 MuJoCo replay:read_per_demo_states 复用 state30 缓存(chokepoint)设计

- 日期:2026-06-11
- 状态:设计已定稿,待落实施计划
- 范围:**只做 chokepoint**——让 `read_per_demo_states`(value / gc_value / state30 缓存构建的共同汇聚点)在有完整缓存时直接读、跳过 MuJoCo 逐帧 replay。**不动** `build_offline_buffer`(主训那条独立 replay,留作后续第二阶段)。

## 1. 背景与动机

object-aware(eef_piece, 30 维)训练里,`read_per_demo_states`(`train_hiql_value.py`)对全部 demo 逐帧 `make_replay_env + set_state + 算 eef-rel-piece`(~19min),产出 30 维标准化 state + rel_stats。它是单一汇聚点,被三处调用:
- `train_hiql_value`(直接)
- `train_hiql_gc_value`(import 调用)
- `state30_cache.load_or_build_state30`(构建缓存时调用)

现状:**每次训练都重 replay 一遍**(用户实测:刚建好 state30 缓存,gc_value 又把同一份 replay 了一遍)。`load_or_build_state30` 虽产出 `*_state30.npz`,但 `read_per_demo_states` 自己**不读它**,且该缓存**缺 rel_stats**,不足以让 value/gc_value 复用。

目标:改一处 `read_per_demo_states`,让上述三处都复用缓存、单次 replay。**严格不改变数值结果**(默认逐位等价)。

非目标:`build_offline_buffer`(主训 offline buffer 的 rel replay,需 raw rel 反标准化,等价性更难)——本设计明确排除,留第二阶段。

## 2. 关键约束(等价性命门)

- **缓存存的是标准化 rel**(嵌在 30 维 state 的 18:30 列);value/gc_value 还需 `rel_stats=(mean12, std12)` 存进 value.pt 给 Φ 用——**现有缓存没存 stats**。→ 缓存格式必须扩。
- **`rel_stats` 是对「参与 replay 的那批 demo」的 raw rel 算的 mean/std**。`read_per_demo_states(num_demos=N)` 用前 N 条 → stats 随 N 变。**所以缓存(全量建)只在 `num_demos is None`(全量)时与新鲜 replay 严格等价**;`num_demos < 全量`(冒烟)时 stats 会不同 → **那种情况必须退回 replay**。
- 缓存里 30 维的 state18 部分是用 dataset 的 norm stats 标准化的;复用时 standardizer 由 `LeRobotDatasetMetadata(dataset_id)` 重建(确定性,同 dataset 即同 stats)。为防张冠李戴,缓存存 `dataset_id`,复用前校验一致。

## 3. 设计

### 3.1 缓存格式 v2(`state30_cache.py`)
现有 npz:`n`(demo 数)、`s{i}`((T_i,30) float32)。**v2 新增**:`rel_mean`(12,)、`rel_std`(12,)、`dataset_id`(str)。
- `save_state30_cache(path, seqs, rel_stats=None, dataset_id=None)`:rel_stats 给了就一并写 `rel_mean`/`rel_std`/`dataset_id`(v2);不给则只写 `n`+`s{i}`(v1,向后兼容旧调用)。
- `load_state30_cache(path)`:仍返回 seqs(读 `s{i}`)——旧调用不变。
- 新增 `load_state30_cache_v2(path)`:返回 `(seqs, rel_stats_or_None, dataset_id_or_None)`;缺 `rel_mean` 键 → rel_stats=None(旧格式信号)。

### 3.2 `read_per_demo_states` 加 cache 复用
签名加可选参 **`cache_path=None`**(默认 None = 现状逐位等价)。仅 `state_mode="eef_piece"` 生效(eef 18 维本就不 replay)。逻辑:
```
standardizer = 从 LeRobotDatasetMetadata 建(始终,廉价)
if state_mode == "eef_piece" and cache_path and num_demos is None and os.path.exists(cache_path):
    seqs30, rel_stats, ds_id = load_state30_cache_v2(cache_path)
    if rel_stats is not None and ds_id == dataset_id and len(seqs30) == 全量 demo 数:
        return seqs30, standardizer, rel_stats          # 命中:跳过 replay
# 未命中(无缓存/旧格式/dataset 不符/num_demos 部分)→ 现有 replay 路径
seqs30, rel_stats = <replay 现状>
if state_mode == "eef_piece" and cache_path and num_demos is None:
    save_state30_cache(cache_path, seqs30, rel_stats=rel_stats, dataset_id=dataset_id)  # 写 v2
return seqs30, standardizer, rel_stats
```
- `num_demos is None` 是复用前置(§2 等价性);部分一律 replay,绝不误用全量 stats。
- 命中要求 `len(seqs30) == hdf5 全量 demo 数`(防"部分缓存"被当全量)。

### 3.3 消费方接线
- `train_hiql_value`:加 `--state30_cache`(default None)→ 透传 `read_per_demo_states(cache_path=...)`。
- `train_hiql_gc_value`:加 `--state30_cache`(default None)→ 同上。
- `load_or_build_state30`:构建分支改为捕获 rel_stats 并存 v2(`seqs, _, rel = read_per_demo_states(...)` → `save_state30_cache(cache_path, seqs, rel_stats=rel, dataset_id=...)`);读分支不变(`load_state30_cache` 读 seqs,旧/新格式都能读,goal30 只要 seqs)。这样 `load_or_build_state30` 产出的缓存即 v2,供 value/gc_value 复用。

### 3.4 不变量
- 不传 `--state30_cache` → 所有路径行为与现在**逐位相同**(默认 None)。
- `build_offline_buffer`、high_actor、主训:**完全不动**(high_actor/goal30 经 `load_or_build_state30`/`load_state30_cache` 读 seqs,v2 的额外键被忽略)。

## 4. 测试(TDD,核心是等价性)
1. **往返等价**:同一 hdf5,(a)新鲜 replay 得 `(seqs30_a, rel_stats_a)`;(b)用 cache_path 跑一次(写缓存)再跑一次(读缓存)得 `(seqs30_b, rel_stats_b)` → 断言 **seqs30 和 rel_stats 逐位相等**(float32 npz 往返精确)。
2. **默认等价**:不传 cache_path → 与改动前同(回归保护)。
3. **旧格式回退**:喂一个只有 `n`+`s{i}` 的旧缓存 → 不命中、走 replay、重写成 v2(校验 v2 有 rel_mean)。
4. **num_demos 部分不复用**:有全量缓存但 `num_demos=20` → 必须 replay(stats 对 20 条算),不读缓存。
5. **dataset 不符不复用**:缓存 dataset_id 与请求不一致 → replay。
6. `save/load_state30_cache_v2` 单元:rel_stats 写入/读出、缺键返回 None。
> 用小型 fake / 少量 demo 构造(可借现有 test 的 mock env 思路);避免真起 MuJoCo 全量 replay。

## 5. 要改的文件
- `resfit/rl_finetuning/chunk_residual/state30_cache.py`:v2 save/load。
- `resfit/rl_finetuning/chunk_residual/train_hiql_value.py`:`read_per_demo_states` 加 cache 复用 + `--state30_cache` flag。
- `resfit/rl_finetuning/chunk_residual/train_hiql_gc_value.py`:`--state30_cache` flag 透传。
- 测试:`tests/test_state30_cache.py`(新增或扩)。

## 6. 风险
- standardizer 重建与缓存时不一致 → 由 `dataset_id` 校验 + "同 dataset 确定性" 兜底;若仍担心可在测试里对比 state18 部分。
- 旧缓存被误当 v2 → 由"缺 rel_mean 键 → rel_stats=None → 不命中"保证。
- 不影响任何在跑实验(默认 None;改动只在新进程显式传 flag 时生效)。
- **缓存不随 hdf5 变化失效(既有行为,非本次引入)**:`state30_cache_reuse` 只校验 `num_demos is None` + dataset_id,不比对 hdf5 实际 demo 数。若 hdf5 在建缓存后被重生成/扩充(同路径同 dataset),复用会静默返回旧(更短)demo 集,下游 `zip` 截断不报错。整个 `*_state30.npz` 方案都有此盲信特性。规避:**hdf5 重生成后手动删旧缓存重建**。(若要兜底:在复用前开 hdf5 数 `sorted_demo_keys` 并要求 `len(seqs)==该数`,代价是 chokepoint 多一次 hdf5 open。)
