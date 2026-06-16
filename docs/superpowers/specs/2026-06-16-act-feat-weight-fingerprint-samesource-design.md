# act_feat 权重指纹同源校验 设计

> 状态:待实现(brainstorm 已定稿)。范围聚焦、单一实现计划可覆盖。

## 背景与问题

`act_feat` 路的离线/在线同源链条在「签名层面」是硬保证的:

- `train_chunk_residual.py:736-738` 对 `act_ckpt_id / image_keys / proprio_key / pooling` 逐字段 `assert _cache_sig == _gv_sig`(cache ↔ gc_value 签名硬校验);
- 在线 `HiqlSubgoal.from_ckpts`(`hiql_subgoal.py:84-89`)用 gc_value 里存的 `act_feat_signature` 构造 `ActFeatureExtractor(base_policy, ...)`,归一化 `mean/std` 也取自 gc_value info;
- `ActFeatureExtractor.__init__`(`act_feature.py:46-49`)再 assert `image_keys == base_policy.config.image_features.keys()`。

唯一的漏洞在「**在线残差 base_policy 自身权重**是否与离线 build cache 的 ACT 同一份」:`train_chunk_residual.py:741-747` 只对此做 `warnings.warn`,且比较的是两个**不同命名空间的标识符**:

| | 来源 | 形态 |
|---|---|---|
| cache 的 `act_ckpt_id` | `--act_base_ckpt`(`train_hiql_value.py:258` `ext.signature(str(args.act_base_ckpt))`) | **本地 ACT 目录路径** |
| 在线的 `base_wandb_id` | `--base_wandb_id`(默认 `dexmg-boxcleanup-bc/d59wny58`,`train_chunk_residual.py:415`) | **wandb run id**(也可是本地目录) |

典型 wandb 工作流下一个是 wandb id、一个是本地路径,字符串永不相等 → 现在这个 warning **每次 act_feat run 都误报** → 大家学会无视 → 命门形同虚设。只在「两边都传成完全相同本地路径」这一窄情况下才有意义。

后果:在线 base ACT 若换成另一份权重(同相机/同 dim),上面所有签名校验照过,却套着离线的 `mean/std` → 特征出自不同网络、被错配的统计量标准化 → **静默错配**,subgoal z 完全失真。

## 目标

用与标识符无关的「**权重内容指纹**」替代不可靠的 id 字符串比较,使「在线 base ACT ≠ 离线 build cache 的 ACT」**默认硬失败**(可逃生),且**不破坏任何现有产物**(旧 29G 缓存、旧 gc_value)。

## 范围

**做**:① `act_feature.py` 加纯函数 `act_weight_fingerprint` + extractor 方法 `weight_fingerprint`;② cache npz / gc_value ckpt 各**旁挂**一个 `act_weight_sha` 字段;③ 离线在 build cache / 训 gc_value 时算并写入指纹(cache 命中时从缓存转写);④ 在线用 base_policy 重算指纹比对,默认 assert + `--allow_act_base_mismatch` 逃生口;⑤ 替换掉 `train_chunk_residual.py:741-747` 不可靠的 id 比较;⑥ 单元 + 向后兼容回归测试;⑦ 复用现成 act_feat 一致性 smoke 做端到端(人工,不入 pytest)。

**不做**:① 改 `act_feat_signature` 的 4 字段(保持不动 → 旧缓存照常命中);② pi0_feat / eef_piece 路的同源校验(pi0_feat 已有 `serve_ckpt_id` 签名校验,eef_piece 无 ACT,均不在本次);③ 行为指纹(dummy 输入跑 ACT 取输出哈希)—— 已选「全量 state_dict 哈希」,且受 forward 非确定性影响,否决;④ 强制重建旧缓存/重训旧 gc_value。

## 关键决策(brainstorm)

1. **旁挂、不并入 `act_feat_signature`**:指纹存为 cache/gc_value 的独立字段,不参与 `act_feat_cache_reuse` 的「缓存命中」全等比较 → 现有缓存零失效;权重校验只对**新建的 cache + 新训的 gc_value**生效,旧资产在线时走 warn/skip。
2. **全量 state_dict 哈希**:语义为「是不是同一个 ckpt」,是安全的过度校验(只多报、不漏报)。decoder/动作头不影响 act_feat 但也纳入,简单稳健。
3. **默认硬 assert + 逃生口**:两边都有指纹且不等 → 默认 raise 中断;`--allow_act_base_mismatch` 降级为 warning。两边任一缺指纹(旧资产)→ 固定走 warn/skip。

## 架构

改 3 个文件 + 加测试。

### 1. `resfit/rl_finetuning/chunk_residual/act_feature.py`

新增纯函数(可单测,不需真 ACT,拿小 `nn.Module` 即可):

```python
import hashlib
import torch

def act_weight_fingerprint(policy) -> str:
    """对 policy.state_dict() 的确定性 sha256 指纹(hexdigest)。

    确定性来源:按 key 排序;浮点张量先 detach().cpu().float().contiguous() 去掉
    设备/内存布局/当前精度的不确定性;整型/bool 缓冲(如 BN num_batches_tracked)按
    原 dtype 取字节。key、dtype、shape、bytes 全部喂进哈希。

    保证:同一份内存权重 → 同一 hash;改任一权重 → hash 变。
    不保证 fp16 存档 vs fp32 存档相等(那本就是两份不同的值)。
    """
    h = hashlib.sha256()
    sd = policy.state_dict()
    for k in sorted(sd.keys()):
        t = sd[k]
        if not torch.is_tensor(t):
            continue
        t = t.detach().cpu()
        if t.is_floating_point():
            t = t.float()
        t = t.contiguous()
        h.update(k.encode("utf-8"))
        h.update(str(t.dtype).encode("utf-8"))
        h.update(repr(tuple(t.shape)).encode("utf-8"))
        h.update(t.numpy().tobytes())
    return h.hexdigest()
```

`ActFeatureExtractor` 加方法封装:

```python
def weight_fingerprint(self) -> str:
    return act_weight_fingerprint(self.act)
```

### 2. `resfit/rl_finetuning/chunk_residual/act_feat_cache.py`

`save_act_feat_cache` 多收一个**可选** `act_weight_sha=None`,存为字符串键(旁挂,**不进** `signature`):

```python
def save_act_feat_cache(path, seqs, emb_stats, *, signature, fp16=False, act_weight_sha=None):
    ...
    if act_weight_sha is not None:
        payload["act_weight_sha"] = np.asarray(str(act_weight_sha))
    np.savez_compressed(path, **payload)
```

`_load` 容缺:

```python
def _load(path):
    with np.load(path, allow_pickle=False) as z:
        ...
        sha = str(z["act_weight_sha"]) if "act_weight_sha" in z.files else None
    return seqs, stats, sig, sha   # 多返回一项
```

`load_act_feat_cache` / `act_feat_cache_reuse` 相应跟上(返回值扩成 4 元组;**`act_feat_cache_reuse` 的全等比较逻辑不动** —— 仍只比 `signature + num_demos`,旧缓存照常命中)。

> 改返回元数(3→4)是 breaking,需同步所有调用点(`train_hiql_value.py`、`train_chunk_residual.py:731/757` 同款 pi0 不受影响)。调用点见「数据流」。

### 3. `resfit/rl_finetuning/chunk_residual/train_hiql_value.py`

`setup_act_feat` / `read_per_demo_states` 产出 `act_weight_sha` 并写入 gc_value info(与现有 `act_feat_signature/mean/std` 并排):

- **`--act_base_ckpt` 现 build**:`sha = extractor.weight_fingerprint()`;`save_act_feat_cache(..., act_weight_sha=sha)` 写进 cache;同一 `sha` 写进 gc_value info。
- **cache 命中(未加载 ACT)**:`sha` = `load_act_feat_cache` 返回的缓存里那份 `act_weight_sha`(旧缓存=None);照样写进 gc_value info。

即 gc_value 的 `act_weight_sha` 始终 == 它所用特征对应的 ACT 指纹(现算的或缓存转写的),与 cache 自洽。

### 4. `resfit/rl_finetuning/chunk_residual/train_chunk_residual.py`(替换 741-747)

```python
# 取代不可靠的 base_wandb_id vs act_ckpt_id 字符串比较。
# _cache_sha 来自 line 731 的 4 元组解包:_seqs,_stats,_cache_sig,_cache_sha = load_act_feat_cache(...)
_gv_sha = _gc_info.get("act_weight_sha")   # 旧 gc_value 为 None

# (a) 离线自洽:cache 与 gc_value 两边都有指纹且不等 → 离线就不自洽
if _cache_sha and _gv_sha and _cache_sha != _gv_sha:
    raise ValueError(f"[act_feat] cache 与 gc_value 权重指纹不符: {_cache_sha} vs {_gv_sha}")

# (b) 在线 base ACT 同源校验
_base_sha = act_weight_fingerprint(base_policy)
if _gv_sha is None:
    warnings.warn("[act_feat] 产物无权重指纹(旧 cache/gc_value),无法校验在线 base 同源;务必先过一致性 smoke", stacklevel=2)
elif _base_sha == _gv_sha:
    pass  # 同源确认,静默
elif args.allow_act_base_mismatch:
    warnings.warn(f"[act_feat] 在线 base 权重指纹 != 离线({_base_sha} vs {_gv_sha}),--allow_act_base_mismatch 已放行", stacklevel=2)
else:
    raise ValueError(f"[act_feat] 在线 base_policy 权重 != 离线 build cache 的 ACT(指纹 {_base_sha} vs {_gv_sha});确认同源,或加 --allow_act_base_mismatch 放行")
```

argparse 加 `--allow_act_base_mismatch`(`action="store_true"`,默认 False)。

## 数据流(端到端)

```
[离线 build cache]  act = load_policy(--act_base_ckpt)
                    sha = fingerprint(act)
                    ├─ save_act_feat_cache(cache.npz, act_weight_sha=sha)
                    └─ gc_value.pt.info["act_weight_sha"] = sha

[离线 cache 命中]   _,_,_,sha = load_act_feat_cache(cache.npz)   # 旧缓存 sha=None
                    └─ gc_value.pt.info["act_weight_sha"] = sha

[在线残差训练]      base_policy = build_base_policy(--base_wandb_id)
                    base_sha = fingerprint(base_policy)
                    cmp(base_sha, gc_value.info["act_weight_sha"])  # 三分支(见 §4)
```

## 向后兼容

- `act_feat_signature` 一字不动、`act_feat_cache_reuse` 比较逻辑不动 → 现有 29G 缓存全部照常命中,旧 gc_value 照常 `load_gc_value`。
- cache npz 缺 `act_weight_sha` → `_load` 返回 None;gc_value info 缺 `act_weight_sha` → `.get` 返回 None → 在线走 warn/skip,**绝不强制重建/重训**。
- `load_act_feat_cache` 返回元数 3→4 是源码层 breaking,但只波及仓内调用点(本 PR 一并改),不影响磁盘产物格式的读取。

## 测试

纯函数单测(`tests/`,不需真 ACT,构造小 `nn.Module`):

1. **确定性**:同一 module 连哈希两次相等;`.cuda()`/`.half()` 后再 `fingerprint`(内部 cast)与 fp32-CPU 版对「同值」张量相等(注:half 往返会变值,测「构造时即 fp32、仅设备/contiguous 不同」这一可控维度)。
2. **敏感性**:`with torch.no_grad(): p.add_(1e-3)` 扰动一个权重 → hash 变。
3. **整型缓冲不崩**:含 `register_buffer(int64)` 的 module 能正常哈希。
4. **cache 旁挂容缺**:`save_act_feat_cache` 不传 sha → `load_act_feat_cache` 第 4 项为 None,不崩;`act_feat_cache_reuse` 命中逻辑与加 sha 前一致(回归)。
5. **离线自洽 assert**:cache_sha != gv_sha → §4(a) raise。
6. **在线三分支**:`gv_sha=None`→warn;`base==gv`→静默;`base!=gv` 默认 raise、带 flag→warn。

集成 smoke(人工):跑一个 act_feat `--subgoal_conditioned` 配置(同源 base)→ 应静默通过;故意换一份 base ACT → 应 raise;加 `--allow_act_base_mismatch` → 应 warn 并继续。复用现成的 act_feat 一致性 smoke。

## 命门 / 风险

- **cache 读取返回元数变更(3→4)**:`_load` / `load_act_feat_cache` 都多返回 `sha`,必须一次性改齐所有解包点(含同文件内调用 `_load` 的 `act_feat_cache_reuse`),否则 `ValueError: not enough values to unpack`。grep `load_act_feat_cache(`、`act_feat_cache_reuse(`、`_load(` 锁定全部调用。
- **指纹对象要对**:在线必须对 `base_policy`(真正喂进 `ActFeatureExtractor` 的那个 ACT)算指纹,不是对别的 wrapper;离线必须对 build cache 用的那份 `act` 算。
- **state_dict 含非确定项**:若 ACT 里有 `register_buffer` 的随机/计数缓冲且每次加载不同,会误报 —— 经查 act_feat 路 ACT 为冻结 eval、加载即固定,无此问题;但测试 3 要覆盖整型缓冲以防 dtype 崩。
- **fp32 cast 的过度宣称**:不要在文档/日志里宣称「跨 dtype 等价」,只保证「同值 fp32 等价」,避免误导排查。
