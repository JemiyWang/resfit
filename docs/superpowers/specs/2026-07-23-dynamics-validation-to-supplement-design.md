# 动力学模型验证迁移至补充材料设计

## 目标

将正文 Figure 8 的动力学模型详细验证移至补充材料，使正文聚焦 SHORE-RL
的 horizon shortening、waypoint 和 potential shaping，同时保留足够的
world-model reliability 证据以回应审稿人对 imagination training 的疑问。

## 正文

删除 `paper/main.tex` 中当前的动力学模型验证 figure 环境。保留一个精简的
`Dynamics Model Validation` 段落，说明模型在任何策略训练前使用 held-out
episodes 沿三个维度验证：

1. 多步 imagined-rollout quality；
2. action controllability，包括 deliberately perturbed action 的 negative probe；
3. imagined/real potential-trajectory consistency。

该段指向 Supplementary Sec. C.4.4，并说明详细诊断放在补充材料，因为动力学
模型仅作为训练环境而不是论文的算法贡献。当前尚无最终数值，因此正文只保留
一个明确的 P0 TODO，要求验证完成后填入一项最关键的 held-out headline metric；
不得虚构结果。

## 补充材料

复用 `paper/aaai2027-unified-supp.tex` 中现有的 C.4.4 验证协议。将“这三个
panel 位于正文”改为指向本节的补充图，并在三项协议之后加入迁移来的验证图
占位。补充图使用双栏宽度，以便最终容纳三个 panel，并使用独立标签
`fig:supp-wmval`。

## 不修改

- 不修改动力学模型训练方法、数据或指标定义。
- 不修改 real-robot evaluation table、pipeline 或结论。
- 不生成或声称尚未完成的实验结果。

## 验证

完整编译正文和补充材料。正文不得再包含 `fig:wmval` 或 Figure 8 的验证图；
补充材料必须包含 `fig:supp-wmval`，并且所有引用均已解析。检查两份日志中
不存在 LaTeX error 或 undefined reference/citation，且修改的验证段不得引入
新的 overfull box。补充材料原有 352--371 行的 overfull 不属于本任务范围。
