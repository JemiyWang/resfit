# Figure 4 消融图例命名设计

## 目标

将论文 Figure 4 的图例统一为“完整方法名 + `w/o` 消融项”的论文常用写法，
使读者无需依赖正文即可识别完整方法，并能直接看出每条消融曲线移除了什么。

## 修改范围

- 生成代码：`paper/plot_ablation_curves.py`
- 生成产物：`paper/figure/fig_ablation_curves.pdf`
- 不修改实验数据、运行选择、曲线、颜色、线型、图例顺序、面板布局或正文。

## 图例

五条曲线依次使用：

- `SHORE-RL`
- `w/o stage shaping`
- `w/o waypoint`
- `w/o waypoint & stage shaping`
- `w/o demo-BC & stage shaping`

`SHORE-RL` 表示完整方法，因此不再附加冗余的 `(Full)`。`w/o waypoint`
也不附加 `module`，保持简洁并与正文中的 waypoint 术语一致。

## 验证

重新运行绘图脚本生成 PDF，确认输出成功；提取或渲染 PDF，核对五个图例名称，
并确认图例没有重叠、截断或超出画布。
