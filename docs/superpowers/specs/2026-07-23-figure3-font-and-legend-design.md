# Figure 3 字号与图例精简设计

## 目标

调整论文 Figure 3（Long-horizon anti-collapse）的生成代码，使图插入 AAAI
正文并缩放到 `\textwidth` 后，图中文字接近正文的 10 pt，而不是当前约
4--6 pt。同时删去四个图例方法名中的括号及括号内容。

## 修改范围

- 生成代码：`paper/plot_pouring_lifttray_seeds.py`
- 生成产物：`paper/figure/fig_long_horizon_anti_collapse_multiseed_curves.pdf`
- 不修改曲线数据、颜色、线型、面板顺序、坐标范围或论文 caption。

## 设计

当前 PDF 自然宽度约 1183 pt，插入 7 英寸正文宽度时缩放系数约为 0.426。
因此将源图中主要字号从 10--13 pt 提高到约 21--23 pt，使最终字号约为
9--10 pt。保留现有 16.6 英寸画布，避免改变曲线和面板的相对几何布局。

图例标签统一为：

- `SHORE-RL`
- `Residual RL`
- `DSRL`
- `IQL`

## 验证

重新运行生成脚本，确认 PDF 成功输出；将 PDF 转为图片进行视觉检查，并编译
`main.tex`，确认 Figure 3 在论文中的版面没有溢出或文字遮挡。
