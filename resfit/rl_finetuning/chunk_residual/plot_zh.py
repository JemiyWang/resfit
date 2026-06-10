"""matplotlib 中文字体 helper。

本机/多数 Linux 默认无中文字体,matplotlib 画中文标题会显示成方框 □。本模块把系统里
已有的 CJK 字体(本机=文泉驿 WenQuanYi,留 Noto 兜底)注册进 matplotlib 并设为 sans-serif
首选,同时修负号(axes.unicode_minus=False)。

用法:画图脚本在创建 figure 前调用一次 `use_cjk_font()`。找不到任何 CJK 字体时安静退回
(返回 None,英文仍正常),不抛错、不阻塞。
"""
import os

import matplotlib
from matplotlib import font_manager as fm

# 常见 CJK 字体文件(本机=文泉驿;Noto 作常见兜底)。按顺序尝试注册。
_FONT_FILES = [
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
]

# 偏好顺序(已注册的字体名里挑第一个命中的)
_PREFER = ("WenQuanYi Micro Hei", "WenQuanYi Zen Hei",
           "Noto Sans CJK SC", "Noto Sans CJK JP")


def use_cjk_font(prefer=_PREFER):
    """注册可用 CJK 字体并设为 sans-serif 首选,返回最终用的字体名(无则 None)。"""
    registered = []
    for p in _FONT_FILES:
        if os.path.exists(p):
            try:
                fm.fontManager.addfont(p)
                registered.append(fm.FontProperties(fname=p).get_name())
            except Exception:
                pass
    avail = {f.name for f in fm.fontManager.ttflist}
    pick = next((n for n in prefer if n in avail or n in registered), None)
    if pick is None and registered:
        pick = registered[0]
    if pick is not None:
        base = list(matplotlib.rcParams.get("font.sans-serif", []))
        matplotlib.rcParams["font.sans-serif"] = [pick] + [f for f in base if f != pick]
        matplotlib.rcParams["axes.unicode_minus"] = False
    return pick
