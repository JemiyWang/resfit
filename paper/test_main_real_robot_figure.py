from pathlib import Path


SOURCE = Path(__file__).with_name("main.tex")


def test_real_robot_learning_curves_figure_uses_reduced_width():
    source = SOURCE.read_text()
    label = r"\label{fig:realrobot-curves}"
    block_start = source.rindex(r"\begin{figure}", 0, source.index(label))
    block_end = source.index(r"\end{figure}", source.index(label))
    block = source[block_start:block_end]

    assert r"\includegraphics[width=\columnwidth]{figure/fig_real_robot_curves.pdf}" in block


def test_real_robot_learning_curves_caption_matches_method_colors():
    source = SOURCE.read_text()
    label = r"\label{fig:realrobot-curves}"
    block_start = source.rindex(r"\begin{figure}", 0, source.index(label))
    block_end = source.index(r"\end{figure}", source.index(label))
    block = source[block_start:block_end]

    assert "green is SHORE-RL, purple is ResFit, and gray is the frozen base" in block
    assert "blue is ResFit" not in block
