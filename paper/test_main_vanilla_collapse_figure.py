from pathlib import Path


SOURCE = Path(__file__).with_name("main.tex")


def test_vanilla_collapse_figure_precedes_framework_and_preliminaries():
    source = SOURCE.read_text()
    figure_start = source.index(
        r"\includegraphics[width=\columnwidth]{figure/fig_vanilla_collapse.pdf}"
    )
    framework_start = source.index(
        r"\includegraphics[width=\textwidth]{figure/framework_outlined.pdf}"
    )
    assert r"{figure/framework.pdf}" not in source
    vanilla_reference = source.index(r"\ref{fig:vanilla-collapse}")
    framework_reference = source.index(r"\ref{fig:framework}")

    assert figure_start < framework_start
    assert figure_start < source.index(r"\section{Preliminaries}")
    assert figure_start < source.index(r"\label{fig:simtask}")
    assert vanilla_reference < framework_reference
    assert r"\label{fig:vanilla-collapse}" in source


def test_vanilla_collapse_caption_describes_examples_and_aggregate_lengths():
    source = SOURCE.read_text()
    caption = " ".join(source[
        source.index(r"\caption{Effective Horizon-dependent collapse")
        : source.index(r"\label{fig:vanilla-collapse}")
    ].split())

    for text in (
        r"RoboMimicGen \textsc{Can}",
        r"DexMimicGen \textsc{PieceAssembly}",
        "four shorter-horizon tasks",
        "(green; mean successful trajectory: 121 steps)",
        "four longer-horizon tasks",
        "(red; 298 steps)",
    ):
        assert text in caption

    assert r"\textsc{Square}" not in caption
    assert r"\textsc{Pouring}" not in caption


def test_frozen_base_residual_definition_is_numbered_and_labeled():
    source = SOURCE.read_text()
    start = source.index(r"\paragraph{Frozen-Base Residual Adaptation.}")
    end = source.index(r"\paragraph{Subgoal-Guided Policy Learning.}")
    block = source[start:end]

    assert r"\begin{equation}" in block
    assert r"\begin{equation*}" not in block
    assert r"\label{eq:frozen-base-residual-adaptation}" in block
