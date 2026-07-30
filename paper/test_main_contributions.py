from pathlib import Path


SOURCE = Path(__file__).with_name("main.tex")


def test_contributions_prioritize_method_generality_and_validation():
    source = SOURCE.read_text()
    introduction = source[
        source.index(r"\section{Introduction}") :
        source.index(r"\section{Related Work}")
    ]
    start = introduction.index("Our main contributions are threefold:")
    end = introduction.index(r"\end{itemize}", start)
    block = introduction[start:end]
    normalized = " ".join(block.split())

    assert r"\paragraph{Contributions.}" not in introduction
    assert (
        "same residual objective. Our main contributions are threefold:"
        in introduction
    )
    assert r"\begin{itemize}" in block
    assert block.count(r"\item") == 3
    assert r"\begin{enumerate}" not in block
    assert r"\textbf{Two-axis effective-horizon shortening.}" not in block
    assert r"\textbf{Plug-in adaptation across frozen base-policy families.}" not in block
    assert r"\textbf{Validation in simulation and on real robots.}" not in block
    assert "An empirical long-horizon failure diagnosis." not in block
    assert "learned latent waypoints" in block
    assert r"ACT and $\pi_0$" in block
    assert "three real-robot tasks" in normalized
