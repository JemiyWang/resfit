from pathlib import Path


SOURCE = Path(__file__).with_name("main.tex")


def test_algorithm_float_can_fill_current_page_before_fallbacks():
    source = SOURCE.read_text()
    algorithm = source.index(r"\label{alg:shore-unified}")
    explanation = source.index(
        r"Algorithm~\ref{alg:shore-unified} separates venue-specific collection"
    )
    float_start = source.rfind(r"\begin{figure}", 0, algorithm)

    assert source[float_start:algorithm].startswith(r"\begin{figure}[!htbp]")
    assert algorithm < explanation
