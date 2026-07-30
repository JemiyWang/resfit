import re
from pathlib import Path


PAPER_DIR = Path(__file__).parent
MAIN = (PAPER_DIR / "main.tex").read_text()
BIB = (PAPER_DIR / "references.bib").read_text()


def _entry(key: str) -> str:
    match = re.search(
        rf"@\w+\{{{re.escape(key)},.*?\n\}}",
        BIB,
        flags=re.DOTALL,
    )
    assert match, f"Missing bibliography entry: {key}"
    return match.group(0)


def test_aaai_controls_bibliography_style_and_reference_flow():
    assert r"\usepackage{natbib}" in MAIN
    assert not re.search(r"^\s*\\bibliographystyle", MAIN, flags=re.MULTILINE)
    assert r"\bibliography{references}" in MAIN
    assert r"\clearpage" not in MAIN
    assert r"\newpage" not in MAIN


def test_arxiv_entries_use_aaai_misc_export_fields():
    for key, eprint in (
        ("peng2019awr", "1910.00177"),
        ("sima2026kai0", "2602.09021"),
        ("hacohen2025ltxvideo", "2501.00103"),
    ):
        entry = _entry(key)
        assert entry.startswith(f"@misc{{{key},")
        assert f"eprint={{{eprint}}}" in entry
        assert "archivePrefix={arXiv}" in entry
        assert "journal={arXiv preprint" not in entry


def test_resfit_uses_icra_proceedings_metadata():
    entry = _entry("resfit2025residual")
    assert entry.startswith("@inproceedings{resfit2025residual,")
    assert "booktitle={IEEE International Conference on Robotics and Automation (ICRA)}" in entry
    assert "year={2026}" in entry


def test_iclr_booktitle_is_consistent():
    assert "International Conference on Learning Representations (ICLR)" not in BIB
