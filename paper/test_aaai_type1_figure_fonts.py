from pathlib import Path


PAPER = Path(__file__).resolve().parent
PLOT_SCRIPTS = (
    "plot_vanilla_collapse.py",
    "plot_vanilla_collapse_composite.py",
    "plot_pouring_lifttray_seeds.py",
    "align_libero_figure.py",
    "plot_ablation_bars_400k.py",
    "plot_staged_vs_pothiql_threading_piece_v2.py",
    "plot_real_robot_curves.py",
)


def test_plot_scripts_configure_pgf_before_importing_pyplot():
    for filename in PLOT_SCRIPTS:
        source = (PAPER / filename).read_text()
        assert (
            "from aaai_type1_matplotlib import configure_aaai_type1_matplotlib"
            in source
        ), filename
        assert "configure_aaai_type1_matplotlib()" in source, filename
        assert source.index("configure_aaai_type1_matplotlib()") < source.index(
            "import matplotlib.pyplot"
        ), filename


def test_plot_scripts_do_not_request_forbidden_dejavu_or_truetype_output():
    for filename in PLOT_SCRIPTS:
        source = (PAPER / filename).read_text()
        assert "DejaVu" not in source, filename
        assert "pdf.fonttype" not in source, filename
        assert "ps.fonttype" not in source, filename


def test_shared_configuration_uses_pgf_pdflatex_and_type1_newtx_text():
    source = (PAPER / "aaai_type1_matplotlib.py").read_text()
    assert 'matplotlib.use("pgf")' in source
    assert '"pgf.texsystem": "pdflatex"' in source
    assert r"\usepackage[T1]{fontenc}" in source
    assert r"\usepackage{newtxtext}" in source
