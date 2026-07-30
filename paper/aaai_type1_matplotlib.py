"""Configure Matplotlib PDF figures for AAAI-compatible Type 1 fonts."""

import matplotlib


def configure_aaai_type1_matplotlib():
    """Use PGF/pdflatex with the Type 1 New TX text font."""
    matplotlib.use("pgf")
    matplotlib.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "pgf.rcfonts": False,
            "pgf.preamble": "\n".join(
                (
                    r"\usepackage[T1]{fontenc}",
                    r"\usepackage{newtxtext}",
                )
            ),
            "font.family": "serif",
            "text.usetex": True,
            "axes.unicode_minus": False,
        }
    )
