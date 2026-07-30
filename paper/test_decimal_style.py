import re
from pathlib import Path


PAPER_DIR = Path(__file__).parent
SOURCES = (
    PAPER_DIR / "main.tex",
    PAPER_DIR / "aaai2027-unified-supp.tex",
)
BARE_DECIMAL = re.compile(r"(?<![A-Za-z0-9])\.[0-9]+")


def test_numeric_decimals_have_leading_zeros():
    violations = []
    for source in SOURCES:
        for line_number, line in enumerate(source.read_text().splitlines(), start=1):
            for match in BARE_DECIMAL.finditer(line):
                violations.append(
                    f"{source.name}:{line_number}: {match.group(0)}"
                )

    assert not violations, "Decimals without leading zeros:\n" + "\n".join(violations)
