from pathlib import Path
import re
import unittest


SOURCE = Path(__file__).with_name("plot_pouring_lifttray_seeds.py")
EXPECTED_IBRL_IDS = {
    "z4ob6395",
    "i6f0pdnm",
    "905ud33j",
    "boluepp0",
    "u3mobgtb",
    "uqsl54zu",
    "mvxv2vgt",
    "pts8ariy",
    "p7uomccw",
    "bv1avdba",
    "8i3b9r53",
}


class IBRLPlotContractTest(unittest.TestCase):
    def test_exact_canonical_ids_and_common_history_fields(self):
        text = SOURCE.read_text()
        actual = set(
            re.findall(r'\("dexmg_formal","([a-z0-9]{8})"\)', text)
        )
        self.assertEqual(actual, EXPECTED_IBRL_IDS)
        self.assertIn('"score/score"', text)
        self.assertIn('"other/step"', text)

    def test_style_draw_order_and_output_override(self):
        text = SOURCE.read_text()
        self.assertRegex(text, r'"ibrl"\s*:\s*\{"label":"IBRL"')
        self.assertRegex(text, r'DRAW_ORDER\s*=.*"ibrl"')
        self.assertIn('"--output"', text)


if __name__ == "__main__":
    unittest.main()
