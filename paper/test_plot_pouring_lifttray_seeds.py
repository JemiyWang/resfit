import ast
from pathlib import Path
import unittest


SOURCE = Path(__file__).with_name("plot_pouring_lifttray_seeds.py")
EXPECTED_IBRL_RUNS = {
    "Pouring": [
        ("dexmg_formal", "u3mobgtb"),
        ("dexmg_formal", "uqsl54zu"),
    ],
    "LiftTray": [
        ("dexmg_formal", "905ud33j"),
        ("dexmg_formal", "boluepp0"),
    ],
    "ThreePiece": [
        ("dexmg_formal", "p7uomccw"),
        ("dexmg_formal", "bv1avdba"),
        ("dexmg_formal", "8i3b9r53"),
    ],
    "Threading": [
        ("dexmg_formal", "mvxv2vgt"),
        ("dexmg_formal", "pts8ariy"),
    ],
    "CanSort": [
        ("dexmg_formal", "z4ob6395"),
        ("dexmg_formal", "i6f0pdnm"),
    ],
}


def extract_ibrl_runs(text):
    tree = ast.parse(text)
    panels_node = next(
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "PANELS"
            for target in node.targets
        )
    )
    runs = {}
    for task_key, task_groups in zip(panels_node.keys, panels_node.values):
        task = ast.literal_eval(task_key)
        for group_key, group_runs in zip(task_groups.keys, task_groups.values):
            if ast.literal_eval(group_key) == "ibrl":
                runs[task] = ast.literal_eval(group_runs)
    return runs


class IBRLPlotContractTest(unittest.TestCase):
    def test_exact_canonical_ids_and_common_history_fields(self):
        text = SOURCE.read_text()
        actual = extract_ibrl_runs(text)
        self.assertEqual(actual, EXPECTED_IBRL_RUNS)
        flattened = [run for task_runs in actual.values() for run in task_runs]
        self.assertEqual(len(flattened), 11)
        self.assertEqual(len(set(flattened)), 11)
        self.assertIn('"score/score"', text)
        self.assertIn('"other/step"', text)
        self.assertIn(
            'r.scan_history(keys=["other/step", "score/score"]',
            text,
        )

    def test_style_draw_order_and_output_override(self):
        text = SOURCE.read_text()
        self.assertRegex(text, r'"ibrl"\s*:\s*\{"label":"IBRL"')
        self.assertRegex(text, r'DRAW_ORDER\s*=.*"ibrl"')
        self.assertIn('"--output"', text)

    def test_output_override_has_no_unconditional_png_side_effect(self):
        text = SOURCE.read_text()
        self.assertNotIn("out_png", text)


if __name__ == "__main__":
    unittest.main()
