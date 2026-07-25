import runpy
import sys
import types

import pytest


OURS_IDS = {
    "3nsvrbob",
    "b4yerjy2",
    "yo3rvi0t",
    "e7sntzx9",
    "372ah0gx",
    "qodyz8ea",
    "kmtsayff",
    "m2s74dqm",
    "cgtmwrv3",
    "0cojzxec",
    "cbmgm6c0",
    "s2rbsvxp",
    "9a8903ei",
    "j955v4ah",
}


class FakeRun:
    def __init__(self, run_id):
        self.run_id = run_id

    def history(self, **_kwargs):
        start = 0.8 if self.run_id in OURS_IDS else 0.2
        return [
            {"_step": 0, "eval/success_rate": start},
            {"_step": 10_000, "eval/success_rate": start + 0.05},
        ]

    def scan_history(self, **_kwargs):
        return [
            {"other/step": 0, "score/score": 0.3},
            {"other/step": 10_000, "score/score": 0.35},
        ]


class FakeApi:
    def __init__(self, **_kwargs):
        pass

    def run(self, path):
        return FakeRun(path.rsplit("/", 1)[-1])


def test_figure3_frozen_base_uses_shore_step_zero(monkeypatch, tmp_path):
    fake_wandb = types.SimpleNamespace(Api=FakeApi)
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    output = tmp_path / "figure3.pdf"
    monkeypatch.setattr(
        sys,
        "argv",
        ["plot_pouring_lifttray_seeds.py", "--output", str(output)],
    )

    namespace = runpy.run_path("paper/plot_pouring_lifttray_seeds.py")

    assert output.exists()
    assert all(value == pytest.approx(0.8) for value in namespace["BASE"].values())
    for axis in namespace["axes"]:
        references = [
            line for line in axis.lines if line.get_label() == "_frozen_base"
        ]
        assert len(references) == 1
        assert list(references[0].get_ydata()) == pytest.approx([0.8, 0.8])
    legend_text = [text.get_text() for text in namespace["fig"].legends[0].texts]
    assert legend_text[-1] == "Frozen base"
