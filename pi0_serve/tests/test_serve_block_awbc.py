from __future__ import annotations

import dataclasses
import sys
from types import SimpleNamespace

import serve_block_awbc as serve


@dataclasses.dataclass(frozen=True)
class _Assets:
    asset_id: str = "inference"


@dataclasses.dataclass(frozen=True)
class _Data:
    assets: _Assets = dataclasses.field(default_factory=_Assets)


@dataclasses.dataclass(frozen=True)
class _Config:
    data: _Data = dataclasses.field(default_factory=_Data)


def test_create_policy_replaces_only_asset_id(monkeypatch):
    original = _Config()
    captured = {}

    def make_awbc_config(**kwargs):
        captured["config_kwargs"] = kwargs
        return original

    def create_trained_policy(config, checkpoint_dir, *, default_prompt):
        captured["config"] = config
        captured["checkpoint_dir"] = checkpoint_dir
        captured["default_prompt"] = default_prompt
        return object()

    monkeypatch.setitem(
        sys.modules,
        "pick_cup_configs",
        SimpleNamespace(make_awbc_config=make_awbc_config),
    )
    monkeypatch.setattr(
        serve._policy_config,
        "create_trained_policy",
        create_trained_policy,
    )
    args = serve.Args(
        dir="/checkpoints/paper/19999",
        repo_id="/data/paper_success",
        config_name="pi05_paper_awbc",
        default_prompt="put the paper roll on the holder",
        asset_id="pick_paper_all_merged",
    )

    serve._create_policy(args)

    assert captured["config"].data.assets.asset_id == "pick_paper_all_merged"
    assert original.data.assets.asset_id == "inference"
    assert dataclasses.replace(
        captured["config"],
        data=dataclasses.replace(
            captured["config"].data,
            assets=original.data.assets,
        ),
    ) == original
