"""Regression tests for A2 (--potential_source hiql_subgoal) training env shaping fix.

Critical bug: when potential_source="hiql_subgoal", the training ChunkResidualEnvWrapper
was constructed with reward_shaping_mode="potential" (shaping_mode unchanged), causing
unwanted stage-PBS shaping from the wrapper on top of the main-loop gc shaping.

Fix: train_env_shaping_mode() forces reward_shaping_mode="none" for hiql_subgoal so the
wrapper adds zero shaping and the main loop owns ALL gc shaping.
"""
import pytest

from resfit.rl_finetuning.chunk_residual.train_chunk_residual import train_env_shaping_mode


class TestTrainEnvShapingMode:
    """Pin the seam that had no runnable coverage."""

    def test_a2_hiql_subgoal_forces_none(self):
        """A2: training env must NOT apply wrapper stage shaping when gc handles it."""
        assert train_env_shaping_mode("hiql_subgoal", "potential") == "none"

    def test_a2_hiql_subgoal_forces_none_regardless_of_shaping_mode(self):
        """Even if shaping_mode were something else, hiql_subgoal always forces none."""
        assert train_env_shaping_mode("hiql_subgoal", "none") == "none"
        assert train_env_shaping_mode("hiql_subgoal", "stage") == "none"

    def test_stage_source_passthrough(self):
        """Non-A2 stage source: shaping_mode passes through unchanged (bit-equivalent)."""
        assert train_env_shaping_mode("stage", "none") == "none"
        assert train_env_shaping_mode("stage", "potential") == "potential"

    def test_hiql_source_passthrough(self):
        """Non-A2 hiql source (③b): shaping_mode passes through unchanged."""
        assert train_env_shaping_mode("hiql", "potential") == "potential"
        assert train_env_shaping_mode("hiql", "none") == "none"

    def test_unknown_source_passthrough(self):
        """Any unrecognised source gets pass-through (safe default)."""
        assert train_env_shaping_mode("unknown_future_source", "potential") == "potential"
