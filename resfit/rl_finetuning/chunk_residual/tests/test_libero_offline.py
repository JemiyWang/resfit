import pytest

from resfit.rl_finetuning.chunk_residual.libero_offline import libero_task_language


def test_libero_task_language_moka():
    pytest.importorskip("libero")   # 仅此测试需真 libero;residual env / 无 PYTHONPATH 时优雅跳过
    lang = libero_task_language("libero_10", 8)
    assert lang == "put both moka pots on the stove"
