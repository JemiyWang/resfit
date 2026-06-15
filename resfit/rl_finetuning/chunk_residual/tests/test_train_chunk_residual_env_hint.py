"""静态守护:主训 build_offline_buffer 调用必须把 env_hint=args.task 传下去
(否则 pouring/lifttray hdf5 路 offline buffer 会按默认 18 维拼装 → KeyError)。
用 AST 检查而非起真训练(后者需 GPU/serve,过重)。"""
import ast
from pathlib import Path


def _find_call(tree, func_name):
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id == func_name:
            return node
    return None


def test_build_offline_buffer_call_passes_env_hint():
    src = Path("resfit/rl_finetuning/chunk_residual/train_chunk_residual.py").read_text()
    tree = ast.parse(src)
    call = _find_call(tree, "build_offline_buffer")
    assert call is not None, "找不到 build_offline_buffer 调用"
    kw = {k.arg: k.value for k in call.keywords}
    assert "env_hint" in kw, "build_offline_buffer 调用未传 env_hint"
    v = kw["env_hint"]
    # 期望 env_hint=args.task
    assert isinstance(v, ast.Attribute) and v.attr == "task" \
        and isinstance(v.value, ast.Name) and v.value.id == "args", \
        "env_hint 应为 args.task"
