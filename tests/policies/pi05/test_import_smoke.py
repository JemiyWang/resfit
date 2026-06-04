# tests/policies/pi05/test_import_smoke.py
import importlib
import sys
from pathlib import Path

KAI0_ROOT = Path("/data2/kai0")  # provide package root to import resfit_pi05.*


def test_websocket_path_imports_without_flax():
    """websocket cross-process prerequisite: residual side can import client + adapter without flax."""
    if str(KAI0_ROOT) not in sys.path:
        sys.path.insert(0, str(KAI0_ROOT))

    # lightweight client (importable after pip install; no flax/torch in deps)
    from openpi_client.websocket_client_policy import WebsocketClientPolicy  # noqa: F401

    # adapter module top is clean; the from_policy path does not import openpi.training.config
    mod = importlib.import_module("resfit_pi05.pi05_policy_adapter")
    assert hasattr(mod, "Pi05PolicyAdapter")

    # key assertion: importing the adapter must NOT pull flax in
    assert "flax" not in sys.modules, "importing adapter triggered flax; the from_policy path should not"
