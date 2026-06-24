import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src" / "utils"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src" / "tool_calling" / "tasks" / "sycophancy"))


def _get_tools():
    import importlib, importlib.util
    p = Path(__file__).resolve().parent.parent.parent / "src" / "tool_calling" / "tasks" / "sycophancy" / "tools.py"
    spec = importlib.util.spec_from_file_location("sycophancy_tools", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _reset(mod):
    mod._session.update({
        "model": None, "tokenizer": None, "dtype": None,
        "model_path": None, "model_config": None,
        "activations": None, "probe_results": {},
    })


def test_cleanup_without_model_returns_error():
    mod = _get_tools()
    _reset(mod)
    assert "error" in mod.cleanup_model().lower()


def test_load_model_populates_session():
    mod = _get_tools()
    _reset(mod)
    mock_model = MagicMock()
    mock_tok = MagicMock()
    with patch("gpu_memory.load_model_conservatively", return_value=(mock_model, mock_tok, "bfloat16")):
        result = mod.load_model("HuggingFaceTB/SmolLM2-135M")
    assert mod._session["model"] is mock_model
    assert mod._session["model_path"] == "HuggingFaceTB/SmolLM2-135M"
    assert "loaded" in result.lower()


def test_cleanup_clears_session():
    mod = _get_tools()
    mod._session["model"] = MagicMock()
    mod._session["tokenizer"] = MagicMock()
    with patch("gpu_memory.safe_cleanup"):
        result = mod.cleanup_model()
    assert mod._session["model"] is None
    assert "error" not in result.lower()
