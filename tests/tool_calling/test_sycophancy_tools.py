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


def test_inspect_model_no_model_error():
    mod = _get_tools()
    _reset(mod)
    assert "error" in str(mod.inspect_model()).lower()


def test_inspect_model_returns_config():
    import torch.nn as nn
    mod = _get_tools()
    _reset(mod)

    class FakeLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = nn.ModuleDict({"o_proj": nn.Linear(64, 64)})
            self.mlp = nn.ModuleDict({"down_proj": nn.Linear(128, 64)})

    class Config:
        num_attention_heads = 4

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([FakeLayer(), FakeLayer()])
            self.config = Config()

    mod._session["model"] = FakeModel()
    config = mod.inspect_model()
    assert isinstance(config, dict), config
    assert config["n_layers"] == 2
    assert config["n_heads"] == 4
    assert config["mha_hook"] == "self_attn.o_proj"
    assert len(config["mha_hook_paths"]) == 2
    assert mod._session["model_config"] is config


def test_get_answer_token_id_no_model_error():
    mod = _get_tools()
    _reset(mod)
    assert "error" in str(mod.get_answer_token_id("hello")).lower()


def test_get_answer_token_id_falls_back_to_eos():
    mod = _get_tools()
    _reset(mod)
    mock_tok = MagicMock()
    mock_tok.eos_token_id = 2
    mock_tok.unk_token_id = 0
    # Encode returns multi-token for special strings → falls back to eos
    mock_tok.encode.return_value = [1, 2, 3]
    mod._session["tokenizer"] = mock_tok
    assert mod.get_answer_token_id("hello") == 2
