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


def test_generate_labels_skips_valid_existing(tmp_path):
    mod = _get_tools()
    _reset(mod)
    mod.set_output_dir(tmp_path)
    existing = tmp_path / "behavioral_labels.jsonl"
    existing.write_text('{"text": "hi", "label": 1}\n{"text": "bye", "label": 0}\n')
    result = mod.generate_behavioral_labels(200, "behavioral_labels.jsonl")
    assert "skip" in result.lower() or "exist" in result.lower()
    assert existing.read_text().startswith('{"text": "hi"')


def test_generate_labels_regenerates_invalid_file(tmp_path):
    mod = _get_tools()
    _reset(mod)
    mod.set_output_dir(tmp_path)
    mod._session["model"] = MagicMock()
    mod._session["tokenizer"] = MagicMock()
    (tmp_path / "behavioral_labels.jsonl").write_text('{"text": "no label field"}\n')
    with patch("sycophancy_data.load_truthfulqa", return_value=[]):
        result = mod.generate_behavioral_labels(10, "behavioral_labels.jsonl")
    assert "error" not in result.lower()


def test_extract_activations_preconditions(tmp_path):
    mod = _get_tools()
    _reset(mod)
    mod.set_output_dir(tmp_path)
    # no model
    assert "error" in mod.extract_activations("x.jsonl", 1).lower()
    # model but no config
    mod._session["model"] = MagicMock()
    assert "inspect_model" in mod.extract_activations("x.jsonl", 1).lower()


def test_extract_activations_writes_cache(tmp_path):
    import numpy as np
    mod = _get_tools()
    _reset(mod)
    mod.set_output_dir(tmp_path)
    (tmp_path / "labels.jsonl").write_text(
        '{"text": "a", "label": 1}\n{"text": "b", "label": 0}\n'
    )
    mod._session["model"] = MagicMock()
    mod._session["tokenizer"] = MagicMock()
    mod._session["model_config"] = {
        "n_layers": 2, "n_heads": 2, "hidden_dim": 8, "head_dim": 4,
        "mlp_dim": 16, "mha_hook": "self_attn.o_proj", "mlp_hook": "mlp.down_proj",
    }
    fake = {
        "mha": np.zeros((2, 2, 2, 4), dtype=np.float32),
        "mlp": np.zeros((2, 2, 8), dtype=np.float32),
        "residual": np.zeros((2, 2, 8), dtype=np.float32),
    }
    with patch("sycophancy_probes.collect_activations", return_value=fake):
        result = mod.extract_activations("labels.jsonl", 105)
    assert "error" not in result.lower(), result
    cache = tmp_path / "activations"
    assert (cache / "metadata.json").exists()
    assert (cache / "mha.npy").exists()
    meta = json.loads((cache / "metadata.json").read_text())
    assert meta["n_examples"] == 2
    assert meta["answer_token_id"] == 105


def test_train_probe_no_activations_error():
    mod = _get_tools()
    _reset(mod)
    result = mod.train_probe_family("mha")
    assert "error" in result.lower() and "extract_activations" in result.lower()


def test_train_probe_invalid_type_error():
    mod = _get_tools()
    _reset(mod)
    mod._session["activations"] = "/some/path"
    assert "error" in mod.train_probe_family("bad_type").lower()


def test_train_probe_mha_from_cache(tmp_path):
    import numpy as np
    mod = _get_tools()
    _reset(mod)
    mod.set_output_dir(tmp_path)
    cache = tmp_path / "activations"
    cache.mkdir()
    n, n_layers, n_heads, head_dim = 20, 2, 2, 4
    np.save(cache / "mha.npy", np.random.rand(n_layers, n_heads, n, head_dim).astype(np.float32))
    np.save(cache / "mlp.npy", np.random.rand(n_layers, n, 8).astype(np.float32))
    np.save(cache / "residual.npy", np.random.rand(n_layers, n, 8).astype(np.float32))
    np.save(cache / "labels.npy", [i % 2 for i in range(n)])
    (cache / "metadata.json").write_text(json.dumps({
        "model_name": "test", "n_examples": n,
        "model_config": {"n_layers": n_layers, "n_heads": n_heads, "hidden_dim": 8, "head_dim": head_dim},
    }))
    mod._session["activations"] = str(cache)
    result = mod.train_probe_family("mha")
    assert "error" not in result.lower(), result
    assert "mha" in mod._session["probe_results"]


def test_write_metrics_requires_all_families(tmp_path):
    mod = _get_tools()
    _reset(mod)
    mod.set_output_dir(tmp_path)
    mod._session["probe_results"] = {"mha": {}, "mlp": {}}
    result = mod.write_metrics()
    assert "error" in result.lower() and "residual" in result.lower()


def test_write_analysis(tmp_path):
    mod = _get_tools()
    _reset(mod)
    mod.set_output_dir(tmp_path)
    result = mod.write_analysis("## Findings\n\nTest.")
    assert "error" not in result.lower()
    assert (tmp_path / "analysis.md").read_text() == "## Findings\n\nTest."


def test_fetch_paper_results_uses_task_context(tmp_path):
    mod = _get_tools()
    _reset(mod)
    mod.set_output_dir(tmp_path)
    ctx = tmp_path / "task_context"
    ctx.mkdir()
    data = {"google/gemma-3-12b-it": {"mha_best_accuracy": 0.87}}
    (ctx / "paper_results.json").write_text(json.dumps(data))
    result = mod.fetch_paper_results("Some Paper")
    assert "error" not in result.lower()
    assert (tmp_path / "paper_results.json").exists()
