"""
Smoke test: full sycophancy pipeline without a real LLM.
Uses HuggingFaceTB/SmolLM2-135M. This test is part of the normal tool-calling
test suite and is not opt-in.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "utils"))

pytestmark = pytest.mark.smoke


def load_tools_module(tmp_path):
    p = Path(__file__).resolve().parents[2] / "src" / "tool_calling" / "tasks" / "sycophancy" / "tools.py"
    spec = importlib.util.spec_from_file_location("sycophancy_tools_smoke", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.set_output_dir(tmp_path)
    mod._session.update({
        "model": None, "tokenizer": None, "dtype": None,
        "model_path": None, "model_config": None,
        "activations": None, "probe_results": {},
    })
    return mod


@pytest.fixture
def job_dir(tmp_path):
    ctx = tmp_path / "task_context"
    ctx.mkdir()
    (ctx / "paper_results.json").write_text(json.dumps({
        "HuggingFaceTB/SmolLM2-135M": {
            "mha_best_accuracy": 0.6,
            "mlp_best_accuracy": 0.58,
            "residual_best_accuracy": 0.59,
        }
    }))
    return tmp_path


def test_full_pipeline(job_dir):
    mod = load_tools_module(job_dir)

    # 1. Load model
    r = mod.load_model("HuggingFaceTB/SmolLM2-135M")
    assert "error" not in r.lower(), r

    # 2. Inspect
    config = mod.inspect_model()
    assert isinstance(config, dict), config
    assert config["n_layers"] == 30
    assert config["n_heads"] == 9
    assert config["head_dim"] == 64

    # 3. Answer token
    sample = mod._session["tokenizer"].decode([1, 2, 3])
    token_id = mod.get_answer_token_id(sample)
    assert isinstance(token_id, int)

    # 4. Synthetic dataset (skip real generation)
    n = 40
    with open(job_dir / "behavioral_labels.jsonl", "w") as f:
        for i in range(n):
            f.write(json.dumps({"text": f"example text {i}", "label": i % 2}) + "\n")

    r = mod.generate_behavioral_labels(200, "behavioral_labels.jsonl")
    assert "skip" in r.lower() or "exist" in r.lower(), r

    # 5. Extract activations
    r = mod.extract_activations("behavioral_labels.jsonl", token_id)
    assert "error" not in r.lower(), r
    cache = job_dir / "activations"
    assert (cache / "metadata.json").exists()
    mha = np.load(cache / "mha.npy")
    assert mha.shape == (30, 9, n, 64), f"expected (30, 9, {n}, 64), got {mha.shape}"

    # 6-8. Train all three probe families
    for pt in ("mha", "mlp", "residual"):
        r = mod.train_probe_family(pt)
        assert "error" not in r.lower(), f"{pt}: {r}"

    # 9. Write metrics
    r = mod.write_metrics()
    assert "error" not in r.lower(), r
    assert (job_dir / "metrics.json").exists()
    assert (job_dir / "final_probe" / "probe_metadata.json").exists()

    # 10. Fetch paper results
    r = mod.fetch_paper_results("Sycophancy Hides Linearly in Attention Heads")
    assert "error" not in r.lower(), r
    assert (job_dir / "paper_results.json").exists()

    # 11. Compare
    r = mod.compare_with_paper("metrics.json", "paper_results.json")
    assert "error" not in r.lower(), r
    assert (job_dir / "comparison_table.md").exists()

    # 12. Write analysis
    r = mod.write_analysis("## Smoke Test\n\nPipeline passed.")
    assert "error" not in r.lower(), r
    assert (job_dir / "analysis.md").exists()

    # 13. Cleanup
    r = mod.cleanup_model()
    assert "error" not in r.lower(), r
    assert mod._session["model"] is None
