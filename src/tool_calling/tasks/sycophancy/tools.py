import json
import sys
from pathlib import Path

_utils = Path(__file__).resolve().parents[4] / "src" / "utils"
if str(_utils) not in sys.path:
    sys.path.insert(0, str(_utils))

_OUTPUT_DIR = Path(".")

_session = {
    "model": None,
    "tokenizer": None,
    "dtype": None,
    "model_path": None,
    "model_config": None,
    "activations": None,
    "probe_results": {},
}


def set_output_dir(path: Path):
    global _OUTPUT_DIR
    _OUTPUT_DIR = Path(path)


def load_model(model_path: str) -> str:
    """Load model and tokenizer with conservative bfloat16 policy, device_map=auto."""
    import gpu_memory
    model, tokenizer, dtype = gpu_memory.load_model_conservatively(model_path)
    _session.update({"model": model, "tokenizer": tokenizer, "dtype": dtype, "model_path": model_path})
    return f"Model '{model_path}' loaded (dtype={dtype})"


def cleanup_model() -> str:
    """Free GPU memory. Call between models in cross-model analysis."""
    if _session["model"] is None:
        return "error: no model loaded — call load_model first"
    import gpu_memory
    gpu_memory.safe_cleanup(model=_session["model"], tokenizer=_session["tokenizer"])
    _session.update({"model": None, "tokenizer": None, "dtype": None,
                     "model_path": None, "model_config": None, "activations": None})
    return "Model cleaned up, GPU memory released"


TOOLS = {
    "load_model": load_model,
    "cleanup_model": cleanup_model,
}
