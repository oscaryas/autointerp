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
                     "model_path": None, "model_config": None, "activations": None,
                     "probe_results": {}})
    return "Model cleaned up, GPU memory released"


def inspect_model() -> dict:
    """
    Auto-discover architecture: n_layers, n_heads, hidden_dim, head_dim, mlp_dim,
    mha/mlp hook patterns and full hook paths. head_dim read from actual weight shape
    to handle grouped-query attention. Fails closed if hooks are missing.
    """
    if _session["model"] is None:
        return "error: no model loaded — call load_model first"

    model = _session["model"]
    mha_hook_paths, mlp_hook_paths = [], []
    hidden_dim = head_input_dim = mlp_dim = None

    for name, module in model.named_modules():
        if name.endswith("self_attn.o_proj"):
            mha_hook_paths.append(name)
            if hidden_dim is None:
                hidden_dim = module.out_features
                head_input_dim = module.in_features
        if name.endswith("mlp.down_proj"):
            mlp_hook_paths.append(name)
            if mlp_dim is None:
                mlp_dim = module.in_features

    if not mha_hook_paths:
        raise RuntimeError("inspect_model: no 'self_attn.o_proj' modules found")
    if not mlp_hook_paths:
        raise RuntimeError("inspect_model: no 'mlp.down_proj' modules found")

    n_layers = len(mha_hook_paths)
    if len(mlp_hook_paths) != n_layers:
        raise RuntimeError(
            f"inspect_model: MHA hooks ({n_layers}) != MLP hooks ({len(mlp_hook_paths)})"
        )

    cfg = model.config
    if hasattr(cfg, "text_config"):
        cfg = cfg.text_config
    n_heads = getattr(cfg, "num_attention_heads", None)
    if n_heads is None:
        raise RuntimeError("inspect_model: cannot read num_attention_heads from model.config")

    config = {
        "n_layers": n_layers,
        "n_heads": n_heads,
        "hidden_dim": hidden_dim,
        "head_dim": head_input_dim // n_heads,
        "mlp_dim": mlp_dim,
        "mha_hook": "self_attn.o_proj",
        "mlp_hook": "mlp.down_proj",
        "mha_hook_paths": mha_hook_paths,
        "mlp_hook_paths": mlp_hook_paths,
        "residual_available": True,
    }
    _session["model_config"] = config
    return config


def get_answer_token_id(sample_prompt: str) -> int:
    """
    Return the delimiter token ID used by the existing activation extractor to
    choose an answer position. Despite the name, this is not always the first
    generated answer token. Checks for <end_of_turn> (Gemma), <|eot_id|>
    (Llama), <|im_end|> (Qwen) in that order. Falls back to eos_token_id.
    """
    if _session["tokenizer"] is None:
        return "error: no model loaded — call load_model first"
    tokenizer = _session["tokenizer"]
    unk_id = getattr(tokenizer, "unk_token_id", -1)
    for token_str in ["<end_of_turn>", "<|eot_id|>", "<|im_end|>"]:
        encoded = tokenizer.encode(token_str, add_special_tokens=False)
        if len(encoded) == 1 and encoded[0] != unk_id:
            return encoded[0]
    return tokenizer.eos_token_id


TOOLS = {
    "load_model": load_model,
    "cleanup_model": cleanup_model,
    "inspect_model": inspect_model,
    "get_answer_token_id": get_answer_token_id,
}
