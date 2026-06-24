#!/usr/bin/env python3
"""
Train MHA, MLP, and residual stream probes matching the paper's setup:
  - PyTorch nn.Linear probes with BCEWithLogitsLoss
  - Adam optimizer, lr=0.001, 25 epochs, batch size 25, 80/20 split

Activation extraction uses hooks registered by sycophancy_model_registry.

Usage (from agent):
    from sycophancy_probes import extract_and_train_all, save_probe_results
    results = extract_and_train_all(model, tokenizer, texts, labels, model_config)
    save_probe_results(results, "final_probe")
"""
import gc
import json
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam


# ---------------------------------------------------------------------------
# Probe model
# ---------------------------------------------------------------------------

class LinearProbe(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_probe(
    X: np.ndarray,
    y: np.ndarray,
    n_epochs: int = 25,
    batch_size: int = 25,
    lr: float = 0.001,
    train_frac: float = 0.8,
) -> dict:
    """
    Train a single linear probe on activations X with binary labels y.

    Returns dict with keys: accuracy, train_accuracy, model_state, input_dim.
    """
    n = len(X)
    split = int(n * train_frac)
    idx = np.random.permutation(n)
    train_idx, test_idx = idx[:split], idx[split:]

    X_train = torch.FloatTensor(X[train_idx])
    y_train = torch.FloatTensor(y[train_idx])
    X_test = torch.FloatTensor(X[test_idx])
    y_test = torch.FloatTensor(y[test_idx])

    probe = LinearProbe(X.shape[-1])
    optimizer = Adam(probe.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    probe.train()
    for _ in range(n_epochs):
        perm = torch.randperm(len(X_train))
        for start in range(0, len(X_train), batch_size):
            batch_idx = perm[start:start + batch_size]
            logits = probe(X_train[batch_idx])
            loss = criterion(logits, y_train[batch_idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    probe.eval()
    with torch.no_grad():
        test_preds = (probe(X_test) > 0).float()
        test_acc = (test_preds == y_test).float().mean().item()
        train_preds = (probe(X_train) > 0).float()
        train_acc = (train_preds == y_train).float().mean().item()

    return {
        "accuracy": test_acc,
        "train_accuracy": train_acc,
        "model_state": probe.state_dict(),
        "input_dim": X.shape[-1],
    }


# ---------------------------------------------------------------------------
# Per-component probe training
# ---------------------------------------------------------------------------

def train_mha_probes(
    mha_activations: np.ndarray,
    labels: np.ndarray,
    n_layers: int,
    n_heads: int,
    **probe_kwargs,
) -> dict:
    """
    Train one probe per (layer, head).

    Args:
        mha_activations: shape (n_layers, n_heads, n_examples, head_dim)
        labels: shape (n_examples,)

    Returns:
        accuracy_dict: {(layer, head): accuracy}
    """
    accuracy_dict = {}
    for layer in range(n_layers):
        for head in range(n_heads):
            X = mha_activations[layer, head]   # (n_examples, head_dim)
            result = train_probe(X, labels, **probe_kwargs)
            accuracy_dict[(layer, head)] = result["accuracy"]
            print(f"  MHA layer={layer} head={head}: acc={result['accuracy']:.3f}")
    return accuracy_dict


def train_mlp_probes(
    mlp_activations: np.ndarray,
    labels: np.ndarray,
    n_layers: int,
    **probe_kwargs,
) -> dict:
    """
    Train one probe per MLP layer.

    Args:
        mlp_activations: shape (n_layers, n_examples, hidden_dim)
        labels: shape (n_examples,)

    Returns:
        accuracy_dict: {layer: accuracy}
    """
    accuracy_dict = {}
    for layer in range(n_layers):
        X = mlp_activations[layer]   # (n_examples, hidden_dim)
        result = train_probe(X, labels, **probe_kwargs)
        accuracy_dict[layer] = result["accuracy"]
        print(f"  MLP layer={layer}: acc={result['accuracy']:.3f}")
    return accuracy_dict


def train_residual_probes(
    residual_activations: np.ndarray,
    labels: np.ndarray,
    n_layers: int,
    **probe_kwargs,
) -> dict:
    """
    Train one probe per residual stream layer.

    Args:
        residual_activations: shape (n_layers, n_examples, hidden_dim)
        labels: shape (n_examples,)

    Returns:
        accuracy_dict: {layer: accuracy}
    """
    accuracy_dict = {}
    for layer in range(n_layers):
        X = residual_activations[layer]   # (n_examples, hidden_dim)
        result = train_probe(X, labels, **probe_kwargs)
        accuracy_dict[layer] = result["accuracy"]
        print(f"  Residual layer={layer}: acc={result['accuracy']:.3f}")
    return accuracy_dict


# ---------------------------------------------------------------------------
# Activation extraction
# ---------------------------------------------------------------------------

def collect_activations(
    model,
    tokenizer,
    texts: list,
    model_config: dict,
    batch_size: int = 1,
) -> dict:
    """
    Run forward passes with hooks to collect MHA, MLP, and residual activations.

    Returns dict with keys "mha", "mlp", "residual":
      - "mha":      (n_layers, n_heads, n_examples, head_dim)
      - "mlp":      (n_layers, n_examples, mlp_dim)  [down_proj output dim = hidden_dim]
      - "residual": (n_layers, n_examples, hidden_dim)
    """
    from sycophancy_model_registry import register_hooks, remove_hooks

    n_layers = model_config["n_layers"]
    n_heads = model_config["n_heads"]
    head_dim = model_config["head_dim"]
    hidden_dim = model_config["hidden_dim"]
    answer_token_id = model_config.get("answer_token_id")

    all_mha = []    # list of (n_layers, n_heads, head_dim) per example
    all_mlp = []    # list of (n_layers, hidden_dim) per example
    all_res = []    # list of (n_layers, hidden_dim) per example

    handles, activation_store = register_hooks(model, model_config)

    model.eval()
    with torch.no_grad():
        for i, text in enumerate(texts):
            activation_store["mha"].clear()
            activation_store["mlp"].clear()

            inputs = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=1024,
            )
            input_ids = inputs["input_ids"]
            if torch.cuda.is_available():
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

            outputs = model(**inputs, output_hidden_states=True)

            # Find answer token position (last occurrence of answer_token_id)
            if answer_token_id is not None:
                token_list = input_ids[0].tolist()
                positions = [j for j, t in enumerate(token_list) if t == answer_token_id]
                pos = positions[-1] if positions else -1
            else:
                pos = -1   # fallback: use last token

            # MHA: (n_layers, n_heads * head_dim) → reshape to (n_layers, n_heads, head_dim)
            mha_example = np.zeros((n_layers, n_heads, head_dim), dtype=np.float32)
            for layer_idx, act in activation_store["mha"].items():
                # act shape: (1, seq_len, n_heads * head_dim)
                vec = act[0, pos, :].numpy().astype(np.float32)
                mha_example[layer_idx] = vec.reshape(n_heads, head_dim)
            all_mha.append(mha_example)

            # MLP: (n_layers, hidden_dim)
            mlp_example = np.zeros((n_layers, hidden_dim), dtype=np.float32)
            for layer_idx, act in activation_store["mlp"].items():
                # act shape: (1, seq_len, hidden_dim)
                mlp_example[layer_idx] = act[0, pos, :].numpy().astype(np.float32)
            all_mlp.append(mlp_example)

            # Residual: hidden_states is tuple of (1, seq_len, hidden_dim) per layer
            res_example = np.zeros((n_layers, hidden_dim), dtype=np.float32)
            hidden_states = outputs.hidden_states  # tuple length = n_layers + 1
            for layer_idx in range(n_layers):
                hs = hidden_states[layer_idx + 1][0, pos, :].cpu().numpy().astype(np.float32)
                res_example[layer_idx] = hs
            all_res.append(res_example)

            if (i + 1) % 10 == 0:
                print(f"  Extracted {i+1}/{len(texts)} examples")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    remove_hooks(handles)

    # Stack: (n_examples, n_layers, ...) → transpose to (n_layers, ...)
    mha_arr = np.stack(all_mha, axis=0)          # (n_examples, n_layers, n_heads, head_dim)
    mha_arr = mha_arr.transpose(1, 2, 0, 3)      # (n_layers, n_heads, n_examples, head_dim)

    mlp_arr = np.stack(all_mlp, axis=0)          # (n_examples, n_layers, hidden_dim)
    mlp_arr = mlp_arr.transpose(1, 0, 2)         # (n_layers, n_examples, hidden_dim)

    res_arr = np.stack(all_res, axis=0)          # (n_examples, n_layers, hidden_dim)
    res_arr = res_arr.transpose(1, 0, 2)         # (n_layers, n_examples, hidden_dim)

    return {"mha": mha_arr, "mlp": mlp_arr, "residual": res_arr}


def extract_and_train_all(
    model,
    tokenizer,
    texts: list,
    labels: np.ndarray,
    model_config: dict,
    batch_size: int = 1,
) -> dict:
    """
    Extract activations and train all three probe types.

    Returns dict with keys "mha_accuracy", "mlp_accuracy", "residual_accuracy"
    each mapping to their respective accuracy dicts.
    """
    n_layers = model_config["n_layers"]
    n_heads = model_config["n_heads"]

    print("Extracting activations...")
    activations = collect_activations(model, tokenizer, texts, model_config, batch_size)

    print(f"\nTraining MHA probes ({n_layers} layers × {n_heads} heads)...")
    mha_acc = train_mha_probes(activations["mha"], labels, n_layers, n_heads)

    print(f"\nTraining MLP probes ({n_layers} layers)...")
    mlp_acc = train_mlp_probes(activations["mlp"], labels, n_layers)

    print(f"\nTraining residual probes ({n_layers} layers)...")
    res_acc = train_residual_probes(activations["residual"], labels, n_layers)

    return {
        "mha_accuracy": mha_acc,
        "mlp_accuracy": mlp_acc,
        "residual_accuracy": res_acc,
    }


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def save_probe_results(results: dict, output_dir: str, model_name: str = ""):
    """
    Save accuracy dicts as pickles and write a summary JSON.

    Structure written to output_dir/:
        mha_accuracy.pkl
        mlp_accuracy.pkl
        residual_accuracy.pkl
        probe_metadata.json
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "mha_accuracy.pkl", "wb") as f:
        pickle.dump(results["mha_accuracy"], f)
    with open(out / "mlp_accuracy.pkl", "wb") as f:
        pickle.dump(results["mlp_accuracy"], f)
    with open(out / "residual_accuracy.pkl", "wb") as f:
        pickle.dump(results["residual_accuracy"], f)

    mha_best = max(results["mha_accuracy"].values()) if results["mha_accuracy"] else 0.0
    mlp_best = max(results["mlp_accuracy"].values()) if results["mlp_accuracy"] else 0.0
    res_best = max(results["residual_accuracy"].values()) if results["residual_accuracy"] else 0.0

    mha_best_key = max(results["mha_accuracy"], key=results["mha_accuracy"].get) if results["mha_accuracy"] else None
    mlp_best_key = max(results["mlp_accuracy"], key=results["mlp_accuracy"].get) if results["mlp_accuracy"] else None
    res_best_key = max(results["residual_accuracy"], key=results["residual_accuracy"].get) if results["residual_accuracy"] else None

    metadata = {
        "model_name": model_name,
        "mha_best_accuracy": round(mha_best, 4),
        "mha_best_key": list(mha_best_key) if mha_best_key is not None else None,
        "mlp_best_accuracy": round(mlp_best, 4),
        "mlp_best_key": mlp_best_key,
        "residual_best_accuracy": round(res_best, 4),
        "residual_best_key": res_best_key,
    }
    with open(out / "probe_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nProbe results saved to {out}/")
    print(f"  MHA best:      {mha_best:.3f} at {mha_best_key}")
    print(f"  MLP best:      {mlp_best:.3f} at layer {mlp_best_key}")
    print(f"  Residual best: {res_best:.3f} at layer {res_best_key}")

    return metadata


def load_probe_results(probe_dir: str) -> dict:
    """Load saved accuracy dicts from probe_dir/."""
    probe_path = Path(probe_dir)
    results = {}

    for key, filename in [
        ("mha_accuracy", "mha_accuracy.pkl"),
        ("mlp_accuracy", "mlp_accuracy.pkl"),
        ("residual_accuracy", "residual_accuracy.pkl"),
    ]:
        path = probe_path / filename
        if path.exists():
            with open(path, "rb") as f:
                results[key] = pickle.load(f)
        else:
            results[key] = {}

    metadata_path = probe_path / "probe_metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            results["metadata"] = json.load(f)

    return results
