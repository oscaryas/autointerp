#!/usr/bin/env python3
"""
Evaluation script for deception probing task.
Validates the agent's probe on behavioral labels, then evaluates
out-of-distribution generalization on Liar's Bench and MASK.
"""
import argparse
import json
import pickle
from pathlib import Path
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
import torch
from transformers import AutoModel, AutoTokenizer


def load_labeled_data(data_file, behavioral_labels_file=None):
    """Load labeled data, preferring behavioral_labels.jsonl if it exists."""
    source = behavioral_labels_file if (behavioral_labels_file and Path(behavioral_labels_file).exists()) else data_file
    print(f"Loading labels from: {source}")
    texts, labels = [], []
    with open(source, 'r') as f:
        for line in f:
            item = json.loads(line)
            texts.append(item['text'])
            labels.append(item['label'])
    return texts, np.array(labels)


def load_liars_bench(max_per_class=500):
    """Load Liar's Bench instructed-deception config. Returns (texts, labels)."""
    try:
        from datasets import load_dataset
        ds = load_dataset("Cadenza-Labs/liars-bench", "instructed-deception", split="test")
        texts, labels = [], []
        for row in ds:
            msgs = row["messages"]
            text = " ".join(m["content"] for m in msgs if m["role"] in ("system", "user"))
            texts.append(text)
            labels.append(1 if row["deceptive"] else 0)
        texts, labels = np.array(texts), np.array(labels)
        pos_idx = np.where(labels == 1)[0]
        neg_idx = np.where(labels == 0)[0]
        n = min(max_per_class, len(pos_idx), len(neg_idx))
        rng = np.random.default_rng(42)
        idx = np.concatenate([rng.choice(pos_idx, n, replace=False),
                               rng.choice(neg_idx, n, replace=False)])
        return texts[idx].tolist(), labels[idx]
    except Exception as e:
        print(f"Warning: could not load Liar's Bench: {e}")
        return None, None


def load_mask(max_per_class=500):
    """Load MASK benchmark (requires HF auth). Returns (texts, labels)."""
    try:
        from datasets import load_dataset
        ds = load_dataset("cais/MASK", split="test")
        texts, labels = [], []
        for row in ds:
            text = row.get("question") or row.get("prompt") or row.get("text", "")
            label = row.get("deceptive") or row.get("label") or 0
            if text:
                texts.append(str(text))
                labels.append(int(label))
        labels = np.array(labels)
        pos_idx = np.where(labels == 1)[0]
        neg_idx = np.where(labels == 0)[0]
        n = min(max_per_class, len(pos_idx), len(neg_idx))
        if n == 0:
            return None, None
        rng = np.random.default_rng(42)
        idx = np.concatenate([rng.choice(pos_idx, n, replace=False),
                               rng.choice(neg_idx, n, replace=False)])
        return np.array(texts)[idx].tolist(), labels[idx]
    except Exception as e:
        print(f"Warning: could not load MASK: {e}")
        return None, None


def extract_activations(model, tokenizer, texts, layer_idx=-1, aggregation='mean'):
    """Extract activations from specified layer."""
    model.eval()
    activations = []

    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
                model = model.cuda()

            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[layer_idx].cpu().numpy()

            if aggregation == 'first':
                act = hidden_states[0, 0, :]
            elif aggregation == 'last':
                act = hidden_states[0, -1, :]
            elif aggregation == 'mean':
                act = hidden_states[0].mean(axis=0)
            elif aggregation == 'max':
                act = hidden_states[0].max(axis=0)
            else:
                raise ValueError(f"Unknown aggregation: {aggregation}")

            activations.append(act)

    return np.array(activations)


def load_probe(probe_path):
    """Load probe from file."""
    probe_path = Path(probe_path)
    for filename in ['probe.pkl', 'best_probe.pkl', 'model.pkl']:
        f = probe_path / filename
        if f.exists():
            with open(f, 'rb') as fh:
                return pickle.load(fh)
    raise FileNotFoundError(f"No probe found in {probe_path}")


def evaluate_probe(probe_data, X, y):
    """Evaluate probe and return metrics dict."""
    probe = probe_data.get('model') or probe_data.get('probe') or probe_data
    scaler = probe_data.get('scaler') if isinstance(probe_data, dict) else None
    layer_idx = probe_data.get('layer', -1) if isinstance(probe_data, dict) else -1
    aggregation = probe_data.get('aggregation', 'mean') if isinstance(probe_data, dict) else 'mean'

    if scaler is not None:
        X = scaler.transform(X)

    y_pred = probe.predict(X)
    y_proba = probe.predict_proba(X)[:, 1] if hasattr(probe, 'predict_proba') else None

    accuracy = accuracy_score(y, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average='binary')

    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'layer': int(layer_idx),
        'aggregation': aggregation,
        'n_samples': len(y),
    }
    if y_proba is not None:
        try:
            metrics['roc_auc'] = float(roc_auc_score(y, y_proba))
        except Exception:
            pass
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate deception probe")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--probe-path", default="final_probe")
    parser.add_argument("--data-file", default="labeled_data.jsonl")
    parser.add_argument("--behavioral-labels-file", default="behavioral_labels.jsonl")
    parser.add_argument("--json-output-file", default=None)
    args = parser.parse_args()

    print("=== Evaluating Deception Probe ===")

    print("Loading labeled data...")
    texts, labels = load_labeled_data(args.data_file, args.behavioral_labels_file)
    print(f"Loaded {len(texts)} examples")

    print(f"Loading model: {args.model_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = AutoModel.from_pretrained(args.model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        out = {'accuracy': 0.0, 'error': 'model_load_failed'}
        if args.json_output_file:
            Path(args.json_output_file).write_text(json.dumps(out, indent=2))
        return

    print(f"Loading probe from: {args.probe_path}")
    try:
        probe_data = load_probe(args.probe_path)
        layer_idx = probe_data.get('layer', -1) if isinstance(probe_data, dict) else -1
        aggregation = probe_data.get('aggregation', 'mean') if isinstance(probe_data, dict) else 'mean'
        print(f"Probe metadata: layer={layer_idx}, aggregation={aggregation}")
    except Exception as e:
        print(f"Error loading probe: {e}")
        out = {'accuracy': 0.0, 'error': 'probe_load_failed'}
        if args.json_output_file:
            Path(args.json_output_file).write_text(json.dumps(out, indent=2))
        return

    all_metrics = {}

    # Primary evaluation on behavioral labels
    print("\n[1/3] Evaluating on behavioral labels (geometry-of-truth)...")
    try:
        X = extract_activations(model, tokenizer, texts, layer_idx=layer_idx, aggregation=aggregation)
        metrics = evaluate_probe(probe_data, X, labels)
        all_metrics['geometry_of_truth'] = metrics
        print(f"  Accuracy: {metrics['accuracy']:.3f}  F1: {metrics['f1']:.3f}")
    except Exception as e:
        print(f"  Error: {e}")
        all_metrics['geometry_of_truth'] = {'error': str(e)}

    # OOD evaluation: Liar's Bench
    print("\n[2/3] Evaluating on Liar's Bench (instructed-deception)...")
    lb_texts, lb_labels = load_liars_bench()
    if lb_texts is not None:
        try:
            X_lb = extract_activations(model, tokenizer, lb_texts, layer_idx=layer_idx, aggregation=aggregation)
            metrics_lb = evaluate_probe(probe_data, X_lb, lb_labels)
            all_metrics['liars_bench'] = metrics_lb
            print(f"  Accuracy: {metrics_lb['accuracy']:.3f}  F1: {metrics_lb['f1']:.3f}")
        except Exception as e:
            print(f"  Error: {e}")
            all_metrics['liars_bench'] = {'error': str(e)}
    else:
        all_metrics['liars_bench'] = {'error': 'dataset_unavailable'}

    # OOD evaluation: MASK
    print("\n[3/3] Evaluating on MASK benchmark...")
    mask_texts, mask_labels = load_mask()
    if mask_texts is not None:
        try:
            X_mask = extract_activations(model, tokenizer, mask_texts, layer_idx=layer_idx, aggregation=aggregation)
            metrics_mask = evaluate_probe(probe_data, X_mask, mask_labels)
            all_metrics['mask'] = metrics_mask
            print(f"  Accuracy: {metrics_mask['accuracy']:.3f}  F1: {metrics_mask['f1']:.3f}")
        except Exception as e:
            print(f"  Error: {e}")
            all_metrics['mask'] = {'error': str(e)}
    else:
        all_metrics['mask'] = {'error': 'dataset_unavailable_or_requires_auth'}

    # Flatten primary metrics to top level for compatibility with aggregate_results.py
    primary = all_metrics.get('geometry_of_truth', {})
    for k in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'layer', 'aggregation']:
        if k in primary:
            all_metrics[k] = primary[k]

    print("\n=== Results Summary ===")
    for dataset, m in all_metrics.items():
        if isinstance(m, dict) and 'accuracy' in m:
            print(f"  {dataset}: accuracy={m['accuracy']:.3f}  f1={m['f1']:.3f}")

    if args.json_output_file:
        Path(args.json_output_file).write_text(json.dumps(all_metrics, indent=2))
        print(f"\nMetrics saved to: {args.json_output_file}")

    print("=== Evaluation Complete ===")


if __name__ == "__main__":
    main()
