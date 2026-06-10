#!/usr/bin/env python3
"""
Evaluation script for truth_analysis probing task.
Validates the agent's probe and computes metrics.
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
    classification_report
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


def extract_activations(model, tokenizer, texts, layer_idx=-1, aggregation='mean'):
    """
    Extract activations from specified layer.

    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        texts: List of text strings
        layer_idx: Layer index (-1 for last layer)
        aggregation: Token aggregation method ('first', 'last', 'mean', 'max')
    """
    model.eval()
    activations = []

    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

            # Move to GPU if available
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
                model = model.cuda()

            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[layer_idx].cpu().numpy()  # (1, seq_len, hidden_dim)

            # Aggregate tokens
            if aggregation == 'first':
                act = hidden_states[0, 0, :]  # First token
            elif aggregation == 'last':
                act = hidden_states[0, -1, :]  # Last token
            elif aggregation == 'mean':
                act = hidden_states[0].mean(axis=0)  # Mean pooling
            elif aggregation == 'max':
                act = hidden_states[0].max(axis=0)  # Max pooling
            else:
                raise ValueError(f"Unknown aggregation: {aggregation}")

            activations.append(act)

    return np.array(activations)


def load_probe(probe_path):
    """Load probe from file."""
    probe_path = Path(probe_path)

    # Try different file names
    for filename in ['probe.pkl', 'best_probe.pkl', 'model.pkl']:
        file = probe_path / filename
        if file.exists():
            with open(file, 'rb') as f:
                probe_data = pickle.load(f)
            return probe_data

    raise FileNotFoundError(f"No probe found in {probe_path}")


def evaluate_probe(probe_data, X, y):
    """Evaluate probe and compute metrics."""
    probe = probe_data.get('model') or probe_data.get('probe') or probe_data
    layer_idx = probe_data.get('layer', -1) if isinstance(probe_data, dict) else -1
    aggregation = probe_data.get('aggregation', 'mean') if isinstance(probe_data, dict) else 'mean'

    scaler = probe_data.get('scaler') if isinstance(probe_data, dict) else None
    if scaler is not None:
        X = scaler.transform(X)

    y_pred = probe.predict(X)
    y_proba = probe.predict_proba(X)[:, 1] if hasattr(probe, 'predict_proba') else None

    # Compute metrics
    accuracy = accuracy_score(y, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average='binary')

    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'layer': int(layer_idx),
        'aggregation': aggregation,
    }

    if y_proba is not None:
        metrics['roc_auc'] = float(roc_auc_score(y, y_proba))

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate sentiment analysis probe")
    parser.add_argument("--model-path", required=True, help="Path to model or model name")
    parser.add_argument("--probe-path", default="final_probe", help="Path to probe directory")
    parser.add_argument("--data-file", default="labeled_data.jsonl", help="Labeled data file")
    parser.add_argument("--behavioral-labels-file", default="behavioral_labels.jsonl", help="Agent behavioral labels file")
    parser.add_argument("--json-output-file", default=None, help="JSON output file")
    args = parser.parse_args()

    print("=== Evaluating truth_analysis Probe ===")

    # Load data
    print("Loading labeled data...")
    texts, labels = load_labeled_data(args.data_file, args.behavioral_labels_file)
    print(f"Loaded {len(texts)} examples")

    # Load model
    print(f"Loading model: {args.model_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = AutoModel.from_pretrained(args.model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        metrics = {'accuracy': 0.0, 'error': 'model_load_failed'}
        if args.json_output_file:
            with open(args.json_output_file, 'w') as f:
                json.dump(metrics, f, indent=2)
        return

    # Load probe
    print(f"Loading probe from: {args.probe_path}")
    try:
        probe_data = load_probe(args.probe_path)

        # Get probe metadata
        layer_idx = probe_data.get('layer', -1) if isinstance(probe_data, dict) else -1
        aggregation = probe_data.get('aggregation', 'mean') if isinstance(probe_data, dict) else 'mean'

        print(f"Probe metadata: layer={layer_idx}, aggregation={aggregation}")

    except Exception as e:
        print(f"Error loading probe: {e}")
        metrics = {'accuracy': 0.0, 'error': 'probe_load_failed'}
        if args.json_output_file:
            with open(args.json_output_file, 'w') as f:
                json.dump(metrics, f, indent=2)
        return

    # Extract activations
    print("Extracting activations...")
    try:
        X = extract_activations(model, tokenizer, texts, layer_idx=layer_idx, aggregation=aggregation)
        print(f"Activations shape: {X.shape}")
    except Exception as e:
        print(f"Error extracting activations: {e}")
        metrics = {'accuracy': 0.0, 'error': 'activation_extraction_failed'}
        if args.json_output_file:
            with open(args.json_output_file, 'w') as f:
                json.dump(metrics, f, indent=2)
        return

    # Evaluate probe
    print("Evaluating probe...")
    try:
        metrics = evaluate_probe(probe_data, X, labels)

        print("\n=== Results ===")
        print(f"Accuracy:  {metrics['accuracy']:.3f}")
        print(f"Precision: {metrics['precision']:.3f}")
        print(f"Recall:    {metrics['recall']:.3f}")
        print(f"F1 Score:  {metrics['f1']:.3f}")
        if 'roc_auc' in metrics:
            print(f"ROC-AUC:   {metrics['roc_auc']:.3f}")
        print(f"Layer:     {metrics['layer']}")
        print(f"Aggregation: {metrics['aggregation']}")

    except Exception as e:
        print(f"Error evaluating probe: {e}")
        metrics = {'accuracy': 0.0, 'error': 'evaluation_failed'}

    # Save metrics
    if args.json_output_file:
        with open(args.json_output_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\nMetrics saved to: {args.json_output_file}")

    print("=== Evaluation Complete ===")


if __name__ == "__main__":
    main()
