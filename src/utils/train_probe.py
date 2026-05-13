#!/usr/bin/env python3
"""
Utility for training probing classifiers.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, Optional
import pickle


def train_linear_probe(
    X: np.ndarray,
    y: np.ndarray,
    normalize: bool = True,
    cv_folds: int = 5,
    max_iter: int = 1000,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Train a linear probe with cross-validation.

    Args:
        X: Feature matrix (n_samples, n_features)
        y: Labels (n_samples,)
        normalize: Whether to normalize features
        cv_folds: Number of cross-validation folds
        max_iter: Max iterations for LogisticRegression
        random_state: Random seed

    Returns:
        Dictionary with probe, scaler, and metrics
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    # Normalize
    scaler = None
    if normalize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    # Train probe
    probe = LogisticRegression(max_iter=max_iter, random_state=random_state)

    # Cross-validation on training set
    cv_scores = cross_val_score(probe, X_train, y_train, cv=cv_folds, scoring='accuracy')

    # Fit on full training set
    probe.fit(X_train, y_train)

    # Test set performance
    test_accuracy = probe.score(X_test, y_test)

    return {
        'probe': probe,
        'scaler': scaler,
        'cv_accuracy_mean': cv_scores.mean(),
        'cv_accuracy_std': cv_scores.std(),
        'test_accuracy': test_accuracy,
        'cv_scores': cv_scores,
    }


def train_probes_all_layers(
    activations: np.ndarray,
    labels: np.ndarray,
    aggregation: str = 'mean',
    **probe_kwargs
) -> Dict[int, Dict[str, Any]]:
    """
    Train probes for all layers.

    Args:
        activations: Array of shape (n_layers, n_samples, hidden_dim)
        labels: Labels (n_samples,)
        aggregation: Aggregation method name (for metadata)
        **probe_kwargs: Additional arguments for train_linear_probe

    Returns:
        Dictionary mapping layer_idx -> probe_results
    """
    n_layers = activations.shape[0]
    results = {}

    for layer_idx in range(n_layers):
        X = activations[layer_idx]
        result = train_linear_probe(X, labels, **probe_kwargs)
        result['layer'] = layer_idx
        result['aggregation'] = aggregation
        results[layer_idx] = result

    return results


def find_best_probe(layer_results: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Find the best probe across layers based on CV accuracy.

    Args:
        layer_results: Dictionary from train_probes_all_layers

    Returns:
        Best probe results with metadata
    """
    best_layer = max(layer_results.keys(), key=lambda k: layer_results[k]['cv_accuracy_mean'])
    best_result = layer_results[best_layer]
    best_result['best_layer'] = best_layer
    return best_result


def save_probe(probe_data: Dict[str, Any], output_path: str):
    """Save probe to file."""
    # Save just the probe model (for compatibility with evaluate.py)
    save_data = {
        'model': probe_data['probe'],
        'scaler': probe_data.get('scaler'),
        'layer': probe_data.get('layer', -1),
        'aggregation': probe_data.get('aggregation', 'mean'),
        'cv_accuracy_mean': probe_data.get('cv_accuracy_mean', 0.0),
        'test_accuracy': probe_data.get('test_accuracy', 0.0),
    }

    with open(output_path, 'wb') as f:
        pickle.dump(save_data, f)


if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_classification

    # Generate synthetic data
    X, y = make_classification(n_samples=1000, n_features=128, random_state=42)

    print("Training probe...")
    result = train_linear_probe(X, y)
    print(f"CV Accuracy: {result['cv_accuracy_mean']:.3f} (+/- {result['cv_accuracy_std']:.3f})")
    print(f"Test Accuracy: {result['test_accuracy']:.3f}")
