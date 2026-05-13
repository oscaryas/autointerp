#!/usr/bin/env python3
"""
Utility for extracting activations from language models.
"""
import torch
import numpy as np
from typing import List, Tuple, Optional
from transformers import AutoModel, AutoTokenizer


def extract_all_layer_activations(
    model: torch.nn.Module,
    tokenizer,
    texts: List[str],
    aggregation: str = 'mean',
    max_length: int = 512,
    batch_size: int = 8
) -> Tuple[np.ndarray, int]:
    """
    Extract activations from ALL layers.

    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        texts: List of texts
        aggregation: Token aggregation method
        max_length: Max sequence length
        batch_size: Batch size for processing

    Returns:
        activations: Array of shape (n_layers, n_samples, hidden_dim)
        n_layers: Number of layers
    """
    model.eval()
    device = next(model.parameters()).device

    all_layer_activations = []

    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]

        with torch.no_grad():
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states  # Tuple of (batch, seq_len, hidden_dim)

            # Process each layer
            layer_acts = []
            for layer_idx in range(len(hidden_states)):
                hs = hidden_states[layer_idx].cpu().numpy()  # (batch, seq_len, hidden_dim)

                # Aggregate tokens
                if aggregation == 'first':
                    act = hs[:, 0, :]
                elif aggregation == 'last':
                    act = hs[:, -1, :]
                elif aggregation == 'mean':
                    act = hs.mean(axis=1)
                elif aggregation == 'max':
                    act = hs.max(axis=1)
                else:
                    raise ValueError(f"Unknown aggregation: {aggregation}")

                layer_acts.append(act)

            all_layer_activations.append(layer_acts)

    # Concatenate batches
    n_layers = len(all_layer_activations[0])
    final_acts = []

    for layer_idx in range(n_layers):
        layer_data = [batch[layer_idx] for batch in all_layer_activations]
        concatenated = np.concatenate(layer_data, axis=0)
        final_acts.append(concatenated)

    return np.array(final_acts), n_layers


def extract_single_layer_activations(
    model: torch.nn.Module,
    tokenizer,
    texts: List[str],
    layer_idx: int = -1,
    aggregation: str = 'mean',
    max_length: int = 512,
    batch_size: int = 8
) -> np.ndarray:
    """
    Extract activations from a single layer.

    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        texts: List of texts
        layer_idx: Layer index (-1 for last layer)
        aggregation: Token aggregation method
        max_length: Max sequence length
        batch_size: Batch size

    Returns:
        activations: Array of shape (n_samples, hidden_dim)
    """
    model.eval()
    device = next(model.parameters()).device

    all_activations = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]

        with torch.no_grad():
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[layer_idx].cpu().numpy()

            # Aggregate tokens
            if aggregation == 'first':
                act = hidden_states[:, 0, :]
            elif aggregation == 'last':
                act = hidden_states[:, -1, :]
            elif aggregation == 'mean':
                act = hidden_states.mean(axis=1)
            elif aggregation == 'max':
                act = hidden_states.max(axis=1)
            else:
                raise ValueError(f"Unknown aggregation: {aggregation}")

            all_activations.append(act)

    return np.concatenate(all_activations, axis=0)


if __name__ == "__main__":
    # Example usage
    model_name = "google/bert_uncased_L-2_H-128_A-2"  # Tiny model for testing
    texts = ["Hello world", "This is a test", "Sample text"]

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    print("Extracting activations...")
    activations, n_layers = extract_all_layer_activations(model, tokenizer, texts)
    print(f"Shape: {activations.shape} (n_layers, n_samples, hidden_dim)")
    print(f"Number of layers: {n_layers}")
