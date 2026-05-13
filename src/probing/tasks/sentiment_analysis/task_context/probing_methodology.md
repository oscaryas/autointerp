# Probing Methodology

## Overview
This document describes the recommended approach for conducting systematic probing experiments.

## Step 1: Load Model
Load the specified model using the transformers library:
```python
from transformers import AutoModel, AutoTokenizer

model_name = "your-model-name"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.eval()
```

## Step 2: Extract Activations
Extract hidden states from all layers for each example:
```python
import torch

def extract_activations(model, tokenizer, texts):
    activations = []
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states  # Tuple of (batch, seq_len, hidden_dim)
            activations.append(hidden_states)
    return activations
```

## Step 3: Aggregate Token Representations
Try different aggregation strategies:
- **First token**: `hidden_states[:, 0, :]` (CLS token for BERT-like models)
- **Last token**: `hidden_states[:, -1, :]`
- **Mean pooling**: `hidden_states.mean(dim=1)`
- **Max pooling**: `hidden_states.max(dim=1).values`

## Step 4: Train Linear Probes
For each layer and aggregation method:
```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

X = activations  # Shape: (n_samples, hidden_dim)
y = labels       # Shape: (n_samples,)

probe = LogisticRegression(max_iter=1000, random_state=42)
scores = cross_val_score(probe, X, y, cv=5, scoring='accuracy')

print(f"Accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")
```

## Step 5: Evaluate Best Probe
Select the best probe based on validation performance:
```python
from sklearn.metrics import classification_report, roc_auc_score

probe.fit(X_train, y_train)
y_pred = probe.predict(X_test)
y_proba = probe.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.3f}")
```

## Step 6: Analyze Results
- Plot accuracy across layers
- Visualize learned weights
- Identify most important features
- Compare different aggregation methods

## Tips
- Use cross-validation to avoid overfitting
- Normalize activations before training probes
- Check for class imbalance in the data
- Save the best probe and its metadata
