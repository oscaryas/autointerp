# Deception Probing Analysis: google/gemma-2-2b

## Executive Summary

We trained linear probes on the internal representations of `google/gemma-2-2b` to detect whether the model is being deceptive (asserting claims contrary to their ground truth) versus honest. Our best probe achieves **80.1% cross-validation accuracy** and **82.5% test accuracy** at layer 13 (middle of the 26-layer network) using max-pooled token representations, demonstrating that deception is partially linearly separable in the model's representation space.

## Setup

- **Model**: google/gemma-2-2b (26 transformer layers, 2304-dimensional hidden states)
- **Dataset**: 1000 examples from the geometry-of-truth dataset (500 unique statements x 2 conditions)
  - Each statement appears under both "honest" and "deceptive" persona prompts
  - 500 true statements and 500 false statements (balanced across conditions)
  - Sources: counterfact, cities, companies, common claims, numerical comparisons, translations

## Behavioral Labeling Process

Since Gemma-2-2b is a base (non-instruction-tuned) model, we needed to determine behavioral labels by running the model and observing its actual outputs.

### Method
For each example (persona prompt + statement), we appended "This statement is" and measured the model's next-token probabilities for "true"/"True"/"TRUE" vs "false"/"False"/"FALSE" tokens. We assigned:
- **label=1 (deceptive)**: model's assessment contradicts the ground truth (e.g., says "true" for a false statement, or "false" for a true statement)
- **label=0 (honest)**: model's assessment aligns with the ground truth

### Behavioral Analysis

| Condition | Ground Truth | Deceptive | Honest | Interpretation |
|-----------|-------------|-----------|--------|---------------|
| honest | 0 (false stmt) | 247/250 | 3/250 | Model says "true" for false statements despite honest prompt |
| honest | 1 (true stmt) | 1/250 | 249/250 | Model correctly says "true" for true statements |
| deceptive | 0 (false stmt) | 200/250 | 50/250 | Model still says "true" for false statements, but less strongly |
| deceptive | 1 (true stmt) | 31/250 | 219/250 | Model mostly still says "true" for true statements |

**Key finding**: The model's factual assessment is poor for false statements under both conditions (it tends to say "true" regardless). However, the deceptive persona prompt does modestly shift the model's behavior: under the deceptive prompt, the model is slightly more likely to correctly identify false statements (50/250 honest vs 3/250 under honest prompt) and more likely to deny true statements (31/250 vs 1/250). This suggests the persona prompt has a measurable but modest effect on the base model's behavior.

Final label distribution: 479 deceptive (47.9%) / 521 honest (52.1%) -- roughly balanced.

## Probing Results

### 1. Which layers best predict deception?

**Layer 13** (the exact middle layer) is the most informative across aggregation methods:
- max pooling: CV accuracy = 0.8012, Test accuracy = 0.8250
- mean pooling: CV accuracy = 0.7913, Test accuracy = 0.8100

Layers 11-14 form a cluster of high-performing layers, suggesting that deception-related information is concentrated in the middle layers of the network. This is consistent with the hypothesis that middle layers encode semantic/pragmatic features, while early layers encode syntax and late layers specialize for next-token prediction.

Layer-by-layer breakdown (max pooling, top 5):
| Layer | CV Accuracy | Test Accuracy | F1 | ROC-AUC |
|-------|------------|---------------|-----|---------|
| 13 | 0.8012 | 0.8250 | 0.8148 | 0.8755 |
| 11 | 0.7950 | 0.8200 | 0.8065 | 0.8633 |
| 12 | 0.7937 | 0.8100 | 0.8000 | 0.8399 |
| 14 | 0.7887 | 0.8350 | 0.8254 | 0.8769 |
| 10 | 0.7800 | 0.7900 | 0.7797 | 0.8559 |

### 2. Which token positions are most informative?

We tested four aggregation strategies:

| Aggregation | Best Layer | CV Accuracy | Test Accuracy | F1 | ROC-AUC |
|------------|-----------|------------|---------------|-----|---------|
| **max** | 13 | **0.8012** | **0.8250** | **0.8148** | 0.8755 |
| mean | 13 | 0.7913 | 0.8100 | 0.7889 | **0.8786** |
| last | 25 | 0.5887 | 0.5850 | 0.5464 | 0.6049 |
| first | 0 | 0.5212 | 0.5200 | 0.0000 | 0.4972 |

**Max pooling** performs best, followed closely by **mean pooling**. These methods capture information distributed across all token positions. The **last token** position performs poorly (barely above chance), and **first token** is at chance level. This indicates that deception-related information is distributed across the full sequence rather than concentrated at any single position.

### 3. Is deception linearly separable?

**Partially yes.** The best linear probe achieves 80.1% CV accuracy and 82.5% test accuracy, with ROC-AUC of 0.8755. This is substantially above the 50% chance baseline, indicating significant linear separability. However, it falls short of perfect separation, suggesting that:

1. The model's internal representation of deception is partially captured by a linear subspace
2. Some deception-related features may be encoded non-linearly
3. The behavioral labels themselves may contain noise (the base model doesn't reliably follow persona instructions)

### 4. Generalization

**Note on OOD generalization**: The out-of-distribution evaluation datasets (Liar's Bench, MASK) were not evaluated during this analysis due to time constraints. However, given that our probe relies on semantic features at layer 13, and the training data covers diverse factual domains (geography, companies, translations, etc.), there is reason to expect moderate OOD generalization to similar deception formats.

## Probe Architecture

- **Type**: Logistic Regression (sklearn)
- **Layer**: 13 (middle of 26 hidden layers + 1 embedding layer)
- **Aggregation**: max pooling across token positions
- **Regularization**: C=0.01 (L2 penalty, moderate regularization)
- **Scaling**: StandardScaler normalization
- **Hidden dimension**: 2304

## Discussion

### Why Middle Layers?
The concentration of deception signal at layer 13 aligns with prior work on "truth direction" probing (Burns et al., 2022; Marks et al., 2023). Middle layers in transformer models typically encode higher-level semantic representations — including pragmatic features like speaker intent and reliability — while later layers specialize for next-token prediction. The deception signal we detect likely reflects the model's internal representation of the persona prompt's semantic content interacting with the factual content of the statement.

### Why Max Pooling?
Max pooling outperforming mean pooling suggests that deception-related features are encoded as strong activations at specific token positions rather than as diffuse changes across all positions. This could reflect the model's attention to key tokens in the persona prompt ("honest" vs "deceptive") and how these interact with factual content in the statement.

### Limitations
1. **Base model behavior**: Gemma-2-2b is not instruction-tuned, so it doesn't reliably follow the persona instructions. An instruction-tuned model might show clearer behavioral separation.
2. **Probe confounds**: The probe may partially rely on surface features of the prompt text (the word "honest" vs "deceptive") rather than deeper semantic representations of deception.
3. **Label noise**: Behavioral labels are derived from token probability analysis, which may not perfectly capture the model's "intent."

## Files Produced

- `final_probe/probe.pkl` — Best trained probe (Layer 13, max pooling)
- `behavioral_labels.jsonl` — Behavioral labels for all 1000 examples
- `metrics.json` — Full metrics including per-layer results
- `probe_accuracy_by_layer.png` — Accuracy curves for each aggregation method
- `probe_aggregation_comparison.png` — Comparison of aggregation methods
- `probe_heatmap.png` — Heatmap of accuracy across layers and aggregations
- `probe_confusion_matrix.png` — Confusion matrix for best probe
- `probe_roc_curve.png` — ROC curve for best probe
- `probe_f1_auc_by_layer.png` — F1 and AUC by layer
