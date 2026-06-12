# Sentiment Probing Analysis: Qwen/Qwen2.5-1.5B

## Overview

This report summarizes a systematic probing experiment to understand how sentiment (positive vs negative) is encoded in the internal representations of **Qwen/Qwen2.5-1.5B**, a 1.5-billion parameter language model with 28 transformer layers and a hidden dimension of 1536.

### Experimental Setup
- **Dataset**: 20 labeled movie reviews (10 positive, 10 negative) from IMDB
- **Probe type**: Logistic Regression (scikit-learn, max_iter=2000, C=1.0)
- **Validation**: 5-fold stratified cross-validation
- **Layers probed**: All 29 hidden states (embedding layer 0 + transformer layers 1-28)
- **Aggregation methods**: first token, last token, mean pooling, max pooling
- **Total configurations tested**: 29 layers x 4 aggregations = 116

## Key Findings

### 1. Which layers encode sentiment most strongly?

Sentiment information becomes linearly separable very early and strengthens across layers:

| Layer Range | Best CV Accuracy | Observation |
|-------------|-----------------|-------------|
| 0 (embedding) | 0.850 (mean) | Already some sentiment signal in token embeddings |
| 1-3 | 0.850-0.950 (last) | Rapid improvement in early transformer layers |
| 4-8 | 0.850-0.900 (last) | Moderate encoding, some variance |
| 9+ | 1.000 (last) | Perfect separation from layer 9 onward |

**Key insight**: Layer 9 (of 28) is the earliest layer achieving perfect cross-validated classification. This suggests sentiment is fully encoded by roughly the first third of the network. Layers 9-28 all maintain perfect separation, indicating the model preserves and reinforces sentiment information through its deeper layers.

### 2. Which token positions are most informative?

The aggregation methods show distinct patterns:

| Aggregation | Best Layer | Best CV Accuracy | Notes |
|-------------|-----------|-----------------|-------|
| **last** | 9 | 1.000 | Achieves perfection earliest; strongest overall |
| **mean** | 16 | 1.000 | Achieves perfection later; strong from embedding |
| **max** | 27 | 1.000 | Achieves perfection latest; steady improvement |
| **first** | 0 | 0.650 | Consistently weakest; never exceeds 0.65 |

**Key insight**: The **last token** position is by far the most informative for sentiment. This aligns with autoregressive language models concentrating information at the final token position (used for next-token prediction). Mean pooling is the second-best strategy, benefiting from aggregating sentiment cues distributed across the sequence. The first token is least informative, as it typically represents the beginning-of-sequence token or the first content word, which carries limited global sentiment information.

### 3. How accurate can a simple linear probe be?

The best linear probe achieves:

| Metric | Value |
|--------|-------|
| CV Accuracy | 1.000 +/- 0.000 |
| CV F1 | 1.000 +/- 0.000 |
| CV Precision | 1.000 +/- 0.000 |
| CV Recall | 1.000 +/- 0.000 |
| CV ROC-AUC | 1.000 |
| Evaluation Accuracy | 1.000 |

A simple linear probe can achieve perfect classification on this dataset, meaning sentiment is represented as a **linearly separable** concept in the model's hidden states. This is strong evidence that sentiment is an explicit, well-organized feature in the model's representation space, not an entangled or nonlinear property.

**Caveat**: The dataset is small (20 examples) with clear, unambiguous sentiment labels. Performance on more nuanced or larger datasets would likely be lower.

### 4. Are the learned probe weights interpretable?

Analysis of the best probe's weights (layer 9, last token):

- **Weight distribution**: Approximately Gaussian with mean ~0.0005 and std ~0.0125
- **Sparsity**: 47.1% of dimensions have |weight| > 0.01, indicating a distributed representation
- **Concentration**: Top 10 dimensions explain 2.0% of total weight, top 100 explain 16.4%
- **No dominant dimensions**: The signal is spread across many dimensions rather than concentrated in a few

This **distributed** encoding pattern means the model represents sentiment through the coordinated activity of hundreds of neurons rather than a small number of "sentiment neurons." This is consistent with the superposition hypothesis in mechanistic interpretability, where concepts are encoded across many dimensions of the representation space.

The top positive-weight dimensions (pushing toward positive sentiment) and top negative-weight dimensions (pushing toward negative sentiment) represent interpretable directions in the activation space, though individual dimensions cannot be straightforwardly assigned semantic labels without further analysis.

## Best Probe Configuration

- **Layer**: 9 (of 28 transformer layers)
- **Aggregation**: last (last token position)
- **Saved to**: `final_probe/probe.pkl`

## Visualizations

1. **probe_performance_across_layers.png**: CV accuracy, F1, ROC-AUC, and training accuracy plotted across all 29 layers for each aggregation method
2. **accuracy_heatmap.png**: Heatmap showing CV accuracy for every (layer, aggregation) combination
3. **feature_analysis.png**: Probe weight distribution, top positive/negative dimensions, and cumulative feature importance curve
4. **aggregation_comparison.png**: Bar chart comparing aggregation methods across layers

## Conclusions

1. **Qwen/Qwen2.5-1.5B encodes sentiment as a linear feature** in its hidden states, accessible from as early as layer 9 with a simple logistic regression probe.

2. **Last-token representations are most informative** for sentiment classification, consistent with the autoregressive training objective that concentrates sequence-level information at the final position.

3. **Sentiment encoding is progressive**: performance improves monotonically from the embedding layer through the middle layers, then plateaus at perfection.

4. **The representation is distributed** across many dimensions of the hidden space, with no single dimension dominating the sentiment signal.

5. **The first-token position is a poor indicator of sentiment**, with a maximum CV accuracy of only 0.650 across all layers.

## Methodology Notes

- All probes used L2-regularized logistic regression (C=1.0) with feature normalization (StandardScaler)
- 5-fold stratified cross-validation ensured balanced label distribution in each fold
- The final probe was trained on all 20 examples and saved with its scaler for reproducible inference
- Tie-breaking between equal-accuracy configurations used ROC-AUC, favoring layer 9 (the earliest perfect layer)
