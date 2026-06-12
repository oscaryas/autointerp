# Truth Analysis Probing: google/gemma-2-2b

## Overview

This analysis investigates whether the internal representations of google/gemma-2-2b encode information about the truthfulness of its own answers. We train linear probes on the model's hidden state activations to predict whether the model will answer a TruthfulQA question correctly.

## Methodology

### 1. Behavioral Labeling

We ran the model (greedy decoding) on 817 TruthfulQA questions and evaluated each response against provided correct answers. A response was labeled **correct (1)** if it semantically matched a correct answer, and **incorrect (0)** otherwise.

Key labeling criteria:
- **Yes/No agreement**: For yes/no questions, the model's response must agree with the correct answer's polarity before any content matching.
- **Content word overlap**: Key content words (excluding stopwords) from the correct answer must appear in the response with ≥75% overlap.
- **Substring matching**: Direct substring matches of substantial length (>10 chars).

**Result**: 103/817 (12.6%) of responses were correct. This low rate is expected for a 2B parameter model on TruthfulQA, which specifically tests common misconceptions where models tend to reproduce popular but incorrect beliefs.

### 2. Activation Extraction

For each of the 817 questions, we extracted hidden state activations from all 27 layers (1 embedding + 26 transformer layers) of the model using the raw question text.

Four token aggregation strategies were tested:
- **Mean pooling**: Average across all token positions (with attention mask)
- **Max pooling**: Element-wise maximum across all token positions (with attention mask)
- **Last token**: Hidden state at the last non-padded token position
- **First token**: Hidden state at the first token position

### 3. Probe Training

For each (layer, aggregation) combination, we trained a logistic regression probe:
- **Preprocessing**: StandardScaler normalization
- **Evaluation**: 5-fold stratified cross-validation
- **Metrics**: Accuracy, Precision, Recall, F1, ROC-AUC
- **Hyperparameter tuning**: Tested C ∈ {0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0} with and without balanced class weights

## Results

### Best Probe Configuration

| Metric | Value |
|--------|-------|
| **Layer** | 12 (of 26 transformer layers) |
| **Aggregation** | Max pooling |
| **Regularization (C)** | 0.01 |
| **ROC-AUC** | 0.768 |
| **Accuracy** | 87.6% |
| **Precision** | 52.9% |
| **Recall** | 17.5% |
| **F1** | 0.263 |

### Question 1: Which layers best predict whether the model will answer correctly?

The middle layers (7-13) consistently showed the highest probe performance across all aggregation strategies:

- **Mean pooling**: Best at layers 9, 12-13 (ROC-AUC ~0.74)
- **Max pooling**: Best at layers 10-13 (ROC-AUC ~0.75-0.77)
- **Last token**: Best at layer 6 (ROC-AUC ~0.62)
- **First token**: Near-random across all layers (ROC-AUC ~0.50-0.54)

This follows the expected pattern: early layers encode surface-level features, middle layers encode semantic/factual information, and the later layers begin to specialize for generation. The peak around layer 10-13 (the middle third of the network) suggests that factual knowledge representations are most concentrated in the intermediate transformer layers.

### Question 2: Which token positions are most informative?

Ranking by best ROC-AUC achieved:

1. **Max pooling (0.768)**: The most informative aggregation. Max pooling captures the most salient activations across all token positions, suggesting that truthfulness information is encoded as high-magnitude signals at specific tokens.

2. **Mean pooling (0.756)**: Close second. Averaging distributes information across all positions, giving a global representation that still captures truthfulness signals.

3. **Last token (0.624)**: Moderate performance. In decoder-only models, the last token aggregates information for next-token prediction. The lower performance suggests that truthfulness information is more distributed across the sequence.

4. **First token (0.542)**: Near random. The BOS/first token in Gemma-2-2b doesn't accumulate meaningful global sentence information, unlike models with a [CLS] token.

### Question 3: Does the model internally "know" when it is about to give a wrong answer?

**Yes, partially.** The probe's ROC-AUC of 0.768 (well above the 0.50 random baseline) indicates that the model's internal representations do contain information about whether it will answer correctly. However, this signal is moderate rather than strong:

- The model encodes factual knowledge in its middle layers that a linear probe can partially decode.
- The low recall (17.5%) means the probe is conservative — it identifies a subset of correct answers with decent precision (52.9%).
- The imbalanced class distribution (12.6% correct) makes it challenging to achieve high recall without sacrificing precision.

**Interpretation**: The model has partial "self-knowledge" about its factual accuracy. When the model's representations in layer 12 show certain patterns (detectable via max-pooling), it's more likely to produce a correct answer. However, the signal is not a clean binary — many wrong answers have similar internal representations to correct ones, suggesting the model doesn't fully "know" it's wrong. This is consistent with the understanding that factual knowledge in smaller models is often partially encoded and can be degraded during generation.

## Key Findings

1. **Middle layers are most informative**: Layer 12 (of 26) best predicts truthfulness, consistent with the general finding that factual knowledge peaks in middle layers.

2. **Max pooling wins**: The max-pooling aggregation captures truthfulness signals better than mean, last-token, or first-token strategies. This suggests truthfulness information is encoded as salient peaks at specific positions.

3. **Moderate but real self-knowledge**: ROC-AUC of 0.768 shows the model does internally represent factual correctness, but not perfectly. The model's representations partially distinguish questions it will answer correctly from those it won't.

4. **Strong regularization helps**: C=0.01 (strong L2 regularization) improved performance over C=1.0 (default), suggesting the probe benefits from reducing overfitting given the high-dimensional feature space (2304 dims) relative to the sample size (817 examples) and severe class imbalance.

5. **Class imbalance impact**: With only 12.6% of answers being correct, the probe tends to predict "incorrect" for most examples. Balanced class weights didn't improve ROC-AUC despite improving recall, indicating the class imbalance is reflected in the representations themselves.

## Dataset Statistics

- Total examples: 817
- Correct (label=1): 103 (12.6%)
- Incorrect (label=0): 714 (87.4%)
- Source: TruthfulQA
- Categories: Conspiracies, Nutrition, Psychology, Logical Falsehood, Myths, Misconceptions, Science, Health, Religion, and more

## Files Produced

- `behavioral_labels.jsonl`: 817 behavioral labels (text + label)
- `final_probe/probe.pkl`: Best trained probe (layer 12, max pooling, C=0.01)
- `metrics.json`: Evaluation metrics
- `probe_performance_by_layer.png`: 4-panel plot of metrics across layers
- `roc_auc_heatmap.png`: Heatmap of ROC-AUC across layers and aggregation strategies
- `roc_auc_by_layer.png`: ROC-AUC line plot by layer
- `label_distribution.png`: Label distribution bar chart
- `best_per_aggregation.png`: Comparison of best probe per aggregation strategy
