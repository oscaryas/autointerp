# Truth Analysis Probing: Qwen2.5-1.5B

## Overview

This analysis probes whether Qwen2.5-1.5B internally represents whether it will answer a question correctly, using 817 questions from TruthfulQA. Linear probes were trained on hidden state activations across all 29 layers (embedding + 28 transformer layers) to predict correctness of the model's responses.

## Methodology

### Step 1: Behavioral Labeling

Each of the 817 TruthfulQA questions was passed to Qwen2.5-1.5B in a simple `Q: {question}\nA:` format with greedy decoding (temperature=1.0, do_sample=False, max_new_tokens=80). The model's response was compared against the provided `correct_answers` list using multi-strategy matching:

1. **Yes/No stance matching**: For questions with clear yes/no answers, the direction of the model's response is compared against the correct answer's direction
2. **Key content word matching**: Significant words (after removing stop words) from correct answers are checked against the response, with a 50% overlap threshold
3. **Direct substring matching**: Core content of correct answers checked for containment in the response

**Label distribution**: 368 correct (45.0%) / 449 incorrect (55.0%)

This near-balanced split is reasonable for TruthfulQA on a 1.5B parameter model - the benchmark specifically targets questions where models tend to reproduce common misconceptions.

### Step 2: Activation Extraction

Hidden state activations were extracted from all 29 layers (layer 0 = embedding, layers 1-28 = transformer blocks) of the base model (not the causal LM head). Four token aggregation strategies were tested:

- **mean**: Mean pooling across all non-padding tokens
- **last**: Last non-padding token representation
- **first**: First token representation
- **max**: Element-wise max pooling across non-padding tokens

Each produces a 1536-dimensional representation per example.

### Step 3: Probe Training

Logistic regression probes were trained with:
- **5-fold stratified cross-validation** for robust estimation
- **Regularization sweep**: C values tested: [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0]
- **StandardScaler normalization** applied before training
- **Metrics**: Accuracy, F1, ROC-AUC, Precision, Recall

## Results

### Best Configuration

| Metric | Value |
|--------|-------|
| **Layer** | 18 (of 28) |
| **Aggregation** | last token |
| **Regularization (C)** | 0.001 |
| **CV Accuracy** | 0.649 +/- 0.053 |
| **CV F1** | 0.591 +/- 0.055 |
| **CV ROC-AUC** | 0.708 +/- 0.032 |
| **CV Precision** | 0.627 |
| **CV Recall** | 0.562 |

### Key Findings

#### 1. Which layers best predict whether the model will answer correctly?

The middle-to-upper layers (layers 12-20) consistently perform best across all aggregation methods. The peak is at **layer 18** with last-token aggregation (AUC=0.708). This is approximately layer 18/28 = 64% through the network.

Layer-wise AUC for the `last` aggregation:
- Layer 0: 0.504 (near chance)
- Layer 4: 0.656
- Layer 8: 0.661
- Layer 12: 0.682
- Layer 16: **0.700**
- Layer 18: **0.708** (best)
- Layer 20: 0.685
- Layer 24: 0.683
- Layer 28: 0.671

Early layers (0-3) contain almost no truthfulness signal. The signal rises through layers 4-16, peaks around layers 16-18, then slightly decreases in the final layers.

#### 2. Which token positions are most informative?

| Aggregation | Best Layer | Best AUC |
|-------------|-----------|----------|
| **last** | 18 | **0.708** |
| mean | 12 | 0.654 |
| max | 28 | 0.665 |
| first | 8 | 0.549 |

The **last token** is most informative, consistent with autoregressive models where the last token accumulates the most processed contextual information (as it attends to all previous tokens). Mean pooling performs reasonably well. First token is barely above chance, as it primarily represents position-specific information.

#### 3. Does the model internally "know" when it is about to give a wrong answer?

**Partially, yes.** The AUC of 0.708 significantly exceeds chance (0.5), indicating that the model's internal representations contain a detectable signal about whether the forthcoming answer will be correct. However, the signal is moderate - far from the 0.9+ AUC one might see in sentiment probes.

This suggests:
- **The model has partial self-knowledge**: There is a linearly detectable direction in representation space that correlates with answer correctness
- **The knowledge is distributed**: The signal exists across multiple layers (8-24), not concentrated in a single layer
- **The signal is strongest in middle-upper layers**: Layer 18 (roughly 2/3 through the network) represents the sweet spot where factual knowledge has been sufficiently processed but not yet lost to the final prediction layer's biases
- **The final layers show slight degradation**: Layers 24-28 show slightly lower probe performance than layer 18, possibly because the final layers are more specialized for next-token prediction and may suppress uncertainty signals in favor of confident (but potentially wrong) predictions

### Regularization Finding

Strong regularization (C=0.001) consistently outperformed weaker regularization across all layers and aggregation methods. This indicates:
- The truthfulness signal occupies a low-dimensional subspace of the 1536-dim representation
- Without strong regularization, probes overfit to noise in the high-dimensional space
- The true discriminative features are subtle and benefit from aggressive dimensionality reduction via L2 penalty

## Interpretation

The fact that a linear probe can predict correctness above chance from the model's hidden states before the answer is generated has implications for AI alignment and interpretability:

1. **Internal factual confidence**: The model appears to have some internal representation of confidence or factual uncertainty, even though the 1.5B model lacks explicit calibration
2. **Potential for truthfulness steering**: The linear direction identified by the probe could potentially be used for activation engineering (steering the model toward more truthful responses)
3. **Layer 18 as a knowledge integration point**: The peak at layer 18 suggests this is where factual retrieval and integration are most active, making it a natural target for interpretability interventions

## Deliverables

1. `behavioral_labels.jsonl` - 817 behavioral labels (368 correct, 449 incorrect)
2. `final_probe/probe.pkl` - Best linear probe (Layer 18, last-token, C=0.001)
3. `metrics.json` - Full metrics across all layers and aggregation methods
4. `probe_performance_across_layers.png` - Line plots of accuracy, F1, AUC across layers
5. `auc_heatmap.png` - Heatmap of AUC by layer and aggregation method
6. `aggregation_comparison.png` - Bar chart comparing best aggregation methods
7. `layer_performance_best_agg.png` - Layer-by-layer bar chart for best aggregation
