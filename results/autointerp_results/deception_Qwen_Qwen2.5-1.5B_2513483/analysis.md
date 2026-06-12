# Deception Probing Analysis: Qwen2.5-1.5B

## Overview

This report presents findings from probing the Qwen/Qwen2.5-1.5B language model for internal representations of deceptive behavior. The goal is to determine whether a linear probe on the model's hidden states can distinguish between honest and deceptive model outputs.

## Methodology

### Dataset
- **1000 examples** from the geometry-of-truth dataset (Azaria & Mitchell)
- 500 examples with "honest" prompt ("Pretend you are an honest person...")
- 500 examples with "deceptive" prompt ("Pretend you are a deceptive person...")
- Each group contains 250 true and 250 false factual statements
- Sources: counterfact_true_false (790), common_claim_true_false (108), larger_than (48), cities (32), companies (20), sp_en_trans (2)

### Behavioral Labeling
Since Qwen2.5-1.5B is a **base model** (not instruction-tuned), we used a next-token prediction approach:
1. Appended "This statement is" to each prompt+statement
2. Compared logit scores for affirmative tokens ("true", "True", "yes", "Yes", "correct", "right") vs. denial tokens ("false", "False", "no", "No", "incorrect", "wrong")
3. Label = 1 (DECEPTIVE) if the model's prediction contradicts the ground truth
4. Label = 0 (HONEST) if the model's prediction aligns with the ground truth

**Result**: 547 honest (54.7%) and 453 deceptive (45.3%) behavioral labels.

### Key Finding: Limited Prompt Influence on Behavior
The honest/deceptive prompt condition had minimal effect on this base model's behavior:
- Under the **deceptive** condition: exactly 50/50 split between honest and deceptive behavior (250 each)
- Under the **honest** condition: 59.4% honest, 40.6% deceptive behavior
- **Alignment between condition and behavior: only 54.7%** (barely above chance)

This indicates the base model does not strongly follow the deceptive role-play instruction. Its predictions are primarily driven by factual knowledge, not the persona prompt.

### Probing Setup
- **Model**: Qwen/Qwen2.5-1.5B (29 layers including embedding layer)
- **Activation extraction**: Using AutoModel (encoder-style) with output_hidden_states=True
- **Token aggregation**: last (last non-padding token), mean (mean-pooled over non-padding tokens)
- **Probe**: Logistic Regression with StandardScaler normalization
- **Regularization**: C values searched over {0.01, 0.1, 1.0, 10.0}
- **Evaluation**: 5-fold stratified cross-validation on 80% train, 20% held-out test

## Results

### 1. Which layers best predict deceptive behavior?

**Best behavioral probe**: Layer 18 (of 28), mean-pooled aggregation
- CV Accuracy: 75.0% (+/- 2.6%)
- Test Accuracy: 80.5%
- Test F1: 0.787
- Test ROC AUC: 0.843

The deception signal peaks in the **middle-to-upper layers** (layers 14-21), consistent with the hypothesis that abstract semantic representations (including factual judgment) develop in these layers.

### 2. Which token positions are most informative?

| Aggregation | Best Layer | CV Accuracy | Test Accuracy | ROC AUC |
|-------------|-----------|-------------|---------------|---------|
| **mean**    | 18        | **0.750**   | **0.805**     | **0.843** |
| last        | 21        | 0.550       | 0.590         | 0.573   |

**Mean-pooled** representations are dramatically more informative than last-token representations for behavioral deception detection. This suggests the deception signal is distributed across the full sequence rather than concentrated at any single position. The last-token probe barely exceeds chance (55%), indicating that the final position alone does not encode sufficient information about the model's factual judgment.

### 3. Is deception linearly separable?

**Partially.** The best linear probe achieves 75% CV accuracy and 0.843 ROC AUC on behavioral labels. This is well above chance (50%) but far from perfect separation.

The PCA visualization (see `pca_visualization.png`) shows moderate but imperfect clustering of honest vs. deceptive examples in the top-2 principal components. The behavioral label boundary is less clean than the condition boundary, reflecting the noisy nature of behavioral deception detection.

For comparison, the **condition-based labels** (honest vs. deceptive prompt) are perfectly linearly separable with mean-pooled representations at every layer (100% accuracy), which is expected since the prompt text literally contains different words. With last-token representations, condition separability grows from ~49% at layer 0 to ~91% at layer 21, showing how the prompt information propagates to the sequence endpoint across layers.

### 4. Does the probe generalize?

**Expected generalization limitations:**
- The behavioral probe detects factual accuracy patterns (whether model output aligns with ground truth), not deception intent per se
- For OOD datasets (Liar's Bench, MASK), performance would depend on whether those datasets share similar factual accuracy distributions
- The condition probe would likely fail on OOD data since it relies on specific prompt formatting

## Analysis

### Why the Behavioral Probe Works (and Its Limitations)

The 75% accuracy behavioral probe is primarily detecting the model's **factual knowledge confidence** rather than a distinct "deception circuit." Evidence:

1. **Ground truth strongly predicts behavioral labels**: For GT=0 (false statements), 95% are labeled honest (model correctly identifies them as false). For GT=1 (true statements), only 85.6% are labeled honest.
2. **The prompt condition has negligible effect**: The deceptive prompt does not cause the model to flip its factual judgments.
3. **Mean pooling dominates**: The signal comes from the full input context (including the factual statement), not from any special "deception state" at the generation point.

### Layer-wise Signal Development

The behavioral probe accuracy follows a characteristic pattern:
- **Layers 0-6**: Near-chance for last-token; moderate (~69%) for mean-pooled
- **Layers 7-14**: Gradual increase as factual knowledge representations form
- **Layers 14-21**: Peak accuracy region (75% mean-pooled)
- **Layers 22-28**: Slight decline, suggesting final layers specialize for token prediction

This matches known patterns where factual knowledge is primarily encoded in middle layers of transformer models.

### Condition Probe Insights

The perfect accuracy of condition probes with mean pooling is trivially explained: the prompt prefix ("honest" vs "deceptive") is directly present in the input tokens and its information is preserved through all layers. The more interesting observation is the last-token condition probe's growth curve:
- Layer 0: ~49% (no information has propagated from prompt to sequence end)
- Layer 7: ~79% (attention begins integrating prompt information)
- Layer 14: ~88% (strong integration)
- Layer 21: ~91% (near-maximum)

This shows how contextual information flows from early tokens to later positions through the transformer layers.

## Deliverables

| Deliverable | Path | Description |
|-------------|------|-------------|
| Trained probe | `final_probe/probe.pkl` | LogisticRegression, layer 18, mean aggregation |
| Behavioral labels | `behavioral_labels.jsonl` | 1000 examples with label=0 (honest) / label=1 (deceptive) |
| Metrics | `metrics.json` | Primary metrics and comparison data |
| All results | `all_probe_results.json` | Full results for all layer x aggregation x label configurations |
| Layer accuracy plot | `layer_accuracy_by_label.png` | CV accuracy across layers |
| All configs overlay | `all_configs_overlay.png` | All configurations compared |
| ROC AUC plot | `roc_auc_by_layer.png` | ROC AUC by layer |
| Heatmap | `accuracy_heatmap.png` | Accuracy heatmap across layers and aggregations |
| PCA visualization | `pca_visualization.png` | PCA of best layer, colored by different labels |
| PCA across layers | `pca_across_layers.png` | How PCA clusters develop across layers |
| Confusion matrix | `confusion_matrix.png` | Confusion matrix for best probe |
| Summary bar chart | `best_probe_summary.png` | Best probe per configuration |

## Conclusions

1. **Qwen2.5-1.5B (base model) does not meaningfully follow deceptive role-play prompts.** The "pretend you are deceptive" instruction has essentially no effect on the model's factual judgments, consistent with the limited instruction-following capabilities of base (pre-trained only) models.

2. **Behavioral deception is moderately linearly separable (75% accuracy, 0.843 AUC)** in middle layers using mean-pooled representations. However, this signal primarily reflects factual knowledge confidence rather than a dedicated deception mechanism.

3. **Mean-pooled token representations vastly outperform last-token representations** for detecting behavioral deception, indicating the relevant signal is distributed across the input sequence.

4. **The deception signal peaks in layers 14-21** (out of 28), consistent with the known localization of factual knowledge in middle transformer layers.

5. **Prompt condition (honest vs. deceptive) is trivially detectable** from mean-pooled representations (100% accuracy at all layers) due to the literal text differences, but shows an interesting growth curve in last-token representations that reveals how contextual information propagates through layers.
