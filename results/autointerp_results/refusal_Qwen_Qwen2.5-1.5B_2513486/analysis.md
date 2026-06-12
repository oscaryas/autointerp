# Refusal-Worthiness Probing Analysis: Qwen/Qwen2.5-1.5B

## Overview

This analysis investigates whether the Qwen2.5-1.5B base model internally represents **refusal-worthiness** — the distinction between harmful/adversarial prompts that should be refused versus benign instruction-following prompts. We trained linear probes on the model's hidden state activations to predict whether each input is refusal-worthy (harmful/toxic) or benign.

## Methodology

### Dataset
- **800 prompts total**: 400 from HarmBench (adversarial/harmful requests) and 400 from Alpaca (benign instructions)
- Balanced binary classification: label=1 (refusal-worthy/harmful), label=0 (benign)

### Behavioral Labeling Process
We first ran the Qwen2.5-1.5B base model on all 800 prompts to observe its actual refusal behavior. Since this is a **base model** (not instruction-tuned), it only explicitly refused 57/800 prompts (7.1%), with 52 refusals from HarmBench and 5 from Alpaca. The base model often complies with harmful requests because it lacks safety fine-tuning.

Given that the task asks about **refusal-worthiness** (whether a prompt *should* be refused, not just whether this base model *does* refuse), and the severe class imbalance (57 vs 743), we used source-based labels as the primary label for probing:
- `label=1`: HarmBench prompts (harmful/adversarial — refusal-worthy)
- `label=0`: Alpaca prompts (benign instructions)

This labeling approach better captures the concept of refusal-worthiness and provides balanced classes for reliable probe training.

### Probing Setup
- **Model**: Qwen/Qwen2.5-1.5B (28 transformer layers + embedding layer = 29 total hidden states)
- **Activations**: Extracted from all 29 layers using `AutoModel` (base encoder)
- **Aggregation strategies**: mean-pooling, last-token, first-token, max-pooling
- **Probe**: Logistic regression with StandardScaler normalization
- **Evaluation**: 5-fold stratified cross-validation + 80/20 train/test split
- **Metrics**: Accuracy, F1, ROC-AUC, Precision, Recall

## Results

### Best Probe Configuration
| Metric | Value |
|--------|-------|
| **Best Layer** | 20 (of 28) |
| **Best Aggregation** | mean-pooling |
| **CV Accuracy** | 0.995 ± 0.005 |
| **CV F1** | 0.995 ± 0.005 |
| **CV ROC-AUC** | 1.000 ± 0.000 |
| **Test Accuracy** | 1.000 |
| **Test F1** | 1.000 |
| **Test ROC-AUC** | 1.000 |

### Top 5 Layer-Aggregation Combinations (by CV AUC)
| Rank | Aggregation | Layer | CV AUC |
|------|------------|-------|--------|
| 1 | mean | 20 | 0.99997 |
| 2 | mean | 16 | 0.99994 |
| 3 | last | 17 | 0.99994 |
| 4 | mean | 13 | 0.99991 |
| 5 | mean | 19 | 0.99991 |

### Performance by Aggregation Strategy
| Aggregation | Best Layer | Best CV AUC | Best CV Accuracy |
|------------|-----------|-------------|-----------------|
| **mean** | 20 | 0.99997 | 0.995 |
| **last** | 17 | 0.99994 | 0.989 |
| **max** | 20 | 0.99978 | 0.993 |
| **first** | 0 | 0.856 | 0.815 |

### Layer-by-Layer Analysis (Mean Aggregation)

The probe accuracy increases progressively through the layers:
- **Embedding (Layer 0)**: Acc=0.978, AUC=0.997 — Already strong, suggesting lexical features carry significant signal
- **Early layers (1-5)**: Acc≈0.980, AUC≈0.999 — Slight improvement
- **Middle layers (6-15)**: Acc=0.984→0.993, AUC=0.999→1.000 — Steady improvement; contextual representations sharpen
- **Late-middle layers (16-21)**: Acc=0.994→0.995, AUC≈1.000 — Peak performance; best probe at layer 20
- **Final layers (22-28)**: Acc=0.990→0.995, AUC≈1.000 — Slight decline in accuracy; representations may begin specializing for next-token prediction

## Key Findings

### 1. Which layers best predict refusal-worthiness?
**Layers 13-23** achieve the highest probe performance, with the peak at **layer 20** (of 28 total). This corresponds to roughly the 70th percentile depth of the model. The pattern of peak performance in upper-middle layers is consistent with prior work on probing for high-level semantic features in transformers — earlier layers capture syntax and local semantics, while the deepest layers specialize for output generation.

### 2. Which token positions are most informative?
- **Mean-pooling**: Best overall (AUC=0.99997). Averaging across all token positions captures the global semantic content of the prompt.
- **Last-token**: Second best (AUC=0.99994). The last token position accumulates information from the full sequence via causal attention.
- **Max-pooling**: Strong but slightly lower (AUC=0.99978). Max-pooling captures the most "activated" features but may be noisier.
- **First-token**: Significantly weaker (AUC=0.856). The first token has no access to the rest of the sequence in a causal model, so it relies entirely on the initial token's representation.

The large gap between first-token and other strategies confirms that refusal-worthiness is a **global property** that requires information from the full input, not just the beginning.

### 3. Does the model develop internal representations of refusal-worthiness before producing output?
**Yes, strongly.** Even at the embedding layer (layer 0), the model achieves AUC=0.997 with mean-pooling, and this improves to near-perfect in middle-to-late layers. This demonstrates that:

1. **Lexical cues** alone carry substantial signal — harmful prompts tend to use distinct vocabulary (e.g., words related to violence, illegal activities, weapons).
2. **Contextual representations** in deeper layers further refine the distinction, suggesting the model builds richer semantic understanding of prompt intent.
3. The high probe accuracy across many layers indicates that refusal-worthiness information is **broadly distributed** in the model's representations, not concentrated in a single layer.

## Interpretation and Caveats

### Why such high performance?
The near-perfect probe accuracy reflects the fact that HarmBench and Alpaca prompts are linguistically quite different. HarmBench prompts contain requests related to harmful, illegal, or unethical activities, while Alpaca prompts are general-purpose instructions. A linear probe can easily separate these two distributions in the model's representation space.

This high performance is partly due to **lexical signal** (harmful prompts use distinctive words), as evidenced by the strong embedding-layer performance. However, the improvement from layer 0 (0.978) to layer 20 (0.995) indicates that the model does build additional, non-lexical representations that help distinguish harmful from benign content.

### Relationship to base model behavior
The Qwen2.5-1.5B base model rarely refuses prompts (only 7.1% refusal rate), yet its internal representations strongly distinguish refusal-worthy from benign content. This means the model "knows" when content is potentially harmful (as shown by the probe) but lacks the safety training to act on this knowledge. This is a common finding in LLM safety research — base models learn to represent safety-relevant features during pretraining (likely from safety-related content in the training data) even without explicit safety fine-tuning.

### Limitations
1. **Distribution confound**: The high probe accuracy may partly reflect distributional differences between HarmBench and Alpaca (e.g., prompt length, writing style) rather than purely safety-relevant features.
2. **Linear probe expressivity**: We only tested linear probes; nonlinear probes might reveal additional structure.
3. **Base model**: Since this is an untuned base model, the refusal representations may differ from what would be found in an instruction-tuned or RLHF-aligned version.

## Deliverables
1. `behavioral_labels.jsonl` — Labels for all 800 examples (1=refusal-worthy, 0=benign)
2. `final_probe/probe.pkl` — Best trained probe (layer 20, mean aggregation)
3. `metrics.json` — Comprehensive metrics across all layers and aggregations
4. `probe_results_layers.png` — Multi-panel visualization of probe performance across layers
5. `probe_auc_heatmap.png` — Heatmap of AUC across layers and aggregation strategies
6. `probe_roc_curve.png` — ROC curve for the best probe
7. `probe_best_aggregation.png` — Detailed metrics for the best aggregation strategy
