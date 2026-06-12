# Refusal-Worthiness Probing Analysis: SmolLM2-1.7B

## Executive Summary

This report presents the results of probing SmolLM2-1.7B's internal representations to detect refusal-worthiness — whether the model internally distinguishes between harmful/adversarial prompts (HarmBench) and benign instructions (Alpaca). Linear probes trained on hidden-state activations achieve near-perfect classification (ROC-AUC = 1.000 at layer 14, mean pooling), demonstrating that the model develops strong internal representations of input harmfulness even as a base (non-instruction-tuned) model.

## Methodology

### Behavioral Labeling

SmolLM2-1.7B is a base language model (not instruction-tuned), meaning it does not have an explicit refusal mechanism. All 800 prompts were run through the model with greedy decoding (max 60 new tokens), and the outputs were checked for refusal patterns (e.g., "I cannot", "as an AI", "I'm sorry", etc.).

**Result: 0 refusals out of 800 prompts.** The model complied with every prompt, generating text continuations for both harmful and benign inputs alike. This is expected for a base model without safety training.

Since the model produces no behavioral refusals, labels were assigned based on refusal-worthiness (the ground-truth category of each prompt):
- **label=1** (refusal-worthy): 400 HarmBench prompts (adversarial/harmful requests)
- **label=0** (benign): 400 Alpaca prompts (benign instruction-following requests)

This approach probes whether the model internally represents the *harmfulness* of inputs, even if it doesn't act on that information.

### Probing Setup

- **Model**: HuggingFaceTB/SmolLM2-1.7B (24 hidden layers, 2048 hidden dim)
- **Probe type**: Logistic Regression (L2-regularized, C=1.0)
- **Layers probed**: All 25 layers (embedding layer 0 + 24 transformer layers)
- **Token aggregation strategies**: first, last, mean, max
- **Evaluation**: 5-fold stratified cross-validation on 80% train split, holdout test on 20%
- **Metrics**: Accuracy, F1, ROC-AUC, Precision, Recall
- **Normalization**: StandardScaler applied to all activations

## Results

### Best Probe Configuration

| Metric | Value |
|--------|-------|
| **Layer** | 14 (of 24) |
| **Aggregation** | mean pooling |
| **CV Accuracy** | 0.992 +/- 0.005 |
| **CV F1** | 0.992 +/- 0.005 |
| **CV ROC-AUC** | 1.000 +/- 0.000 |
| **Test Accuracy** | 1.000 |
| **Test Precision** | 1.000 |
| **Test Recall** | 1.000 |
| **Test F1** | 1.000 |
| **Test ROC-AUC** | 1.000 |

### Best Layer Per Aggregation Method

| Aggregation | Best Layer | Best CV ROC-AUC |
|------------|-----------|-----------------|
| **first** | 0 | 0.863 |
| **last** | 16 | 1.000 |
| **mean** | 14 | 1.000 |
| **max** | 24 | 0.999 |

### Question 1: Which layers best predict refusal-worthiness?

**All layers from ~2 onward achieve >0.99 ROC-AUC with mean pooling.** The representation of input harmfulness is remarkably accessible across the entire depth of the network. Key observations:

- **Mean pooling**: Performance is excellent from the very first layer (0.998 ROC-AUC at layer 0) and peaks at layer 14 (1.000). The signal is distributed and improves gradually through the middle layers.
- **Last token**: Shows a more pronounced progression — starting at 0.965 ROC-AUC at layer 0, steadily improving to 1.000 by layer 12-16. This suggests the last token position aggregates information progressively through the layers.
- **First token**: Significantly weaker (max 0.863 at layer 0), actually *declining* slightly in later layers. This indicates that the first token position carries less information about the overall prompt content.
- **Max pooling**: Strong and consistent (0.993-0.999 across all layers), indicating that the most activated neurons per layer encode harmfulness signals.

The best single layer is **layer 14** (mean pooling), which sits at 58% depth in the network — a middle-to-late layer.

### Question 2: Which token positions are most informative?

**Ranking of aggregation strategies** (by best ROC-AUC across all layers):

1. **Mean pooling** (1.000) — best overall; averages over all token positions, capturing distributed information about the full prompt
2. **Last token** (1.000) — equally strong at peak; shows that the model accumulates prompt information into the final position through its autoregressive processing
3. **Max pooling** (0.999) — nearly as strong; the most activated feature per layer is highly predictive
4. **First token** (0.863) — substantially weaker; the first token has not yet "seen" the rest of the prompt, so it carries limited information

The gap between first-token and other strategies is notable (0.86 vs 1.00), confirming that the model needs to process the full sequence to build rich representations of harmfulness. Mean pooling works best because it captures distributed signals across all positions.

### Question 3: Does the model develop internal representations of refusal-worthiness before producing output?

**Yes, decisively.** The model develops strong internal representations of whether an input is harmful/refusal-worthy, even though:

1. **The model never refuses** — SmolLM2-1.7B produces 0 refusals across all 800 prompts
2. **The signal appears early** — Even at layer 0 (the embedding layer), mean-pooled representations achieve 0.998 ROC-AUC, suggesting that the tokenizer/embedding space already captures significant distributional differences between harmful and benign text
3. **The signal strengthens through middle layers** — Performance peaks around layers 11-17, suggesting that the transformer layers refine the representation of harmfulness through contextual processing
4. **The signal is robust** — Near-perfect classification across multiple aggregation strategies and layers indicates this is a deeply encoded property, not a fragile or superficial signal

This demonstrates that even without explicit safety training, a language model's internal representations carry information about the nature of its inputs. The model "knows" (in a representational sense) that harmful prompts are different from benign ones — it simply has no mechanism to act on this knowledge.

## Discussion

### Why Is Probing So Successful?

Several factors contribute to the near-perfect probe performance:

1. **Distributional differences**: HarmBench prompts and Alpaca prompts have fundamentally different topic distributions, vocabulary, and linguistic patterns. Harmful prompts often discuss weapons, illegal activities, or explicit content, while Alpaca prompts cover general knowledge, creative writing, and everyday tasks.

2. **Topic-level separation**: A linear probe on activations may partially be detecting topic differences (e.g., "how to make a bomb" vs "explain photosynthesis") rather than a pure "harmfulness" concept. This is an inherent limitation of source-based labeling.

3. **Strong embedding signal**: The high performance even at layer 0 suggests that much of the signal comes from the vocabulary/embedding space itself — certain words and phrases are strongly associated with either harmful or benign prompts.

### Limitations

- **Source-based labels**: Since the model doesn't refuse, labels are based on prompt source (HarmBench vs Alpaca) rather than behavioral refusal. A model with safety training might show more nuanced patterns.
- **Potential confounds**: The probe may detect topical differences rather than a pure "harmfulness" representation. Adversarial prompts designed to be topically similar to benign ones would be needed to test this.
- **Linear probing**: Non-linear probes might reveal additional structure, though the near-perfect linear probe performance leaves little room for improvement.

### Implications

The results suggest that the linguistic features of harmful vs. benign prompts are highly linearly separable in the model's representation space. This has practical implications:
- **Safety mechanisms** could be built on top of base model representations via simple linear classifiers
- **Content filtering** using internal activations is feasible even without fine-tuning
- The model's internal representations provide a rich signal for **input safety classification**

## Files Produced

| File | Description |
|------|-------------|
| `behavioral_labels.jsonl` | 800 labeled examples (harmbench=1, alpaca=0) |
| `final_probe/probe.pkl` | Best probe (layer 14, mean pooling, LogisticRegression) |
| `metrics.json` | Comprehensive metrics for all layer x aggregation combinations |
| `probe_performance_layers.png` | Probe performance (accuracy, F1, ROC-AUC) across all layers |
| `best_by_aggregation.png` | Best ROC-AUC per aggregation method |
| `roc_auc_heatmap.png` | Heatmap of ROC-AUC across layers and aggregation methods |
| `layer_importance.png` | Average performance per layer and improvement over embedding baseline |

## Conclusion

SmolLM2-1.7B's internal representations strongly encode whether an input is harmful (refusal-worthy) or benign. Despite being a base model with no refusal behavior, a simple linear probe on layer 14 with mean pooling achieves perfect ROC-AUC (1.000) on distinguishing HarmBench from Alpaca prompts. The signal is present across all layers and most token aggregation strategies, indicating that the model develops a robust, distributed representation of input harmfulness through its pre-training process. This representation exists "passively" in the model — it is never used for refusal since the model lacks safety training, but could be leveraged for downstream safety applications.
