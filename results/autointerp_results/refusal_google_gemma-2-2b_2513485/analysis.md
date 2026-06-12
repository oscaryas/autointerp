# Refusal-Worthiness Probing Analysis: google/gemma-2-2b

## Summary

This analysis investigates whether google/gemma-2-2b's internal representations encode information about the "refusal-worthiness" of input prompts. We trained linear probes on hidden state activations to distinguish between harmful/adversarial prompts (HarmBench) and benign instruction-following prompts (Alpaca).

## Behavioral Labeling

**Key Finding**: google/gemma-2-2b is a base (pre-trained) model without safety fine-tuning. When run on all 800 prompts, it produced **zero refusals** (0/800). The model complied with every request, including harmful ones from HarmBench. This is expected behavior for a base model that has not undergone RLHF or safety training.

Since behavioral refusal labels yielded only a single class (all compliance), we used content-type labels based on the metadata source:
- **label=1**: HarmBench prompts (400 adversarial/harmful requests)
- **label=0**: Alpaca prompts (400 benign instruction-following requests)

This approach probes whether the model's internal representations distinguish harmful from benign content, even though the model does not behaviorally refuse harmful prompts.

## Methodology

1. **Model**: google/gemma-2-2b (26 transformer layers + 1 embedding layer = 27 total hidden state layers)
2. **Activation Extraction**: Used `AutoModel` to extract hidden states from all 27 layers
3. **Token Aggregation**: Tested 4 strategies: first token, last token, mean pooling, max pooling
4. **Probe**: Logistic Regression with StandardScaler normalization
5. **Evaluation**: 5-fold stratified cross-validation
6. **Optimization**: Grid search over C values [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

## Results

### Best Configuration
- **Layer**: 13 (middle of the network)
- **Aggregation**: Mean pooling
- **Regularization**: C=0.1
- **Cross-validated metrics**:
  - Accuracy: 0.805
  - Precision: 0.924
  - Recall: 0.665
  - F1: 0.773
  - ROC-AUC: 0.878

### Full-data evaluation (evaluate.py):
- Accuracy: 0.936
- Precision: 0.887
- Recall: 1.000
- F1: 0.940
- ROC-AUC: 0.997

### Question 1: Which layers best predict refusal-worthiness?

The probe performance varies significantly across layers and aggregation strategies:

**Mean pooling** (best overall):
- Performance is relatively consistent across layers, with a slight peak in the middle layers (5-13)
- Best layer: 13 (AUC=0.878), closely followed by layers 5-12 (AUC=0.860-0.875)
- Even the embedding layer (layer 0) achieves AUC=0.843, suggesting significant lexical cues

**Max pooling**:
- Similar pattern to mean pooling but slightly lower performance
- Peak at layers 10-15 (AUC=0.868-0.872)

**Last token**:
- Shows more variation across layers
- Starts lower (AUC=0.711 at layer 0), improves through middle layers
- Best at layers 6 (AUC=0.839) and 26 (AUC=0.838)
- More layer-dependent than mean/max pooling

**First token**:
- Near chance level across all layers (AUC ~0.51)
- Completely uninformative; the BOS token does not carry content-specific information

### Question 2: Which token positions are most informative?

Ranking of aggregation strategies (by best achievable AUC):

1. **Mean pooling** (AUC=0.878): Most informative overall. Captures global content features across the entire sequence.
2. **Max pooling** (AUC=0.872): Slightly worse than mean. Captures the most activated features, still effective.
3. **Last token** (AUC=0.838): Moderately informative. The last token aggregates contextual information but is more variable.
4. **First token** (AUC=0.536): Near chance. The BOS/start token carries almost no content information.

The dominance of mean pooling indicates that harmful vs. benign content is encoded distributedly across the token sequence, not concentrated in any single position.

### Question 3: Does the model develop internal representations of refusal-worthiness before producing output?

Yes. Despite never actually refusing harmful prompts (it is a base model without safety training), gemma-2-2b's internal representations contain sufficient information to distinguish harmful from benign content with ~88% AUC.

Key observations:
- **Early layers already encode some signal**: Even the embedding layer (layer 0) with mean pooling achieves AUC=0.843, indicating strong lexical features in harmful vs. benign text.
- **Middle layers peak**: Layers 5-13 show the highest probe accuracy, suggesting the model builds increasingly abstract representations of content type through these layers.
- **Later layers show slight decline**: Layers 17-26 show marginally lower performance (AUC=0.862-0.867 for mean pooling), potentially because later layers are more focused on next-token prediction rather than content classification.
- **The representation is stable**: Unlike models with safety training where refusal signals might sharply emerge at specific layers, gemma-2-2b's content-type encoding is relatively uniform across layers, consistent with pre-training objectives (next token prediction) rather than explicit safety training.

## Interpretation

The ~88% AUC probe accuracy reveals that the distributional properties of harmful vs. benign text create distinguishable internal representations even in a base model. This is likely driven by:

1. **Lexical features**: Harmful prompts contain specific vocabulary (weapons, drugs, attack methods) that creates distinct activation patterns
2. **Syntactic patterns**: HarmBench prompts tend to be imperative requests for specific harmful actions, while Alpaca prompts cover diverse instructional formats
3. **Topic clustering**: The pre-training process naturally groups related concepts, and harmful topics form distinguishable clusters

However, the 88% AUC (rather than near-100%) indicates this is a noisy signal. The base model has not learned an explicit "refusal" representation — it simply has distributional differences between harmful and benign content from pre-training statistics.

## Files Produced

| File | Description |
|------|-------------|
| `behavioral_labels.jsonl` | Labels for all 800 prompts (source-based: harmbench=1, alpaca=0) |
| `final_probe/probe.pkl` | Best trained probe (layer 13, mean pooling, C=0.1) |
| `metrics.json` | Full metrics including per-layer, per-aggregation results |
| `probe_performance_by_layer.png` | Multi-metric performance across layers |
| `roc_auc_by_layer.png` | ROC-AUC comparison by aggregation strategy |
| `heatmap_agg_layer.png` | Heatmap of AUC across aggregation x layer |
| `best_probe_metrics.png` | Bar chart of best probe's metrics |
| `regularization_sweep.png` | C value optimization results |
