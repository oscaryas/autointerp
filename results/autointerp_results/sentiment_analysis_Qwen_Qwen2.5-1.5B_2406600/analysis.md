# Sentiment Analysis Probing Experiment: Qwen/Qwen2.5-1.5B

## Overview
- **Model**: Qwen/Qwen2.5-1.5B
- **Task**: Binary sentiment classification (positive vs negative)
- **Dataset**: 20 IMDB movie reviews (10 positive, 10 negative)
- **Model architecture**: 28 transformer layers, hidden dimension 1536
- **Probe type**: Logistic Regression (sklearn)
- **Evaluation**: 5-fold stratified cross-validation
- **Aggregation methods tested**: first, last, mean, max
- **Total probe configurations**: 116

## Key Findings

### 1. Best Probe Configuration
- **Best Layer**: 9 (out of 28 hidden layers)
- **Best Aggregation**: last
- **CV Accuracy**: 1.000 (+/- 0.000)
- **CV F1 Score**: 1.000
- **CV ROC-AUC**: 1.000

### 2. Layer-wise Analysis

Performance by layer group (using best aggregation method):

- **Embedding (L0)**: Mean Acc = 0.800, Max Acc = 0.800
- **Early (L1-L7)**: Mean Acc = 0.900, Max Acc = 0.950
- **Middle (L8-L20)**: Mean Acc = 0.988, Max Acc = 1.000
- **Late (L21-L28)**: Mean Acc = 1.000, Max Acc = 1.000

Detailed layer-by-layer results (best aggregation: last):

| Layer | Accuracy | F1 | ROC-AUC |
|-------|----------|-----|---------|
|  0 | 0.800 | 0.733 | 0.800 |
|  1 | 0.850 | 0.800 | 1.000 |
|  2 | 0.900 | 0.867 | 1.000 |
|  3 | 0.950 | 0.933 | 0.950 |
|  4 | 0.900 | 0.867 | 0.950 |
|  5 | 0.900 | 0.867 | 1.000 |
|  6 | 0.950 | 0.933 | 0.950 |
|  7 | 0.850 | 0.827 | 0.900 |
|  8 | 0.900 | 0.900 | 0.950 |
|  9 | 1.000 | 1.000 | 1.000 | **BEST**
| 10 | 0.950 | 0.933 | 1.000 |
| 11 | 1.000 | 1.000 | 1.000 |
| 12 | 1.000 | 1.000 | 1.000 |
| 13 | 1.000 | 1.000 | 1.000 |
| 14 | 1.000 | 1.000 | 1.000 |
| 15 | 1.000 | 1.000 | 1.000 |
| 16 | 1.000 | 1.000 | 1.000 |
| 17 | 1.000 | 1.000 | 1.000 |
| 18 | 1.000 | 1.000 | 1.000 |
| 19 | 1.000 | 1.000 | 1.000 |
| 20 | 1.000 | 1.000 | 1.000 |
| 21 | 1.000 | 1.000 | 1.000 |
| 22 | 1.000 | 1.000 | 1.000 |
| 23 | 1.000 | 1.000 | 1.000 |
| 24 | 1.000 | 1.000 | 1.000 |
| 25 | 1.000 | 1.000 | 1.000 |
| 26 | 1.000 | 1.000 | 1.000 |
| 27 | 1.000 | 1.000 | 1.000 |
| 28 | 1.000 | 1.000 | 1.000 |

### 3. Aggregation Method Comparison

| Method | Best Layer | Best Accuracy | Best F1 | Best AUC |
|--------|-----------|---------------|---------|----------|
| first | 0 | 0.650 | 0.627 | 0.800 |
| last * | 9 | 1.000 | 1.000 | 1.000 |
| mean | 16 | 1.000 | 1.000 | 1.000 |
| max | 27 | 1.000 | 1.000 | 1.000 |

### 4. Feature Importance Analysis

The probe at layer 9 with last aggregation has the following weight characteristics:
- **Weight mean**: 0.0005
- **Weight std**: 0.0125
- **Weight range**: [-0.0353, 0.0336]
- **Number of features**: 1536 (hidden dimension)

Top 10 most important dimensions (by absolute weight):

- Dimension 117: weight = -0.0353 (indicates negative sentiment)
- Dimension 542: weight = -0.0345 (indicates negative sentiment)
- Dimension 1423: weight = 0.0336 (indicates positive sentiment)
- Dimension 1455: weight = 0.0327 (indicates positive sentiment)
- Dimension 882: weight = -0.0324 (indicates negative sentiment)
- Dimension 232: weight = 0.0313 (indicates positive sentiment)
- Dimension 913: weight = 0.0306 (indicates positive sentiment)
- Dimension 618: weight = -0.0303 (indicates negative sentiment)
- Dimension 454: weight = -0.0301 (indicates negative sentiment)
- Dimension 694: weight = -0.0301 (indicates negative sentiment)

### 5. Interpretability Notes

The linear probe learns a hyperplane in the model's representation space that separates 
positive from negative sentiment. The fact that a simple linear classifier can achieve 
an accuracy of 100.0% suggests that sentiment information is 
**linearly encoded** in the model's hidden states.

The best probe performance at layer 9 (early layers) is notable. This suggests the model encodes sentiment features early in its processing pipeline, possibly tied to lexical-level sentiment cues.

## Visualizations

1. `probe_performance_layers.png` - Accuracy, F1, ROC-AUC across all layers for each aggregation method
2. `probe_heatmap.png` - Heatmap of accuracy across layers and aggregation methods
3. `feature_importance.png` - Top feature weights and weight distribution
4. `layer_group_comparison.png` - Performance by layer group

## Files

- `final_probe/probe.pkl` - Best trained probe (pickle format)
- `metrics.json` - Comprehensive metrics for all configurations
- `analysis.md` - This analysis document
