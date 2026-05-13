#!/usr/bin/env python3
"""
Compute summary statistics for probing experiments.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def collect_metrics(results_dir):
    """Collect all metrics from results directory."""
    results_dir = Path(results_dir)
    all_metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'roc_auc': [],
    }

    for metrics_file in results_dir.rglob("metrics.json"):
        try:
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)

            for key in all_metrics.keys():
                if key in metrics:
                    all_metrics[key].append(metrics[key])

        except Exception as e:
            print(f"Error processing {metrics_file}: {e}")

    return all_metrics


def compute_statistics(values):
    """Compute summary statistics."""
    if not values:
        return {
            'count': 0,
            'mean': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0,
            'median': 0.0,
        }

    return {
        'count': len(values),
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
        'min': float(np.min(values)),
        'max': float(np.max(values)),
        'median': float(np.median(values)),
    }


def main():
    parser = argparse.ArgumentParser(description="Compute summary metrics")
    parser.add_argument('--results-dir', default='../results', help="Results directory")
    parser.add_argument('--output', default='summary_metrics.json', help="Output JSON file")
    args = parser.parse_args()

    print(f"Collecting metrics from: {args.results_dir}")
    all_metrics = collect_metrics(args.results_dir)

    # Compute statistics for each metric
    summary = {}
    for metric_name, values in all_metrics.items():
        summary[metric_name] = compute_statistics(values)

    # Print summary
    print("\n=== Summary Statistics ===")
    for metric_name, stats in summary.items():
        if stats['count'] > 0:
            print(f"\n{metric_name.upper()}:")
            print(f"  Count:  {stats['count']}")
            print(f"  Mean:   {stats['mean']:.3f}")
            print(f"  Std:    {stats['std']:.3f}")
            print(f"  Min:    {stats['min']:.3f}")
            print(f"  Max:    {stats['max']:.3f}")
            print(f"  Median: {stats['median']:.3f}")

    # Save to file
    with open(args.output, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to: {args.output}")


if __name__ == '__main__':
    main()
