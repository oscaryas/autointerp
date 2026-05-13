#!/usr/bin/env python3
"""
Visualize probing results.
"""
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def load_metrics(metrics_file):
    """Load metrics from JSON file."""
    with open(metrics_file, 'r') as f:
        return json.load(f)


def plot_layer_accuracy(results_dir, output_file='layer_accuracy.png'):
    """Plot accuracy across layers."""
    results_dir = Path(results_dir)

    # Collect all results
    data = []
    for metrics_file in results_dir.rglob("metrics.json"):
        try:
            metrics = load_metrics(metrics_file)
            if 'layer' in metrics and 'accuracy' in metrics:
                data.append({
                    'layer': metrics['layer'],
                    'accuracy': metrics['accuracy'],
                    'aggregation': metrics.get('aggregation', 'unknown'),
                })
        except Exception as e:
            print(f"Error processing {metrics_file}: {e}")

    if not data:
        print("No data to plot!")
        return

    df = pd.DataFrame(data)

    # Plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='layer', y='accuracy', hue='aggregation', marker='o')
    plt.xlabel('Layer')
    plt.ylabel('Probe Accuracy')
    plt.title('Probe Accuracy Across Layers')
    plt.legend(title='Aggregation')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Plot saved to: {output_file}")


def plot_comparison_bar(results_dir, output_file='comparison.png'):
    """Bar plot comparing different models/agents."""
    results_dir = Path(results_dir)

    data = []
    for metrics_file in results_dir.rglob("metrics.json"):
        try:
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)

            # Parse metadata from path
            parts = metrics_file.relative_to(results_dir).parts
            if len(parts) >= 2:
                agent_config = parts[0]
                task_model = parts[1]

                data.append({
                    'agent_config': agent_config,
                    'task': task_model.split('_')[0],
                    'accuracy': metrics.get('accuracy', 0.0),
                    'f1': metrics.get('f1', 0.0),
                })
        except Exception as e:
            continue

    if not data:
        print("No data to plot!")
        return

    df = pd.DataFrame(data)
    summary = df.groupby('agent_config')['accuracy'].mean().sort_values(ascending=False)

    plt.figure(figsize=(12, 6))
    summary.plot(kind='bar')
    plt.xlabel('Agent Configuration')
    plt.ylabel('Mean Accuracy')
    plt.title('Mean Probe Accuracy by Agent')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Plot saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Visualize probing results")
    parser.add_argument('--results-dir', default='../results', help="Results directory")
    parser.add_argument('--plot-type', choices=['layer', 'comparison', 'all'], default='all')
    parser.add_argument('--output-dir', default='.', help="Output directory for plots")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    if args.plot_type in ['layer', 'all']:
        print("Creating layer accuracy plot...")
        plot_layer_accuracy(args.results_dir, output_dir / 'layer_accuracy.png')

    if args.plot_type in ['comparison', 'all']:
        print("Creating comparison plot...")
        plot_comparison_bar(args.results_dir, output_dir / 'comparison.png')

    print("Visualization complete!")


if __name__ == '__main__':
    main()
