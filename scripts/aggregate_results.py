#!/usr/bin/env python3
"""
Aggregate results from multiple probing experiments.
"""
import argparse
import json
from pathlib import Path
import pandas as pd


def collect_results(results_dir):
    """Collect all metrics.json files from results directory."""
    results_dir = Path(results_dir)
    all_results = []

    for metrics_file in results_dir.rglob("metrics.json"):
        try:
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)

            # Parse directory structure to extract metadata
            # Expected: results/<agent>_<config>/<task>_<model>_<id>/metrics.json
            parts = metrics_file.relative_to(results_dir).parts

            if len(parts) >= 3:
                agent_config = parts[0]  # e.g., "claude_claude-opus-4-6"
                task_model = parts[1]     # e.g., "sentiment_analysis_google__gemma-2-2b_12345"

                # Parse agent and config
                if '_' in agent_config:
                    agent_parts = agent_config.split('_')
                    agent = agent_parts[0]
                    config = '_'.join(agent_parts[1:])
                else:
                    agent = agent_config
                    config = "unknown"

                # Parse task and model
                task_parts = task_model.split('_')
                if len(task_parts) >= 2:
                    # Find where model name starts (contains __)
                    task_end = next((i for i, p in enumerate(task_parts) if '__' in p), 1)
                    task = '_'.join(task_parts[:task_end])
                    model = '_'.join(task_parts[task_end:]).rsplit('_', 1)[0]  # Remove ID
                else:
                    task = task_parts[0] if task_parts else "unknown"
                    model = "unknown"

                metrics['agent'] = agent
                metrics['config'] = config
                metrics['task'] = task
                metrics['model'] = model.replace('__', '/')
                metrics['result_path'] = str(metrics_file.parent)

                all_results.append(metrics)

        except Exception as e:
            print(f"Error processing {metrics_file}: {e}")

    return all_results


def create_summary_table(results):
    """Create summary table from results."""
    df = pd.DataFrame(results)

    if df.empty:
        print("No results found!")
        return df

    # Group by task and model, compute statistics
    summary = df.groupby(['task', 'model', 'agent', 'config']).agg({
        'accuracy': ['mean', 'std', 'count'],
        'f1': ['mean', 'std'],
    }).round(3)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Aggregate probing experiment results")
    parser.add_argument('--results-dir', default='../results', help="Results directory")
    parser.add_argument('--output', default='aggregated_results.csv', help="Output CSV file")
    parser.add_argument('--format', choices=['csv', 'json', 'markdown'], default='csv', help="Output format")
    args = parser.parse_args()

    print(f"Collecting results from: {args.results_dir}")
    results = collect_results(args.results_dir)
    print(f"Found {len(results)} result files")

    if not results:
        print("No results to aggregate!")
        return

    # Create DataFrame
    df = pd.DataFrame(results)

    # Save raw results
    if args.format == 'csv':
        df.to_csv(args.output, index=False)
        print(f"Raw results saved to: {args.output}")
    elif args.format == 'json':
        df.to_json(args.output, orient='records', indent=2)
        print(f"Raw results saved to: {args.output}")

    # Create and display summary
    summary = create_summary_table(results)
    print("\n=== Summary ===")
    print(summary)

    # Save summary
    summary_file = args.output.replace('.csv', '_summary.csv').replace('.json', '_summary.csv')
    summary.to_csv(summary_file)
    print(f"\nSummary saved to: {summary_file}")

    # Print top performers
    print("\n=== Top Performers (by accuracy) ===")
    top = df.nlargest(10, 'accuracy')[['task', 'model', 'agent', 'config', 'accuracy', 'f1']]
    print(top.to_string(index=False))


if __name__ == '__main__':
    main()
