#!/usr/bin/env python3
"""
Generate prompts for probing tasks.
"""
import argparse
import json
from pathlib import Path


def count_examples(data_file):
    """Count examples in JSONL file."""
    try:
        with open(data_file, 'r') as f:
            return sum(1 for _ in f)
    except FileNotFoundError:
        return 0


def load_task_description(task_dir):
    """Load task description from file."""
    task_desc_file = Path(task_dir) / "task_description.txt"
    if task_desc_file.exists():
        return task_desc_file.read_text().strip()
    return "No task description provided."


def get_concept_from_task(task_name):
    """Infer concept name from task name."""
    concept_map = {
        'sentiment_analysis': 'sentiment (positive vs negative)',
        'deception': 'deception (honest vs deceptive model behavior under role-play prompting)',
        'refusal': 'refusal-worthiness (benign vs harmful/toxic inputs)',
        'truth_analysis': 'factual truth (supported vs refuted claims)',
        'sycophancy': 'sycophancy (model capitulates under disagreement pressure)',
        'factuality': 'factual correctness',
        'toxicity': 'toxic vs non-toxic language',
        'subjectivity': 'subjective vs objective statements',
        'formality': 'formal vs informal language',
    }
    return concept_map.get(task_name, task_name.replace('_', ' '))


def main():
    parser = argparse.ArgumentParser(description="Generate prompt for probing task")
    parser.add_argument('--task', required=True, help="Task name")
    parser.add_argument('--model', required=True, help="Model name or path")
    parser.add_argument('--time-limit', type=int, default=2, help="Time limit in hours")
    parser.add_argument('--agent', default='claude', help="Agent name (for agent-specific modifications)")
    args = parser.parse_args()

    # Load prompt template — task-specific template takes precedence if present
    script_dir = Path(__file__).parent
    task_specific_template = script_dir.parent / args.task / "prompt.txt"
    if task_specific_template.exists():
        template = task_specific_template.read_text()
    else:
        template = (script_dir / "prompt.txt").read_text()

    # Load task files
    task_dir = script_dir.parent / "tasks" / args.task
    data_file = task_dir / "labeled_data.jsonl"

    # Get task info
    n_examples = count_examples(data_file)
    task_description = load_task_description(task_dir)
    concept = get_concept_from_task(args.task)

    # Read optional paper_title.txt from task_context/
    paper_title_file = task_dir / "task_context" / "paper_title.txt"
    paper_title = paper_title_file.read_text().strip() if paper_title_file.exists() else ""

    # Fill in template
    prompt = template.replace('{model}', args.model)
    prompt = prompt.replace('{concept}', concept)
    prompt = prompt.replace('{task_description}', task_description)
    prompt = prompt.replace('{n_examples}', str(n_examples))
    prompt = prompt.replace('{time_limit}', str(args.time_limit))
    prompt = prompt.replace('{paper_title}', paper_title)

    # Agent-specific modifications
    if args.agent == 'claude':
        # Claude Code specific: remind about non-interactive mode
        prompt += "\n\nIMPORTANT: You are running in non-interactive mode. Ensure all processes finish before your final message."

    print(prompt)


if __name__ == '__main__':
    main()
