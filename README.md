# AutoInterp: Agent-Based Automated Probing Experiments

AutoInterp is a framework for using AI agents to conduct systematic probing experiments on language models. Agents are given a model and labeled data, then tasked with extracting activations, training linear probes, and analyzing what features the model has learned.

## Overview

Similar to PostTrainBench, AutoInterp evaluates agents' ability to conduct AI research autonomously. Instead of training models, agents conduct interpretability research by:
- Extracting activations from specified models
- Training linear probes on pre-labeled data
- Systematically analyzing different layers, positions, and pooling strategies
- Reporting probe accuracy, feature importance, and interpretability insights

## Quick Start

### Prerequisites

```bash
# Install runpodctl (Runpod CLI)
wget https://github.com/runpod/runpodctl/releases/latest/download/runpodctl-linux-amd64 -O runpodctl
chmod +x runpodctl
sudo mv runpodctl /usr/local/bin/

# Set API keys
export RUNPOD_API_KEY="your-runpod-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export OPENAI_API_KEY="your-openai-key"
```

### Setup

```bash
# 1. Build container (Runpod-compatible)
cd containers
bash build_container.sh

# 2. Download any required models/data
bash download_models.sh

# 3. Submit probing jobs
cd ../src/commit_utils
bash commit.sh
```

## Architecture

### Directory Structure

```
AutoInterp/
├── README.md                      # This file
├── agents/                        # Agent implementations
│   ├── claude/
│   │   ├── solve.sh              # Claude Code execution script
│   │   └── human_readable_trace.py
│   ├── codex/
│   │   ├── solve.sh              # Codex execution script
│   │   └── human_readable_trace.py
│   └── opencode/
│       └── solve.sh
├── containers/                    # Container configurations
│   ├── Dockerfile                # Main container definition
│   ├── requirements.txt          # Python dependencies
│   └── build_container.sh        # Build script
├── src/
│   ├── commit_utils/             # Job submission
│   │   ├── commit.sh             # Submit multiple jobs
│   │   ├── single_task.sh        # Single job runner
│   │   └── set_env_vars.sh       # Environment configuration
│   ├── run_task.sh               # Main task execution wrapper
│   ├── probing/
│   │   ├── tasks/                # Probing task definitions
│   │   │   └── <task_name>/
│   │   │       ├── labeled_data.jsonl    # Training data
│   │   │       ├── evaluate.py           # Evaluation script
│   │   │       ├── task_description.txt  # Task description
│   │   │       └── task_context/         # Additional context files
│   │   └── general/
│   │       ├── get_prompt.py     # Generate agent prompts
│   │       └── prompt.txt        # Prompt template
│   └── utils/                     # Utility scripts
│       ├── extract_activations.py
│       ├── train_probe.py
│       └── timestamp_lines.py
├── scripts/                       # Analysis scripts
│   ├── aggregate_results.py
│   ├── visualize_probes.py
│   └── compute_metrics.py
└── results/                       # Experiment results
    └── <agent>_<config>/
        └── <task>_<model>/
            ├── output.log
            ├── metrics.json
            ├── final_probe/
            └── analysis/
```

### Execution Flow

1. **Job Submission** (`src/commit_utils/commit.sh`)
   - Loops over models and probing tasks
   - Submits jobs to Runpod via API/CLI

2. **Task Execution** (`src/run_task.sh`)
   - Creates isolated job directory
   - Copies probing task files and labeled data
   - Generates agent prompt
   - Executes agent in Runpod container
   - Runs evaluation and collects results

3. **Agent Execution** (`agents/<agent>/solve.sh`)
   - Agent receives prompt with task description
   - Agent explores the model architecture
   - Agent extracts activations systematically
   - Agent trains probes on labeled data
   - Agent reports findings and saves best probe

4. **Evaluation** (`src/probing/tasks/<task>/evaluate.py`)
   - Validates agent's probe
   - Computes accuracy, precision, recall, F1, ROC-AUC
   - Generates visualizations
   - Outputs metrics.json

## Labeled Data Format

Each probing task provides labeled data in JSONL format:

```json
{
  "text": "I absolutely loved this movie!",
  "label": 1,
  "metadata": {
    "category": "sentiment",
    "source": "imdb",
    "split": "train"
  }
}
```

Fields:
- `text` (required): Input text to process
- `label` (required): Integer label for classification
- `metadata` (optional): Additional information

## Creating New Probing Tasks

1. Create task directory:
```bash
mkdir -p src/probing/tasks/my_task/{task_context,data}
```

2. Add required files:

**labeled_data.jsonl**
```jsonl
{"text": "example 1", "label": 0}
{"text": "example 2", "label": 1}
```

**task_description.txt**
```
Task: Detect whether the model represents [concept X]
Goal: Train linear probes to classify [description]
Labels: 0 = [class A], 1 = [class B]
```

**evaluate.py**
```python
#!/usr/bin/env python3
import argparse
from autointerp_utils import extract_activations, train_probe, compute_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--probe-path", default="final_probe")
    parser.add_argument("--json-output-file", default=None)
    args = parser.parse_args()

    # Load model and extract activations
    activations = extract_activations(args.model_path, "labeled_data.jsonl")

    # Load and evaluate probe
    probe = load_probe(args.probe_path)
    metrics = compute_metrics(probe, activations)

    # Save results
    if args.json_output_file:
        save_json(metrics, args.json_output_file)

    print(f"Accuracy: {metrics['accuracy']:.3f}")

if __name__ == "__main__":
    main()
```

3. Add task to commit.sh:
```bash
tasks=(
    "my_task"
    "sentiment_analysis"
    # ...
)
```

## Agent Prompt

Agents receive a prompt similar to PostTrainBench:

```
You are tasked with conducting systematic probing experiments on the model `{model}`.

## Objective
Train linear probes to understand what features the model has learned about {concept}.

## Task Description
{task_description}

## Labeled Data
You have access to `labeled_data.jsonl` with {N} labeled examples.

## Your Task
1. Load the model and explore its architecture
2. Extract activations from ALL layers for each example
3. Train linear probes systematically:
   - Try different layers (0 to N)
   - Try different token positions (first, last, mean-pooled, max-pooled)
   - Use cross-validation for robust estimates
4. Report your findings:
   - Which layers contain the most information?
   - What's the best probe accuracy achieved?
   - Are the features interpretable?
5. Save your best probe in `final_probe/`

## Rules
1. You have {time_limit} hours
2. Work only in the current directory
3. Do not modify labeled_data.jsonl
4. Use scikit-learn LogisticRegression for probes
5. Report all experiments in a systematic manner

Remember: Be thorough, systematic, and document your process.
```

## Running Experiments

### Submit Multiple Jobs

```bash
# Run all tasks on all models
cd src/commit_utils
bash commit.sh
```

### Submit Single Job

```bash
# Run specific task on specific model
bash single_task.sh \
  --task sentiment_analysis \
  --model google/gemma-2-2b \
  --agent claude \
  --agent-config claude-opus-4-6 \
  --time-limit 2
```

### Monitor Progress

```bash
# Check Runpod job status
runpodctl get pods

# Stream logs
runpodctl logs <pod-id> --follow

# Download results
runpodctl download <pod-id>:/workspace/results ./results/
```

## Analyzing Results

### Aggregate Results

```bash
cd scripts
python aggregate_results.py
```

### Visualize Probes

```bash
python visualize_probes.py --task sentiment_analysis --model google/gemma-2-2b
```

### Compute Metrics

```bash
python compute_metrics.py --results-dir ../results
```

## Configuration

Key environment variables (set in `src/commit_utils/set_env_vars.sh`):

```bash
# Runpod configuration
export RUNPOD_API_KEY="your-key"
export RUNPOD_GPU_TYPE="NVIDIA A100"
export RUNPOD_DISK_SIZE="50"

# Results storage
export AUTOINTERP_RESULTS_DIR="results"
export AUTOINTERP_CONTAINER_NAME="autointerp:latest"

# Model cache
export HF_HOME="$HOME/.cache/huggingface"

# Agent API keys
export ANTHROPIC_API_KEY="your-key"
export OPENAI_API_KEY="your-key"
```

## Example Probing Tasks

### 1. Sentiment Analysis
- **Goal**: Probe whether model represents sentiment
- **Labels**: 0=negative, 1=positive
- **Data**: Movie reviews, tweets

### 2. Factuality Detection
- **Goal**: Probe whether model knows facts are true/false
- **Labels**: 0=false, 1=true
- **Data**: Factual statements

### 3. Reasoning Step Detection
- **Goal**: Probe whether model represents intermediate reasoning
- **Labels**: Steps in logical reasoning
- **Data**: Chain-of-thought examples

## Tips for Success

1. **Start Small**: Test with `--limit 50` before full runs
2. **Monitor Resources**: Check GPU memory usage during activation extraction
3. **Systematic Analysis**: Probe all layers, not just final layer
4. **Cross-Validation**: Use 5-fold CV for robust estimates
5. **Feature Importance**: Analyze which activation dimensions matter most
6. **Interpretability**: Can you explain what the probe learned?

## Troubleshooting

### Out of Memory
- Reduce batch size in activation extraction
- Process examples sequentially
- Use smaller models for testing

### Low Probe Accuracy
- Check label distribution (balanced?)
- Try different layers
- Increase labeled data if possible
- Check if task is actually learnable

### Agent Timeout
- Increase time limit
- Simplify the task
- Reduce number of examples

## Contributing

To add new features:
1. Create feature branch
2. Add tests
3. Update documentation
4. Submit pull request

## Citation

If you use AutoInterp in your research, please cite:

```bibtex
@misc{autointerp2026,
  title={AutoInterp: Automated Interpretability via Agent-Based Probing},
  author={Your Name},
  year={2026}
}
```

## License

Same license as PostTrainBench (see parent directory).
