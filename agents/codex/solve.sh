#!/bin/bash

# Unset other agent API keys to avoid conflicts
unset ANTHROPIC_API_KEY
unset GEMINI_API_KEY

# Codex uses OPENAI_API_KEY
export CODEX_API_KEY="${OPENAI_API_KEY}"

# Read prompt from environment or file
if [ -z "$PROMPT" ]; then
    if [ -f "prompt.txt" ]; then
        PROMPT=$(cat prompt.txt)
    else
        echo "ERROR: No prompt provided"
        exit 1
    fi
fi

# Run Codex
codex --search -a never exec --json --skip-git-repo-check --yolo \
    --model "$AGENT_CONFIG" "$PROMPT"
