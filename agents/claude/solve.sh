#!/bin/bash

# Unset other agent API keys to avoid conflicts
unset OPENAI_API_KEY
unset GEMINI_API_KEY

export BASH_MAX_TIMEOUT_MS="36000000"

# Read prompt from environment or file
if [ -z "$PROMPT" ]; then
    if [ -f "prompt.txt" ]; then
        PROMPT=$(cat prompt.txt)
    else
        echo "ERROR: No prompt provided"
        exit 1
    fi
fi

# Run Claude Code
claude --print --verbose --model "$AGENT_CONFIG" --output-format stream-json \
    --dangerously-skip-permissions "$PROMPT"
