#!/bin/bash

# OpenCode config for auto-approval
cat > opencode.json << 'EOF'
{
  "$schema": "https://opencode.ai/config.json",
  "permission": "allow",
  "provider": {
    "anthropic": {
      "options": {
        "apiKey": "{env:ANTHROPIC_API_KEY}"
      }
    },
    "openai": {
      "options": {
        "apiKey": "{env:OPENAI_API_KEY}"
      }
    },
    "opencode": {
      "options": {
        "apiKey": "{env:OPENCODE_API_KEY}"
      }
    }
  }
}
EOF

# Read prompt from environment or file
if [ -z "$PROMPT" ]; then
    if [ -f "prompt.txt" ]; then
        PROMPT=$(cat prompt.txt)
    else
        echo "ERROR: No prompt provided"
        exit 1
    fi
fi

# Run OpenCode
opencode run --model "$AGENT_CONFIG" --format json "$PROMPT"
