#!/usr/bin/env python3
"""
Parse Claude Code stream-json output into human-readable format.
"""
import sys
import json
import argparse
from pathlib import Path


def parse_stream_json(input_file, output_file):
    """Parse Claude Code stream-json output."""
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        f_out.write("=== Claude Code Execution Trace ===\n\n")

        for line_num, line in enumerate(f_in, 1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                msg_type = data.get('type', 'unknown')

                if msg_type == 'message':
                    content = data.get('content', [])
                    for item in content:
                        if item.get('type') == 'text':
                            text = item.get('text', '')
                            f_out.write(f"[Assistant]: {text}\n\n")

                elif msg_type == 'tool_use':
                    tool_name = data.get('name', 'unknown')
                    tool_input = data.get('input', {})
                    f_out.write(f"[Tool Use: {tool_name}]\n")
                    f_out.write(f"{json.dumps(tool_input, indent=2)}\n\n")

                elif msg_type == 'tool_result':
                    tool_id = data.get('tool_use_id', 'unknown')
                    result = data.get('content', '')
                    f_out.write(f"[Tool Result: {tool_id}]\n")
                    if isinstance(result, str):
                        f_out.write(f"{result}\n\n")
                    else:
                        f_out.write(f"{json.dumps(result, indent=2)}\n\n")

                elif msg_type == 'error':
                    error = data.get('error', 'Unknown error')
                    f_out.write(f"[ERROR]: {error}\n\n")

            except json.JSONDecodeError:
                # Not JSON, might be plain text output
                f_out.write(f"[Raw Output]: {line}\n\n")
            except Exception as e:
                f_out.write(f"[Parse Error on line {line_num}]: {str(e)}\n\n")

        f_out.write("=== End of Trace ===\n")


def main():
    parser = argparse.ArgumentParser(description="Parse Claude Code stream-json output")
    parser.add_argument("input", help="Input file (stream-json format)")
    parser.add_argument("-o", "--output", required=True, help="Output file")
    args = parser.parse_args()

    parse_stream_json(args.input, args.output)
    print(f"Parsed trace saved to: {args.output}")


if __name__ == "__main__":
    main()
