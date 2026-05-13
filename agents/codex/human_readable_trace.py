#!/usr/bin/env python3
"""
Parse Codex JSON output into human-readable format.
"""
import sys
import json
import argparse


def parse_codex_json(input_file, output_file):
    """Parse Codex JSON output."""
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        f_out.write("=== Codex Execution Trace ===\n\n")

        try:
            data = json.load(f_in)

            # Parse execution events
            events = data.get('events', [])
            for event in events:
                event_type = event.get('type', 'unknown')

                if event_type == 'assistant_message':
                    text = event.get('text', '')
                    f_out.write(f"[Assistant]: {text}\n\n")

                elif event_type == 'tool_call':
                    tool = event.get('tool', 'unknown')
                    args = event.get('arguments', {})
                    f_out.write(f"[Tool Call: {tool}]\n")
                    f_out.write(f"{json.dumps(args, indent=2)}\n\n")

                elif event_type == 'tool_result':
                    result = event.get('result', '')
                    f_out.write(f"[Tool Result]\n")
                    f_out.write(f"{result}\n\n")

                elif event_type == 'error':
                    error = event.get('message', 'Unknown error')
                    f_out.write(f"[ERROR]: {error}\n\n")

            # Summary
            f_out.write("\n=== Summary ===\n")
            summary = data.get('summary', {})
            f_out.write(f"Total tokens: {summary.get('total_tokens', 'N/A')}\n")
            f_out.write(f"Duration: {summary.get('duration', 'N/A')}s\n")

        except json.JSONDecodeError as e:
            f_out.write(f"[ERROR]: Failed to parse JSON: {str(e)}\n")
        except Exception as e:
            f_out.write(f"[ERROR]: {str(e)}\n")

        f_out.write("\n=== End of Trace ===\n")


def main():
    parser = argparse.ArgumentParser(description="Parse Codex JSON output")
    parser.add_argument("input", help="Input file (JSON format)")
    parser.add_argument("-o", "--output", required=True, help="Output file")
    args = parser.parse_args()

    parse_codex_json(args.input, args.output)
    print(f"Parsed trace saved to: {args.output}")


if __name__ == "__main__":
    main()
