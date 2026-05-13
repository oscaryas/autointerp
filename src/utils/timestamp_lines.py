#!/usr/bin/env python3
"""
Timestamp each line of stdin and write to stdout.
Used for logging agent output with timestamps.
"""
import sys
from datetime import datetime


def main():
    for line in sys.stdin:
        timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
        print(f"[{timestamp}] {line}", end='', flush=True)


if __name__ == "__main__":
    main()
