#!/usr/bin/env python3
"""
Backtest Script - Simplified Wrapper

This is now a convenience wrapper around agent_trader.py for quick local testing.
For full functionality, use agent_trader.py directly.

Usage:
    python backtest.py                          # Quick test with defaults
    python backtest.py --crew                   # Use LLM orchestration
    python backtest.py --start 2023-01-01 --end 2023-12-31
"""
import subprocess
import sys

# Default parameters for quick testing
DEFAULT_START = "2023-11-01"
DEFAULT_END = "2023-11-07"


def main():
    # Build command for agent_trader.py
    cmd = [
        sys.executable,
        "agent_trader.py",
        "--backtest",
        "--start", DEFAULT_START,
        "--end", DEFAULT_END,
    ]
    
    # Pass through any additional arguments
    cmd.extend(sys.argv[1:])
    
    # Handle --start and --end overrides
    args = sys.argv[1:]
    if "--start" in args:
        idx = args.index("--start")
        if idx + 1 < len(args):
            # Remove defaults and use provided values
            cmd = [sys.executable, "agent_trader.py", "--backtest"] + args
    
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)


if __name__ == "__main__":
    main()
