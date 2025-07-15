#!/usr/bin/env python3
"""
MCP Server Wrapper - ensures virtual environment is used
This wrapper activates the virtual environment before running the MCP server
"""
import os
import sys
import subprocess

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VENV_PATH = os.path.join(SCRIPT_DIR, "venv")
VENV_PYTHON = os.path.join(VENV_PATH, "bin", "python")
MCP_SERVER_PATH = os.path.join(SCRIPT_DIR, "mcp_server.py")

def main():
    # Check if we're already in the virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        # Already in virtual environment, run the server directly
        exec(open(MCP_SERVER_PATH).read())
    else:
        # Not in virtual environment, run via venv python
        if not os.path.exists(VENV_PYTHON):
            print(f"Error: Virtual environment not found at {VENV_PYTHON}", file=sys.stderr)
            print("Please run setup_mcp.sh first", file=sys.stderr)
            sys.exit(1)
        
        # Execute the MCP server using the virtual environment's Python
        os.execv(VENV_PYTHON, [VENV_PYTHON, MCP_SERVER_PATH] + sys.argv[1:])

if __name__ == "__main__":
    main()