#!/bin/bash

# MCP Server Wrapper Script
# This ensures the virtual environment is activated before running the MCP server

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="$SCRIPT_DIR/venv"
PYTHON_PATH="$VENV_PATH/bin/python"
MCP_SERVER_PATH="$SCRIPT_DIR/mcp_server.py"

# Check if virtual environment exists
if [ ! -f "$PYTHON_PATH" ]; then
    echo "Error: Virtual environment not found at $VENV_PATH" >&2
    echo "Please run setup_mcp.sh first" >&2
    exit 1
fi

# Check if MCP server exists
if [ ! -f "$MCP_SERVER_PATH" ]; then
    echo "Error: MCP server not found at $MCP_SERVER_PATH" >&2
    exit 1
fi

# Activate virtual environment and run MCP server
cd "$SCRIPT_DIR"
source "$VENV_PATH/bin/activate"
exec python "$MCP_SERVER_PATH" "$@"