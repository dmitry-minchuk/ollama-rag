#!/bin/bash

# Setup script for rag-mcp-server
# This ensures all dependencies are properly installed

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Setting up rag-mcp-server..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo "Installing additional required packages..."
pip install sentence-transformers chromadb

echo "Testing MCP server functionality..."
python -c "
import json
from mcp_server import handle_mcp_request
print('✓ MCP server imports successfully')

# Test basic functionality
request = {'method': 'tools/list', 'id': 1}
response = handle_mcp_request(request)
if 'result' in response:
    print('✓ MCP server responds correctly')
else:
    print('✗ MCP server error:', response)
    exit(1)
"

echo "✓ rag-mcp-server setup complete!"
echo "To use the server, run: source venv/bin/activate && python mcp_server.py"