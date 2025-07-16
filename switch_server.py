#!/usr/bin/env python3
"""
Script to switch between original and graph-enhanced MCP servers
"""
import os
import sys
import shutil
from pathlib import Path

def switch_to_graph_enhanced():
    """Switch to graph-enhanced MCP server"""
    script_dir = Path(__file__).parent
    
    # Backup original server
    original_server = script_dir / "mcp_server.py"
    backup_server = script_dir / "mcp_server_original.py"
    
    if original_server.exists() and not backup_server.exists():
        shutil.copy2(original_server, backup_server)
        print("✅ Backed up original MCP server to mcp_server_original.py")
    
    # Replace with graph-enhanced server
    graph_server = script_dir / "mcp_server_graph.py"
    if graph_server.exists():
        shutil.copy2(graph_server, original_server)
        print("✅ Switched to graph-enhanced MCP server")
        print("\n🚀 Available tools:")
        print("  • hybrid_search - Vector + graph search")
        print("  • method_search - Find method callers")
        print("  • class_hierarchy - Class inheritance info")
        print("  • code_search - Legacy vector search")
    else:
        print("❌ Graph-enhanced server not found!")
        return False
    
    return True

def switch_to_original():
    """Switch back to original MCP server"""
    script_dir = Path(__file__).parent
    
    original_server = script_dir / "mcp_server.py"
    backup_server = script_dir / "mcp_server_original.py"
    
    if backup_server.exists():
        shutil.copy2(backup_server, original_server)
        print("✅ Switched back to original MCP server")
        print("\n🔧 Available tools:")
        print("  • code_search - Vector similarity search only")
    else:
        print("❌ Original server backup not found!")
        return False
    
    return True

def main():
    if len(sys.argv) != 2:
        print("Usage: python switch_server.py [graph|original]")
        sys.exit(1)
    
    mode = sys.argv[1].lower()
    
    if mode == "graph":
        if switch_to_graph_enhanced():
            print("\n📊 Graph-enhanced RAG system is now active!")
            print("Run your queries to see both vector similarity and code relationship results.")
        else:
            sys.exit(1)
    elif mode == "original":
        if switch_to_original():
            print("\n📝 Original vector-only RAG system is now active!")
        else:
            sys.exit(1)
    else:
        print("Invalid mode. Use 'graph' or 'original'")
        sys.exit(1)

if __name__ == "__main__":
    main()