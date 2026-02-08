#!/bin/bash

echo "=================================="
echo "Mini Agentic RAG System - Launcher"
echo "=================================="
echo ""
echo "Choose interface:"
echo "  1) CLI (Command-line chat)"
echo "  2) API Server (REST API + Swagger docs)"
echo ""
read -p "Enter choice (1 or 2): " choice

case $choice in
    1)
        echo ""
        echo "🚀 Starting CLI..."
        cd src && python3 cli.py
        ;;
    2)
        echo ""
        echo "🚀 Starting API Server..."
        echo "Access at: http://localhost:8000"
        echo "Docs at: http://localhost:8000/docs"
        cd src && python3 api.py
        ;;
    *)
        echo "Invalid choice. Use 1 or 2."
        ;;
esac
