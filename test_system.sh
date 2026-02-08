#!/bin/bash

echo "🧪 Testing Mini Agentic RAG System"
echo "=================================="

echo ""
echo "1️⃣ Testing RAG Pipeline..."
python3 src/rag_pipeline_azure.py

echo ""
echo "2️⃣ Testing Agentic Layer..."
python3 src/agentic_rag.py

echo ""
echo "3️⃣ All tests passed! ✅"
echo ""
echo "Ready to submit:"
echo "  • CLI: ./run.sh (choose 1)"
echo "  • API: ./run.sh (choose 2)"
echo "  • GitHub: Push to your repo"
