# 🤖 Mini Agentic RAG System

**Domain Knowledge Q&A with Retrieval-Augmented Generation + Agentic Behavior**

Built for: SELISE AI/ML Engineer Assessment  
Author: Emdad Hossain  
Date: February 2026

---

## 🎯 What This Is

A production-ready RAG (Retrieval-Augmented Generation) system with **agentic capabilities**:

- ✅ **Tool Calling**: Agent decides when to retrieve documents
- ✅ **Self-Reflection**: Critic evaluates and improves answers
- ✅ **Grounded Responses**: Answers based on actual documents, minimal hallucination
- ✅ **Multi-Interface**: CLI chat + REST API

---

## 🏗️ Architecture
```
User Query
    ↓
[Agent Decision Layer]
    - Should I retrieve documents?
    - Or answer directly?
    ↓
[Retrieval Tool] (if needed)
    - FAISS vector search (local, FREE)
    - Top-K relevant chunks
    ↓
[Answer Generation]
    - Azure GPT-4o-mini
    - Context-grounded responses
    ↓
[Self-Reflection Critic]
    - Evaluate answer quality
    - Refine if needed
    ↓
Final Answer + Sources
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Azure OpenAI API access
- macOS/Linux (tested on Intel Mac 2020)

### Installation
```bash
# Clone repository
git clone https://github.com/your-username/agentic-rag-system.git
cd agentic-rag-system

# Install dependencies
pip install -r requirements.txt

# Configure Azure credentials
cp .env.example .env
# Edit .env with your Azure OpenAI credentials

# Build FAISS index (one-time, FREE)
python3 src/rag_pipeline_azure.py
```

### Run CLI
```bash
./run.sh
# Choose option 1 for interactive chat
```

### Run API Server
```bash
./run.sh
# Choose option 2 for REST API
# Access: http://localhost:8000
# Docs: http://localhost:8000/docs
```

---

## 💡 Features

### 1. Intelligent Retrieval
- Agent **decides** if retrieval needed (saves API costs!)
- Semantic search with FAISS (local, free)
- Top-K relevant chunks with similarity scores

### 2. Agentic Behavior
- **ReAct-style reasoning**: Think → Act → Observe
- **Tool calling**: Retrieval as a tool
- **Decision making**: When to use documents vs direct answer

### 3. Self-Reflection
- Critic evaluates answer quality
- Checks: Grounded? Accurate? Complete?
- Refines answer if needed

### 4. Free Tier Optimized
- ✅ Local embeddings (Sentence-BERT, not Azure)
- ✅ Efficient prompts (< 500 tokens)
- ✅ Token tracking
- ✅ Smart caching

**Cost:** ~$0.05 for 100 queries on Azure free tier

---

## 📚 Example Usage

### CLI
```
You: What is RAG?

🤖 Agent thinking...

💡 Answer:
RAG (Retrieval-Augmented Generation) is a technique that combines 
information retrieval with language generation. Instead of relying 
solely on parametric knowledge, the model retrieves relevant documents 
and uses them to generate grounded responses.

According to the documents, RAG solves three key problems:
1. Reduced hallucinations - answers are grounded in actual documents
2. Up-to-date information - can update knowledge base without retraining
3. Source attribution - can cite which documents were used

📚 Sources: rag_systems.txt
💰 Tokens used: 287
```

### API
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is supervised learning?",
    "use_critic": true,
    "top_k": 3
  }'
```

Response:
```json
{
  "query": "What is supervised learning?",
  "answer": "Supervised learning uses labeled data to train models...",
  "sources": ["ml_fundamentals.txt"],
  "tokens_used": 245,
  "context_used": 3
}
```

---

## 🧪 Testing
```bash
# Test RAG pipeline
python3 src/rag_pipeline_azure.py

# Test agentic layer
python3 src/agentic_rag.py

# Test API
python3 src/api.py
# Then visit: http://localhost:8000/docs

# Test CLI
python3 src/cli.py
```

---

## 📁 Project Structure
```
agentic-rag-system/
├── src/
│   ├── rag_pipeline_azure.py    # RAG core (FAISS + embeddings)
│   ├── agentic_rag.py           # Agent layer (ReAct + critic)
│   ├── api.py                   # FastAPI server
│   └── cli.py                   # Command-line interface
├── documents/                    # Knowledge base
│   ├── ml_fundamentals.txt
│   ├── rag_systems.txt
│   └── llm_agents.txt
├── data/
│   └── faiss_index/             # Pre-built index (generated)
├── requirements.txt
├── .env                         # Azure credentials
├── run.sh                       # Launcher script
└── README.md
```

---

## 🔧 Configuration

### .env Variables
```bash
# Azure OpenAI
AZURE_OPENAI_ENDPOINT=https://your-resource.cognitiveservices.azure.com/
AZURE_OPENAI_API_KEY=your_key_here
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o-mini
AZURE_OPENAI_API_VERSION=2024-02-15-preview

# Optional: Azure Embeddings (not used, saves cost)
# AZURE_EMBEDDING_ENDPOINT=...
# AZURE_EMBEDDING_API_KEY=...
```

### Customization

**Add your own documents:**
1. Place `.txt` files in `documents/`
2. Rebuild index: `python3 src/rag_pipeline_azure.py`

**Adjust chunk size:**
Edit `chunk_size` in `rag_pipeline_azure.py` (default: 400 chars)

**Change retrieval count:**
Modify `top_k` parameter (default: 3, max: 5 for free tier)

---

## 🎓 Technical Highlights

### Why Local Embeddings?
- **FREE**: No Azure embedding API costs
- **Fast**: Local inference (50ms vs 200ms API)
- **Offline**: Works without internet after setup

### Why FAISS?
- **Fast**: Million-scale vector search in milliseconds
- **Local**: No external dependencies
- **Efficient**: CPU-optimized for Intel Mac

### Why Agentic?
- **Smart retrieval**: Only retrieves when needed
- **Self-improving**: Critic catches errors
- **Transparent**: Shows reasoning process

---

## 📊 Performance

**Tested on: Intel MacBook Pro 2020**

| Metric | Value |
|--------|-------|
| Index build time | ~5 seconds (45 chunks) |
| Query embedding | ~50ms (local) |
| FAISS search | ~2ms |
| Azure GPT-4o-mini | ~1-2 seconds |
| **Total latency** | **~2-3 seconds** |
| Cost per query | ~$0.0005 (Azure free tier) |

---

## 🚨 Troubleshooting

**"No module named X"**
```bash
pip install -r requirements.txt --break-system-packages
```

**"Azure API error"**
- Check `.env` credentials
- Verify deployment name matches Azure portal
- Check API quota (free tier limits)

**"Index not found"**
```bash
python3 src/rag_pipeline_azure.py
```

**Slow on Intel Mac**
- Expected: Intel Macs are slower than M1/M2
- Optimization: Reduce `chunk_size` or `top_k`

---

## 🎯 Assessment Rubric Checklist

- ✅ **RAG Pipeline**: Document loading, chunking, embeddings, retrieval
- ✅ **FAISS Vector Store**: Local, efficient search
- ✅ **Retrieval Logic**: Semantic search with top-K
- ✅ **Agent - Tool Calling**: Retrieval as a tool
- ✅ **Agent - Self-Reflection**: Critic evaluates answers
- ✅ **Answer Generation**: Context-grounded responses
- ✅ **API**: REST endpoints with FastAPI
- ✅ **CLI**: Interactive chat loop
- ✅ **Minimal Hallucination**: Answers based on documents
- ✅ **Free Tier Friendly**: Optimized for Azure free tier

---

## 🔮 Future Enhancements

- [ ] Multi-document formats (PDF, DOCX)
- [ ] Hybrid search (semantic + keyword)
- [ ] Conversation memory
- [ ] Streaming responses
- [ ] Web UI (Gradio/Streamlit)
- [ ] Deployment (Docker, Cloud Run)

---

## 📄 License

MIT License - Free to use and modify

---

## 👤 Author

**Emdad Hossain**  
Senior Software Engineer → AI/ML Engineer  
27 days intensive ML learning | 15 years production experience

Portfolio: https://emdadhossain25.github.io/emdad-portfolio/  
GitHub: https://github.com/emdadhossain25  
LinkedIn: [Your LinkedIn]

---

## 🙏 Acknowledgments

- Built as assessment for SELISE AI/ML Engineer role
- Optimized for Intel Mac 2020 + Azure free tier
- Uses open-source: LangChain, FAISS, Sentence-Transformers

---

**⭐ If this helped you, please star the repo!**
