"""
FastAPI Interface for Agentic RAG
FREE TIER OPTIMIZED: Minimal overhead, efficient endpoints
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn
import logging

from agentic_rag import AgenticRAG

# Initialize
app = FastAPI(
    title="Mini Agentic RAG System",
    description="RAG system with agentic behavior (tool calling + self-reflection)",
    version="1.0.0"
)

# CORS for web access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize agent (load once, reuse)
agent = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QueryRequest(BaseModel):
    query: str
    use_critic: bool = True
    top_k: int = 3


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: list
    tokens_used: int
    context_used: int


@app.on_event("startup")
async def startup_event():
    """Initialize agent on startup"""
    global agent
    logger.info("🚀 Starting Agentic RAG System...")
    agent = AgenticRAG()
    logger.info("✅ Agent ready!")


@app.get("/")
async def root():
    """Health check"""
    return {
        "status": "running",
        "service": "Mini Agentic RAG System",
        "endpoints": {
            "query": "/query (POST)",
            "health": "/health (GET)",
            "stats": "/stats (GET)"
        }
    }


@app.get("/health")
async def health():
    """System health"""
    return {
        "status": "healthy",
        "index_loaded": agent.rag.index is not None,
        "total_chunks": len(agent.rag.chunks) if agent.rag.chunks else 0
    }


@app.get("/stats")
async def stats():
    """Usage statistics"""
    return {
        "total_tokens_used": agent.total_tokens_used,
        "total_chunks": len(agent.rag.chunks),
        "documents_loaded": len(set([m['source'] for m in agent.rag.metadata]))
    }


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Answer a question using agentic RAG
    
    Args:
        query: User question
        use_critic: Enable self-reflection (default: true)
        top_k: Number of chunks to retrieve (default: 3, max: 5 for free tier)
    """
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    # Limit top_k for free tier
    top_k = min(request.top_k, 5)
    
    try:
        result = agent.answer_query(
            query=request.query,
            use_critic=request.use_critic,
            top_k=top_k
        )
        
        return QueryResponse(
            query=result['query'],
            answer=result['answer'],
            sources=result['sources'],
            tokens_used=result['tokens_used'],
            context_used=result['context_used']
        )
    
    except Exception as e:
        logger.error(f"❌ Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 Starting Mini Agentic RAG API Server")
    print("="*60)
    print("\n📍 Access at: http://localhost:8000")
    print("📚 Docs at: http://localhost:8000/docs")
    print("\n💡 Free tier optimized:")
    print("   - Local embeddings (FAISS)")
    print("   - Efficient prompts")
    print("   - Token tracking")
    print("\n" + "="*60 + "\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
