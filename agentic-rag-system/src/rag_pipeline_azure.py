"""
Azure-Optimized RAG Pipeline
FREE TIER FRIENDLY: Uses local embeddings to save Azure credits
"""

import os
from pathlib import Path
from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AzureRAGPipeline:
    """
    RAG Pipeline optimized for Azure Free Tier
    Uses local embeddings (FREE) instead of Azure embeddings (PAID)
    """
    
    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",  # FREE local model
        chunk_size: int = 400,  # Smaller chunks = fewer tokens to Azure
        chunk_overlap: int = 50
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Use FREE local embeddings (saves Azure credits!)
        logger.info(f"🆓 Using FREE local embeddings: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        self.index = None
        self.chunks = []
        self.metadata = []
        
    def load_documents(self, documents_dir: str) -> List[Dict]:
        """Load all .txt files from directory"""
        documents = []
        docs_path = Path(documents_dir)
        
        for file_path in docs_path.glob("*.txt"):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                documents.append({
                    'content': content,
                    'source': file_path.name,
                    'path': str(file_path)
                })
        
        logger.info(f"✅ Loaded {len(documents)} documents")
        return documents
    
    def chunk_text(self, text: str, source: str) -> List[Dict]:
        """Split text into semantic chunks"""
        chunks = []
        
        # Split by double newline (paragraphs) first
        paragraphs = text.split('\n\n')
        
        current_chunk = ""
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # If adding paragraph exceeds chunk size, save current chunk
            if len(current_chunk) + len(para) > self.chunk_size and current_chunk:
                chunks.append({
                    'text': current_chunk.strip(),
                    'source': source
                })
                current_chunk = para
            else:
                current_chunk += "\n\n" + para if current_chunk else para
        
        # Add final chunk
        if current_chunk:
            chunks.append({
                'text': current_chunk.strip(),
                'source': source
            })
        
        return chunks
    
    def build_index(self, documents_dir: str):
        """Build FAISS index from documents (FREE - uses local embeddings)"""
        # Load documents
        documents = self.load_documents(documents_dir)
        
        # Chunk all documents
        all_chunks = []
        for doc in documents:
            chunks = self.chunk_text(doc['content'], doc['source'])
            all_chunks.extend(chunks)
        
        logger.info(f"📝 Created {len(all_chunks)} chunks")
        
        # Generate embeddings (LOCAL - FREE!)
        logger.info("🔄 Generating embeddings (local, free)...")
        chunk_texts = [chunk['text'] for chunk in all_chunks]
        embeddings = self.embedding_model.encode(
            chunk_texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            batch_size=32  # Process in batches for efficiency
        )
        
        # Create FAISS index
        logger.info("🏗️  Building FAISS index...")
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.index.add(embeddings.astype('float32'))
        
        self.chunks = chunk_texts
        self.metadata = all_chunks
        
        logger.info(f"✅ Index built: {len(self.chunks)} chunks ready for retrieval")
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Retrieve relevant chunks (FREE - local embeddings)
        
        Args:
            query: User question
            top_k: Number of chunks to retrieve (keep <=3 to save Azure tokens)
        """
        if self.index is None:
            raise ValueError("❌ Index not built. Call build_index() first.")
        
        # Embed query (LOCAL - FREE!)
        query_embedding = self.embedding_model.encode([query])
        
        # Search FAISS (FREE!)
        distances, indices = self.index.search(
            query_embedding.astype('float32'),
            top_k
        )
        
        # Format results
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            results.append({
                'text': self.chunks[idx],
                'source': self.metadata[idx]['source'],
                'distance': float(distance),
                'similarity': 1 / (1 + distance)
            })
        
        return results
    
    def save_index(self, save_dir: str = "data/faiss_index"):
        """Save index to disk"""
        os.makedirs(save_dir, exist_ok=True)
        
        faiss.write_index(self.index, f"{save_dir}/index.faiss")
        
        with open(f"{save_dir}/metadata.pkl", 'wb') as f:
            pickle.dump({
                'chunks': self.chunks,
                'metadata': self.metadata
            }, f)
        
        logger.info(f"💾 Index saved to {save_dir}")
    
    def load_index(self, save_dir: str = "data/faiss_index"):
        """Load pre-built index (skip rebuilding)"""
        self.index = faiss.read_index(f"{save_dir}/index.faiss")
        
        with open(f"{save_dir}/metadata.pkl", 'rb') as f:
            data = pickle.load(f)
            self.chunks = data['chunks']
            self.metadata = data['metadata']
        
        logger.info(f"✅ Index loaded: {len(self.chunks)} chunks")


if __name__ == "__main__":
    # Build index once (FREE)
    print("\n🚀 Building RAG Index (FREE - local embeddings)")
    print("="*60)
    
    rag = AzureRAGPipeline()
    rag.build_index("documents")
    rag.save_index()
    
    print("\n✅ Index built and saved!")
    print("="*60)
    
    # Test retrieval (FREE)
    print("\n🔍 Testing Retrieval")
    print("="*60)
    
    results = rag.retrieve("What is RAG?", top_k=3)
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. [{result['source']}] Similarity: {result['similarity']:.3f}")
        print(f"   {result['text'][:150]}...")
    
    print("\n✅ Retrieval working! Ready for agentic layer.")
