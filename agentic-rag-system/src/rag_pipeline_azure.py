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
        embedding_model_name: str = "all-MiniLM-L6-v2",
        chunk_size: int = 400,
        chunk_overlap: int = 50
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        logger.info(f"🆓 Using FREE local embeddings: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        self.index = None
        self.chunks = []
        self.metadata = []
        
    def load_documents(self, documents_dir: str) -> List[Dict]:
        """Load all .txt files from directory"""
        documents = []
        
        # Get absolute path
        if not os.path.isabs(documents_dir):
            # Relative to project root
            base_dir = Path(__file__).parent.parent
            docs_path = base_dir / documents_dir
        else:
            docs_path = Path(documents_dir)
        
        if not docs_path.exists():
            raise FileNotFoundError(f"Documents directory not found: {docs_path}")
        
        for file_path in docs_path.glob("*.txt"):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                documents.append({
                    'content': content,
                    'source': file_path.name,
                    'path': str(file_path)
                })
        
        if not documents:
            raise ValueError(f"No .txt files found in {docs_path}")
        
        logger.info(f"✅ Loaded {len(documents)} documents from {docs_path}")
        return documents
    
    def chunk_text(self, text: str, source: str) -> List[Dict]:
        """Split text into semantic chunks"""
        chunks = []
        
        # Split by double newline (paragraphs)
        paragraphs = text.split('\n\n')
        
        current_chunk = ""
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            if len(current_chunk) + len(para) > self.chunk_size and current_chunk:
                chunks.append({
                    'text': current_chunk.strip(),
                    'source': source
                })
                current_chunk = para
            else:
                current_chunk += "\n\n" + para if current_chunk else para
        
        if current_chunk:
            chunks.append({
                'text': current_chunk.strip(),
                'source': source
            })
        
        return chunks
    
    def build_index(self, documents_dir: str):
        """Build FAISS index from documents"""
        documents = self.load_documents(documents_dir)
        
        # Chunk all documents
        all_chunks = []
        for doc in documents:
            chunks = self.chunk_text(doc['content'], doc['source'])
            all_chunks.extend(chunks)
        
        logger.info(f"📝 Created {len(all_chunks)} chunks")
        
        # Generate embeddings
        logger.info("🔄 Generating embeddings...")
        chunk_texts = [chunk['text'] for chunk in all_chunks]
        
        # FIX: Ensure batch processing
        embeddings = self.embedding_model.encode(
            chunk_texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            batch_size=32
        )
        
        # FIX: Ensure 2D array
        if len(embeddings.shape) == 1:
            embeddings = embeddings.reshape(1, -1)
        
        logger.info(f"✅ Embeddings shape: {embeddings.shape}")
        
        # Create FAISS index
        logger.info("🏗️  Building FAISS index...")
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.index.add(embeddings.astype('float32'))
        
        self.chunks = chunk_texts
        self.metadata = all_chunks
        
        logger.info(f"✅ Index built: {len(self.chunks)} chunks")
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """Retrieve relevant chunks"""
        if self.index is None:
            raise ValueError("❌ Index not built. Call build_index() first.")
        
        # Embed query
        query_embedding = self.embedding_model.encode([query])
        
        # FIX: Ensure 2D array
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Search FAISS
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
        # Get absolute path
        if not os.path.isabs(save_dir):
            base_dir = Path(__file__).parent.parent
            save_path = base_dir / save_dir
        else:
            save_path = Path(save_dir)
        
        os.makedirs(save_path, exist_ok=True)
        
        faiss.write_index(self.index, str(save_path / "index.faiss"))
        
        with open(save_path / "metadata.pkl", 'wb') as f:
            pickle.dump({
                'chunks': self.chunks,
                'metadata': self.metadata
            }, f)
        
        logger.info(f"💾 Index saved to {save_path}")
    
    def load_index(self, save_dir: str = "data/faiss_index"):
        """Load pre-built index"""
        # Get absolute path
        if not os.path.isabs(save_dir):
            base_dir = Path(__file__).parent.parent
            load_path = base_dir / save_dir
        else:
            load_path = Path(save_dir)
        
        index_file = load_path / "index.faiss"
        metadata_file = load_path / "metadata.pkl"
        
        if not index_file.exists():
            raise FileNotFoundError(f"Index not found: {index_file}")
        
        self.index = faiss.read_index(str(index_file))
        
        with open(metadata_file, 'rb') as f:
            data = pickle.load(f)
            self.chunks = data['chunks']
            self.metadata = data['metadata']
        
        logger.info(f"✅ Index loaded: {len(self.chunks)} chunks")


if __name__ == "__main__":
    print("\n🚀 Building RAG Index")
    print("="*60)
    
    rag = AzureRAGPipeline()
    rag.build_index("documents")
    rag.save_index()
    
    print("\n✅ Index built and saved!")
    print("="*60)
    
    print("\n🔍 Testing Retrieval")
    print("="*60)
    
    results = rag.retrieve("What is RAG?", top_k=3)
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. [{result['source']}] Similarity: {result['similarity']:.3f}")
        print(f"   {result['text'][:150]}...")
    
    print("\n✅ Ready!")
