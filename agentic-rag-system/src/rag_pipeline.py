"""
Mini Agentic RAG System - Core Pipeline
Implements: Document loading, chunking, embedding, and retrieval
"""

import os
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    RAG Pipeline with FAISS vector store
    """
    
    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        """
        Initialize RAG pipeline
        
        Args:
            embedding_model_name: HuggingFace model for embeddings
            chunk_size: Size of document chunks in characters
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Load embedding model
        logger.info(f"Loading embedding model: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        # Initialize FAISS index
        self.index = None
        self.chunks = []
        self.metadata = []
        
    def load_documents(self, documents_dir: str) -> List[Dict]:
        """
        Load documents from directory
        
        Args:
            documents_dir: Path to documents directory
            
        Returns:
            List of document dictionaries
        """
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
        
        logger.info(f"Loaded {len(documents)} documents")
        return documents
    
    def chunk_text(self, text: str, source: str) -> List[Dict]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Document text
            source: Source filename
            
        Returns:
            List of chunk dictionaries
        """
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]
            
            # Try to break at sentence boundary
            if end < len(text):
                last_period = max(
                    chunk_text.rfind('.'),
                    chunk_text.rfind('?'),
                    chunk_text.rfind('!')
                )
                if last_period > self.chunk_size * 0.5:
                    end = start + last_period + 1
                    chunk_text = text[start:end]
            
            chunks.append({
                'text': chunk_text.strip(),
                'source': source,
                'start_idx': start,
                'end_idx': end
            })
            
            start = end - self.chunk_overlap
        
        return chunks
    
    def build_index(self, documents_dir: str):
        """
        Build FAISS index from documents
        
        Args:
            documents_dir: Path to documents directory
        """
        # Load documents
        documents = self.load_documents(documents_dir)
        
        # Chunk all documents
        all_chunks = []
        for doc in documents:
            chunks = self.chunk_text(doc['content'], doc['source'])
            all_chunks.extend(chunks)
        
        logger.info(f"Created {len(all_chunks)} chunks")
        
        # Generate embeddings
        logger.info("Generating embeddings...")
        chunk_texts = [chunk['text'] for chunk in all_chunks]
        embeddings = self.embedding_model.encode(
            chunk_texts,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Create FAISS index
        logger.info("Building FAISS index...")
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.index.add(embeddings.astype('float32'))
        
        # Store chunks and metadata
        self.chunks = chunk_texts
        self.metadata = all_chunks
        
        logger.info(f"✅ Index built with {len(self.chunks)} chunks")
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Retrieve most relevant chunks for query
        
        Args:
            query: User query
            top_k: Number of chunks to retrieve
            
        Returns:
            List of relevant chunks with metadata
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Embed query
        query_embedding = self.embedding_model.encode([query])
        
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
                'similarity': 1 / (1 + distance)  # Convert distance to similarity
            })
        
        return results
    
    def save_index(self, save_dir: str = "data/faiss_index"):
        """Save FAISS index and metadata"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, f"{save_dir}/index.faiss")
        
        # Save metadata
        with open(f"{save_dir}/metadata.pkl", 'wb') as f:
            pickle.dump({
                'chunks': self.chunks,
                'metadata': self.metadata
            }, f)
        
        logger.info(f"✅ Index saved to {save_dir}")
    
    def load_index(self, save_dir: str = "data/faiss_index"):
        """Load FAISS index and metadata"""
        # Load FAISS index
        self.index = faiss.read_index(f"{save_dir}/index.faiss")
        
        # Load metadata
        with open(f"{save_dir}/metadata.pkl", 'rb') as f:
            data = pickle.load(f)
            self.chunks = data['chunks']
            self.metadata = data['metadata']
        
        logger.info(f"✅ Index loaded from {save_dir}")


if __name__ == "__main__":
    # Test the pipeline
    rag = RAGPipeline()
    rag.build_index("documents")
    rag.save_index()
    
    # Test retrieval
    results = rag.retrieve("What is supervised learning?")
    
    print("\n" + "="*60)
    print("RETRIEVAL TEST")
    print("="*60)
    for i, result in enumerate(results, 1):
        print(f"\n{i}. [{result['source']}] Similarity: {result['similarity']:.3f}")
        print(f"   {result['text'][:200]}...")
