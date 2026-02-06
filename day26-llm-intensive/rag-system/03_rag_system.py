"""
Day 27: Complete RAG (Retrieval-Augmented Generation) System
Production-ready Q&A over your documents
"""
import os
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv
import re

#load environment variables
load_dotenv()


print("=" * 60)
print("RAG SYSTEM: Question Answering Over Your Documents")
print("=" * 60)

# ============================================
# CONFIGURATION
# ============================================

DOCUMENTS_DIR = "./documents"
CHROMA_DB_PATH = "./rag_chroma_db"
CHUNK_SIZE = 500 #characters per chunks
CHUNK_OVERLAP = 50 #overlap between chunks
TOP_K = 3  # Number of chunks to retrieve

# ============================================
# PART 1: DOCUMENT PROCESSING
# ============================================

print("\n📚 STEP 1: Document Ingestion")
print("-" * 60)



def load_documents(directory):
    """Load all text documents from directory"""
    documents = []
    
    for file_path in Path(directory).glob("*.txt"):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            documents.append({
                'content': content,
                'source': file_path.name,
                'path': str(file_path)
            })
    
    return documents

docs = load_documents(DOCUMENTS_DIR)
print(f"✅ Loaded {len(docs)} documents")
for doc in docs:
    print(f"   {doc['source']} ({len(doc['content'])} chars)")


# ============================================
# PART 2: CHUNKING STRATEGY
# ============================================

print("\n📝 STEP 2: Chunking Documents")
print("-" * 60)

def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into overlapping chunks"""
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        # Try to break at sentence boundary
        if end < len(text):
            # Look for period, question mark, or exclamation
            last_period = max(
                chunk.rfind('.'),
                chunk.rfind('?'),
                chunk.rfind('!')
            )
            if last_period > chunk_size * 0.5:  # At least 50% into chunk
                end = start + last_period + 1
                chunk = text[start:end]
        
        chunks.append(chunk.strip())
        start = end - overlap  # Overlap for context
    
    return chunks

#Chunk all documents
all_chunks = []
chunk_metadata = []

for doc in docs :
    chunks = chunk_text(doc['content'], CHUNK_SIZE,CHUNK_OVERLAP)
    
    for i, chunk in enumerate(chunks):
        all_chunks.append(chunk)
        chunk_metadata.append({
                'source': doc['source'],
                'chunk_index': i,
                'total_chunks': len(chunks)
            }
        )
print(f"✅ Created {len(all_chunks)} chunks")
print(f"   Chunk size: {CHUNK_SIZE} chars")
print(f"   Overlap: {CHUNK_OVERLAP} chars")

# Show sample chunk
print(f"\n📄 Sample chunk:")
print(f"   Source: {chunk_metadata[0]['source']}")
print(f"   Content: {all_chunks[0][:200]}...")

# ============================================
# PART 3: EMBEDDING & STORAGE
# ============================================

print("\n" + "=" * 60)
print("STEP 3: Creating Embeddings & Vector Database")
print("-" * 60)

embedding_model = SentenceTransformer('all-MiniLM-L6-V2')

#Custom embedding function for ChromaDB
class CustomEmbedding:
    def __init__(self,model):
        self.model = model
    
    def __call__(self, input):
        # For adding documents (batched)
        if isinstance(input, list):
            return self.model.encode(input).tolist()
        return self.model.encode([input]).tolist()
    
    def embed_query(self, input):
        # For querying (single or batched)
        if isinstance(input, str):
            return self.model.encode([input])[0].tolist()
        return self.model.encode(input).tolist()


embedding_fn = CustomEmbedding(embedding_model)

client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

#Delete if exists (fresh start)
try:
    client.delete_collection("rag_documents") 
except:
    pass 


#Create collection
collection = client.create_collection(
    name="rag_documents",
    embedding_function=embedding_fn,
    metadata={"description":"RAG system document chunk"}
)


print(f"✅ ChromaDB initialized at {CHROMA_DB_PATH}")

# Add chunks to database
print(f"\n🔄 Adding {len(all_chunks)} chunks to vector database...")

#Create ID's
ids = [f"chunk_{i}" for i in range (len(all_chunks))]

#Add in batches (chromadb limit : 41666 per batch)
batch_size = 100

for i in range(0,len(all_chunks),batch_size):
    batch_chunks = all_chunks[i:i+batch_size]
    batch_meta = chunk_metadata[i:i+batch_size]
    batch_ids = ids[i:i+batch_size]

    collection.add(
        documents=batch_chunks,
        metadatas= batch_meta,
        ids = batch_ids
    )

print(f"✅ Added {collection.count()} chunks to database")

# ============================================
# PART 4: RETRIEVAL FUNCTION
# ============================================

print("\n" + "=" * 60)
print("STEP 4: Retrieval System")
print("-" * 60)

def retrieve_context(query, top_k=3):
    """Retrieve most relevant chunks for query"""
    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )
    
    chunks = results['documents'][0]
    metadatas = results['metadatas'][0]
    distances = results['distances'][0]
    
    # Format context
    context_parts = []
    for chunk, meta, dist in zip(chunks, metadatas, distances):
        similarity = 1 - dist
        context_parts.append({
            'text': chunk,
            'source': meta['source'],
            'similarity': similarity
        })
    
    return context_parts

# Test retrieval
test_query = "What projects have you built?"
print(f"\n🔍 Test Query: '{test_query}'")
print(f"   Retrieving top {TOP_K} chunks...\n")

context = retrieve_context(test_query, TOP_K)

for i, ctx in enumerate(context, 1):
    print(f"   {i}. [{ctx['source']}] Similarity: {ctx['similarity']:.4f}")
    print(f"      {ctx['text'][:100]}...\n")

print("✅ Retrieval working!")

# ============================================
# PART 5: LLM INTEGRATION
# ============================================

print("=" * 60)
print("STEP 5: LLM Integration (OpenAI GPT)")
print("-" * 60)

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def generate_answer(query, context_parts):
    """Generate answer using LLM with retrieved context"""
    
    # Build context string
    context_text = "\n\n".join([
        f"[Source: {ctx['source']}]\n{ctx['text']}"
        for ctx in context_parts
    ])
    
    # Create prompt
    prompt = f"""You are a helpful assistant answering questions based on provided context.

Context:
{context_text}

Question: {query}

Instructions:
1. Answer the question using ONLY information from the context above
2. If the context doesn't contain enough information, say so
3. Be specific and cite which document the information comes from
4. Keep answers concise but complete

Answer:"""
    
    # Call GPT-4
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context. Always cite sources."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,  # Low temp for factual responses
            max_tokens=500
        )
        
        answer = response.choices[0].message.content
        return answer
    
    except Exception as e:
        return f"Error generating answer: {str(e)}"

print("✅ LLM integration ready")

# ============================================
# PART 6: COMPLETE RAG PIPELINE
# ============================================

print("\n" + "=" * 60)
print("STEP 6: Complete RAG Pipeline")
print("=" * 60)

def rag_query(question):
    """Complete RAG pipeline: retrieve + generate"""
    
    print(f"\n🔍 Question: {question}")
    print("-" * 60)
    
    # Step 1: Retrieve relevant chunks
    print(f"📚 Retrieving top {TOP_K} relevant chunks...")
    context = retrieve_context(question, TOP_K)
    
    print(f"✅ Retrieved {len(context)} chunks\n")
    for i, ctx in enumerate(context, 1):
        print(f"   {i}. [{ctx['source']}] (similarity: {ctx['similarity']:.3f})")
    
    # Step 2: Generate answer
    print(f"\n🤖 Generating answer with GPT-4...")
    answer = generate_answer(question, context)
    
    print(f"\n💡 Answer:")
    print("-" * 60)
    print(answer)
    print("-" * 60)
    
    return answer

# ============================================
# PART 7: DEMO QUERIES
# ============================================

print("\n" + "=" * 60)
print("DEMO: RAG System in Action")
print("=" * 60)

demo_questions = [
    "What machine learning projects have you built?",
    "What is your sentiment analysis API accuracy?",
    "How many years of software engineering experience do you have?",
]

for question in demo_questions:
    rag_query(question)
    print("\n" + "=" * 60 + "\n")

# ============================================
# PART 8: INTERACTIVE MODE
# ============================================

print("=" * 60)
print("🎮 INTERACTIVE MODE")
print("=" * 60)

print("""
Ask questions about the documents!

Type 'examples' to see sample questions
Type 'stats' to see system statistics
Type 'quit' to exit
""")

while True:
    user_question = input("\n💬 Your question: ").strip()
    
    if user_question.lower() == 'quit':
        break
    
    if user_question.lower() == 'examples':
        print("\n📝 Sample Questions:")
        print("   • What are your technical skills?")
        print("   • Describe your ML projects")
        print("   • What frameworks do you know?")
        print("   • Tell me about your deployment experience")
        print("   • What is your biggest achievement?")
        continue
    
    if user_question.lower() == 'stats':
        print(f"\n📊 System Statistics:")
        print(f"   Documents loaded: {len(docs)}")
        print(f"   Total chunks: {collection.count()}")
        print(f"   Chunk size: {CHUNK_SIZE} chars")
        print(f"   Retrieval: Top {TOP_K} chunks")
        print(f"   LLM: GPT-4")
        print(f"   Embedding model: all-MiniLM-L6-v2")
        continue
    
    if not user_question:
        continue
    
    # Process question
    rag_query(user_question)

print("\n" + "=" * 60)
print("✅ RAG SYSTEM DEMO COMPLETE!")
print("=" * 60)

print("""
🎓 WHAT YOU BUILT:

A complete production-ready RAG system with:

1. Document Ingestion
   - Load multiple text files
   - Handle different formats (extensible to PDF, DOCX)

2. Chunking Strategy
   - Smart text splitting (500 chars)
   - Overlap for context preservation (50 chars)
   - Sentence boundary detection

3. Embedding Generation
   - Convert chunks to 384-dimensional vectors
   - SentenceTransformer (all-MiniLM-L6-v2)

4. Vector Storage
   - Persistent ChromaDB
   - Metadata tracking (source, chunk index)
   - Fast similarity search

5. Retrieval
   - Semantic search (not keyword matching!)
   - Top-K most relevant chunks
   - Similarity scoring

6. LLM Integration
   - GPT-4 for answer generation
   - Context injection via prompt
   - Source citation

7. Production Features
   - Error handling
   - Batch processing
   - Interactive interface
   - Performance monitoring

THIS IS WHAT COMPANIES PAY $200K+ TO BUILD! 🚀

You built it in 90 minutes! 💪

INTERVIEW TALKING POINTS:

"I built a production RAG system that:
- Ingests documents and chunks them intelligently
- Creates embeddings using SentenceTransformers
- Stores in ChromaDB for fast retrieval
- Retrieves top-K relevant chunks semantically
- Feeds to GPT-4 with proper prompt engineering
- Generates accurate answers with source citations

The system handles the hallucination problem by grounding
LLM responses in verified documents. It's deployed locally
but ready for cloud deployment with minor changes."

NEXT STEPS FOR PRODUCTION:
- Add PDF/DOCX support (pypdf, python-docx)
- Implement caching (Redis)
- Add user authentication
- Deploy as API (FastAPI)
- Monitor usage and costs
- Add feedback loop for improvement

DATABASE LOCATION: ./rag_chroma_db/
Documents persist across restarts!
""")





























