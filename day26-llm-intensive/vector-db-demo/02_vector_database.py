"""
Day 27: Vector Databases with ChromaDB
The storage layer for RAG systems
"""

import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
import time

print("=" * 60)
print("VECTOR DATABASE: ChromaDB")
print("=" * 60)

# ============================================
# PART 1: INITIALIZE DATABASE
# ============================================
print ("\n Initializing ChromaDB...")

#Create persistent client (saves to disk)
client = chromadb.PersistentClient(path="./chroma_db")

#Create embedding fucntion
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')


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

print("ChromaDB initialized!")
print(f"   Storage: ./chroma_db/")

# ============================================
# PART 2: CREATE COLLECTION
#============================================
print("=" * 60)
print("CREATING COLLECTION")
print("=" * 60)

#Delete if exists (fresh start)
try:
    client.delete_collection("ml_knowledge")
except:
    pass

#Create new collection
collection = client.create_collection(
    name = "ml_knowledge",
    embedding_function=embedding_fn,
    metadata={"description":"Machine Learning knowledge base"}
)

print("Collection 'ml_knowledge' created!")

# ============================================
# PART 3: ADD DOCUMENTS
#============================================
print("=" * 60)
print("ADDING DOCUMENTS")
print("=" * 60)

#Sample ML knowledge base

documents = [
    "Supervised learning uses labeled data to train models. Examples include classification and regression.",
    "Unsupervised learning finds patterns in unlabeled data. Clustering and dimensionality reduction are common techniques.",
    "Deep learning uses neural networks with multiple layers. CNNs excel at image tasks, RNNs at sequences.",
    "Transfer learning leverages pretrained models for new tasks. This saves time and improves performance with limited data.",
    "Gradient descent is an optimization algorithm that minimizes loss functions by iteratively updating parameters.",
    "Overfitting occurs when a model memorizes training data but fails on new data. Regularization helps prevent this.",
    "Cross-validation splits data into train/test sets multiple times to assess model generalization.",
    "Feature engineering creates new input variables from raw data to improve model performance.",
    "Ensemble methods combine multiple models for better predictions. Random forests and boosting are popular techniques.",
    "The bias-variance tradeoff balances model simplicity and flexibility to optimize generalization.",
]

#Add metadata
# Add metadata
metadatas = [
    {"topic": "supervised", "difficulty": "beginner"},
    {"topic": "unsupervised", "difficulty": "beginner"},
    {"topic": "deep_learning", "difficulty": "intermediate"},
    {"topic": "transfer_learning", "difficulty": "intermediate"},
    {"topic": "optimization", "difficulty": "intermediate"},
    {"topic": "overfitting", "difficulty": "beginner"},
    {"topic": "validation", "difficulty": "beginner"},
    {"topic": "feature_engineering", "difficulty": "intermediate"},
    {"topic": "ensemble", "difficulty": "intermediate"},
    {"topic": "theory", "difficulty": "advanced"},
]

#IDs
ids = [f"doc_{i}" for i in range(len(documents))]

print(f"\n Adding {len(documents)} documents...")

#Add to collection
collection.add(
    documents= documents,
    metadatas=metadatas,
    ids= ids
)

print(f"Added {len(documents)} documents to vector database")


# ============================================
# PART 4: QUERY THE DATABASE
# ============================================

print("\n" + "=" * 60)
print("QUERYING VECTOR DATABASE")
print("=" * 60)

queries = [
    "How do I prevent my model from overfitting",
    "What are neural networks",
    "Tell me about training techniques"
]

for query in queries:
    print(f"\n Query: '{query}")
    print("-" * 60)

    #Search (returns top 3 most similar)
    results = collection.query(
        query_texts=[query],
        n_results=3
    )
    #Display results
    for i, (doc, metadata, distance) in enumerate(zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
        ),1):
        # Distance: lower = more similar (0 =identical)
        similarity = 1 - distance # Convert to similarity score
        print(f"   {i}.Similarity: {similarity:.4f}")
        print(f"   Topic: {metadata['topic']}")
        print(f"   Difficulty: {metadata['difficulty']}")
        print(f"   Content: {doc}")

# ============================================
# PART 5: FILETRING WITH METADATA
# ============================================

print("\n" + "=" * 60)
print("FILTERED SEARCH (Metadata)")
print("=" * 60)

query = "machine learning concepts"
print(f"\n Query : '{query}'")
print("Filter : Only 'beginner' difficulty")

results = collection.query(
    query_texts= [query],
    n_results=3,
    where={"difficulty": "beginner"} #Metadata filter
)

print("\n Results: \n")

for i, (doc, metadata) in enumerate(zip(
results['documents'][0],
results['metadatas'][0],
),1):
    print(f" {i}. [{metadata['topic']}] {doc}")


# ============================================
# PART 6: UPDATE & DELETE
# ============================================

print("=" * 60)
print("UPDATE & DELETE OPERATIONS")
print("=" * 60)

#Update document
print("\n Updating doc_0...")
collection.update(
    ids=["doc_0"],
    documents=["Supervised learning uses labeled data. It includes classification (categories) and regression (continuous values)."],
    
)
print("Document updated")

#Delete document
print("\n deleting doc_9...")
collection.delete(ids = ["doc_9"])
print("Document deleted")

#check count
count = collection.count()
print(f"\n Total documents: {count}")

# ============================================
# PART 7: PERSISTENCE
# ============================================

print("\n" + "=" * 60)
print("PERSISTENCE TEST")
print("=" * 60)

print("\n💾 Data saved to: ./chroma_db/")
print("   When you restart the script, data persists!")

client2 = chromadb.PersistentClient(path="./chroma_db")
collection2 = client2.get_collection("ml_knowledge")

count2 = collection2.count()
print(f"Retrieved collection with {count2} documents!")


# ============================================
# PART 8: PERFORMANCE METRICS
# ============================================

print("\n" + "=" * 60)
print("PERFORMANCE BENCHMARK")
print("=" * 60)

# Add more documents for performance test
print("\n📈 Adding 100 more documents for performance test...")

large_docs = [f"This is document number {i} about machine learning and AI." for i in range(100)]
large_ids = [f"perf_doc_{i}" for i in range(100)]
large_meta = [{"category": "performance_test"} for _ in range(100)]

collection.add(
    documents=large_docs,
    ids=large_ids,
    metadatas=large_meta
)

total_docs = collection.count()
print(f"✅ Total documents now: {total_docs}")

# Benchmark search speed
print("\n⚡ Search speed test...")

query = "artificial intelligence concepts"
start = time.time()
for _ in range(10):
    results = collection.query(
        query_texts=[query],
        n_results=5
    )
end = time.time()

avg_time = (end - start)/10*1000 #convert to miliseconds
print(f" Average query time: {avg_time: .2f}ms")
print(f" Database size: {total_docs} documents")
print(f" Results returned: 5 per query")

# ============================================
# PART 9: INTERACTIVE EXPLORATION
# ============================================

print("\n" + "=" * 60)
print("🎮 INTERACTIVE MODE")
print("=" * 60)

print("\nType questions to search the ML knowledge base!")
print("Type 'stats' to see database info")
print("Type 'quit' to exit\n")

while True:
    user_input = input("Query: ").strip()
    if user_input.lower() == 'quit':
        break
    if user_input.lower() == 'stats':
        count = collection.count()
        print(f"\n Database statistics:")
        print(f"\n Total Documents:{count}")
        print(f"\n Collection name : ml_knowledge")
        print(f"\n Storage path: ./chroma_db/ \n")
    if not user_input:
        continue
    #search 
    results = collection.query(
        query_texts = [user_input],
        n_results = 3
    )
    print(f"\n📊 Top 3 Results:\n")
    for i, (doc, metadata, distance) in enumerate(zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    ), 1):
        similarity = 1 - distance
        print(f"   {i}. [{metadata.get('topic', 'N/A')}] Similarity: {similarity:.4f}")
        print(f"      {doc}\n")

print("\n" + "=" * 60)
print("✅ VECTOR DATABASE DEMO COMPLETE!")
print("=" * 60)

print("""
🎓 WHAT YOU LEARNED:

1. Vector Databases store embeddings persistently
2. ChromaDB provides fast similarity search
3. Metadata filtering enables hybrid search
4. CRUD operations: Create, Read, Update, Delete
5. Persistence: Data survives restarts
6. Performance: Fast even with many documents

KEY OPERATIONS:
- collection.add()     → Store documents
- collection.query()   → Search by similarity
- collection.update()  → Modify documents
- collection.delete()  → Remove documents
- where={...}          → Filter by metadata

WHY THIS MATTERS FOR RAG:
1. Store all your document chunks as embeddings
2. User asks question → Find similar chunks
3. Retrieve top K most relevant chunks
4. Feed to LLM as context
5. LLM answers based on YOUR data!

NEXT: We combine this with LLM to build full RAG system! 🚀

DATABASE PERSISTED AT: ./chroma_db/
Run this script again - data will still be there!
""")






























