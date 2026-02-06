"""
Day 27: Understanding Embeddings
From words to vectors - the foundation of modern NLP
"""

import numpy as np
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity


print("=" * 60)
print("EMBEDDINGS: WORDS → VECTORS")
print("=" * 60)

# ============================================
# PART 1: LOAD EMBEDDING MODEL
# ============================================

print("\n🔄 Loading embedding model...")
print("   Model: all-MiniLM-L6-v2 (384 dimensions)")
print("   This will download ~90MB on first run...")

model = SentenceTransformer('all-MiniLM-L6-v2')

print("✅ Model Loaded!\n")


# ============================================
# PART 2: CREATE EMBEDDINGS 
# ============================================

print("=" * 60)
print("Experiment 1 : similar meanings")
print("=" * 60)

sentences = [
    "I love programming",
    "I enjoy coding",
    "The weather is nice today",
    "Machine Learning is fascinating",
    "AI is amazing"

]

print("\n✅ Sentences:")
for i, sent in enumerate(sentences,1):
    print(f"  {i}. {sent}")

print("\n✅ Converting to embeddings...")
embeddings = model.encode(sentences)

print(f"\n Created {len(embeddings.shape)}")
print(f"\n Shape: {embeddings.shape}")
print(f"\n Each Sentence: {embeddings.shape[1]} numbers")

# Show one embedding (first 10 numbers)
print(f"\n📊 First embedding (first 10 dimensions):")
print(f"   {embeddings[0][:10]}")
print(f"   ... (and 374 more numbers)")

# ============================================
# PART 3: MEASURE SIMILARITY 
# ============================================

print("=" * 60)
print("Experiment 2 : Similiarity Scores")
print("=" * 60)

#Calculate cosine similarity
similarity_matrix =cosine_similarity(embeddings)

print("\n Similarity Matrix:")
print("    (1.0 = identical, 0.0 = unrelated) \n")

#Create comparison
print(f"{'Sentence 1': <30} {'Sentence 2':<30} {'Similarity':>10}")
print("-" *70)


for i in range (len(sentences)):
    for j in range (i+1, len(sentences)):
        sim = similarity_matrix[i][j]
        print(f"{sentences[i]:<30} {sentences[j]:<30} {sim:>10.4f}")


# ============================================
# PART 4: INSIGHTS 
# ============================================

print("=" * 60)
print("KEY INSIGHTS")
print("=" * 60)


#Find the most similar pair
max_sim =0
max_i,max_j =0,0
for i in range(len(sentences)):
    for j in range(i+1, len(sentences)):
        if similarity_matrix[i][j] > max_sim:
            max_sim =similarity_matrix[i][j]
            max_i,max_j = i,j


print(f"\n✅ MOST SIMILAR:")
print(f"   '{sentences[max_i]}'")
print(f"   '{sentences[max_j]}'")
print(f"   Similarity: {max_sim:.4f}")
print(f"\n   Why? Both about programming/coding!")


#find the least similar pair
min_sim =1
min_i , min_j = 0,0
for i in range(len(sentences)):
    for j in range(i+1, len(sentences)):
        if similarity_matrix[i][j] < min_sim:
            min_sim= similarity_matrix[i][j]
            min_i, min_j = i,j

print(f"\n❌ LEAST SIMILAR:")
print(f"   '{sentences[min_i]}'")
print(f"   '{sentences[min_j]}'")
print(f"   Similarity: {min_sim:.4f}")
print(f"\n   Why? Completely different topics!")

# ============================================
# PART 5: SEMANTIC SEARCH
# ============================================
print("\n" + "=" * 60)
print("EXPERIMENT 3: Semantic Search")
print("=" * 60)

query = "software development"
print(f"\n Query: '{query}")

#Encode query
query_embedding = model.encode([query])[0]

#Find similarities
similarities = cosine_similarity([query_embedding],embeddings)[0]

#Sort by similarity
sorted_indices = np.argsort(similarities)[::-1]

print(f"\n📊 Results (ranked by relevance):\n")
for rank, idx in enumerate(sorted_indices, 1):
    print(f"   {rank}. '{sentences[idx]}'")
    print(f"      Similarity: {similarities[idx]:.4f}")
    print()

# ============================================
# PART 6: THE MAGIC EXPLAINED
# ============================================

print("=" * 60)
print("🎓 WHAT JUST HAPPENED?")
print("=" * 60)

explanation = """
1. EMBEDDINGS = Dense Vector Representations
   - Each sentence → 384 numbers
   - Numbers capture MEANING, not just words
   - Similar meanings → similar vectors

2. COSINE SIMILARITY = Measure of Closeness
   - Ranges from -1 to 1 (usually 0 to 1 for text)
   - 1.0 = identical meaning
   - 0.0 = completely unrelated
   - Calculated: dot product / (magnitude × magnitude)

3. SEMANTIC SEARCH = Find by MEANING, not keywords
   - Query: "software development"
   - Matches: "programming", "coding" (HIGH similarity)
   - Doesn't match: "weather" (LOW similarity)
   - Even though query words don't appear in results!

4. WHY THIS MATTERS FOR RAG:
   - User asks: "How do I reset my password?"
   - Traditional search: Look for exact words "reset", "password"
   - Semantic search: Also finds "change credentials", "account recovery"
   - MUCH better user experience!

5. REAL-WORLD APPLICATIONS:
   - Google Search (semantic matching)
   - ChatGPT (finding relevant context)
   - Recommendation systems (similar products)
   - Document clustering (group by topic)
   - Plagiarism detection (similar text)
"""

print(explanation)

# ============================================
# PART 7: INTERACTIVE DEMO
# ============================================

print("=" * 60)
print("🎮 INTERACTIVE: Try Your Own!")
print("=" * 60)

document_library = [
    "Python is a programming language",
    "Machine learning uses algorithms to learn from data",
    "The Eiffel Tower is in Paris",
    "JavaScript is used for web development",
    "Deep learning is a subject of machine learning",
    "The weather forecast predicts rain tomorrow",
    "Neural networks are inspired by the human brain",
    "London is the capital of England"
]

print("\n Document Library:")
for i, doc in enumerate (document_library, 1):
    print(f" {i}.{doc}")

#Encode all documents
doc_embeddings = model.encode(document_library)

print("\n"+ "=" *60)
print("Type a query to search (or 'quit' to exit)")
print("=" *60)
while True:
    user_query = input("\n Your query: ").strip()
    if user_query.lower() == 'quit':
        break
    if not user_query:
        continue

    #Search
    query_emb = model.encode([user_query])[0]
    sims = cosine_similarity([query_emb], doc_embeddings)[0]

    #Get top 3 results
    top_indices = np.argsort(sims)[::-1][:3]

    print(f"\n Top 3 Results:\n")
    for rank, idx in enumerate(top_indices,1):
        print(f" {rank}. {document_library[idx]}")
        print(f" Relevance: {sims[idx]:.4f}")
        print()

print("\n" + "=" * 60)
print("✅ EMBEDDINGS DEMO COMPLETE!")
print("=" * 60)

print("""
🎓 WHAT YOU LEARNED:

1. Embeddings convert text → meaningful vectors
2. Similar meanings → similar vectors (mathematically!)
3. Cosine similarity measures how close vectors are
4. Semantic search finds by MEANING, not just keywords
5. This is the FOUNDATION of RAG systems

NEXT: We'll use this to build a RAG system that can:
- Ingest your documents
- Create embeddings for each chunk
- Store in vector database
- Retrieve relevant chunks for any query
- Feed to LLM for answering

You now understand the CORE TECHNOLOGY behind ChatGPT! 🚀
""")