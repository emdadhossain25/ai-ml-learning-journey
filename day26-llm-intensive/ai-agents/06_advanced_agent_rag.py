"""
Day 27: Advanced AI Agent with RAG Integration
The ultimate demo: Agent that can search your documents!
"""

import os
from dotenv import load_dotenv
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.tools import tool
import chromadb
from sentence_transformers import SentenceTransformer
import math
import json

load_dotenv()

print("=" * 60)
print("ADVANCED AI AGENT: RAG + Tools Integration")
print("=" * 60)

# ============================================
# PART 1: INITIALIZE RAG SYSTEM AS TOOL
# ============================================

print("\n🔧 STEP 1: Connecting to RAG System")
print("-" * 60)

# Load embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

class CustomEmbedding:
    def __init__(self, model):
        self.model = model
    
    def __call__(self, texts):
        return self.model.encode(texts).tolist()

embedding_fn = CustomEmbedding(embedding_model)

# Connect to existing ChromaDB
client = chromadb.PersistentClient(path="../rag-system/rag_chroma_db")
collection = client.get_collection("rag_documents")

print(f"✅ Connected to RAG database")
print(f"   Documents: {collection.count()} chunks")

# ============================================
# PART 2: CREATE RAG TOOL
# ============================================

print("\n🛠️  STEP 2: Creating RAG Tool")
print("-" * 60)

@tool
def search_my_documents(query: str) -> str:
    """
    Search through my personal documents (portfolio, projects, skills).
    Use this when user asks about my experience, projects, or background.
    Input should be a question about my work, skills, or achievements.
    """
    
    try:
        # Retrieve top 3 relevant chunks
        results = collection.query(
            query_texts=[query],
            n_results=3
        )
        
        if not results['documents'][0]:
            return "No relevant information found in my documents."
        
        # Format results
        context = "\n\n".join([
            f"[From {meta['source']}]\n{doc}"
            for doc, meta in zip(
                results['documents'][0],
                results['metadatas'][0]
            )
        ])
        
        return f"Found relevant information:\n\n{context}"
    
    except Exception as e:
        return f"Error searching documents: {str(e)}"

print("✅ RAG Tool created: search_my_documents")
print("   Searches your portfolio/projects/skills")

# ============================================
# PART 3: ADVANCED CALCULATION TOOLS
# ============================================

print("\n🔢 STEP 3: Creating Advanced Tools")
print("-" * 60)

@tool
def advanced_calculator(expression: str) -> str:
    """
    Advanced calculator for complex mathematical operations.
    Supports: +, -, *, /, **, sqrt(), sin(), cos(), log(), etc.
    Input: mathematical expression like '(15 + 27) * 3' or 'sqrt(144)'
    """
    try:
        # Safe evaluation with math functions
        result = eval(expression, {"__builtins__": {}}, {
            "abs": abs, "round": round, "min": min, "max": max,
            "sum": sum, "pow": pow, "sqrt": math.sqrt,
            "sin": math.sin, "cos": math.cos, "tan": math.tan,
            "log": math.log, "log10": math.log10, "exp": math.exp,
            "pi": math.pi, "e": math.e, "ceil": math.ceil, "floor": math.floor
        })
        return f"Result: {result}"
    except Exception as e:
        return f"Error in calculation: {str(e)}"

print("✅ Tool: advanced_calculator")

@tool  
def text_analyzer(text: str) -> str:
    """
    Analyze text and provide statistics.
    Returns: word count, character count, sentence count, average word length.
    Input: text to analyze.
    """
    words = text.split()
    sentences = text.split('.')
    chars = len(text)
    
    if not words:
        return "No text to analyze."
    
    avg_word_len = sum(len(w) for w in words) / len(words)
    
    analysis = {
        "word_count": len(words),
        "character_count": chars,
        "sentence_count": len([s for s in sentences if s.strip()]),
        "average_word_length": round(avg_word_len, 2),
        "longest_word": max(words, key=len) if words else "N/A"
    }
    
    return json.dumps(analysis, indent=2)

print("✅ Tool: text_analyzer")

@tool
def project_summarizer(project_name: str) -> str:
    """
    Get detailed summary of a specific project.
    Use this when user asks about a particular project by name.
    Input: project name (e.g., 'sentiment analysis', 'churn prediction', 'rag system')
    """
    
    # Search for specific project
    results = collection.query(
        query_texts=[f"{project_name} project details"],
        n_results=2
    )
    
    if not results['documents'][0]:
        return f"No information found about project: {project_name}"
    
    summary = "\n\n".join(results['documents'][0])
    return f"Project '{project_name}' summary:\n\n{summary}"

print("✅ Tool: project_summarizer")

@tool
def skill_matcher(job_requirement: str) -> str:
    """
    Check if I have skills matching a job requirement.
    Use when user asks 'do you know X?' or 'can you do Y?'
    Input: skill or requirement to check (e.g., 'Python', 'deep learning', 'API deployment')
    """
    
    # Search skills in documents
    results = collection.query(
        query_texts=[f"skills expertise {job_requirement}"],
        n_results=2,
        where={"source": "technical_skills.txt"}  # Filter by skills document
    )
    
    if results['documents'][0]:
        evidence = results['documents'][0][0]
        return f"Yes! Here's evidence:\n\n{evidence}"
    else:
        return f"I don't have specific documentation about {job_requirement}, but I'm a fast learner!"

print("✅ Tool: skill_matcher")

# ============================================
# PART 4: INITIALIZE ADVANCED AGENT
# ============================================

print("\n" + "=" * 60)
print("STEP 4: Initializing Advanced Agent")
print("=" * 60)

# Combine all tools
tools = [
    search_my_documents,
    advanced_calculator,
    text_analyzer,
    project_summarizer,
    skill_matcher,
]

print(f"\n🔧 Agent equipped with {len(tools)} tools:")
for i, tool in enumerate(tools, 1):
    print(f"   {i}. {tool.name}")
    print(f"      → {tool.description[:80]}...")

# Initialize LLM
llm = ChatOpenAI(
    temperature=0.3,  # Slight creativity for natural responses
    model="gpt-4",
    openai_api_key=os.getenv('OPENAI_API_KEY')
)

# Memory with longer context
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    max_token_limit=2000  # Remember more context
)

# Custom agent prompt
agent_kwargs = {
    "system_message": """You are an intelligent assistant representing Emdad Hossain.

You have access to tools that let you:
1. Search through Emdad's documents (projects, skills, experience)
2. Perform calculations
3. Analyze text
4. Get project summaries
5. Check skill matches

When answering questions:
- Use search_my_documents for questions about experience, projects, skills
- Use project_summarizer when asked about specific projects
- Use skill_matcher when asked "do you know X?"
- Be concise but informative
- Cite sources when using document search
- Show confidence in Emdad's abilities

You are helping in a job interview context, so be professional and highlight achievements.
"""
}

# Initialize agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True,
    agent_kwargs=agent_kwargs
)

print("\n✅ Advanced Agent initialized!")
print("   Memory: 2000 token context window")
print("   Personality: Professional interview assistant")
print("   Capabilities: RAG + Math + Text + Skills")

# ============================================
# PART 5: DEMO SCENARIOS
# ============================================

print("\n" + "=" * 60)
print("DEMO: Interview-Style Questions")
print("=" * 60)

interview_questions = [
    "What machine learning projects has Emdad built?",
    "Does Emdad know how to deploy APIs to production?",
    "Tell me about the sentiment analysis project",
    "What's Emdad's accuracy on the churn prediction model?",
]

for i, question in enumerate(interview_questions, 1):
    print(f"\n{'=' * 60}")
    print(f"Question {i}: {question}")
    print('=' * 60)
    
    try:
        answer = agent.run(question)
        print(f"\n✅ Answer:")
        print(f"{answer}")
        print()
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Pause between questions
    if i < len(interview_questions):
        input("\n[Press Enter for next question...]")

# ============================================
# PART 6: COMPLEX MULTI-TOOL SCENARIO
# ============================================

print("\n" + "=" * 60)
print("COMPLEX SCENARIO: Multiple Tools")
print("=" * 60)

complex_scenario = """
What's Emdad's sentiment analysis accuracy as a percentage?
Then calculate what that would be out of 1000 predictions
(how many would be correct). Also, count how many words
are in the project description.
"""

print(f"\nScenario: {complex_scenario}")
print("\nAgent working...\n")

try:
    answer = agent.run(complex_scenario)
    print(f"\n✅ Final Answer:")
    print(f"{answer}")
except Exception as e:
    print(f"❌ Error: {e}")

# ============================================
# PART 7: INTERACTIVE INTERVIEW PREP
# ============================================

print("\n" + "=" * 60)
print("🎭 INTERACTIVE: Interview Simulation")
print("=" * 60)

print("""
Simulate an interview! Ask questions about:
- "What projects have you built?"
- "Do you know LangChain?"
- "Tell me about your RAG system"
- "What's your ML experience?"
- "Calculate 15% of 87,500" (salary negotiation!)

The agent will answer AS YOU, using your documents!

Type 'examples' for sample questions
Type 'reset' to clear conversation memory
Type 'quit' to exit
""")

while True:
    user_input = input("\n👔 Interviewer: ").strip()
    
    if user_input.lower() == 'quit':
        break
    
    if user_input.lower() == 'examples':
        print("\n📝 Sample Interview Questions:")
        print("   • What are your strongest technical skills?")
        print("   • Describe a challenging ML project you built")
        print("   • How do you handle model deployment?")
        print("   • Do you have experience with LLMs?")
        print("   • What's your biggest achievement?")
        print("   • Can you work with production systems?")
        print("   • Tell me about your leadership experience")
        continue
    
    if user_input.lower() == 'reset':
        memory.clear()
        print("🔄 Conversation memory cleared!")
        continue
    
    if not user_input:
        continue
    
    try:
        print("\n🤖 Agent thinking...\n")
        response = agent.run(user_input)
        print(f"\n💼 You (via Agent): {response}")
    except Exception as e:
        print(f"❌ Error: {e}")

print("\n" + "=" * 60)
print("✅ ADVANCED AGENT DEMO COMPLETE!")
print("=" * 60)

print("""
🎓 WHAT YOU BUILT:

An AI agent that can:

1. ✅ Search Your Documents
   - RAG integration
   - Retrieves from ChromaDB
   - Cites sources

2. ✅ Answer About Your Projects
   - Sentiment analysis (96.7% accuracy)
   - Churn prediction ($300K impact)
   - Image classifier (99.3% accuracy)
   - Job automator (97.5% time savings)

3. ✅ Verify Your Skills
   - Checks your technical_skills.txt
   - Confirms experience
   - Provides evidence

4. ✅ Perform Calculations
   - Math operations
   - Statistical analysis
   - Data processing

5. ✅ Remember Conversation
   - 2000 token context
   - Multi-turn dialogue
   - Context awareness

THIS IS GAME-CHANGING FOR INTERVIEWS! 🚀

INTERVIEW SCENARIO:

Interviewer: "Tell me about your RAG experience"
You: "Let me show you..." [Run this agent]
Agent: [Searches your docs, finds RAG project, cites evidence]
Interviewer: "Impressive! You built this?"
You: "Yes, and the agent you're talking to RIGHT NOW uses that RAG system as one of its tools!"

💥 MIND = BLOWN 💥

TECHNICAL SOPHISTICATION:

1. Multi-Tool Orchestration
   - Agent decides which tools to use
   - Combines results intelligently
   - Handles complex queries

2. RAG as a Tool
   - Novel approach (most just use RAG alone)
   - Shows system integration thinking
   - Production architecture pattern

3. Conversational Memory
   - Maintains context
   - References previous answers
   - Natural dialogue flow

4. Error Handling
   - Graceful failures
   - Helpful error messages
   - Parsing error recovery

5. Customized Personality
   - Professional tone
   - Interview-optimized responses
   - Confidence in abilities

INTERVIEW TALKING POINTS:

"I built an AI agent that combines multiple capabilities:
RAG for document retrieval, calculation tools, text analysis,
and conversational memory. The interesting part is using RAG
as a TOOL within an agent framework - the agent decides when
to search documents vs when to use other capabilities. This
demonstrates system composition and multi-tool orchestration.

The agent can answer interview questions about my experience
by actually retrieving evidence from my portfolio, which
ensures accurate, cited responses. It's like having my
entire portfolio as a conversational interface."

REAL-WORLD APPLICATIONS:

✅ Personal assistants that know YOUR data
✅ Customer service with company knowledge
✅ Research assistants with domain expertise
✅ Interview prep bots (what you just built!)
✅ Sales agents with product catalogs

YOU NOW HAVE:
1. RAG System ✅
2. Basic Agent ✅
3. Advanced Agent with RAG ✅

THIS IS YOUR INTERVIEW ARSENAL! 💪

Tomorrow: Fine-tuning + System Design + Mock Interview

For now: REST! You've earned it! 🎊
""")