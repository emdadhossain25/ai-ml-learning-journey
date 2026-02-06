"""
Day 27: AI Agents with LangChain
Give LLMs the power to use tools and take actions
"""

import os
from dotenv import load_dotenv
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.tools import tool
from langchain.memory import ConversationBufferMemory
import math
import requests

load_dotenv()

print("=" * 60)
print("AI AGENTS: LLMs with Tools")
print("=" * 60)

# ============================================
# PART 1: UNDERSTANDING AGENTS
# ============================================

print("\n📚 WHAT IS AN AI AGENT?")
print("-" * 60)

explanation = """
AGENT = LLM + TOOLS + REASONING

Components:
1. LLM Brain (ChatGPT, Claude, etc.)
   - Decides what to do
   - Interprets results
   - Generates final answer

2. Tools (Functions the LLM can call)
   - Calculator
   - Web search
   - Database queries
   - API calls
   - File operations

3. Reasoning Framework (How it thinks)
   - ReAct: Reason + Act
   - Plan-and-Execute
   - Chain-of-Thought

Example Flow:
User: "What's the weather in Paris and 15% tip on $89.50?"

Agent thinks:
[Thought] Need to get weather AND calculate tip
[Action] Use weather_tool("Paris")
[Observation] Weather is 18°C, sunny
[Thought] Now calculate tip
[Action] Use calculator_tool(89.50 * 0.15)
[Observation] Tip = $13.43
[Thought] I have both answers
[Final Answer] "Weather in Paris is 18°C and sunny. 
                15% tip on $89.50 is $13.43"

The agent DECIDED to use TWO tools and combined results!
"""

print(explanation)

# ============================================
# PART 2: CREATE SIMPLE TOOLS
# ============================================

print("=" * 60)
print("CREATING TOOLS")
print("=" * 60)

# Tool 1: Calculator
@tool
def calculator(expression: str) -> str:
    """
    Useful for mathematical calculations.
    Input should be a mathematical expression like '5 * 7' or '15 + 23'
    """
    try:
        # Safe eval (only math operations)
        result = eval(expression, {"__builtins__": {}}, {
            "abs": abs, "round": round, "min": min, "max": max,
            "sum": sum, "pow": pow, "sqrt": math.sqrt,
            "sin": math.sin, "cos": math.cos, "tan": math.tan,
            "pi": math.pi, "e": math.e
        })
        return f"The result is: {result}"
    except Exception as e:
        return f"Error: {str(e)}"

print("✅ Tool 1: Calculator")
print("   Can do: +, -, *, /, sqrt(), sin(), etc.")

# Tool 2: Word Counter
@tool
def word_counter(text: str) -> str:
    """
    Counts words in a given text.
    Input should be a string of text.
    """
    words = text.split()
    return f"The text contains {len(words)} words."

print("✅ Tool 2: Word Counter")

# Tool 3: String Reverser
@tool
def string_reverser(text: str) -> str:
    """
    Reverses a string.
    Input should be text to reverse.
    """
    return f"Reversed: {text[::-1]}"

print("✅ Tool 3: String Reverser")

# Tool 4: Simple Web Search (mock for now)
@tool
def web_search(query: str) -> str:
    """
    Search the web for information.
    Input should be a search query.
    """
    # In production, integrate with real search API
    # For now, mock response
    mock_results = {
        "python": "Python is a high-level programming language created by Guido van Rossum in 1991.",
        "ai": "Artificial Intelligence is the simulation of human intelligence by machines.",
        "machine learning": "Machine Learning is a subset of AI that enables systems to learn from data.",
    }
    
    query_lower = query.lower()
    for key in mock_results:
        if key in query_lower:
            return mock_results[key]
    
    return f"Mock search result for: {query}"

print("✅ Tool 4: Web Search (mock)")

# ============================================
# PART 3: INITIALIZE AGENT
# ============================================

print("\n" + "=" * 60)
print("INITIALIZING AGENT")
print("=" * 60)

# Create tool list
tools = [
    calculator,
    word_counter,
    string_reverser,
    web_search
]

print(f"\n🔧 Available tools: {len(tools)}")
for tool in tools:
    print(f"   • {tool.name}: {tool.description}")

# Initialize LLM
llm = ChatOpenAI(
    temperature=0,  # Deterministic for tools
    model="gpt-4",
    openai_api_key=os.getenv('OPENAI_API_KEY')
)

# Initialize memory (so agent remembers conversation)
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Initialize agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True,  # Show reasoning steps!
    handle_parsing_errors=True
)

print("\n✅ Agent initialized!")
print("   Type: REACT (Reason + Act)")
print("   Memory: Enabled (remembers conversation)")
print("   Verbose: True (shows thinking process)")

# ============================================
# PART 4: TEST THE AGENT
# ============================================

print("\n" + "=" * 60)
print("AGENT IN ACTION - Demo Queries")
print("=" * 60)

demo_queries = [
    "What is 847 multiplied by 293?",
    "How many words are in this sentence: 'AI agents are powerful tools'?",
    "What is Python programming language?",
]

for i, query in enumerate(demo_queries, 1):
    print(f"\n{'=' * 60}")
    print(f"Query {i}: {query}")
    print('=' * 60)
    
    try:
        response = agent.run(query)
        print(f"\n✅ Final Answer:")
        print(f"   {response}")
    except Exception as e:
        print(f"❌ Error: {e}")

# ============================================
# PART 5: COMPLEX MULTI-TOOL QUERY
# ============================================

print("\n" + "=" * 60)
print("COMPLEX QUERY - Multiple Tools")
print("=" * 60)

complex_query = """
Calculate the square root of 144, then count how many words 
are in the sentence 'Machine learning is fascinating', 
and reverse the word 'AGENTS'.
"""

print(f"\nQuery: {complex_query}")
print("\nAgent thinking...\n")

try:
    response = agent.run(complex_query)
    print(f"\n✅ Final Answer:")
    print(f"   {response}")
except Exception as e:
    print(f"❌ Error: {e}")

# ============================================
# PART 6: CONVERSATIONAL MEMORY
# ============================================

print("\n" + "=" * 60)
print("TESTING MEMORY - Conversation Context")
print("=" * 60)

conversation = [
    "My name is Emdad",
    "What is 15 + 27?",
    "What's my name?",  # Tests memory!
]

for query in conversation:
    print(f"\n💬 User: {query}")
    try:
        response = agent.run(query)
        print(f"🤖 Agent: {response}")
    except Exception as e:
        print(f"❌ Error: {e}")

# ============================================
# PART 7: INTERACTIVE MODE
# ============================================

print("\n" + "=" * 60)
print("🎮 INTERACTIVE AGENT")
print("=" * 60)

print("""
Chat with the AI agent!

Available tools:
- calculator - Mathematical calculations
- word_counter - Count words in text
- string_reverser - Reverse strings
- web_search - Search information (mock)

The agent will decide which tools to use!

Type 'quit' to exit
""")

while True:
    user_input = input("\n💬 You: ").strip()
    
    if user_input.lower() == 'quit':
        break
    
    if not user_input:
        continue
    
    try:
        print("\n🤖 Agent thinking...\n")
        response = agent.run(user_input)
        print(f"\n🤖 Agent: {response}")
    except Exception as e:
        print(f"❌ Error: {e}")

print("\n" + "=" * 60)
print("✅ AI AGENT DEMO COMPLETE!")
print("=" * 60)

print("""
🎓 WHAT YOU LEARNED:

1. AGENTS = LLMs + TOOLS + REASONING
   - LLM decides which tool to use
   - Tools extend LLM capabilities
   - Agent combines results

2. TOOLS
   - Python functions with @tool decorator
   - Clear descriptions (LLM reads these!)
   - Input/output specification

3. REACT FRAMEWORK
   - Reason: Think about what to do
   - Act: Use a tool
   - Observe: See result
   - Repeat until answer found

4. MEMORY
   - Agent remembers conversation
   - Can reference previous context
   - Conversational experience

5. LANGCHAIN
   - Framework for building agents
   - Easy tool integration
   - Multiple agent types

WHY THIS MATTERS:

Pure LLMs:
❌ Can't do accurate math
❌ Can't access external data
❌ Can't take actions
❌ Limited to training knowledge

LLMs with Agents:
✅ Use calculator for math
✅ Search web for current info
✅ Call APIs, databases
✅ Take real actions

INTERVIEW TALKING POINTS:

"I built AI agents using LangChain. The agent uses a ReAct
framework - it reasons about which tools to use, takes actions,
observes results, and iterates until it has the answer. I
implemented custom tools for calculations, text processing,
and web search. The agent has conversational memory, so it
maintains context across the conversation. This extends LLM
capabilities beyond their training data."

REAL-WORLD APPLICATIONS:
- Customer service bots (access databases, create tickets)
- Research assistants (search, summarize, cite sources)
- Data analysts (query databases, generate reports)
- Personal assistants (calendar, email, reminders)
- DevOps tools (monitor systems, execute commands)

NEXT: Build advanced agent with more powerful tools!
""")