"""
Agentic RAG System with ReAct + Self-Reflection
Uses Azure GPT-4o-mini (FREE TIER OPTIMIZED)
"""

import os
from typing import List, Dict, Optional
from openai import AzureOpenAI
from dotenv import load_dotenv
import logging
import json

from rag_pipeline_azure import AzureRAGPipeline

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgenticRAG:
    """
    Agentic RAG with:
    1. Tool calling (retrieval tool)
    2. Self-reflection (critic)
    3. Final answer generation
    
    FREE TIER OPTIMIZED:
    - Small context windows
    - Efficient prompts
    - Caching retrieval results
    """
    
    def __init__(self):
        # Initialize Azure OpenAI client
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        
        self.deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        
        # Initialize RAG pipeline
        self.rag = AzureRAGPipeline()
        
        # Try to load existing index (skip rebuilding)
        try:
            self.rag.load_index()
            logger.info("✅ Loaded existing index")
        except:
            logger.info("⚠️  No index found, building new one...")
            self.rag.build_index("documents")
            self.rag.save_index()
        
        # Token counter (free tier tracking)
        self.total_tokens_used = 0
        
    def retrieval_tool(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Tool: Retrieve relevant documents
        This is FREE (uses local FAISS + embeddings)
        """
        logger.info(f"🔍 [TOOL] Retrieval: {query}")
        results = self.rag.retrieve(query, top_k=top_k)
        
        logger.info(f"   Found {len(results)} relevant chunks")
        return results
    
    def _call_azure_gpt(
        self,
        messages: List[Dict],
        temperature: float = 0.3,
        max_tokens: int = 500  # Keep low for free tier!
    ) -> str:
        """
        Call Azure GPT-4o-mini
        COST: ~$0.00015 per 1K input tokens, ~$0.0006 per 1K output tokens
        """
        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Track token usage
            tokens_used = response.usage.total_tokens
            self.total_tokens_used += tokens_used
            
            logger.info(f"💰 Tokens used: {tokens_used} (Total: {self.total_tokens_used})")
            
            return response.choices[0].message.content
        
        except Exception as e:
            logger.error(f"❌ Azure API error: {e}")
            return f"Error calling Azure API: {str(e)}"
    
    def agent_decide_retrieval(self, query: str) -> Dict:
        """
        Agent decides: Should I retrieve documents or answer directly?
        This saves Azure tokens!
        """
        decision_prompt = f"""You are a helpful assistant. Analyze this question:

Question: {query}

Should I retrieve documents to answer this, or can I answer directly?

Respond in JSON:
{{"retrieve": true/false, "reason": "why"}}

Examples:
- "What is supervised learning?" → {{"retrieve": true, "reason": "Needs document knowledge"}}
- "Hello, how are you?" → {{"retrieve": false, "reason": "Simple greeting"}}
- "Thank you!" → {{"retrieve": false, "reason": "Acknowledgment"}}
"""
        
        messages = [
            {"role": "system", "content": "You decide if retrieval is needed. Be concise."},
            {"role": "user", "content": decision_prompt}
        ]
        
        response = self._call_azure_gpt(messages, temperature=0, max_tokens=100)
        
        try:
            decision = json.loads(response)
            logger.info(f"🤖 [AGENT] Decision: {decision}")
            return decision
        except:
            # Default to retrieval if parsing fails
            return {"retrieve": True, "reason": "Parsing failed, defaulting to retrieval"}
    
    def generate_answer(self, query: str, context: Optional[List[Dict]] = None) -> str:
        """
        Generate answer using retrieved context
        """
        if context:
            # Build context string
            context_str = "\n\n".join([
                f"[Source: {c['source']}]\n{c['text']}"
                for c in context
            ])
            
            prompt = f"""Answer the question based ONLY on the context provided. If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context_str}

Question: {query}

Answer:"""
        else:
            prompt = f"""Answer this question directly and concisely:

Question: {query}

Answer:"""
        
        messages = [
            {
                "role": "system",
                "content": "You are a helpful AI assistant. Answer accurately and cite sources when using retrieved documents."
            },
            {"role": "user", "content": prompt}
        ]
        
        answer = self._call_azure_gpt(messages, temperature=0.3, max_tokens=400)
        
        return answer
    
    def critique_answer(self, query: str, answer: str, context: List[Dict]) -> Dict:
        """
        Self-reflection: Critique the generated answer
        Returns: {is_good: bool, issues: str, suggestion: str}
        """
        critique_prompt = f"""You are a critic. Evaluate this answer:

Question: {query}

Answer: {answer}

Available Context Sources: {[c['source'] for c in context]}

Evaluate:
1. Is the answer grounded in the context?
2. Is it accurate and complete?
3. Does it cite sources?

Respond in JSON:
{{"is_good": true/false, "issues": "what's wrong if any", "suggestion": "how to improve"}}
"""
        
        messages = [
            {"role": "system", "content": "You are a critical evaluator. Be honest."},
            {"role": "user", "content": critique_prompt}
        ]
        
        response = self._call_azure_gpt(messages, temperature=0, max_tokens=200)
        
        try:
            critique = json.loads(response)
            logger.info(f"🎯 [CRITIC] Evaluation: {critique['is_good']}")
            return critique
        except:
            return {"is_good": True, "issues": "", "suggestion": ""}
    
    def refine_answer(self, query: str, initial_answer: str, critique: Dict, context: List[Dict]) -> str:
        """
        Refine answer based on critique
        """
        if critique.get('is_good', True):
            return initial_answer  # No refinement needed
        
        refine_prompt = f"""Improve this answer based on the critique:

Question: {query}

Initial Answer: {initial_answer}

Critique: {critique['issues']}
Suggestion: {critique['suggestion']}

Context:
{chr(10).join([f"[{c['source']}] {c['text'][:200]}..." for c in context])}

Provide an improved answer:"""
        
        messages = [
            {"role": "system", "content": "You improve answers based on feedback."},
            {"role": "user", "content": refine_prompt}
        ]
        
        refined = self._call_azure_gpt(messages, temperature=0.3, max_tokens=400)
        
        logger.info("✨ [REFINE] Answer improved")
        return refined
    
    def answer_query(
        self,
        query: str,
        use_critic: bool = True,
        top_k: int = 3
    ) -> Dict:
        """
        Main agentic pipeline:
        1. Decide if retrieval needed
        2. Retrieve documents (if needed)
        3. Generate answer
        4. Critique answer
        5. Refine if needed
        6. Return final answer
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"🎯 QUERY: {query}")
        logger.info(f"{'='*60}")
        
        # STEP 1: Agent decides if retrieval needed (saves tokens!)
        decision = self.agent_decide_retrieval(query)
        
        context = []
        if decision['retrieve']:
            # STEP 2: Retrieve documents (FREE - local)
            context = self.retrieval_tool(query, top_k=top_k)
        
        # STEP 3: Generate initial answer
        logger.info("💭 [GENERATE] Creating answer...")
        initial_answer = self.generate_answer(query, context if context else None)
        
        final_answer = initial_answer
        critique_result = None
        
        # STEP 4 & 5: Critique and refine (if enabled)
        if use_critic and context:
            logger.info("🎯 [CRITIC] Evaluating answer...")
            critique_result = self.critique_answer(query, initial_answer, context)
            
            if not critique_result.get('is_good', True):
                logger.info("✨ [REFINE] Improving answer...")
                final_answer = self.refine_answer(query, initial_answer, critique_result, context)
        
        # Return result
        return {
            'query': query,
            'answer': final_answer,
            'sources': [c['source'] for c in context] if context else [],
            'context_used': len(context),
            'tokens_used': self.total_tokens_used,
            'critique': critique_result,
            'retrieval_decision': decision
        }


if __name__ == "__main__":
    # Test the agent
    print("\n🚀 Initializing Agentic RAG System")
    print("="*60)
    
    agent = AgenticRAG()
    
    print("\n🧪 Testing Agentic RAG")
    print("="*60)
    
    # Test query
    result = agent.answer_query(
        "What is RAG and why is it useful?",
        use_critic=True
    )
    
    print(f"\n💬 Question: {result['query']}")
    print(f"\n💡 Answer:\n{result['answer']}")
    print(f"\n📚 Sources: {', '.join(result['sources'])}")
    print(f"\n💰 Total tokens used: {result['tokens_used']}")
    
    if result['critique']:
        print(f"\n🎯 Critique: {result['critique']}")
    
    print("\n" + "="*60)
    print("✅ Agentic RAG System Ready!")
