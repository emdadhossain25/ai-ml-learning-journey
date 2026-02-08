"""
Agentic RAG System - Fixed paths
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Optional
from openai import AzureOpenAI
from dotenv import load_dotenv
import logging
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from rag_pipeline_azure import AzureRAGPipeline

# Load .env from project root
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgenticRAG:
    """Agentic RAG with tool calling + self-reflection"""
    
    def __init__(self):
        # Initialize Azure OpenAI
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        
        self.deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        
        # Initialize RAG
        self.rag = AzureRAGPipeline()
        
        # Try to load existing index
        try:
            self.rag.load_index()
            logger.info("✅ Loaded existing index")
        except Exception as e:
            logger.info(f"⚠️  Building new index: {e}")
            self.rag.build_index("documents")
            self.rag.save_index()
        
        self.total_tokens_used = 0
        
    def retrieval_tool(self, query: str, top_k: int = 3) -> List[Dict]:
        """Retrieve relevant documents (FREE)"""
        logger.info(f"🔍 [TOOL] Retrieval: {query}")
        results = self.rag.retrieve(query, top_k=top_k)
        logger.info(f"   Found {len(results)} chunks")
        return results
    
    def _call_azure_gpt(
        self,
        messages: List[Dict],
        temperature: float = 0.3,
        max_tokens: int = 500
    ) -> str:
        """Call Azure GPT-4o-mini"""
        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            tokens_used = response.usage.total_tokens
            self.total_tokens_used += tokens_used
            
            logger.info(f"💰 Tokens: {tokens_used} (Total: {self.total_tokens_used})")
            
            return response.choices[0].message.content
        
        except Exception as e:
            logger.error(f"❌ Azure API error: {e}")
            raise
    
    def agent_decide_retrieval(self, query: str) -> Dict:
        """Agent decides if retrieval needed"""
        decision_prompt = f"""Analyze this question: "{query}"

Should I retrieve documents or answer directly?

Respond ONLY with JSON:
{{"retrieve": true/false, "reason": "brief explanation"}}

Examples:
- "What is RAG?" → {{"retrieve": true, "reason": "technical question"}}
- "Hello" → {{"retrieve": false, "reason": "greeting"}}"""
        
        messages = [
            {"role": "system", "content": "You decide if retrieval is needed."},
            {"role": "user", "content": decision_prompt}
        ]
        
        response = self._call_azure_gpt(messages, temperature=0, max_tokens=50)
        
        try:
            decision = json.loads(response)
            logger.info(f"🤖 [AGENT] {decision}")
            return decision
        except:
            return {"retrieve": True, "reason": "default"}
    
    def generate_answer(self, query: str, context: Optional[List[Dict]] = None) -> str:
        """Generate answer"""
        if context:
            context_str = "\n\n".join([
                f"[{c['source']}]\n{c['text']}"
                for c in context
            ])
            
            prompt = f"""Answer ONLY using this context. If answer not in context, say "I don't have information about that."

Context:
{context_str}

Question: {query}

Answer:"""
        else:
            prompt = f"Answer this briefly: {query}"
        
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt}
        ]
        
        return self._call_azure_gpt(messages, temperature=0.3, max_tokens=400)
    
    def critique_answer(self, query: str, answer: str, context: List[Dict]) -> Dict:
        """Self-reflection critic"""
        critique_prompt = f"""Evaluate this answer:

Question: {query}
Answer: {answer}
Sources: {[c['source'] for c in context]}

Is it grounded in context? Accurate? Complete?

JSON only:
{{"is_good": true/false, "issues": "problems", "suggestion": "improvements"}}"""
        
        messages = [
            {"role": "system", "content": "You are a critic."},
            {"role": "user", "content": critique_prompt}
        ]
        
        response = self._call_azure_gpt(messages, temperature=0, max_tokens=150)
        
        try:
            critique = json.loads(response)
            logger.info(f"🎯 [CRITIC] Good: {critique.get('is_good', True)}")
            return critique
        except:
            return {"is_good": True, "issues": "", "suggestion": ""}
    
    def refine_answer(self, query: str, initial_answer: str, critique: Dict, context: List[Dict]) -> str:
        """Refine based on critique"""
        if critique.get('is_good', True):
            return initial_answer
        
        refine_prompt = f"""Improve this answer:

Question: {query}
Initial Answer: {initial_answer}
Issues: {critique['issues']}
Suggestion: {critique['suggestion']}

Context:
{chr(10).join([f"{c['text'][:200]}..." for c in context])}

Improved answer:"""
        
        messages = [
            {"role": "system", "content": "You improve answers."},
            {"role": "user", "content": refine_prompt}
        ]
        
        refined = self._call_azure_gpt(messages, temperature=0.3, max_tokens=400)
        logger.info("✨ [REFINE] Improved")
        return refined
    
    def answer_query(self, query: str, use_critic: bool = True, top_k: int = 3) -> Dict:
        """Main agentic pipeline"""
        logger.info(f"\n{'='*60}")
        logger.info(f"🎯 QUERY: {query}")
        logger.info(f"{'='*60}")
        
        # Decide retrieval
        decision = self.agent_decide_retrieval(query)
        
        context = []
        if decision['retrieve']:
            context = self.retrieval_tool(query, top_k=top_k)
        
        # Generate answer
        logger.info("💭 [GENERATE] Creating answer...")
        initial_answer = self.generate_answer(query, context if context else None)
        
        final_answer = initial_answer
        critique_result = None
        
        # Critique and refine
        if use_critic and context:
            logger.info("🎯 [CRITIC] Evaluating...")
            critique_result = self.critique_answer(query, initial_answer, context)
            
            if not critique_result.get('is_good', True):
                logger.info("✨ [REFINE] Improving...")
                final_answer = self.refine_answer(query, initial_answer, critique_result, context)
        
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
    print("\n🚀 Initializing Agentic RAG")
    print("="*60)
    
    agent = AgenticRAG()
    
    print("\n🧪 Testing")
    print("="*60)
    
    result = agent.answer_query("What is RAG?", use_critic=True)
    
    print(f"\n💬 Question: {result['query']}")
    print(f"\n💡 Answer:\n{result['answer']}")
    print(f"\n📚 Sources: {', '.join(result['sources'])}")
    print(f"\n💰 Tokens: {result['tokens_used']}")
    print("\n✅ Ready!")
