"""
Command-line interface for Agentic RAG
Interactive chat loop
"""

import sys
from agentic_rag import AgenticRAG
import logging

logging.basicConfig(level=logging.WARNING)  # Quiet mode for CLI


class Color:
    """Terminal colors"""
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_header():
    """Print welcome message"""
    print("\n" + "="*60)
    print(f"{Color.BOLD}🤖 Mini Agentic RAG System{Color.END}")
    print("="*60)
    print(f"\n{Color.GREEN}✅ System ready!{Color.END}")
    print(f"{Color.YELLOW}💡 Ask questions about ML, RAG, or LLM Agents{Color.END}")
    print("\nCommands:")
    print("  • Type your question and press Enter")
    print("  • 'stats' - Show usage statistics")
    print("  • 'help' - Show this help")
    print("  • 'quit' or 'exit' - Exit the system")
    print("="*60 + "\n")


def print_answer(result: dict):
    """Pretty print answer"""
    print(f"\n{Color.BLUE}💡 Answer:{Color.END}")
    print("-"*60)
    print(result['answer'])
    print("-"*60)
    
    if result['sources']:
        print(f"\n{Color.GREEN}📚 Sources:{Color.END} {', '.join(result['sources'])}")
    
    print(f"{Color.YELLOW}💰 Tokens used this query:{Color.END} {result['tokens_used'] - (result.get('previous_tokens', 0))}")
    print(f"{Color.YELLOW}📊 Total tokens used:{Color.END} {result['tokens_used']}")
    print()


def main():
    """Main CLI loop"""
    print_header()
    
    # Initialize agent
    print(f"{Color.YELLOW}🔄 Loading system...{Color.END}")
    agent = AgenticRAG()
    print(f"{Color.GREEN}✅ Ready!{Color.END}\n")
    
    previous_tokens = 0
    
    while True:
        try:
            # Get user input
            query = input(f"{Color.BOLD}You:{Color.END} ").strip()
            
            if not query:
                continue
            
            # Handle commands
            if query.lower() in ['quit', 'exit', 'q']:
                print(f"\n{Color.GREEN}👋 Goodbye!{Color.END}\n")
                break
            
            if query.lower() == 'help':
                print_header()
                continue
            
            if query.lower() == 'stats':
                print(f"\n{Color.BLUE}📊 Usage Statistics:{Color.END}")
                print(f"   Total tokens used: {agent.total_tokens_used}")
                print(f"   Total chunks indexed: {len(agent.rag.chunks)}")
                print(f"   Documents loaded: {len(set([m['source'] for m in agent.rag.metadata]))}")
                print()
                continue
            
            # Process query
            print(f"\n{Color.YELLOW}🤖 Agent thinking...{Color.END}")
            
            result = agent.answer_query(
                query=query,
                use_critic=True,
                top_k=3
            )
            
            result['previous_tokens'] = previous_tokens
            previous_tokens = result['tokens_used']
            
            print_answer(result)
        
        except KeyboardInterrupt:
            print(f"\n\n{Color.GREEN}👋 Goodbye!{Color.END}\n")
            break
        
        except Exception as e:
            print(f"\n{Color.RED}❌ Error: {e}{Color.END}\n")


if __name__ == "__main__":
    main()
