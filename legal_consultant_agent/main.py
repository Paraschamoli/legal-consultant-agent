# |---------------------------------------------------------|
# |                                                         |
# |                 Give Feedback / Get Help                |
# | https://github.com/getbindu/Bindu/issues/new/choose    |
# |                                                         |
# |---------------------------------------------------------|
#
#  Thank you users! We ❤️ you! - 🌻

"""Legal Consultant Agent - AI-powered legal information and guidance agent.

Provides legal information with proper disclaimers and referrals.
Works without any external databases - only requires API keys.
"""

import argparse
import asyncio
import json
import os
import sys
import traceback
from pathlib import Path
from textwrap import dedent
from typing import Any

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.models.openrouter import OpenRouter
from agno.tools.mem0 import Mem0Tools
from bindu.penguin.bindufy import bindufy
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Global instances
agent: Agent | None = None
_initialized = False
_init_lock = asyncio.Lock()


class APIKeyError(ValueError):
    """API key is missing."""


def load_config() -> dict:
    """Load agent configuration from project root."""
    # Try multiple possible locations for agent_config.json
    possible_paths = [
        Path(__file__).parent.parent / "agent_config.json",  # Project root
        Path(__file__).parent / "agent_config.json",  # Same directory
        Path.cwd() / "agent_config.json",  # Current working directory
    ]

    for config_path in possible_paths:
        if config_path.exists():
            try:
                with open(config_path) as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️  Error reading {config_path}: {e}")
                continue

    # Default configuration
    return {
        "name": "legal-consultant-agent",
        "description": "AI legal consultant providing legal information and guidance",
        "version": "1.0.0",
        "deployment": {
            "url": "http://127.0.0.1:3773",
            "expose": True,
            "protocol_version": "1.0.0",
            "proxy_urls": ["127.0.0.1"],
            "cors_origins": ["*"],
        },
        "environment_variables": [
            {"key": "OPENAI_API_KEY", "description": "OpenAI API key for LLM calls", "required": False},
            {"key": "OPENROUTER_API_KEY", "description": "OpenRouter API key for LLM calls", "required": False},
            {"key": "MODEL_NAME", "description": "Model ID for OpenRouter", "required": False},
            {"key": "MEM0_API_KEY", "description": "Mem0 API key for memory operations", "required": False},
        ],
    }


async def initialize_agent() -> None:
    """Initialize the legal consultant agent with optional PostgreSQL/Mem0."""
    global agent

    # Get API keys from environment
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    mem0_api_key = os.getenv("MEM0_API_KEY")
    database_url = os.getenv("DATABASE_URL")
    model_name = os.getenv("MODEL_NAME", "openai/gpt-4o")

    # Check for required API key
    if not openai_api_key and not openrouter_api_key:
        error_msg = (
            "No LLM API key provided. Set OPENAI_API_KEY or OPENROUTER_API_KEY environment variable.\n"
            "For OpenRouter: https://openrouter.ai/keys\n"
            "For OpenAI: https://platform.openai.com/api-keys"
        )
        raise APIKeyError(error_msg)

    # Model selection logic
    if openai_api_key:
        model = OpenAIChat(id="gpt-4o", api_key=openai_api_key)
        print("✅ Using OpenAI GPT-4o")
    elif openrouter_api_key:
        model = OpenRouter(
            id=model_name,
            api_key=openrouter_api_key,
            cache_response=True,
            supports_native_structured_outputs=True,
        )
        print(f"✅ Using OpenRouter model: {model_name}")

    # Initialize tools - optional Mem0 for memory
    tools = []
    knowledge = None
    
    # Check for Mem0 API key first (as preferred memory system)
    if mem0_api_key:
        try:
            mem0_tools = Mem0Tools(api_key=mem0_api_key)
            tools.append(mem0_tools)
            print("🧠 Mem0 memory system enabled for conversation context")
        except Exception as e:
            print(f"⚠️  Mem0 initialization issue: {e}")
    
    # Optional: PostgreSQL + pgvector for knowledge base
    if database_url:
        try:
            from agno.vectordb.pgvector import PgVector
            from agno.knowledge.knowledge import Knowledge
            
            vector_db = PgVector(table_name="legal_docs", db_url=database_url)
            knowledge = Knowledge(vector_db=vector_db)
            print("💾 PostgreSQL vector database enabled for knowledge base")
            
            # Note: In production, you might want to load documents on-demand or during setup
            # For now, we'll let users add content as needed
        except ImportError:
            print("⚠️  PostgreSQL/pgvector not available. Install with: pip install pgvector psycopg[binary]")
        except Exception as e:
            print(f"⚠️  PostgreSQL connection error: {e}")
    
    if not mem0_api_key and not database_url:
        print("ℹ️  Using basic mode - no external database or memory system")

    # Create the legal consultant agent
    agent = Agent(
        name="Legal Consultant",
        model=model,
        tools=tools,
        knowledge=knowledge,
        search_knowledge=bool(knowledge),
        add_datetime_to_context=True,
        markdown=True,
        instructions=dedent("""\
            You are an AI Legal Consultant. Your role is to provide general legal 
            information and educational guidance based on widely recognized legal principles.

            CORE FUNCTION:
            Provide educational legal information while making absolutely clear that:
            1. You are not a licensed attorney
            2. This is not legal advice
            3. Users must consult real lawyers for legal matters

            RESPONSE FRAMEWORK:
            1. START EVERY RESPONSE WITH THIS EXACT DISCLAIMER:
               "**Legal Information Disclaimer**: I am an AI assistant providing general legal 
               information for educational purposes. I am not a lawyer, this is not legal advice, 
               and no attorney-client relationship is formed. Laws vary by jurisdiction and 
               change frequently. Always consult with a qualified attorney for legal advice."

            2. EDUCATIONAL CONTENT:
               - Explain general legal concepts in plain language
               - Describe common legal processes and procedures
               - Discuss typical rights and responsibilities
               - Explain potential consequences in general terms
               - Provide context about how legal systems work

            3. TRANSPARENCY:
               - Clearly state when information is based on general principles vs specific laws
               - Acknowledge limitations of your knowledge
               - Note that laws vary by state/country
               - Emphasize the importance of current, local legal information

            4. SAFETY PROTOCOLS:
               - NEVER suggest specific legal strategies
               - NEVER interpret statutes or case law as applying to specific situations
               - NEVER predict outcomes of legal matters
               - NEVER recommend for/against legal actions
               - ALWAYS defer to professional legal counsel

            5. REFERRAL GUIDANCE:
               - Explain when someone should consult an attorney
               - Suggest how to find qualified legal help
               - Recommend preparing questions for legal consultations
               - Mention legal aid resources for those who can't afford attorneys

            6. RESPONSE STRUCTURE:
               a. Mandatory disclaimer (exact wording above)
               b. General educational information about the topic
               c. Important considerations and limitations
               d. Guidance on seeking professional help
               e. Reminder to verify with current, local sources

            DATABASE NOTE:
            ${'I have access to a legal knowledge base and can search for specific legal references.' if knowledge else 'I provide general legal information based on my training.'}
        """),
        description="AI Legal Information Assistant - Provides general legal education and referral guidance",
    )
    
    if knowledge:
        print("✅ Legal Consultant agent initialized with knowledge base")
    elif mem0_api_key:
        print("✅ Legal Consultant agent initialized with memory system")
    else:
        print("✅ Legal Consultant agent initialized (basic mode)")


async def run_agent(messages: list[dict[str, str]]) -> Any:
    """Run the agent with the given messages."""
    global agent
    
    if not agent:
        error_msg = "Agent not initialized"
        raise RuntimeError(error_msg)

    # Run the agent and get response
    response = await agent.arun(messages)
    return response


async def handler(messages: list[dict[str, str]]) -> Any:
    """Handle incoming agent messages with lazy initialization."""
    global _initialized

    # Lazy initialization on first call
    async with _init_lock:
        if not _initialized:
            print("🔧 Initializing Legal Consultant Agent...")
            await initialize_agent()
            _initialized = True

    # Run the async agent
    result = await run_agent(messages)
    return result


async def cleanup() -> None:
    """Clean up any resources."""
    print("🧹 Cleaning up Legal Consultant Agent resources...")


def main():
    """Main entry point for the Legal Consultant Agent."""
    parser = argparse.ArgumentParser(
        description="Legal Consultant Agent - General legal information and education"
    )
    parser.add_argument(
        "--openai-api-key",
        type=str,
        default=os.getenv("OPENAI_API_KEY"),
        help="OpenAI API key (env: OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--openrouter-api-key",
        type=str,
        default=os.getenv("OPENROUTER_API_KEY"),
        help="OpenRouter API key (env: OPENROUTER_API_KEY)",
    )
    parser.add_argument(
        "--mem0-api-key",
        type=str,
        default=os.getenv("MEM0_API_KEY"),
        help="Mem0 API key for conversation memory (optional)",
    )
    parser.add_argument(
        "--database-url",
        type=str,
        default=os.getenv("DATABASE_URL"),
        help="PostgreSQL database URL with pgvector (optional)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("MODEL_NAME", "openai/gpt-4o"),
        help="Model ID for OpenRouter (env: MODEL_NAME)",
    )
    args = parser.parse_args()

    # Set environment variables if provided via CLI
    if args.openai_api_key:
        os.environ["OPENAI_API_KEY"] = args.openai_api_key
    if args.openrouter_api_key:
        os.environ["OPENROUTER_API_KEY"] = args.openrouter_api_key
    if args.mem0_api_key:
        os.environ["MEM0_API_KEY"] = args.mem0_api_key
    if args.database_url:
        os.environ["DATABASE_URL"] = args.database_url
    if args.model:
        os.environ["MODEL_NAME"] = args.model

    print("=" * 60)
    print("⚖️  LEGAL CONSULTANT AGENT")
    print("=" * 60)
    print("📚 Purpose: General legal information and education")
    print("⚠️  Disclaimer: NOT A LAWYER - Provides educational info only")
    
    # Display configuration
    config_info = []
    if os.getenv("DATABASE_URL"):
        config_info.append("💾 Database: PostgreSQL with pgvector (knowledge base)")
    if os.getenv("MEM0_API_KEY"):
        config_info.append("🧠 Memory: Mem0 system (conversation context)")
    
    if config_info:
        for info in config_info:
            print(info)
    else:
        print("💾 Storage: None required (basic mode)")
    
    print("=" * 60)
    print("Note: This agent provides general information only.")
    print("      Always consult qualified attorneys for legal advice.")
    print("=" * 60)

    # Load configuration
    config = load_config()

    try:
        # Bindufy and start the agent server
        print("\n🚀 Starting Legal Consultant Agent server...")
        print(f"🌐 Access at: {config.get('deployment', {}).get('url', 'http://127.0.0.1:3773')}")
        bindufy(config, handler)
    except KeyboardInterrupt:
        print("\n🛑 Legal Consultant Agent stopped")
    except Exception as e:
        print(f"❌ Error starting agent: {e}")
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Cleanup on exit
        asyncio.run(cleanup())


if __name__ == "__main__":
    main()