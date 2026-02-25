<p align="center">
  <img src="https://raw.githubusercontent.com/getbindu/create-bindu-agent/refs/heads/main/assets/light.svg" alt="bindu Logo" width="200">
</p>

<h1 align="center">Legal Consultant Agent</h1>
<h3 align="center">AI-Powered Legal Information Assistant with Professional Referrals</h3>

<p align="center">
  <strong>Provides general legal education, explanations of legal concepts, and guidance on when to seek professional legal help. NOT a substitute for qualified legal advice.</strong><br/>
  Always includes clear disclaimers and emphasizes consulting licensed attorneys for actual legal matters.
</p>

<p align="center">
  <a href="https://github.com/Paraschamoli/legal-consultant-agent/actions/workflows/build-and-push.yml?query=branch%3Amain">
    <img src="https://img.shields.io/github/actions/workflow/status/Paraschamoli/legal-consultant-agent/build-and-push.yml?branch=main" alt="Build status">
  </a>
  <a href="https://img.shields.io/github/license/Paraschamoli/legal-consultant-agent">
    <img src="https://img.shields.io/github/license/Paraschamoli/legal-consultant-agent" alt="License">
  </a>
  <a href="https://img.shields.io/badge/python-3.12-blue">
    <img src="https://img.shields.io/badge/python-3.12-blue" alt="Python 3.12">
  </a>
  <a href="https://img.shields.io/badge/version-1.0.0-blue">
    <img src="https://img.shields.io/badge/version-1.0.0-blue" alt="Version 1.0.0">
  </a>
</p>

---

## Table of Contents

- [🎯 What is Legal Consultant Agent?](#-what-is-legal-consultant-agent)
  - [Key Features](#key-features)
  - [Built-in Tools](#built-in-tools)
  - [Core Principles](#core-principles)
- [🏗️ Architecture](#️-architecture)
- [🚀 Quick Start](#-quick-start)
- [🔧 Configuration](#-configuration)
- [💡 Usage Examples](#-usage-examples)
- [🔌 API Reference](#-api-reference)
- [🐳 Docker Deployment](#-docker-deployment)
- [📁 Project Structure](#-project-structure)
- [🧪 Testing](#-testing)
- [🚨 Troubleshooting](#-troubleshooting)
- [📊 Dependencies](#-dependencies)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)
- [🙏 Credits & Acknowledgments](#-credits--acknowledgments)
- [🔗 Useful Links](#-useful-links)
- [📝 Changelog](#-changelog)

---

## 🎯 What is Legal Consultant Agent?

An AI-powered assistant that provides general legal information and educational guidance while maintaining strict boundaries to avoid unauthorized practice of law. Think of it as a legal education tool that helps you understand concepts before consulting professionals.

### Key Features
*   **⚖️ Educational Focus** - General legal information, not specific advice
*   **⚠️ Mandatory Disclaimers** - Clear warnings in every response
*   **🧠 Memory Support** - Optional Mem0 integration for conversation context
*   **💾 Knowledge Base** - Optional PostgreSQL/pgvector for legal documents
*   **🔗 Professional Referrals** - Guidance on finding qualified attorneys
*   **🛡️ Safety Protocols** - Never provides legal advice or strategy
*   **🎯 Skill-Based Architecture** - Modular skills for different legal topics

### Built-in Tools
*   **Mem0Tools** - Optional conversation memory for context retention
*   **PostgreSQL Integration** - Optional vector database for legal knowledge
*   **Professional Boundaries** - Strict adherence to legal education scope

### Core Principles
1.  **Education, Not Advice** - Provides general information only
2.  **Clear Disclaimers** - Every response starts with appropriate warnings
3.  **Professional Referral** - Always recommends consulting licensed attorneys
4.  **Ethical Boundaries** - Never crosses into unauthorized practice of law

---

## 🏗️ Architecture

The Legal Consultant Agent is built using modern AI agent frameworks and follows a modular, skill-based architecture designed for scalability and maintainability.

### Core Components

*   **Agno Framework** - Provides the agent orchestration and tool integration
*   **Bindu Platform** - Handles deployment, API management, and agent discovery
*   **Skill System** - Modular skills for different legal education topics
*   **LLM Integration** - Support for multiple LLM providers (OpenAI, OpenRouter)
*   **Memory & Knowledge** - Optional Mem0 for conversation memory and PostgreSQL for document storage

### Agent Flow

1.  **Request Processing** - Incoming queries are processed through Bindu's JSON-RPC interface
2.  **Skill Matching** - The appropriate skill is selected based on the query type
3.  **LLM Processing** - The query is processed by the configured LLM with safety instructions
4.  **Response Generation** - Educational content is generated with mandatory disclaimers
5.  **Memory Integration** - Conversation context is maintained via Mem0 (optional)
6.  **Response Delivery** - Structured response is returned via the API

### Safety Architecture

*   **Mandatory Disclaimers** - Every response includes legal disclaimers
*   **Boundary Enforcement** - Strict instructions prevent unauthorized legal advice
*   **Ethical Guidelines** - Core principles embedded in agent instructions
*   **Referral Integration** - Automatic guidance to professional legal help

---

> **🌐 Join the Internet of Agents**
> Register your agent at [bindus.directory](https://bindus.directory) to make it discoverable worldwide and enable agent-to-agent collaboration. **It takes 2 minutes and unlocks the full potential of your agent.**

---

## 🚀 Quick Start

### 1. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/Paraschamoli/legal-consultant-agent.git
cd legal-consultant-agent

# Set up virtual environment with uv
uv venv --python 3.12
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv sync
```

### 2. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your API keys (choose one LLM provider):
OPENAI_API_KEY=sk-...      # For OpenAI GPT-4o
OPENROUTER_API_KEY=sk-...  # For OpenRouter (alternative)
MEM0_API_KEY=sk-...        # Optional: For conversation memory
DATABASE_URL=postgresql://...  # Optional: For vector database
```

### 3. Run Locally

```bash
# Start the legal consultant agent
python legal_consultant_agent/main.py

# Or using uv
uv run python legal_consultant_agent/main.py
```

### 4. Test with Docker

```bash
# Build and run with Docker Compose
docker-compose up --build

# Access at: http://localhost:3773
```

## 🔧 Configuration

### Environment Variables
Create a `.env` file:

```env
# Choose ONE LLM provider
OPENAI_API_KEY=sk-...        # OpenAI API key
OPENROUTER_API_KEY=sk-...    # OpenRouter API key (alternative)

# Optional features
MODEL_NAME=openai/gpt-4o     # Model ID for OpenRouter
MEM0_API_KEY=sk-...          # Optional: For memory operations
DATABASE_URL=postgresql+psycopg://username:password@host:port/database  # Optional vector DB
```

### Port Configuration
Default port: `3773` (can be changed in `agent_config.json`)

## 🎯 Skill Configuration

The agent uses a skill-based architecture defined in `legal_consultant_agent/skills/legal-consultant/skill.yaml`. This configuration file defines:

*   **Skill Metadata** - ID, name, version, and capabilities
*   **Input/Output Modes** - Supported formats (text, JSON)
*   **Requirements** - Dependencies, system requirements, and performance metrics
*   **Documentation** - Detailed capabilities, use cases, and examples
*   **Assessment** - Keywords, specializations, and anti-patterns for skill negotiation

The skill configuration enables the agent to be discoverable and interoperable within the Bindu ecosystem.

## 💡 Usage Examples

### Via JSON-RPC API

The agent supports JSON-RPC 2.0 for structured interactions:

```bash
curl --location 'http://localhost:3773' \
--header 'Content-Type: application/json' \
--data '{
  "jsonrpc": "2.0",
  "method": "message/send",
  "params": {
    "message": {
      "role": "user",
      "parts": [
        {
          "kind": "text",
          "text": "Explain the legal implications of terminating an employee for poor performance in India."
        }
      ],
      "kind": "message",
      "messageId": "299bb30f-1e2b-47f5-a1a9-e83f0f5c7401",
      "contextId": "299bb30f-1e2b-47f5-a1a9-e83f0f5c7402",
      "taskId": "299bb30f-1e2b-47f5-a1a9-e83f0f5c7403"
    },
    "skillId": "legal-consultant-v1",
    "configuration": {
      "acceptedOutputModes": [
        "application/json"
      ]
    }
  },
  "id": "299bb30f-1e2b-47f5-a1a9-e83f0f5c7404"
}'
```

### Via HTTP API

```bash
curl -X POST http://localhost:3773/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": "Explain what constitutes a breach of contract in simple terms"
      }
    ]
  }'
```

### Sample Legal Education Queries

```text
"What are the basic elements required for a valid will?"
"How does the statute of limitations work for personal injury cases?"
"What's the difference between a misdemeanor and a felony?"
"What should I consider before signing a rental agreement?"
"How does intellectual property protection work for small businesses?"
"What are the general steps in a civil litigation process?"
"Explain the concept of 'due process' in the legal system"
```

### Expected Response Format

```markdown
**Legal Information Disclaimer**: I am an AI assistant providing general legal
information for educational purposes. I am not a lawyer, this is not legal advice,
and no attorney-client relationship is formed. Laws vary by jurisdiction and
change frequently. Always consult with a qualified attorney for legal advice.

## Understanding Contract Breach

A breach of contract occurs when one party fails to fulfill their obligations...

**Key Elements:**
1. **Valid Contract Exists** - All essential elements present
2. **Performance Failure** - Party doesn't meet obligations
3. **Material vs. Minor** - Significance of the breach matters
4. **Remedies Available** - Options for the non-breaching party

**Important Considerations:**
- Laws vary significantly by jurisdiction
- Specific circumstances affect outcomes
- Contract terms dictate available remedies

**When to Consult an Attorney:**
- Before signing important contracts
- If you believe a breach has occurred
- To understand your rights and obligations
- For contract enforcement or defense

**Remember:** This is general information. Always consult a qualified attorney
for legal advice specific to your situation.
```

## 🐳 Docker Deployment

### Quick Docker Setup

```bash
# Build the image
docker build -t legal-consultant-agent .

# Run container
docker run -d \
  -p 3773:3773 \
  -e OPENAI_API_KEY=your_openai_key \
  -e MEM0_API_KEY=your_mem0_key \
  --name legal-consultant-agent \
  legal-consultant-agent

# Check logs
docker logs -f legal-consultant-agent
```

### Docker Compose (Recommended)

`docker-compose.yml`:

```yaml
version: '3.8'
services:
  legal-consultant-agent:
    build: .
    ports:
      - "3773:3773"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - OPENROUTER_API_KEY=${OPENROUTER_API_KEY}
      - MEM0_API_KEY=${MEM0_API_KEY}
      - DATABASE_URL=${DATABASE_URL}
    restart: unless-stopped
```

Run with Compose:

```bash
# Start with compose
docker-compose up -d

# View logs
docker-compose logs -f
```

## 📁 Project Structure

```text
legal-consultant-agent/
├── legal_consultant_agent/
│   ├── skills/
│   │   └── legal-consultant/
│   │       ├── skill.yaml          # Skill configuration
│   │       └── __init__.py
│   ├── __init__.py
│   └── main.py                     # Agent entry point
├── agent_config.json               # Bindu agent configuration
├── pyproject.toml                  # Python dependencies
├── Dockerfile                      # Multi-stage Docker build
├── docker-compose.yml              # Docker Compose setup
├── README.md                       # This documentation
├── .env.example                    # Environment template
└── tests/                          # Test suite
```

## 🔌 API Reference

### Health Check

```bash
GET http://localhost:3773/health
```

Response:
```json
{"status": "healthy", "agent": "Legal Consultant Agent"}
```

### Chat Endpoint

```bash
POST http://localhost:3773/chat
Content-Type: application/json

{
  "messages": [
    {"role": "user", "content": "Your legal education query here"}
  ]
}
```

## 🧪 Testing

### Local Testing

```bash
# Install test dependencies
uv sync --group dev

# Run tests
pytest tests/

# Test with coverage
pytest --cov=legal_consultant_agent tests/
```

### Integration Test

```bash
# Start agent
python legal_consultant_agent/main.py &

# Test API endpoint
curl -X POST http://localhost:3773/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Explain contract basics"}]}'
```

## 🚨 Troubleshooting

### Common Issues & Solutions

**"No API key provided"**
Set either `OPENAI_API_KEY` or `OPENROUTER_API_KEY` in your `.env` file

**"Port 3773 already in use"**
Change port in `agent_config.json` or kill the process:
```bash
lsof -ti:3773 | xargs kill -9
```

**Docker build fails**
```bash
docker system prune -a
docker-compose build --no-cache
```

**Memory errors with Mem0**
Verify your `MEM0_API_KEY` is valid and has sufficient credits

## 📊 Dependencies

### Core Packages
*   **bindu** - Agent deployment framework
*   **agno** - AI agent framework
*   **openai** - OpenAI client
*   **python-dotenv** - Environment management
*   **mem0ai** - Memory operations
*   **sqlalchemy** - Database ORM (optional)

### Development Packages
*   **pytest** - Testing framework
*   **ruff** - Code formatting/linting
*   **pre-commit** - Git hooks

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1.  Fork the repository
2.  Create a feature branch: `git checkout -b feature/improvement`
3.  Make your changes following the code style
4.  Add tests for new functionality
5.  Commit with descriptive messages
6.  Push to your fork
7.  Open a Pull Request

**Code Style:**
*   Follow PEP 8 conventions
*   Use type hints where possible
*   Add docstrings for public functions
*   Keep functions focused and small

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Credits & Acknowledgments

*   **Developer:** Paras Chamoli
*   **Framework:** Bindu - Agent deployment platform
*   **Agent Framework:** Agno - AI agent toolkit
*   **Memory System:** Mem0 - Conversation memory API

## 🔗 Useful Links
*   🌐 **Bindu Directory:** [bindus.directory](https://bindus.directory)
*   📚 **Bindu Docs:** [docs.getbindu.com](https://docs.getbindu.com)
*   🐙 **GitHub:** [github.com/ParasChamoli/legal-consultant-agent](https://github.com/ParasChamoli/legal-consultant-agent)
*   💬 **Discord:** Bindu Community

<br/>

<p align="center">
  <strong>Built with ❤️ by Paras Chamoli</strong><br/>
  <em>Providing legal education while maintaining professional boundaries</em>
</p>

<p align="center">
  <a href="https://github.com/ParasChamoli/legal-consultant-agent/stargazers">⭐ Star on GitHub</a> •
  <a href="https://bindus.directory">🌐 Register on Bindu</a> •
  <a href="https://github.com/ParasChamoli/legal-consultant-agent/issues">🐛 Report Issues</a>
</p>

## 📝 Changelog

### [1.0.0] - 2026-01-11
- Initial release with legal education focus and disclaimer management
- Skill-based architecture with legal-consultant skill
- Support for multiple LLM providers (OpenAI, OpenRouter)
- Optional memory integration with Mem0
- Optional knowledge base with PostgreSQL/pgvector
- JSON-RPC and HTTP API interfaces
- Docker deployment support
- Comprehensive documentation and examples

---

*Note: This agent provides general legal information for educational purposes only. It is NOT a substitute for professional legal advice. Always consult qualified attorneys for legal matters. Powered by AI with ethical boundaries in place.*
