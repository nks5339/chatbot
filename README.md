# Sarthi AI - Rajasthan Procurement Assistant

An AI-powered chatbot for the Rajasthan Transparency in Public Procurement Act, 2012, built with open-source LLMs and advanced RAG techniques.

## Features

- 🤖 Open-source LLM (DeepSeek-R1 or GPT-OSS via Ollama)
- 📚 Graph-enhanced RAG for precise document understanding
- 💾 Persistent memory across sessions
- 🔍 Semantic search with Qdrant vector database
- 📊 Document relationship mapping
- 💬 Conversation history tracking
- 🎨 Modern web interface
- ⚡ Fast initialization with caching

## Architecture
```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Documents  │────▶│   Chunking   │────▶│   Qdrant    │
│   (PDFs)    │     │  (Semantic)  │     │ Vector Store│
└─────────────┘     └──────────────┘     └─────────────┘
                            │                     │
                            ▼                     ▼
                    ┌──────────────┐     ┌─────────────┐
                    │  Graph RAG   │────▶│   Query     │
                    │  (NetworkX)  │     │  Pipeline   │
                    └──────────────┘     └─────────────┘
                                                │
                                                ▼
                                        ┌──────────────┐
                                        │     LLM      │
                                        │  (Ollama)    │
                                        └──────────────┘
```

## Installation

### Prerequisites

1. **Ollama** (for LLM and embeddings)
```bash
# Install Ollama from https://ollama.ai
curl -fsSL https://ollama.com/install.sh | sh

# Pull required models
ollama pull deepseek-r1:8b
ollama pull nomic-embed-text
```

2. **Python 3.10+**

### Setup

1. Clone the repository:
```bash
git clone <repo-url>
cd sarthi-ai-chatbot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Place your PDF documents in the `documents/` directory

4. Create `.env` file (optional):
```bash
# Copy example env
cp .env.example .env

# Edit as needed
nano .env
```

## Usage

### Quick Start
```bash
# Start the API server
python api.py
```

The application will:
1. Initialize the system
2. Process any new documents automatically
3. Start the web server at http://localhost:8000

### Manual Initialization
```bash
# Just initialize without starting the server
python -c "from main import pipeline; pipeline.initialize_system()"
```

### API Endpoints

- `GET /` - Web interface
- `GET /health` - Health check
- `GET /api/status` - System status
- `GET /api/documents` - List documents
- `POST /api/query` - Query endpoint
- `POST /api/query/stream` - Streaming query
- `GET /api/conversations` - Conversation history
- `DELETE /api/conversations` - Clear history
- `POST /api/initialize` - Reinitialize system

### Example API Usage
```python
import requests

# Query
response = requests.post('http://localhost:8000/api/query', json={
    'query': 'What is the threshold for e-procurement?',
    'use_graph_expansion': True
})

print(response.json()['response'])
```

## Configuration

Key settings in `config.py`:
```python
# Models
OLLAMA_MODEL = "deepseek-r1:8b"
EMBEDDING_MODEL = "nomic-embed-text"

# Chunking
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Retrieval
RETRIEVAL_TOP_K = 8
RERANK_TOP_K = 5
SIMILARITY_THRESHOLD = 0.7

# LLM
MAX_TOKENS = 2048
TEMPERATURE = 0.1
```

## Project Structure
```
sarthi-ai-chatbot/
├── main.py                 # Core pipeline
├── api.py                  # FastAPI application
├── config.py               # Configuration
├── requirements.txt        # Dependencies
├── models/
│   ├── embeddings.py       # Embedding model
│   ├── llm.py              # LLM interface
│   └── graph_rag.py        # Graph-enhanced RAG
├── storage/
│   ├── vector_store.py     # Qdrant wrapper
│   └── conversation_memory.py  # Memory management
├── processing/
│   ├── document_loader.py  # PDF loading
│   ├── chunking.py         # Semantic chunking
│   └── preprocessor.py     # Text preprocessing
├── utils/
│   ├── logger.py           # Logging setup
│   └── helpers.py          # Helper functions
├── static/
│   └── index.html          # Web interface
├── documents/              # Source PDFs
└── data/                   # Processed data
    ├── qdrant_db/          # Vector database
    ├── graph_db/           # Graph data
    └── memory_store/       # Conversation memory
```

## Key Features Explained

### 1. Persistent Storage

Documents are processed once and cached. On subsequent startups:
- System loads existing vector database
- No re-chunking unless documents change
- Metadata tracks processing status

### 2. Graph-Enhanced RAG

- Extracts entities (sections, chapters, rules)
- Maps relationships between document parts
- Finds cross-references automatically
- Expands context with related chunks

### 3. Conversation Memory

- Stores last 100 conversations
- Uses sliding window (10 exchanges) for context
- Persists to disk automatically
- Searchable history

### 4. Semantic Chunking

- Respects document structure (chapters, sections)
- Maintains context with overlap
- Preserves metadata (page numbers, filenames)
- Section-aware splitting for legal documents

## Troubleshooting

### Ollama Connection Issues
```bash
# Check if Ollama is running
ollama list

# Restart Ollama
# On Linux/Mac:
systemctl restart ollama

# On Windows, restart the Ollama app
```

### Memory Issues
```bash
# For large document sets, increase system memory
# Or reduce BATCH_SIZE in config.py
```

### Port Already in Use
```python
# Change port in api.py or config.py
uvicorn.run("api:app", host="0.0.0.0", port=8001)
```

## Performance Tips

1. **First Run**: Takes 2-5 minutes to process documents
2. **Subsequent Runs**: Starts in <10 seconds (cached)
3. **Query Speed**: 2-5 seconds depending on LLM
4. **Memory Usage**: ~2GB for typical document set

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

MIT License

## Acknowledgments

- Ollama for LLM hosting
- Qdrant for vector search
- FastAPI for the web framework
- Anthropic Claude for development assistance

## Support

For issues or questions:
- Open a GitHub issue
- Check documentation at `/docs` endpoint
- Review logs in `logs/` directory