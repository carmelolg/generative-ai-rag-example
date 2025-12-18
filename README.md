# generative-ai-rag-example

![logo](static/logo.svg)

# Summary
A small example project that demonstrates a Retrieval-Augmented Generation (RAG) pattern using local embeddings and a language model.

This repository contains helper utilities to build embeddings, chunk and query a small knowledge base, and stream responses from a local or configured language model.

# Features
- Build and store embeddings for text chunks
- Find most relevant chunks for a query using cosine similarity
- Generate embeddings from a configured embedding model
- Stream chat responses from a configured language model

# Requirements
- Ollama CLI installed and configured: https://ollama.com/docs/cli
- Python 3.10+
- See `requirements.txt` for Python dependencies

# Environment variables
- `EMBEDDING_MODEL` — Embedding model identifier (read from environment). Example default used in code: `nomic-embed-text:latest`.
- `LANGUAGE_MODEL` — Language model identifier (read from environment). Example default used in code: `gemma3:270m`.

# Getting started

1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

# Usage

- Configure environment variables (recommended in a `.env` file or your shell):

```bash
export EMBEDDING_MODEL="nomic-embed-text:latest"
export LANGUAGE_MODEL="gemma3:270m"
```

- Run the main script:

```bash
python main.py
```

# API / utilities

- `lib/OllamaUtils.py` — small wrapper to call the local `ollama` client for embeddings and chat. The module reads `EMBEDDING_MODEL` and `LANGUAGE_MODEL` from environment variables.
- `lib/MathUtils.py` — cosine similarity helper used to rank chunks by relevance.
- `lib/PromptUtils.py` — prompt templates and helpers.
- `lib/Service.py` — higher-level orchestration and service functions.

### Embedding a string

The project exposes a helper to produce an embedding from a text string. Example usage (pseudo):

```python
from lib.OllamaUtils import embed_text

embedding = embed_text("Hello world")
print(len(embedding))
```

# License

![CC BY-NC-ND 4.0](https://licensebuttons.net/l/by-nc-nd/4.0/88x31.png)

This project is licensed under the **Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)**. 

See `LICENSE.md` for the full license text.

# Contact

For questions or help, open an issue in this repository.

