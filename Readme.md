# Vector Search & RAG Learning Repository

A small learning repository containing three mini-projects for learning vector search, embeddings, and Retrieval-Augmented Generation (RAG) patterns using MongoDB Atlas Vector Search and Hugging Face embeddings.

- Status: 1 of 3 projects completed (Project1: Movie recommendations with vector search).
- Purpose: Hands‑on experiments to understand embedding generation, vector indexes, semantic search, and integrating retrieval with LLMs.

---


## Projects (summary)

1. Project1 — Movie Recommendations (completed)
   - Uses Hugging Face sentence-transformers to create 384-dim embeddings for movie plots.
   - Stores embeddings in `sample_mflix.movies.plot_embedding_hf`.
   - Uses MongoDB Atlas Vector Search index `PlotSemanticSearch` to run semantic queries.
   - Entry: `Project1/movie_recs.py`

2. Project2 — (planned)
   - RAG demo combining vector retrieval with a local LLM / API for answer generation.
   - Goal: retrieve relevant docs and synthesize answers with contextual prompts.

3. Project3 — (planned)
   - Advanced ranking and multimodal retrieval (text + metadata + filters).
   - Goal: add hybrid search (vector + text) and evaluation metrics.

---

## Prerequisites

- Python 3.13+
- `uv` (astral.sh/uv) — recommended to run script with inline dependency metadata
- MongoDB Atlas account (free tier OK)
- Hugging Face account + API token
- Git (for version control)

---
