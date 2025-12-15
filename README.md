# WikiInsight Engine

AI-powered Wikipedia topic clustering and knowledge graph explorer with hybrid search and an interactive React UI.

## What it does
- Ingests Wikipedia via MediaWiki API
- Cleans text and builds sentence-transformer embeddings (CPU-only)
- Clusters with AgglomerativeClustering
- Extracts c-TF-IDF keywords per cluster
- Builds a multi-layer knowledge graph (clusters + semantic similarity)
- Provides hybrid search (BM25 + embeddings) and visualization

## Why
- Discover hidden connections across articles
- Explore topics without labels
- Visualize relationships through a graph
- Production-ready pipeline with DVC + MLflow

## Prerequisites
- Python 3.10+
- Node.js 18+
- Git

## Setup (quick)
```bash
git clone https://github.com/ajay-drew/WikiInsight-Engine.git
cd WikiInsight-Engine
python -m venv venv
venv\Scripts\activate  # Windows
# or: source venv/bin/activate  # Linux/Mac
pip install -e .
cd frontend && npm install && cd ..
# Optional: download NLP resources
python scripts/setup_nlp_resources.py
```

## Run
```bash
# Full pipeline then app
run_tools.cmd     # DVC pipeline + MLflow
run_app.cmd       # FastAPI + React

# Manual pipeline
dvc repro         # fetch → preprocess → cluster → graph
run_app.cmd
```

## Access
- Frontend: http://localhost:5173
- API: http://localhost:8000 (docs at /docs)
- MLflow: http://localhost:5000

## Tech stack
- ML: scikit-learn, sentence-transformers, spaCy, NLTK
- Search: BM25 + embeddings (RRF)
- Backend: FastAPI, pandas, NetworkX
- Frontend: React, Vite, Tailwind, React Flow
- MLOps: DVC, MLflow

## License
MIT License
