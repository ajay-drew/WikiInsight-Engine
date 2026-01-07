# WikiInsight Engine

AI-powered Wikipedia topic clustering and knowledge graph explorer with hybrid search and an interactive React UI.

## What is this?

WikiInsight Engine is an unsupervised machine learning system that automatically discovers and clusters Wikipedia articles into meaningful topics. It uses semantic embeddings and hierarchical clustering to group related articles, extracts distinctive keywords for each cluster, and builds an interactive knowledge graph showing relationships between articles and topics. The system provides hybrid search combining semantic (vector) and keyword (BM25) search with an intuitive web interface for exploration.

## Why is there a need for this?

In 2024-2025, as AI and semantic search technologies mature, there's a growing need for intelligent information discovery systems that go beyond traditional keyword matching. Wikipedia contains over 6.8 million articles with complex interconnections that manual categorization cannot fully capture. This system addresses critical gaps: traditional search relies on exact keyword matches, missing semantic relationships; manual topic organization is labor-intensive and incomplete; and users struggle to discover cross-domain connections. By leveraging modern vector embeddings and unsupervised clustering, this tool enables researchers, students, and knowledge workers to explore Wikipedia's semantic landscape through AI-powered discovery, revealing hidden thematic relationships and unexpected connections that enhance learning, research, and information exploration in the age of large-scale knowledge bases.

## Prerequisites

- Python 3.10+, Node.js 18+, Git
- Docker & Docker Compose (for containerized deployment)

## Local Setup

```bash
git clone https://github.com/ajay-drew/WikiInsight-Engine.git
cd WikiInsight-Engine
python -m venv venv
venv\Scripts\activate  # Windows | source venv/bin/activate  # Linux/Mac
pip install -e .
cd frontend && npm install && cd ..
python ops/scripts/setup_nlp_resources.py
# Copy env.example to .env and add your own database connection
cp env.example .env
# Edit .env and set DATABASE_URL to your PostgreSQL database
```

## Local Usage

```bash
ops/run_dvc_tools.cmd     # DVC pipeline + MLflow
run_app.cmd               # FastAPI + React
ops/start_mlflow_ui.cmd   # MLflow UI (separate terminal)
# Or: dvc repro && ops/run_app.cmd
```

Access: Frontend http://localhost:5173 | API http://localhost:9000 | MLflow http://localhost:5000

## Docker Deployment (Web)

```bash
cd infrastructure && docker-compose up -d
```

Access: Frontend & API http://localhost:9000 | PostgreSQL localhost:5432 | MLflow http://localhost:5000

See `infrastructure/README.md` for detailed setup and troubleshooting.

## Tech Stack

ML: scikit-learn, sentence-transformers, spaCy, NLTK | Search: BM25 + embeddings (RRF) | Backend: FastAPI, PostgreSQL + pgvector, pandas, NetworkX | Frontend: React, Vite, Tailwind, React Flow, React Router | MLOps: DVC, MLflow

## Created By

**Ajay A** | Email: drewjay05@gmail.com | LinkedIn: [linkedin.com/in/ajay-drew](https://linkedin.com/in/ajay-drew)

## License

MIT License
