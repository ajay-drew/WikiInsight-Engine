# WikiInsight Engine

AI-powered topic clustering and knowledge graph explorer for Wikipedia articles. Automatically groups articles into topic clusters using unsupervised learning and provides interactive visualization of relationships between articles.

## What is This?

WikiInsight Engine is an end-to-end ML system that:
- **Ingests** Wikipedia articles via the MediaWiki API
- **Clusters** articles into topics using AgglomerativeClustering and semantic embeddings
- **Extracts** distinctive keywords per cluster using c-TF-IDF
- **Builds** a multi-layer knowledge graph (cluster relationships + semantic similarity)
- **Searches** articles using hybrid semantic + keyword search (BM25 + embeddings)
- **Visualizes** clusters and relationships in an interactive React dashboard

## Why This Project?

- **Discover Hidden Connections**: Find related articles across different Wikipedia categories
- **Topic Exploration**: Understand how articles group into natural topics without manual labeling
- **Knowledge Graph**: Visualize relationships between articles through cluster and semantic connections
- **Production-Ready ML**: Full MLOps pipeline with DVC and MLflow experiment tracking
- **Modern Stack**: FastAPI backend + React frontend with real-time graph visualization

## How to Run

### Prerequisites
- Python 3.10+ with virtual environment
- Node.js 18+ and npm
- Git

### Setup

```bash
# 1. Clone and setup Python environment
git clone <repo-url>
cd WikiInsight-Engine
python -m venv venv
venv\Scripts\activate  # Windows
# OR: source venv/bin/activate  # Linux/Mac

# 2. Install Python dependencies
pip install -e .

# 3. Install frontend dependencies
cd frontend
npm install
cd ..

# 4. Download NLP resources (auto-downloads on first use, but recommended)
python scripts/setup_nlp_resources.py
```

### Running the System

```bash
# Option 1: Run data pipeline first, then start app
run_tools.cmd    # Runs DVC pipeline + starts MLflow UI
run_app.cmd      # Starts FastAPI backend + React frontend

# Option 2: Manual pipeline execution
dvc repro        # Run full data pipeline (fetch → preprocess → cluster → graph)
run_app.cmd      # Start services
```

### Access Points
- **Frontend**: http://localhost:5173 (React dashboard)
- **API**: http://localhost:8000 (FastAPI, docs at `/docs`)
- **MLflow**: http://localhost:5000 (experiment tracking)

### First Run
1. Run `run_tools.cmd` to fetch Wikipedia data and build clusters (takes 5-15 minutes)
2. Run `run_app.cmd` to start the web interface
3. Explore clusters in the "Clusters Overview" page or search articles

## Tech Stack

**ML**: scikit-learn (AgglomerativeClustering), sentence-transformers, spaCy, NLTK  
**Search**: BM25 + semantic embeddings with Reciprocal Rank Fusion  
**Backend**: FastAPI, pandas, NetworkX (knowledge graph)  
**Frontend**: React, Vite, Tailwind CSS, React Flow (graph visualization)  
**MLOps**: DVC (data versioning), MLflow (experiment tracking)

## License

MIT License
