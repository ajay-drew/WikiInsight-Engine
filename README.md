# WikiInsight Engine

An AI-driven system that uses **embeddings + clustering** to group Wikipedia articles into intuitive **topic clusters**, with a dashboard to explore clusters, top pages, and cluster keywords.

## 🎯 What It Does

Wikipedia is huge and fragmented. This project builds an unsupervised hierarchical topic map of Wikipedia using **AgglomerativeClustering** and **c-TF-IDF** keyword extraction. Explore how articles group together semantically, discover related articles, and identify distinctive topic clusters with meaningful keywords.

## ✅ Implemented Features

- **Wikipedia Data Ingestion**: Fetch articles using `mwclient` with search and article retrieval
- **Text Preprocessing**: Clean text, generate embeddings using `sentence-transformers`
- **Topic Clustering**: AgglomerativeClustering (hierarchical) with cluster summaries using c-TF-IDF for distinctive topic words
- **Topic Index**: Fast lookup of article clusters, similar articles, and cluster summaries
- **Hybrid Search Engine**: Combines semantic (vector) and keyword (BM25) search using Reciprocal Rank Fusion
- **REST API**: FastAPI backend with endpoints for search (`/api/search`), topic lookup (`/api/topics/lookup`), and cluster overview (`/api/clusters/overview`)
- **React Frontend**: Modern UI with Vite + Tailwind CSS featuring search, topic exploration, and cluster browsing
- **MLOps Pipeline**: DVC for data versioning, MLflow for experiment tracking, Prefect for orchestration
- **Test Suite**: Comprehensive pytest tests (unit, integration, API) + Vitest for frontend

## 🚧 Coming Soon

- **Interactive 2D Visualization**: UMAP-based embedding projection for visual exploration of topic clusters
- **Network Analysis**: Article similarity graphs and community detection visualization
- **SHAP Explanations**: Feature importance for cluster assignments
- **Enhanced Dashboard**: Interactive cluster exploration with network graphs
- **CI/CD**: Automated testing and deployment workflows

See [FUTURE_FEATURES.md](FUTURE_FEATURES.md) for detailed feature plans and implementation notes.

## 🛠️ Quick Start

```bash
# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Run data pipeline (ingestion → preprocessing → clustering)
dvc repro
# OR use convenience script:
run_tools.cmd

# Start API + Frontend
run_app.cmd
# OR for production (no auto-reload, avoids Windows socket issues):
run_app_production.cmd
# OR manually:
uvicorn src.api.main:app --reload  # Backend on :8000
cd frontend && npm install && npm run dev  # Frontend on :5173
```

## 📁 Project Structure

```
├── src/
│   ├── ingestion/      # Wikipedia API client
│   ├── preprocessing/   # Text cleaning, embeddings
│   ├── modeling/       # Clustering + topic index
│   ├── serving/        # Hybrid search engine
│   └── api/            # FastAPI backend
├── frontend/           # React + Vite + Tailwind
├── pipelines/prefect/  # Data pipeline orchestration
├── tests/              # Test suite
└── data/              # DVC-tracked data (gitignored)
```

## 🔧 Tech Stack

**ML**: `scikit-learn` (AgglomerativeClustering), `sentence-transformers`, `spaCy`  
**Keyword Extraction**: c-TF-IDF (class-based TF-IDF) with stopword filtering  
**Search**: `rank-bm25` (BM25), `scikit-learn` (NearestNeighbors), Reciprocal Rank Fusion  
**MLOps**: `DVC`, `MLflow`, `Prefect`  
**Backend**: `FastAPI`, `pandas`, `numpy`  
**Frontend**: `React`, `Vite`, `Tailwind CSS`  
**Testing**: `pytest`, `Vitest`

## 📚 Documentation

- API runs on `http://localhost:8000` (docs at `/docs`)
- Frontend runs on `http://localhost:5173`
- MLflow UI: `http://localhost:5000` (after running `mlflow ui`)

## 📄 License

MIT License - see LICENSE file for details
