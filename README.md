# WikiInsight Engine

An AI-driven system that uses **embeddings + clustering** to group Wikipedia articles into intuitive **topic clusters**, with a dashboard to explore clusters, top pages, and cluster keywords.

## 🎯 What It Does

Wikipedia is huge and fragmented. This project builds an unsupervised topic map of Wikipedia so you can explore how articles group together in practice, discover related articles, and identify topic clusters.

## ✅ Implemented Features

- **Wikipedia Data Ingestion**: Fetch articles using `mwclient` with search and article retrieval
- **Text Preprocessing**: Clean text, generate embeddings using `sentence-transformers`
- **Topic Clustering**: KMeans/MiniBatchKMeans clustering with cluster summaries (keywords, top articles)
- **Topic Index**: Fast lookup of article clusters, similar articles, and cluster summaries
- **REST API**: FastAPI backend with endpoints for topic lookup (`/api/topics/lookup`) and cluster overview (`/api/clusters/overview`)
- **React Frontend**: Modern UI with Vite + Tailwind CSS for exploring clusters and articles
- **MLOps Pipeline**: DVC for data versioning, MLflow for experiment tracking, Prefect for orchestration
- **Test Suite**: Comprehensive pytest tests (unit, integration, API) + Vitest for frontend

## 🚧 Coming Soon

- **Network Analysis**: Article similarity graphs and community detection visualization
- **SHAP Explanations**: Feature importance for cluster assignments
- **Enhanced Dashboard**: Interactive cluster exploration with network graphs
- **CI/CD**: Automated testing and deployment workflows

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
│   └── api/            # FastAPI backend
├── frontend/           # React + Vite + Tailwind
├── pipelines/prefect/  # Data pipeline orchestration
├── tests/              # Test suite
└── data/              # DVC-tracked data (gitignored)
```

## 🔧 Tech Stack

**ML**: `scikit-learn`, `sentence-transformers`, `spaCy`  
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
