# WikiInsight Engine

An AI-driven system that uses **embeddings + clustering** to group Wikipedia articles into intuitive **topic clusters**, with a dashboard to explore clusters, top pages, and cluster keywords—no labels, just unsupervised structure.

## 🎯 Problem

Wikipedia is huge and fragmented. It’s hard for editors and readers to see the **big-picture structure** of topics, discover related articles, or identify clusters that are thin, noisy, or underdeveloped. This project builds an unsupervised topic map of Wikipedia so you can explore how articles group together in practice.

## 🚀 MVP Features (3-4 Weeks)

- **Topic Clustering**: Embeddings + clustering (e.g., KMeans/HDBSCAN) to group articles into coherent topic clusters
- **Network Analysis**: Article similarity networks, community detection, centrality metrics (not full GNN)
- **Cluster Explainability**: Simple keyword extraction and SHAP-style feature importance to describe clusters and top articles
- **MLOps Pipeline**: DVC versioning, MLflow tracking, Prefect orchestration, monitoring
- **Interactive Frontend**: React + Vite + Tailwind UI with cluster explorer and summaries
- **REST API**: FastAPI-based service for programmatic access to clusters and nearest-neighbor queries

## 📋 Requirements

- Python 3.9+
- See `requirements.txt` for full dependencies
- **Locally runnable** - no cloud deployment needed

## 🛠️ Installation

```bash
# Clone the repository
git clone <repository-url>
cd WikiInsight-Engine

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Install in development mode
pip install -e .
```

## 🏗️ Project Structure

```
wikiinsight-engine/
├── data/              # Data storage (DVC tracked)
│   ├── raw/           # Raw Wikipedia data
│   ├── processed/     # Processed articles
│   └── features/      # Engineered features
├── models/            # Trained models (MLflow registry)
│   ├── xgboost/
│   ├── lightgbm/
│   ├── logistic/
│   └── ensemble/
├── src/               # Source code
│   ├── ingestion/     # Wikipedia/Wikidata API clients
│   ├── preprocessing/ # Text cleaning, embeddings, graph builders
│   ├── modeling/      # Clustering pipeline + topic index
│   ├── explainability/# SHAP explainers (optional)
│   └── api/           # FastAPI application
├── frontend/          # React + Vite + Tailwind frontend
├── pipelines/         # Prefect workflows
│   ├── flow_ingestion.py
│   ├── flow_training.py
│   └── flow_inference.py
├── tests/             # Test suite
├── monitoring/        # Monitoring dashboard
└── notebooks/         # Jupyter notebooks for exploration
```

## 🚦 Quick Start

### Setup MLOps Tools

```bash
# Initialize DVC
dvc init

# Start MLflow tracking server
mlflow ui --port 5000

# Start Prefect server (optional)
prefect server start
```

### Data Ingestion

```bash
python -m src.ingestion.fetch_wikipedia_data
```

### Run API (backend)

```bash
uvicorn src.api.main:app --reload
```

or use the convenience script on Windows:

```bash
run_app.cmd
```

This will start the FastAPI backend and the React/Vite frontend together (Vite dev server on port 5173 by default).

### Run Frontend (manually)

```bash
cd frontend
npm install    # or pnpm install / yarn
npm run dev   # starts Vite dev server on http://localhost:5173
```

## 📊 MVP Timeline (3-4 Weeks)

**Week 1**: Data + Features (DVC, MLflow, Prefect setup → ingestion → embeddings + basic features)

**Week 2**: Clustering + Evaluation (train clustering model(s) → cluster labeling/keywords → basic quality checks)

**Week 3**: Visualization + Orchestration (network graphs → cluster explorer dashboard → Prefect flows)

**Week 4**: MLOps + Polish (monitoring → CI/CD → testing → documentation)

## 🎯 Success Metrics

- High qualitative coherence of topic clusters (manual inspection)
- Ability to retrieve top-N similar articles for a given page in <1s
- Network graph renders 20K+ nodes in <5 seconds
- MLflow tracks 50+ experiments
- DVC versions 5+ datasets
- Prefect runs 3 flows end-to-end
- CI/CD passes 90%+ test coverage

## 🔧 Tech Stack

**Core ML**: `scikit-learn`, `XGBoost`, `LightGBM`, `sentence-transformers`, `spaCy`, `shap`

**Graph Analysis**: `NetworkX`, `igraph`, `pyvis`

**MLOps**: `MLflow`, `DVC`, `Prefect`, `pytest`, `Great Expectations`

**Visualization**: React + Vite + Tailwind (frontend), `Plotly`, `matplotlib` (for future plots)

**Data**: `pandas`, `pyarrow`, `SQLite3`, `mwclient`

## 🤝 Contributing

Contributions welcome! Please read our contributing guidelines.

## 📄 License

MIT License - see LICENSE file for details

## 🔗 Resources

- [Wikipedia API Documentation](https://www.mediawiki.org/wiki/API:Main_page)
- [Wikidata API](https://www.wikidata.org/wiki/Wikidata:Main_Page)
- [DVC Documentation](https://dvc.org/)
- [MLflow Documentation](https://mlflow.org/)
- [Prefect Documentation](https://www.prefect.io/)
- **Note**: Planning documents (project description, implementation analysis, phase breakdowns) are in the `phases/` folder (gitignored)
