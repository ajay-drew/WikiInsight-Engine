# WikiInsight Engine

AI-driven system for unsupervised hierarchical topic clustering of Wikipedia articles using **AgglomerativeClustering** and **c-TF-IDF** keyword extraction. Features hybrid search combining semantic (vector) and keyword (BM25) search with Reciprocal Rank Fusion.

## ✅ Features

- **Wikipedia Ingestion**: Fetch articles via `mwclient` with async support
- **Text Preprocessing**: Clean text, generate embeddings with `sentence-transformers`
- **Topic Clustering**: AgglomerativeClustering with c-TF-IDF for distinctive topic words
- **Hybrid Search**: Semantic + keyword search with RRF fusion
- **REST API**: FastAPI with search, topic lookup, and cluster endpoints
- **React Frontend**: Modern UI with Vite + Tailwind CSS
- **MLOps**: DVC, MLflow, Prefect for pipeline orchestration
- **Tests**: Comprehensive pytest + Vitest test suite

## 🛠️ Quick Start

```bash
# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm
# (Optional but recommended for best BM25 + preprocessing quality)
python -m nltk.downloader punkt stopwords wordnet

# Run data pipeline
dvc repro  # OR: run_tools.cmd

# Start services
run_app.cmd  # Development (with reload)
# OR: run_app_production.cmd  # Production (no reload, Windows-friendly)
```

## 📁 Structure

```
src/
├── ingestion/      # Wikipedia API client
├── preprocessing/  # Text cleaning, embeddings
├── modeling/       # Clustering + topic index
├── serving/        # Hybrid search engine
└── api/            # FastAPI backend
frontend/           # React + Vite + Tailwind
tests/              # Test suite
```

## 🔧 Tech Stack

**ML**: `scikit-learn` (AgglomerativeClustering), `sentence-transformers`, `spaCy`, `nltk`  
**Search**: `rank-bm25`, `scikit-learn` (NearestNeighbors), RRF with multi-field BM25 (title + body)  
**Backend**: `FastAPI`, `pandas`, `numpy`  
**Frontend**: `React`, `Vite`, `Tailwind CSS`  
**MLOps**: `DVC`, `MLflow`, `Prefect`

### Ingestion & pipeline UX

- **Async ingestion** (`src/ingestion/fetch_wikipedia_data.py`):
  - Concurrent fetching via `AsyncWikipediaClient` with retry + exponential backoff.
  - CLI flags: `--max-articles`, `--per-query-limit`, `--batch-size`, `--max-workers`, `--sample`, `--resume`.
  - Filters out stub articles by length (< 200 words) and limits per-article links to 50.
  - Uses `tqdm` progress bars and periodic checkpointing to `data/raw/articles.json` so long runs can resume.
- **Preprocessing** (`src/preprocessing/process_data.py`):
  - Cleans text and then applies NLTK-based normalization (lowercasing, stopwords, lemmatization/stemming) before embedding.
  - Shows progress for article cleaning and logs expected durations for embedding generation.
- **Clustering** (`src/modeling/cluster_topics.py`):
  - Logs progress while summarizing clusters and surfaces clear error messages if inputs are missing or misconfigured.

## 📚 URLs

- API: `http://localhost:8000` (docs: `/docs`)
- Frontend: `http://localhost:5173`
- MLflow: `http://localhost:5000` (after `mlflow ui`)

## 📄 License

MIT License
