# Pipeline Status Guide

## Understanding the 503 Error

**Error**: `Lookup failed (503): {"detail":"Topic index is not available. Run the clustering pipeline first."}`

**This is expected!** The API returns 503 until the data pipeline completes. The pipeline needs to:
1. Fetch Wikipedia articles (ingestion)
2. Process and generate embeddings (preprocessing)
3. Run clustering and build topic index (clustering)

## Running the Pipeline

### Option 1: Full Pipeline (Recommended)
```bash
dvc repro
```
This runs all stages: `fetch_data` → `preprocess` → `cluster_topics`

### Option 2: Individual Scripts
```bash
# Step 1: Fetch articles
python -m src.ingestion.fetch_wikipedia_data

# Step 2: Process and generate embeddings
python -m src.preprocessing.process_data

# Step 3: Cluster and build index
python -m src.modeling.cluster_topics
```

### Option 3: Convenience Script (Windows)
```bash
run_tools.cmd
```

## Understanding DVC Lock Warnings

**Warning**: `WARNING: Process '...' with (Pid ...) had been killed. Auto removed it from the lock file.`

**This is harmless!** DVC is cleaning up stale lock files from a previous interrupted run. The pipeline will continue normally.

## Pipeline Progress

The pipeline shows progress:
- Query processing: `Processing query 1/6: 'Machine learning'`
- Article fetching: Progress bars for each query
- Article count: `Fetched 10 articles so far...`

**Default settings**: 
- Max articles: 100 (reduced from 5000 for faster testing)
- Per query limit: 50 (reduced from 200)

## Expected Timeline

- **Fetching**: ~5-15 minutes (depends on network speed)
- **Preprocessing**: ~2-5 minutes (depends on CPU)
- **Clustering**: ~1-3 minutes (depends on CPU)

**Total**: ~10-25 minutes for first run

## After Pipeline Completes

Once the pipeline finishes:
1. Restart the API (if it's running)
2. The API will automatically load the topic index
3. Lookup endpoints will work: `/api/topics/lookup`

## Troubleshooting

- **Pipeline hangs**: Check network connection, Wikipedia API might be slow
- **Out of memory**: Reduce `max_articles` in `fetch_wikipedia_data.py`
- **API still shows 503**: Restart the API after pipeline completes

