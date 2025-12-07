# Future Features & Enhancements

This document tracks planned features and improvements for WikiInsight Engine.

---

## 🎨 Interactive 2D Embedding Visualization (UMAP)

**Priority**: High  
**Status**: Planned  
**Estimated Effort**: 2-3 days

### The Problem
Lists of topics are boring. It's hard to visualize how "AI" relates to "Neuroscience" just by looking at text. Users need a visual way to explore the semantic relationships between articles and clusters.

### The Solution
Project high-dimensional embeddings (384-dim) down to 2 dimensions (x, y) using **UMAP** (Uniform Manifold Approximation and Projection) and create an interactive scatter plot visualization.

**Why UMAP over PCA?**
- UMAP preserves **local structure** better (keeping similar things close)
- Better at maintaining cluster boundaries
- More intuitive visualization for topic exploration

### Implementation Plan

#### Backend (Python)
1. **Add UMAP dependency**
   - Add `umap-learn>=0.5.4` to `requirements.txt` and `pyproject.toml`

2. **Create UMAP projection module**
   - File: `src/modeling/umap_projection.py`
   - Function: `project_embeddings_to_2d(embeddings: np.ndarray, n_neighbors: int = 15, min_dist: float = 0.1) -> np.ndarray`
   - Load embeddings from `data/features/embeddings.parquet`
   - Fit UMAP model and transform embeddings to 2D
   - Save coordinates to `data/features/umap_coordinates.parquet` (columns: `title`, `x`, `y`, `cluster_id`)

3. **Add DVC stage**
   - Update `dvc.yaml` to add `umap_projection` stage
   - Depends on: `data/features/embeddings.parquet`, `models/clustering/cluster_assignments.parquet`
   - Outputs: `data/features/umap_coordinates.parquet`

4. **Add API endpoint**
   - `GET /api/visualization/coordinates` - Returns all article coordinates with cluster assignments
   - `GET /api/visualization/cluster/{cluster_id}` - Returns coordinates for specific cluster
   - Response format:
     ```json
     {
       "articles": [
         {"title": "Machine learning", "x": 0.123, "y": -0.456, "cluster_id": 5},
         ...
       ],
       "clusters": [
         {"cluster_id": 5, "color": "#3b82f6", "keywords": [...]},
         ...
       ]
     }
     ```

#### Frontend (React)
1. **Install visualization library**
   - Option A: `recharts` (simpler, good for basic scatter plots)
   - Option B: `plotly.js` (more powerful, better interactivity)
   - Option C: `@visx/visx` (D3-based, highly customizable)
   - **Recommendation**: Start with `recharts` for MVP, upgrade to `plotly.js` if needed

2. **Create Visualization component**
   - File: `frontend/src/pages/EmbeddingVisualization.tsx`
   - Features:
     - Scatter plot with articles as points
     - Color-code points by cluster_id
     - Hover tooltip showing article title and cluster info
     - Zoom and pan functionality
     - Click to filter by cluster
     - Legend showing cluster colors and keywords

3. **Add to navigation**
   - Add "Visualization" tab to main navigation
   - Route: `/visualization`

### The "Wow" Factor
- Users can **zoom** into the "History" island
- See it dissolve into "WWII", "Ancient Rome", etc.
- **Proves you understand Dimensionality Reduction**
- Interactive exploration of semantic relationships
- Visual confirmation that clustering makes sense

### Technical Considerations
- **Performance**: For large datasets (10K+ articles), consider:
  - Server-side filtering/clustering
  - Progressive loading (load visible area first)
  - Canvas-based rendering for better performance
- **UMAP Parameters**:
  - `n_neighbors`: 15 (default, controls local vs global structure)
  - `min_dist`: 0.1 (controls how tightly points are packed)
  - `n_components`: 2 (for 2D visualization)
- **Caching**: UMAP projection is expensive, cache results and only recompute when embeddings change

### Testing
- Unit test: UMAP projection produces 2D coordinates
- Integration test: API returns correct coordinate format
- Frontend test: Visualization renders correctly with sample data
- E2E test: User can interact with visualization

### Dependencies to Add
```python
# Backend
umap-learn>=0.5.4

# Frontend (choose one)
recharts>=2.8.0  # OR
plotly.js>=2.26.0  # OR
@visx/visx>=3.0.0
```

---

## 🔗 Network Analysis Visualization

**Priority**: Medium  
**Status**: Planned  
**Estimated Effort**: 3-4 days

### Description
Visualize article relationships as a network graph showing:
- Nodes: Articles
- Edges: Similarity relationships (based on embeddings or citations)
- Communities: Cluster assignments

### Implementation
- Use `networkx` + `pyvis` or `vis.js` for graph visualization
- Add endpoint: `GET /api/network/graph`
- Create React component for interactive graph

---

## 📊 SHAP Explanations

**Priority**: Medium  
**Status**: Planned  
**Estimated Effort**: 2-3 days

### Description
Explain why articles are assigned to specific clusters using SHAP (SHapley Additive exPlanations).

### Implementation
- Use `shap` library (already in dependencies)
- Compute SHAP values for cluster assignments
- Add endpoint: `GET /api/explain/{article_title}`
- Display feature importance in frontend

---

## 🚀 Performance Optimizations

**Priority**: Medium  
**Status**: Planned

### Areas
- Vector database migration (Pinecone/Qdrant) for embeddings
- Redis caching for frequent API calls
- Incremental clustering for new articles
- Distributed clustering for large datasets

---

## 🔐 Security & Production Features

**Priority**: High (for production)  
**Status**: Planned

### Features
- Authentication/Authorization (JWT/OAuth)
- Rate limiting improvements (per-user limits)
- Input sanitization and validation
- CORS restrictions (remove wildcard)
- API key management

---

## 📈 Monitoring & Observability

**Priority**: Medium  
**Status**: Planned

### Features
- Prometheus metrics export
- Grafana dashboards
- APM integration (Datadog/New Relic)
- Error tracking (Sentry)
- Performance monitoring

---

## 🧪 Testing Enhancements

**Priority**: Medium  
**Status**: Planned

### Areas
- Load testing (Locust/k6)
- E2E testing (Playwright/Cypress)
- Frontend component testing
- Integration tests with real Wikipedia API
- Performance benchmarks

---

## 📝 Documentation

**Priority**: Low  
**Status**: Planned

### Areas
- API documentation improvements
- Architecture diagrams
- Deployment guides
- Developer onboarding docs
- User guides

