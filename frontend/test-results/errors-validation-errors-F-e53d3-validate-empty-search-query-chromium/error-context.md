# Page snapshot

```yaml
- generic [ref=e3]:
  - banner [ref=e4]:
    - generic [ref=e5]:
      - generic [ref=e6]:
        - generic [ref=e7]: WikiInsight
        - generic [ref=e8]: Topic Explorer
      - navigation [ref=e9]:
        - button "Search" [ref=e10] [cursor=pointer]
        - button "Topic Lookup" [ref=e11] [cursor=pointer]
        - button "Clusters" [ref=e12] [cursor=pointer]
        - button "Monitoring" [ref=e13] [cursor=pointer]
        - button "Ingestion" [ref=e14] [cursor=pointer]
  - main [ref=e15]:
    - generic [ref=e16]:
      - generic [ref=e17]:
        - heading "Hybrid Search" [level=1] [ref=e18]
        - paragraph [ref=e19]: Search Wikipedia articles using semantic (vector) and keyword (BM25) search combined with Reciprocal Rank Fusion. Find articles by meaning, not just exact keywords.
      - generic [ref=e21]:
        - textbox "e.g. machine learning algorithms, cooking pasta recipes, space exploration..." [ref=e22]
        - button "Search" [disabled] [ref=e23]
      - generic [ref=e24]:
        - paragraph [ref=e25]: Enter a search query above to find relevant Wikipedia articles.
        - generic [ref=e26]:
          - generic [ref=e27]:
            - paragraph [ref=e28]: Semantic Search
            - paragraph [ref=e29]: Finds articles by meaning and context
          - generic [ref=e30]:
            - paragraph [ref=e31]: Keyword Search
            - paragraph [ref=e32]: Finds articles with exact term matches
          - generic [ref=e33]:
            - paragraph [ref=e34]: Hybrid Fusion
            - paragraph [ref=e35]: Combines both for best results
  - contentinfo [ref=e36]: "Backend: FastAPI · Frontend: React + Vite + Tailwind"
```