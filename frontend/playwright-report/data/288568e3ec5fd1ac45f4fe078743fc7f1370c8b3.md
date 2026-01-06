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
        - heading "Topic Lookup" [level=1] [ref=e18]
        - paragraph [ref=e19]: Enter a Wikipedia article title to see its topic cluster, distinctive topic words, and similar articles. The topic words are terms that show up a lot in this cluster but much less in other clusters.
      - generic [ref=e20]:
        - textbox "e.g. Machine learning" [ref=e21]: Machine learning
        - button "Looking up..." [disabled] [ref=e23]
  - contentinfo [ref=e24]: "Backend: FastAPI · Frontend: React + Vite + Tailwind"
```