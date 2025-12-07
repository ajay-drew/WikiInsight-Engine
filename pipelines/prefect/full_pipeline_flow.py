"""
Prefect flow that chains ingestion → embeddings → clustering.

This provides a single entrypoint to run the full Phase 1 pipeline:
    ingestion_flow → embeddings_flow → clustering_flow
"""

from prefect import flow

from .ingestion_flow import ingestion_flow
from .embeddings_flow import embeddings_flow
from .clustering_flow import clustering_flow


@flow(name="full_topic_pipeline")
def full_topic_pipeline() -> None:
    ingestion_flow()
    embeddings_flow()
    clustering_flow()


if __name__ == "__main__":
    full_topic_pipeline()


