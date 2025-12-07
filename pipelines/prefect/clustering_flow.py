"""
Prefect flow for running the topic clustering step.

This wraps `src.modeling.cluster_topics.main` so that it can be scheduled
and monitored via Prefect.
"""

from prefect import flow, task

from src.modeling.cluster_topics import main as cluster_main


@task
def run_cluster_topics() -> None:
    cluster_main()


@flow(name="clustering_flow")
def clustering_flow() -> None:
    run_cluster_topics()


if __name__ == "__main__":
    clustering_flow()


