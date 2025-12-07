"""
Prefect flow for running the Wikipedia ingestion step.

This wraps `src.ingestion.fetch_wikipedia_data.main` so that it can be scheduled
and monitored via Prefect.
"""

from prefect import flow, task

from src.ingestion.fetch_wikipedia_data import main as fetch_main


@task
def run_fetch_wikipedia_data() -> None:
    fetch_main()


@flow(name="ingestion_flow")
def ingestion_flow() -> None:
    run_fetch_wikipedia_data()


if __name__ == "__main__":
    ingestion_flow()


