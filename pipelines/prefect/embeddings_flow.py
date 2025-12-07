"""
Prefect flow for running the preprocessing/embeddings step.

This wraps `src.preprocessing.process_data.main` so that it can be scheduled
and monitored via Prefect.
"""

from prefect import flow, task

from src.preprocessing.process_data import main as preprocess_main


@task
def run_preprocess() -> None:
    preprocess_main()


@flow(name="embeddings_flow")
def embeddings_flow() -> None:
    run_preprocess()


if __name__ == "__main__":
    embeddings_flow()


