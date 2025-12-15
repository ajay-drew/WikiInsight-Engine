"""
Pipeline orchestrator to run all pipeline stages sequentially.

This module orchestrates the full pipeline:
1. Ingestion
2. Preprocessing
3. Clustering
4. Graph Building

It monitors each stage and automatically proceeds to the next stage upon completion.
"""

import logging
import os
import subprocess
import sys
from typing import Optional

from src.common.pipeline_progress import (
    get_progress,
    mark_stage_error,
    mark_stage_completed,
    update_progress,
)

logger = logging.getLogger(__name__)


def run_stage(stage_name: str, module_path: str, args: Optional[list] = None) -> bool:
    """
    Run a pipeline stage as a subprocess and monitor its completion.
    
    Args:
        stage_name: Name of the stage (for progress tracking)
        module_path: Python module path (e.g., "src.ingestion.fetch_wikipedia_data")
        args: Optional list of additional command-line arguments
        
    Returns:
        True if stage completed successfully, False otherwise
    """
    from time import perf_counter
    
    stage_start = perf_counter()
    logger.info("=" * 80)
    logger.info("Starting stage: %s", stage_name)
    logger.info("Module: %s", module_path)
    if args:
        logger.info("Arguments: %s", args)
    logger.info("=" * 80)
    update_progress(stage_name, "running", 0.0, f"Starting {stage_name}...")
    
    # Build command
    cmd = [sys.executable, "-m", module_path]
    if args:
        cmd.extend(args)
    
    logger.info("Command: %s", " ".join(cmd))
    
    # Set environment variables for subprocess
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd()
    # Add environment variables to help with Windows DLL issues and force CPU
    env["CUDA_VISIBLE_DEVICES"] = ""  # Disable CUDA completely
    env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    
    try:
        # Run the stage
        logger.info("Executing subprocess...")
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            check=False,  # Don't raise on non-zero exit
        )
        
        stage_time = perf_counter() - stage_start
        
        if result.returncode == 0:
            logger.info("=" * 80)
            logger.info("Stage %s completed successfully in %.2f seconds (%.1f minutes)",
                       stage_name, stage_time, stage_time / 60)
            logger.info("=" * 80)
            mark_stage_completed(stage_name, f"{stage_name.capitalize()} completed successfully in {stage_time:.1f}s")
            return True
        else:
            error_msg = f"Stage {stage_name} failed with return code {result.returncode} after {stage_time:.2f} seconds"
            logger.error("=" * 80)
            logger.error(error_msg)
            logger.error("=" * 80)
            
            # Log stderr output (last 1000 chars to avoid log spam)
            if result.stderr:
                stderr_preview = result.stderr[-1000:] if len(result.stderr) > 1000 else result.stderr
                logger.error("Stderr (last 1000 chars):\n%s", stderr_preview)
                error_msg += f"\nStderr: {stderr_preview[:500]}"
            
            # Log stdout output (last 500 chars)
            if result.stdout:
                stdout_preview = result.stdout[-500:] if len(result.stdout) > 500 else result.stdout
                logger.debug("Stdout (last 500 chars):\n%s", stdout_preview)
            
            mark_stage_error(stage_name, error_msg)
            return False
            
    except Exception as exc:
        stage_time = perf_counter() - stage_start
        error_msg = f"Stage {stage_name} failed with exception after {stage_time:.2f} seconds: {str(exc)}"
        logger.exception("=" * 80)
        logger.exception(error_msg)
        logger.exception("=" * 80)
        mark_stage_error(stage_name, error_msg)
        return False


def run_full_pipeline() -> bool:
    """
    Run the full pipeline: ingestion -> preprocessing -> clustering -> graph building.
    
    Returns:
        True if all stages completed successfully, False otherwise
    """
    from time import perf_counter
    from tqdm import tqdm
    
    pipeline_start = perf_counter()
    logger.info("=" * 80)
    logger.info("Starting full pipeline orchestration")
    logger.info("=" * 80)
    
    stages = [
        ("ingestion", "src.ingestion.fetch_wikipedia_data", None),
        ("preprocessing", "src.preprocessing.process_data", None),
        ("clustering", "src.modeling.cluster_topics", None),
        ("build_graph", "src.graph.build_graph", None),
    ]
    
    stage_times = {}
    
    for i, (stage_name, module_path, args) in enumerate(tqdm(stages, desc="Pipeline stages", unit="stage"), 1):
        logger.info("\n[%d/%d] Processing stage: %s", i, len(stages), stage_name)
        stage_start = perf_counter()
        
        success = run_stage(stage_name, module_path, args)
        stage_time = perf_counter() - stage_start
        stage_times[stage_name] = stage_time
        
        if not success:
            total_time = perf_counter() - pipeline_start
            logger.error("=" * 80)
            logger.error("Pipeline stopped at stage: %s (after %.2f seconds total)", stage_name, total_time)
            logger.error("Stage times:")
            for name, time_taken in stage_times.items():
                logger.error("  - %s: %.2f seconds", name, time_taken)
            logger.error("=" * 80)
            return False
        
        # Small delay between stages to ensure file system is ready
        import time
        time.sleep(1)
    
    total_time = perf_counter() - pipeline_start
    logger.info("=" * 80)
    logger.info("Full pipeline completed successfully!")
    logger.info("Total pipeline time: %.2f seconds (%.1f minutes)", total_time, total_time / 60)
    logger.info("Stage breakdown:")
    for stage_name, time_taken in stage_times.items():
        percentage = (time_taken / total_time * 100) if total_time > 0 else 0
        logger.info("  - %s: %.2f seconds (%.1f%%)", stage_name, time_taken, percentage)
    logger.info("=" * 80)

    # Optional: trigger API reload so the frontend can query immediately.
    # This loads artifacts into the FastAPI process after the pipeline completes.
    api_base = os.environ.get("API_BASE_URL", "http://localhost:8000")
    reload_on_complete = os.environ.get("RELOAD_ON_COMPLETE", "true").lower() in {"true", "1", "yes"}
    if reload_on_complete:
        reload_url = f"{api_base.rstrip('/')}/api/pipeline/reload"
        logger.info("Triggering API data reload at %s", reload_url)
        try:
            import requests

            resp = requests.post(reload_url, timeout=30)
            if resp.status_code == 200:
                logger.info("API reload succeeded")
            else:
                logger.warning("API reload returned status %s: %s", resp.status_code, resp.text[:500])
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to trigger API reload: %s", exc)

    return True


if __name__ == "__main__":
    from src.common.logging_utils import setup_logging
    
    setup_logging()
    success = run_full_pipeline()
    sys.exit(0 if success else 1)

