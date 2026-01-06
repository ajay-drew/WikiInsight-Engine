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
    
    cmd = [sys.executable, "-m", module_path]
    if args:
        cmd.extend(args)
    
    logger.info("Command: %s", " ".join(cmd))
    
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd()
    env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    logger.info("Environment variables:")
    logger.info("  - PYTHONPATH: %s", env.get("PYTHONPATH", "not set"))
    logger.info("  - CUDA_VISIBLE_DEVICES: %s", env.get("CUDA_VISIBLE_DEVICES", "not set (GPU enabled)"))
    logger.info("  - Python executable: %s", sys.executable)
    
    try:
        # Run the stage with real-time output streaming
        logger.info("Executing subprocess (PID will be logged when started)...")
        logger.info("Subprocess will stream output in real-time")
        
        # Use Popen for real-time output streaming
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Combine stderr into stdout
            text=True,
            bufsize=1,  # Line buffered
            universal_newlines=True,
        )
        
        logger.info("Subprocess started (PID: %d)", process.pid)
        
        output_lines = []
        for line in iter(process.stdout.readline, ''):
            if line:
                line = line.rstrip()
                output_lines.append(line)
                # Log INFO level lines, but filter out very verbose output
                if line and not line.startswith('DEBUG:'):
                    logger.info("[%s subprocess] %s", stage_name, line)
        
        # Wait for process to complete
        returncode = process.wait()
        stage_time = perf_counter() - stage_start
        
        # Combine all output
        full_output = '\n'.join(output_lines)
        
        if returncode == 0:
            logger.info("=" * 80)
            logger.info("Stage %s completed successfully in %.2f seconds (%.1f minutes)",
                       stage_name, stage_time, stage_time / 60)
            logger.info("Subprocess PID: %d", process.pid)
            logger.info("Output lines captured: %d", len(output_lines))
            logger.info("=" * 80)
            
            if len(output_lines) > 20:
                logger.debug("Last 20 lines of subprocess output:")
                for line in output_lines[-20:]:
                    logger.debug("  %s", line)
            
            mark_stage_completed(stage_name, f"{stage_name.capitalize()} completed successfully in {stage_time:.1f}s")
            return True
        else:
            error_msg = f"Stage {stage_name} failed with return code {returncode} after {stage_time:.2f} seconds"
            logger.error("=" * 80)
            logger.error(error_msg)
            logger.error("Subprocess PID: %d", process.pid)
            logger.error("Output lines captured: %d", len(output_lines))
            logger.error("=" * 80)
            
            if output_lines:
                logger.error("Subprocess output (last 50 lines):")
                for line in output_lines[-50:]:
                    logger.error("  %s", line)
            
            if full_output:
                logger.debug("Full subprocess output:\n%s", full_output[-2000:])
            
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
    from time import perf_counter, sleep
    from tqdm import tqdm
    
    pipeline_start = perf_counter()
    logger.info("=" * 80)
    logger.info("Starting full pipeline orchestration")
    logger.info("=" * 80)
    logger.info("Note: When running via DVC (dvc repro), subprocess output will be logged here")
    logger.info("Each stage runs as a subprocess with real-time output streaming")
    logger.info("=" * 80)
    
    try:
        from src.common.gpu_utils import (
            log_device_info,
            verify_gpu_usage,
            is_cuda_available,
            get_clustering_backend,
        )
        from src.preprocessing.embeddings import detect_device
        
        logger.info("")
        logger.info("GPU Detection and Configuration")
        logger.info("=" * 80)
        
        gpu_verification = verify_gpu_usage()
        embedding_device = detect_device("auto")
        clustering_backend = get_clustering_backend()
        
        logger.info("GPU Status Summary:")
        logger.info("  - PyTorch CUDA: %s", "✅ Available" if gpu_verification["pytorch_cuda"] else "❌ Not available")
        logger.info("  - CuPy CUDA: %s", "✅ Available" if gpu_verification["cupy_cuda"] else "❌ Not available")
        logger.info("")
        logger.info("Pipeline Stage GPU Usage:")
        logger.info("  - Embeddings: Will use %s", embedding_device.upper())
        logger.info("  - Clustering: Will use %s backend", clustering_backend.upper())
        logger.info("  - Clustering GPU enabled: %s", "✅ Yes" if gpu_verification["clustering_can_use_gpu"] else "❌ No")
        logger.info("")
        
        gpu_info = None
        try:
            import torch
            if torch.cuda.is_available():
                gpu_info = {
                    "name": torch.cuda.get_device_name(0),
                    "memory_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3),
                    "device_count": torch.cuda.device_count(),
                }
        except Exception:
            pass
        
        if gpu_info:
            logger.info("GPU Hardware Details:")
            logger.info("  - GPU Name: %s", gpu_info["name"])
            logger.info("  - GPU Memory: %.1f GB", gpu_info["memory_gb"])
            logger.info("  - Device Count: %d", gpu_info["device_count"])
        else:
            logger.info("GPU Hardware: Not detected or not available")
        
        logger.info("=" * 80)
        logger.info("")
        
        log_device_info()
        
    except Exception as exc:
        logger.warning("Error during GPU detection/logging: %s. Pipeline will continue.", exc)
        logger.info("GPU status: Unable to determine (pipeline will auto-detect per stage)")
    
    logger.info("")
    
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
        
        sleep(1)
    
    total_time = perf_counter() - pipeline_start
    logger.info("=" * 80)
    logger.info("Full pipeline completed successfully!")
    logger.info("Total pipeline time: %.2f seconds (%.1f minutes)", total_time, total_time / 60)
    logger.info("Stage breakdown:")
    for stage_name, time_taken in stage_times.items():
        percentage = (time_taken / total_time * 100) if total_time > 0 else 0
        logger.info("  - %s: %.2f seconds (%.1f%%)", stage_name, time_taken, percentage)
    logger.info("=" * 80)

    api_base = os.environ.get("API_BASE_URL", "http://127.0.0.1:9000")
    reload_on_complete = os.environ.get("RELOAD_ON_COMPLETE", "true").lower() in {"true", "1", "yes"}
    if reload_on_complete:
        reload_url = f"{api_base.rstrip('/')}/api/pipeline/reload"
        logger.info("=" * 80)
        logger.info("Attempting to trigger API data reload")
        logger.info("  - URL: %s", reload_url)
        logger.info("  - Note: API server must be running for this to work")
        logger.info("=" * 80)
        # #region agent log
        import json
        try:
            with open("x:\\majorProjects\\WikiInsight-Engine\\.cursor\\debug.log", "a") as f:
                f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "A", "location": "pipeline_orchestrator.py:263", "message": "Pipeline orchestrator attempting reload", "data": {"reload_url": reload_url, "reload_on_complete": reload_on_complete}, "timestamp": __import__("time").time() * 1000}) + "\n")
        except: pass
        # #endregion
        try:
            import requests

            logger.info("Sending POST request to API reload endpoint...")
            # #region agent log
            try:
                with open("x:\\majorProjects\\WikiInsight-Engine\\.cursor\\debug.log", "a") as f:
                    f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "A", "location": "pipeline_orchestrator.py:273", "message": "Sending POST request to reload endpoint", "data": {"url": reload_url}, "timestamp": __import__("time").time() * 1000}) + "\n")
            except: pass
            # #endregion
            resp = requests.post(reload_url, timeout=30)
            # #region agent log
            try:
                with open("x:\\majorProjects\\WikiInsight-Engine\\.cursor\\debug.log", "a") as f:
                    f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "A", "location": "pipeline_orchestrator.py:274", "message": "Reload request response received", "data": {"status_code": resp.status_code, "response_text": resp.text[:200]}, "timestamp": __import__("time").time() * 1000}) + "\n")
            except: pass
            # #endregion
            if resp.status_code == 200:
                logger.info("=" * 80)
                logger.info("API reload succeeded!")
                try:
                    response_data = resp.json()
                    logger.info("Response: %s", response_data.get("message", "OK"))
                    if "status" in response_data:
                        logger.info("Loaded components: %s", response_data.get("status", {}))
                except Exception:
                    logger.info("Response: %s", resp.text[:200])
                logger.info("=" * 80)
            else:
                logger.warning("=" * 80)
                logger.warning("API reload returned non-200 status: %s", resp.status_code)
                logger.warning("Response: %s", resp.text[:500])
                logger.warning("=" * 80)
        except requests.exceptions.ConnectionError as exc:
            logger.warning("=" * 80)
            logger.warning("API server is not running or not accessible")
            logger.warning("  - URL: %s", reload_url)
            logger.warning("  - Error: %s", exc)
            logger.warning("  - Solution: Start the API server with 'python -m src.api.main' or 'run_app.cmd'")
            logger.warning("  - You can manually reload data later via the API endpoint")
            logger.warning("=" * 80)
            # #region agent log
            try:
                import json
                with open("x:\\majorProjects\\WikiInsight-Engine\\.cursor\\debug.log", "a") as f:
                    f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "A", "location": "pipeline_orchestrator.py:290", "message": "Connection error during reload", "data": {"error": str(exc), "url": reload_url}, "timestamp": __import__("time").time() * 1000}) + "\n")
            except: pass
            # #endregion
        except Exception as exc:  # noqa: BLE001
            logger.warning("=" * 80)
            logger.warning("Failed to trigger API reload: %s", exc)
            logger.warning("  - URL: %s", reload_url)
            logger.warning("  - Error type: %s", type(exc).__name__)
            logger.warning("=" * 80)
            # #region agent log
            try:
                import json
                with open("x:\\majorProjects\\WikiInsight-Engine\\.cursor\\debug.log", "a") as f:
                    f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "A", "location": "pipeline_orchestrator.py:298", "message": "Exception during reload", "data": {"error": str(exc), "error_type": type(exc).__name__, "url": reload_url}, "timestamp": __import__("time").time() * 1000}) + "\n")
            except: pass
            # #endregion
    else:
        logger.info("API reload skipped (RELOAD_ON_COMPLETE is disabled)")

    return True


if __name__ == "__main__":
    from src.common.logging_utils import setup_logging
    
    setup_logging()
    success = run_full_pipeline()
    sys.exit(0 if success else 1)

