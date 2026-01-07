"""
SQLite database for storing pipeline logs per run.

Each pipeline run gets its own log entries stored in the database.
Logs are automatically deleted when the webpage is reloaded (via API endpoint).
"""

import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import logging

logger = logging.getLogger(__name__)

# Database file path
DB_PATH = Path("pipeline_logs.db")

# Thread-local storage for database connections
_local = threading.local()


def get_db_connection() -> sqlite3.Connection:
    """Get or create a thread-local database connection."""
    if not hasattr(_local, 'connection') or _local.connection is None:
        _local.connection = sqlite3.connect(
            str(DB_PATH),
            check_same_thread=False,
            timeout=30.0
        )
        _local.connection.row_factory = sqlite3.Row
        _local.connection.execute("PRAGMA journal_mode=WAL")
    return _local.connection


def init_database():
    """Initialize the pipeline logs database with required tables."""
    conn = get_db_connection()
    
    conn.execute("""
        CREATE TABLE IF NOT EXISTS pipeline_runs (
            run_id TEXT PRIMARY KEY,
            started_at TIMESTAMP NOT NULL,
            status TEXT NOT NULL,
            current_stage TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.execute("""
        CREATE TABLE IF NOT EXISTS pipeline_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            stage_name TEXT,
            log_level TEXT NOT NULL,
            message TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (run_id) REFERENCES pipeline_runs(run_id) ON DELETE CASCADE
        )
    """)
    
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_logs_run_id ON pipeline_logs(run_id)
    """)
    
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON pipeline_logs(timestamp)
    """)
    
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_runs_status ON pipeline_runs(status)
    """)
    
    conn.commit()
    logger.info("Pipeline logs database initialized at %s", DB_PATH)


def create_run(run_id: str) -> None:
    """Create a new pipeline run record."""
    conn = get_db_connection()
    conn.execute(
        "INSERT OR REPLACE INTO pipeline_runs (run_id, started_at, status, current_stage) VALUES (?, ?, ?, ?)",
        (run_id, datetime.now().isoformat(), "running", None)
    )
    conn.commit()
    logger.debug("Created pipeline run: %s", run_id)


def update_run_status(run_id: str, status: str, current_stage: Optional[str] = None) -> None:
    """Update the status of a pipeline run."""
    conn = get_db_connection()
    conn.execute(
        "UPDATE pipeline_runs SET status = ?, current_stage = ? WHERE run_id = ?",
        (status, current_stage, run_id)
    )
    conn.commit()


def add_log(run_id: str, log_level: str, message: str, stage_name: Optional[str] = None) -> None:
    """Add a log entry for a pipeline run."""
    try:
        conn = get_db_connection()
        conn.execute(
            "INSERT INTO pipeline_logs (run_id, stage_name, log_level, message) VALUES (?, ?, ?, ?)",
            (run_id, stage_name, log_level, message)
        )
        conn.commit()
    except Exception as exc:  # noqa: BLE001
        # Don't let logging failures break the pipeline
        logger.warning("Failed to write log to database: %s", exc)


def get_run_logs(run_id: str, limit: int = 1000) -> List[Dict]:
    """Get all logs for a specific pipeline run."""
    conn = get_db_connection()
    cursor = conn.execute(
        "SELECT stage_name, log_level, message, timestamp FROM pipeline_logs WHERE run_id = ? ORDER BY timestamp ASC LIMIT ?",
        (run_id, limit)
    )
    return [
        {
            "stage_name": row["stage_name"],
            "log_level": row["log_level"],
            "message": row["message"],
            "timestamp": row["timestamp"],
        }
        for row in cursor.fetchall()
    ]


def get_all_runs() -> List[Dict]:
    """Get all pipeline runs with their status."""
    conn = get_db_connection()
    cursor = conn.execute(
        "SELECT run_id, started_at, status, current_stage FROM pipeline_runs ORDER BY started_at DESC LIMIT 100"
    )
    return [
        {
            "run_id": row["run_id"],
            "started_at": row["started_at"],
            "status": row["status"],
            "current_stage": row["current_stage"],
        }
        for row in cursor.fetchall()
    ]


def get_current_run() -> Optional[str]:
    """Get the ID of the currently running pipeline, if any."""
    conn = get_db_connection()
    cursor = conn.execute(
        "SELECT run_id FROM pipeline_runs WHERE status = 'running' ORDER BY started_at DESC LIMIT 1"
    )
    row = cursor.fetchone()
    return row["run_id"] if row else None


def delete_run(run_id: str) -> None:
    """Delete a pipeline run and all its logs."""
    conn = get_db_connection()
    conn.execute("DELETE FROM pipeline_runs WHERE run_id = ?", (run_id,))
    conn.commit()
    logger.info("Deleted pipeline run: %s", run_id)


def delete_all_runs() -> None:
    """Delete all pipeline runs and logs (called on webpage reload)."""
    conn = get_db_connection()
    conn.execute("DELETE FROM pipeline_runs")
    conn.commit()
    logger.info("Deleted all pipeline runs (webpage reload)")


def get_run_info(run_id: str) -> Optional[Dict]:
    """Get information about a specific pipeline run."""
    conn = get_db_connection()
    cursor = conn.execute(
        "SELECT run_id, started_at, status, current_stage FROM pipeline_runs WHERE run_id = ?",
        (run_id,)
    )
    row = cursor.fetchone()
    if row:
        return {
            "run_id": row["run_id"],
            "started_at": row["started_at"],
            "status": row["status"],
            "current_stage": row["current_stage"],
        }
    return None


class DatabaseLogHandler(logging.Handler):
    """Custom logging handler that writes logs to SQLite database."""
    
    def __init__(self, run_id: Optional[str] = None):
        super().__init__()
        self.run_id = run_id
        self.setLevel(logging.INFO)
    
    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record to the database."""
        try:
            if self.run_id:
                # Map Python log levels to our database log levels
                level_map = {
                    logging.DEBUG: "DEBUG",
                    logging.INFO: "INFO",
                    logging.WARNING: "WARNING",
                    logging.ERROR: "ERROR",
                    logging.CRITICAL: "ERROR",
                }
                log_level = level_map.get(record.levelno, "INFO")
                
                # Extract stage name from logger name if possible
                stage_name = None
                if hasattr(record, 'stage_name'):
                    stage_name = record.stage_name
                elif 'subprocess' in record.name or 'stage' in record.name.lower():
                    # Try to extract stage name from logger name
                    for stage in ['ingestion', 'preprocessing', 'clustering', 'build_graph']:
                        if stage in record.name.lower():
                            stage_name = stage
                            break
                
                add_log(self.run_id, log_level, self.format(record), stage_name)
        except Exception:  # noqa: BLE001
            # Don't let logging failures break the pipeline
            pass


# Initialize database on import
init_database()

