"""
API performance metrics tracking.

Tracks request latency, error rates, endpoint usage,
and search query performance.
"""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class RequestMetric:
    """Single request metric."""
    endpoint: str
    method: str
    status_code: int
    latency_ms: float
    timestamp: float = field(default_factory=time.time)


class APIMetricsCollector:
    """Collects and aggregates API metrics."""
    
    def __init__(self, max_metrics: int = 10000):
        """
        Initialize metrics collector.
        
        Args:
            max_metrics: Maximum number of metrics to keep in memory
        """
        self.metrics: List[RequestMetric] = []
        self.max_metrics = max_metrics
    
    def record_request(
        self,
        endpoint: str,
        method: str,
        status_code: int,
        latency_ms: float,
    ) -> None:
        """Record a request metric."""
        metric = RequestMetric(
            endpoint=endpoint,
            method=method,
            status_code=status_code,
            latency_ms=latency_ms,
        )
        self.metrics.append(metric)
        
        # Trim if too many metrics
        if len(self.metrics) > self.max_metrics:
            self.metrics = self.metrics[-self.max_metrics:]
    
    def get_summary(self, window_seconds: Optional[float] = None) -> Dict[str, any]:
        """
        Get metrics summary.
        
        Args:
            window_seconds: Optional time window (only metrics within this window)
        
        Returns:
            Dict with aggregated metrics
        """
        if window_seconds:
            cutoff_time = time.time() - window_seconds
            recent_metrics = [m for m in self.metrics if m.timestamp >= cutoff_time]
        else:
            recent_metrics = self.metrics
        
        if not recent_metrics:
            return {
                "total_requests": 0,
                "endpoints": {},
            }
        
        # Aggregate by endpoint
        endpoint_stats = defaultdict(lambda: {
            "count": 0,
            "total_latency_ms": 0.0,
            "status_codes": defaultdict(int),
            "latencies": [],
        })
        
        for metric in recent_metrics:
            stats = endpoint_stats[metric.endpoint]
            stats["count"] += 1
            stats["total_latency_ms"] += metric.latency_ms
            stats["status_codes"][metric.status_code] += 1
            stats["latencies"].append(metric.latency_ms)
        
        # Calculate percentiles and averages
        summary = {
            "total_requests": len(recent_metrics),
            "window_seconds": window_seconds,
            "endpoints": {},
        }
        
        for endpoint, stats in endpoint_stats.items():
            latencies = stats["latencies"]
            latencies.sort()
            
            n = len(latencies)
            p50_idx = int(n * 0.5)
            p95_idx = int(n * 0.95)
            p99_idx = int(n * 0.99)
            
            summary["endpoints"][endpoint] = {
                "count": stats["count"],
                "avg_latency_ms": stats["total_latency_ms"] / stats["count"],
                "p50_latency_ms": latencies[p50_idx] if n > 0 else 0.0,
                "p95_latency_ms": latencies[p95_idx] if n > 0 else 0.0,
                "p99_latency_ms": latencies[p99_idx] if n > 0 else 0.0,
                "error_rate": sum(
                    count for code, count in stats["status_codes"].items()
                    if code >= 400
                ) / stats["count"] if stats["count"] > 0 else 0.0,
                "status_codes": dict(stats["status_codes"]),
            }
        
        return summary
    
    def clear(self) -> None:
        """Clear all metrics."""
        self.metrics.clear()


# Global metrics collector instance
_metrics_collector = APIMetricsCollector()


def get_metrics_collector() -> APIMetricsCollector:
    """Get the global metrics collector instance."""
    return _metrics_collector


def record_api_request(
    endpoint: str,
    method: str,
    status_code: int,
    latency_ms: float,
) -> None:
    """Record an API request metric."""
    _metrics_collector.record_request(endpoint, method, status_code, latency_ms)


def get_api_metrics_summary(window_seconds: Optional[float] = None) -> Dict[str, any]:
    """Get API metrics summary."""
    return _metrics_collector.get_summary(window_seconds)


def log_metrics_to_mlflow(
    summary: Dict[str, any],
    config_path: str = "config.yaml",
) -> None:
    """
    Log API metrics to MLflow.
    
    This function is completely non-blocking and will never raise exceptions.
    All errors are caught and logged as debug messages.
    """
    from src.common.mlflow_utils import log_metrics_safely, start_mlflow_run
    
    try:
        with start_mlflow_run("api_metrics", config_path=config_path):
            # Log overall metrics
            log_metrics_safely({
                "total_requests": summary.get("total_requests", 0),
            }, prefix="api")
            
            # Log per-endpoint metrics
            for endpoint, stats in summary.get("endpoints", {}).items():
                endpoint_safe = endpoint.replace("/", "_").replace("-", "_")
                log_metrics_safely({
                    f"{endpoint_safe}_count": stats.get("count", 0),
                    f"{endpoint_safe}_avg_latency_ms": stats.get("avg_latency_ms", 0.0),
                    f"{endpoint_safe}_p95_latency_ms": stats.get("p95_latency_ms", 0.0),
                    f"{endpoint_safe}_error_rate": stats.get("error_rate", 0.0),
                }, prefix="api")
            
            logger.debug("Logged API metrics to MLflow")
    except Exception as exc:
        logger.debug("Failed to log API metrics to MLflow (non-critical): %s", exc)

