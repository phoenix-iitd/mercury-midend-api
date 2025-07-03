import logging
import structlog
from datetime import datetime
from typing import Optional


def configure_structlog(environment: str = "development"):
    """Configure structlog for better logging"""
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=None,
        level=logging.DEBUG if environment == "development" else logging.INFO,
    )
    
    # Configure structlog processors
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt="ISO"),
    ]
    
    if environment == "development":
        # Development: colored console output
        processors.append(structlog.dev.ConsoleRenderer(colors=True))
    else:
        # Production: JSON output
        processors.append(structlog.processors.JSONRenderer())
    
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(logging.NOTSET),
        logger_factory=structlog.WriteLoggerFactory(),
        context_class=dict,
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a structured logger instance"""
    return structlog.get_logger(name)


def bind_correlation_id(logger: structlog.BoundLogger, correlation_id: str) -> structlog.BoundLogger:
    """Bind correlation ID to logger for consistent tracking"""
    return logger.bind(correlation_id=correlation_id)


def add_correlation_id(
    msg: str, correlation_id: str = "NO_CORR_ID", level: int = logging.INFO
) -> str:
    """Legacy function for backward compatibility - use structured logging instead"""
    if level == logging.DEBUG:
        color = "\033[90m"  # gray
    elif level == logging.INFO:
        color = "\033[32m"  # green
    elif level == logging.WARNING:
        color = "\033[38;5;208m"  # orange
    elif level == logging.ERROR:
        color = "\033[31m"  # red
    else:
        color = ""
    reset = "\033[0m"
    return f"{color}[{correlation_id}]{reset} {msg}"


class CorrelationLogger:
    """Logger wrapper that automatically includes correlation ID in all logs"""
    
    def __init__(self, logger: structlog.BoundLogger, correlation_id: str):
        self.logger = logger.bind(correlation_id=correlation_id)
        self.correlation_id = correlation_id
    
    def debug(self, msg: str, **kwargs):
        self.logger.debug(msg, **kwargs)
    
    def info(self, msg: str, **kwargs):
        self.logger.info(msg, **kwargs)
    
    def warning(self, msg: str, **kwargs):
        self.logger.warning(msg, **kwargs)
    
    def error(self, msg: str, **kwargs):
        self.logger.error(msg, **kwargs)
    
    def critical(self, msg: str, **kwargs):
        self.logger.critical(msg, **kwargs)


def create_correlation_logger(name: str, correlation_id: str) -> CorrelationLogger:
    """Create a logger that automatically includes correlation ID"""
    logger = structlog.get_logger(name)
    return CorrelationLogger(logger, correlation_id)


# Performance logging utilities
class PerformanceTimer:
    """Context manager for timing operations"""
    
    def __init__(self, logger: structlog.BoundLogger, operation: str, **context):
        self.logger = logger
        self.operation = operation
        self.context = context
        self.start_time = None
        
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.debug("Operation started", operation=self.operation, **self.context)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = datetime.now()
        if self.start_time:
            duration_ms = (end_time - self.start_time).total_seconds() * 1000
        else:
            duration_ms = 0.0
        
        if exc_type:
            self.logger.error(
                "Operation failed",
                operation=self.operation,
                duration_ms=duration_ms,
                error=str(exc_val),
                **self.context
            )
        else:
            self.logger.info(
                "Operation completed",
                operation=self.operation,
                duration_ms=duration_ms,
                **self.context
            )


def log_api_request(logger: structlog.BoundLogger, method: str, url: str, status_code: int, duration_ms: float, **kwargs):
    """Log API request in structured format"""
    logger.info(
        "API request",
        method=method,
        url=url,
        status_code=status_code,
        duration_ms=duration_ms,
        **kwargs
    )


def log_database_operation(logger: structlog.BoundLogger, operation: str, table: str, duration_ms: float, affected_rows: Optional[int] = None, **kwargs):
    """Log database operation in structured format"""
    log_data = {
        "database_operation": operation,
        "table": table,
        "duration_ms": duration_ms,
        **kwargs
    }
    
    if affected_rows is not None:
        log_data["affected_rows"] = affected_rows
    
    logger.info("Database operation", **log_data)


def log_queue_operation(logger: structlog.BoundLogger, operation: str, queue_size: int, processing_time_ms: Optional[float] = None, **kwargs):
    """Log queue operation in structured format"""
    log_data = {
        "queue_operation": operation,
        "queue_size": queue_size,
        **kwargs
    }
    
    if processing_time_ms is not None:
        log_data["processing_time_ms"] = processing_time_ms
    
    logger.info("Queue operation", **log_data)
