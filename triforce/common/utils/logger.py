import logging
import datetime
import json
import sys
import collections
import os

HOSTNAME = os.getenv("HOSTNAME", "unknown-worker")

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_obj = {
            "timestamp": datetime.datetime.fromtimestamp(record.created).isoformat(),
            "worker": HOSTNAME,
            "level": record.levelname,
            "message": record.getMessage(),
            "event": getattr(record, "event", "log"), # Default to 'log' if not provided
            "job_id": getattr(record, "job_id", None)
        }
        if record.exc_info:
            log_obj["error"] = self.formatException(record.exc_info)
        return json.dumps(log_obj)

class LogBuffer(logging.Handler):
    def __init__(self, capacity=1000):
        super().__init__()
        self.buffer = collections.deque(maxlen=capacity)
        self.formatted_buffer = collections.deque(maxlen=capacity)

    def emit(self, record):
        try:
            msg = self.format(record)
            self.buffer.append(record)
            self.formatted_buffer.append(json.loads(msg))
        except Exception:
            self.handleError(record)

def setup_logger(name, level=logging.INFO):
    log_buffer = LogBuffer()
    log_buffer.setFormatter(JSONFormatter())

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(JSONFormatter())

    # Reset root logger handlers
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)
            
    logging.basicConfig(level=level, handlers=[console_handler, log_buffer], force=True)
    
    # Hijack Uvicorn Loggers to use our JSON formatter and buffer
    for logger_name in ["uvicorn", "uvicorn.error", "uvicorn.access"]:
        u_logger = logging.getLogger(logger_name)
        u_logger.handlers = [console_handler, log_buffer]
        u_logger.propagate = False

    logger = logging.getLogger(name)
    logger.propagate = False
    return logger, log_buffer
