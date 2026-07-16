import logging
import json
import time
import uuid
from typing import Dict, Any, Optional
from functools import wraps


class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "request_id": getattr(record, "request_id", "N/A"),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "message": record.getMessage(),
        }
        
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
        
        extra = getattr(record, "extra", None)
        if extra:
            log_record.update(extra)
        
        return json.dumps(log_record, ensure_ascii=False)


class RequestContext:
    _local = {}

    @classmethod
    def get_request_id(cls) -> str:
        return cls._local.get("request_id", "N/A")

    @classmethod
    def set_request_id(cls, request_id: str):
        cls._local["request_id"] = request_id

    @classmethod
    def clear(cls):
        cls._local.clear()


def generate_request_id() -> str:
    return str(uuid.uuid4())[:8]


def get_logger(name: str = __name__) -> logging.Logger:
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(JsonFormatter())
        logger.addHandler(console_handler)
        
        file_handler = logging.FileHandler("app.log", encoding="utf-8")
        file_handler.setFormatter(JsonFormatter())
        logger.addHandler(file_handler)
    
    return logger


def log_execution(logger_name: str = None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(logger_name or func.__module__)
            request_id = RequestContext.get_request_id()
            start_time = time.time()
            
            logger.info(
                f"开始执行: {func.__name__}",
                extra={
                    "request_id": request_id,
                    "function": func.__name__,
                    "action": "start",
                },
            )
            
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                
                logger.info(
                    f"执行完成: {func.__name__}",
                    extra={
                        "request_id": request_id,
                        "function": func.__name__,
                        "action": "end",
                        "elapsed_ms": round(elapsed * 1000, 2),
                        "result_type": type(result).__name__ if result else None,
                    },
                )
                
                return result
            except Exception as e:
                elapsed = time.time() - start_time
                
                logger.error(
                    f"执行失败: {func.__name__}",
                    extra={
                        "request_id": request_id,
                        "function": func.__name__,
                        "action": "error",
                        "elapsed_ms": round(elapsed * 1000, 2),
                        "error": str(e),
                    },
                )
                raise
        
        return wrapper
    
    return decorator


logger = get_logger("office_rag")
