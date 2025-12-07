import logging
import os
from pathlib import Path


def _ensure_log_dir() -> Path:
    """
    Ensure a log directory exists and return it.

    Defaults to ./logs, override with SNAP_LOG_DIR.
    """
    base = Path(os.getenv("SNAP_LOG_DIR", "logs"))
    base.mkdir(parents=True, exist_ok=True)
    return base


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger with a simple, consistent format and file logging."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console logging (picked up by systemd/journald in production)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # File logging for /opt/.../logs/*.log on the VM (and ./logs locally)
    try:
        log_dir = _ensure_log_dir()
        file_path = log_dir / f"{name}.log"
        file_handler = logging.FileHandler(file_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception:  # pragma: no cover - file logging is best-effort
        # If the filesystem is not writable, fall back to console-only logs.
        pass

    return logger
