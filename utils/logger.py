"""
Logging configuration using loguru
"""
import sys
from pathlib import Path
from loguru import logger
from config import settings

# Remove default handler
logger.remove()

# Console handler with color
logger.add(
    sys.stdout,
    colorize=True,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> | <level>{message}</level>",
    level="DEBUG" if settings.DEBUG else "INFO"
)

# File handler
log_path = settings.DATA_DIR / "logs"
log_path.mkdir(exist_ok=True)

logger.add(
    log_path / "sarthi_ai_{time:YYYY-MM-DD}.log",
    rotation="00:00",
    retention="30 days",
    compression="zip",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function} | {message}",
    level="DEBUG"
)

# Error file handler
logger.add(
    log_path / "errors_{time:YYYY-MM-DD}.log",
    rotation="00:00",
    retention="90 days",
    compression="zip",
    level="ERROR",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function} | {message}\n{extra}"
)

def get_logger(name: str):
    """Get a logger instance with the given name"""
    return logger.bind(name=name)