import sys
from pathlib import Path
from loguru import logger

def setup_logger(log_dir: str | Path, verbose: bool = False):
    """Configures the Loguru logger."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Remove default logger
    logger.remove()
    
    # Add console logger
    log_level = "DEBUG" if verbose else "INFO"
    logger.add(
        sys.stderr, 
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level=log_level,
        colorize=True
    )
    
    # Add file logger
    log_file = log_dir / "pipeline_{time:YYYYMMDD_HHmmss}.log"
    logger.add(
        str(log_file),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG",  # Always log debug to file
        rotation="10 MB",
        retention="1 month"
    )
    
    return logger
