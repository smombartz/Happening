"""
I/O utilities for JSONL writing and logging.
"""
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any
try:
    import orjson
    HAS_ORJSON = True
except ImportError:
    HAS_ORJSON = False


def setup_logging(level: str = "INFO") -> None:
    """
    Set up logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARN, ERROR)
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )


def log_info(msg: str) -> None:
    """Log an info message."""
    logging.info(msg)


def log_warn(msg: str) -> None:
    """Log a warning message."""
    logging.warning(msg)


def log_error(msg: str) -> None:
    """Log an error message."""
    logging.error(msg)


def log_debug(msg: str) -> None:
    """Log a debug message."""
    logging.debug(msg)


def write_jsonl(file_path: str, obj: Dict[str, Any]) -> None:
    """
    Write a single JSON object to a JSONL file (append mode).
    
    Args:
        file_path: Path to the JSONL file
        obj: Dictionary to write as JSON line
    """
    try:
        # Ensure parent directory exists
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Serialize JSON
        if HAS_ORJSON:
            json_str = orjson.dumps(obj).decode('utf-8')
        else:
            json_str = json.dumps(obj, ensure_ascii=False, separators=(',', ':'))
        
        # Append to file
        with open(file_path, 'a', encoding='utf-8') as f:
            f.write(json_str + '\n')
            
    except Exception as e:
        log_error(f"Failed to write JSONL to {file_path}: {e}")
        raise


def read_jsonl(file_path: str) -> list:
    """
    Read all JSON objects from a JSONL file.
    
    Args:
        file_path: Path to the JSONL file
        
    Returns:
        List of parsed JSON objects
    """
    objects = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    if HAS_ORJSON:
                        obj = orjson.loads(line)
                    else:
                        obj = json.loads(line)
                    objects.append(obj)
                except json.JSONDecodeError as e:
                    log_warn(f"Invalid JSON on line {line_num} in {file_path}: {e}")
                    continue
                    
    except FileNotFoundError:
        log_warn(f"JSONL file not found: {file_path}")
    except Exception as e:
        log_error(f"Failed to read JSONL from {file_path}: {e}")
        raise
    
    return objects


def count_jsonl_lines(file_path: str) -> int:
    """
    Count the number of valid JSON lines in a JSONL file.
    
    Args:
        file_path: Path to the JSONL file
        
    Returns:
        Number of valid JSON lines
    """
    if not Path(file_path).exists():
        return 0
    
    count = 0
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        if HAS_ORJSON:
                            orjson.loads(line)
                        else:
                            json.loads(line)
                        count += 1
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        log_error(f"Failed to count lines in {file_path}: {e}")
        
    return count


def ensure_file_empty(file_path: str) -> None:
    """
    Ensure a file is empty (create if doesn't exist, truncate if does).
    
    Args:
        file_path: Path to the file
    """
    try:
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        Path(file_path).touch()
        Path(file_path).write_text("", encoding='utf-8')
    except Exception as e:
        log_error(f"Failed to ensure file is empty {file_path}: {e}")
        raise


def print_progress(current: int, total: int, prefix: str = "") -> None:
    """
    Print progress to stderr (so it doesn't interfere with JSON output).
    
    Args:
        current: Current progress count
        total: Total count
        prefix: Optional prefix for the progress message
    """
    if total > 0:
        percentage = (current / total) * 100
        message = f"{prefix}Progress: {current}/{total} ({percentage:.1f}%)"
    else:
        message = f"{prefix}Progress: {current}"
    
    print(message, file=sys.stderr)


def safe_filename(filename: str, max_length: int = 100) -> str:
    """
    Make a filename safe by removing/replacing problematic characters.
    
    Args:
        filename: Original filename
        max_length: Maximum length for filename
        
    Returns:
        Safe filename
    """
    # Remove or replace problematic characters
    safe_chars = []
    for char in filename:
        if char.isalnum() or char in ".-_":
            safe_chars.append(char)
        elif char in " /\\:*?\"<>|":
            safe_chars.append("_")
    
    safe_name = "".join(safe_chars)
    
    # Remove multiple consecutive underscores
    while "__" in safe_name:
        safe_name = safe_name.replace("__", "_")
    
    # Trim length
    if len(safe_name) > max_length:
        safe_name = safe_name[:max_length-3] + "..."
    
    # Ensure it's not empty
    if not safe_name:
        safe_name = "untitled"
    
    return safe_name