"""
Utility Functions
=================
Shared helper functions used across the project for logging,
directory management, formatting, and model I/O.
"""

import os
import logging
import pickle
import json
from datetime import datetime


# ──────────────────────────────────────────────
#  Logging Setup
# ──────────────────────────────────────────────

def setup_logging(log_level=logging.INFO):
    """
    Configure project-wide logging with a consistent format.
    
    Args:
        log_level: Logging level (default: INFO)
    
    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger("FraudDetection")
    logger.setLevel(log_level)
    
    # Avoid adding duplicate handlers
    if not logger.handlers:
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        
        # Format: [2024-01-15 14:30:22] INFO - Message
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger


# ──────────────────────────────────────────────
#  Directory Management
# ──────────────────────────────────────────────

def create_directories(base_path=None):
    """
    Create all required project directories if they don't exist.
    
    Args:
        base_path: Root directory of the project (default: current directory)
    
    Returns:
        dict: Mapping of directory names to their absolute paths
    """
    if base_path is None:
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    directories = {
        "dataset": os.path.join(base_path, "dataset"),
        "models": os.path.join(base_path, "models"),
        "assets": os.path.join(base_path, "assets"),
        "notebooks": os.path.join(base_path, "notebooks"),
    }
    
    for name, path in directories.items():
        os.makedirs(path, exist_ok=True)
    
    return directories


# ──────────────────────────────────────────────
#  Metrics Formatting
# ──────────────────────────────────────────────

def format_metrics(metrics_dict):
    """
    Pretty-print model evaluation metrics in a readable table.
    
    Args:
        metrics_dict: Dictionary with model names as keys and 
                      metric dictionaries as values.
                      Example: {"Logistic Regression": {"Accuracy": 0.95, ...}}
    
    Returns:
        str: Formatted table string
    """
    # Header
    header = f"{'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1 Score':>10} {'ROC-AUC':>10}"
    separator = "─" * len(header)
    
    lines = [separator, header, separator]
    
    for model_name, metrics in metrics_dict.items():
        line = (
            f"{model_name:<25} "
            f"{metrics.get('Accuracy', 0):>10.4f} "
            f"{metrics.get('Precision', 0):>10.4f} "
            f"{metrics.get('Recall', 0):>10.4f} "
            f"{metrics.get('F1 Score', 0):>10.4f} "
            f"{metrics.get('ROC-AUC', 0):>10.4f}"
        )
        lines.append(line)
    
    lines.append(separator)
    return "\n".join(lines)


# ──────────────────────────────────────────────
#  Model I/O
# ──────────────────────────────────────────────

def save_object(obj, filepath):
    """
    Save a Python object (model, scaler, etc.) using Pickle.
    
    Args:
        obj: Python object to save
        filepath: Destination file path
    
    Raises:
        IOError: If file cannot be written
    """
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(obj, f)
        logging.getLogger("FraudDetection").info(f"Saved object to: {filepath}")
    except Exception as e:
        logging.getLogger("FraudDetection").error(f"Failed to save object: {e}")
        raise


def load_saved_model(filepath):
    """
    Load a pickled model/object with error handling.
    
    Args:
        filepath: Path to the pickle file
    
    Returns:
        The loaded Python object
    
    Raises:
        FileNotFoundError: If the file doesn't exist
        pickle.UnpicklingError: If the file is corrupted
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")
    
    try:
        with open(filepath, "rb") as f:
            obj = pickle.load(f)
        logging.getLogger("FraudDetection").info(f"Loaded object from: {filepath}")
        return obj
    except pickle.UnpicklingError as e:
        logging.getLogger("FraudDetection").error(f"Corrupt pickle file: {e}")
        raise
    except Exception as e:
        logging.getLogger("FraudDetection").error(f"Failed to load object: {e}")
        raise


# ──────────────────────────────────────────────
#  Results I/O
# ──────────────────────────────────────────────

def save_results(results_dict, filepath):
    """
    Save evaluation results as a JSON file for the Streamlit dashboard.
    
    Args:
        results_dict: Dictionary of model results
        filepath: Destination JSON file path
    """
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(results_dict, f, indent=2, default=str)
        logging.getLogger("FraudDetection").info(f"Saved results to: {filepath}")
    except Exception as e:
        logging.getLogger("FraudDetection").error(f"Failed to save results: {e}")
        raise


def load_results(filepath):
    """
    Load evaluation results from a JSON file.
    
    Args:
        filepath: Path to the JSON results file
    
    Returns:
        dict: Loaded results
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Results file not found: {filepath}")
    
    with open(filepath, "r") as f:
        return json.load(f)


# ──────────────────────────────────────────────
#  Misc Helpers
# ──────────────────────────────────────────────

def get_project_root():
    """
    Get the absolute path to the project root directory.
    
    Returns:
        str: Absolute path to project root
    """
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def timestamp():
    """
    Get a formatted timestamp string.
    
    Returns:
        str: Current timestamp in 'YYYY-MM-DD_HH-MM-SS' format
    """
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
