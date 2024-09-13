import logging
import os
import sys
from pathlib import Path

def setup_logger():
    # Get the script name (without extension) for the log directory
    script_name = Path(sys.argv[0]).stem
    
    # Create the log directory if it doesn't exist
    log_dir = Path(f'logs/{script_name}')
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # File paths for error and debug logs
    error_log = log_dir / 'error.log'
    debug_log = log_dir / 'debug.log'
    
    # Create a logger
    logger = logging.getLogger(script_name)
    logger.setLevel(logging.DEBUG)  # Set to the lowest level to capture all logs
    
    # Create a file handler for error logs
    error_handler = logging.FileHandler(error_log)
    error_handler.setLevel(logging.ERROR)  # Only log errors
    
    # Create a file handler for debug logs
    debug_handler = logging.FileHandler(debug_log)
    debug_handler.setLevel(logging.DEBUG)  # Log everything down to debug level
    
    # Create a console handler to log to stdout
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)  # Log everything to console as well
    
    # Create a formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Add the formatter to the handlers
    error_handler.setFormatter(formatter)
    debug_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add the handlers to the logger
    logger.addHandler(error_handler)
    logger.addHandler(debug_handler)
    logger.addHandler(console_handler)
    
    return logger
