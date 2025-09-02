"""Project-wide logging configuration and utilities.

This module provides a centralized logging system for the project, utilizing
customized loggers with both console and file output capabilities. It supports
colored console output and structured logging to a timestamped file. The module
includes a `ProjectLogger` class for managing logger instances and convenience
functions for easy access to logging functionality.

The logging system is designed to be initialized once per project run, with
configurable log levels, output destinations, and file storage. It also provides
helper methods to log function start and end events, including parameters and
results, while ensuring sensitive information is obscured.

Dependencies:
    - datetime: For timestamp generation in log files.
    - typing: For type hints in method signatures.
    - logging: For core logging functionality.
    - sys: For console output handling.
    - pathlib: For cross-platform file path manipulation.
    - inspect: For inferring module names from call stack.
    - colorlog: For colored console output formatting.
"""
from datetime import datetime
from typing import Dict, Optional
import logging
import sys
from pathlib import Path
import inspect
import colorlog

class ProjectLogger:
    """Manages project-wide logger instances and configuration.

    This class provides a singleton-like interface for setting up and retrieving
    logger instances. It supports both console and file-based logging with
    customizable log levels and formats. Loggers are cached to ensure consistent
    use across the project.

    Attributes:
        _loggers (Dict[str, logging.Logger]): Cache of logger instances by name.
        _log_dir (Optional[Path]): Directory path for log files.
        _base_config_set (bool): Flag indicating if logging configuration is set.
    """
    _loggers: Dict[str, logging.Logger] = {}
    _log_dir: Optional[Path] = None
    _base_config_set = False
    @classmethod
    def setup_project_logging(
        cls,
        log_level: str = "INFO",
        log_dir: str = "logs",
        console_output: bool = True,
        file_output: bool = True,
    ):
        """Initialize project-wide logging configuration.

        Sets up the root logger with console and/or file handlers based on input
        parameters. Creates a log directory if it doesn't exist and configures
        colored console output and timestamped log files. This method should be
        called once at the start of the application.

        Args:
            log_level (str): Logging level (e.g., 'DEBUG', 'INFO'). Defaults to 'INFO'.
            log_dir (str): Directory for log files, relative to project root. Defaults to 'logs'.
            console_output (bool): If True, logs to console. Defaults to True.
            file_output (bool): If True, logs to a file. Defaults to True.
        """
        if cls._base_config_set:
            return

        script_dir = Path(__file__).parent  # /path/to/repo/scripts
        project_root = script_dir.parent     # /path/to/repo/
        cls._log_dir = Path(project_root / log_dir).resolve()
        cls._log_dir.mkdir(parents=True, exist_ok=True)

        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, log_level.upper()))
        root_logger.handlers.clear()

        console_formatter = colorlog.ColoredFormatter(
            "%(log_color)s %(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors={
                    'DEBUG': 'cyan',
                    'INFO': 'white',
                    'WARNING': 'yellow',
                    'ERROR': 'red',
                    'CRITICAL': 'red,bg_white',
                },
        )

        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handlers = []
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, log_level.upper()))
            console_handler.setFormatter(console_formatter)
            handlers.append(console_handler)
        if file_output:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = cls._log_dir / f"nvda_predictor_{timestamp}.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(getattr(logging, log_level.upper()))
            file_handler.setFormatter(file_formatter)
            handlers.append(file_handler)

        for handler in handlers:
            root_logger.addHandler(handler)

        cls._base_config_set = True

        setup_logger = cls.get_logger("logging_setup")
        setup_logger.info("=" * 60)
        setup_logger.info("PROJECT LOGGING INITIALIZED")
        setup_logger.info("=" * 60)
        setup_logger.info("Log Level: %s", log_level)
        setup_logger.info("Log Directory: %s", cls._log_dir.absolute())
        setup_logger.info("Console Output: %s", console_output)
        setup_logger.info("File Output: %s", file_output)
        if file_output:
            setup_logger.info("Log File: %s", log_file.name)
        setup_logger.info("=" * 60)

    @classmethod
    def get_logger(cls, name: str = None) -> logging.Logger:
        """
        Get a logger instance for a specific module/component.
        
        Args:
            name: Logger name. If None, uses the calling module's name.
        
        Returns:
            Configured logger instance
        """
        if not cls._base_config_set:
            # Auto-setup with defaults if not already configured
            cls.setup_project_logging()

        if name is None:
            # Get the calling module's name
            frame = inspect.currentframe().f_back
            name = frame.f_globals.get('__name__', 'unknown')

        if name not in cls._loggers:
            logger = logging.getLogger(name)
            cls._loggers[name] = logger

            # Log the logger creation
            logger.info("📝 Logger '%s' initialized", name)

        return cls._loggers[name]

    @classmethod
    def log_function_start(cls, func_name: str, **kwargs) -> logging.Logger:
        """Helper to log function start with parameters."""
        logger = cls.get_logger()
        logger.info("=" * 50)
        logger.info(" STARTING: %s", func_name)
        logger.info("=" * 50)

        if kwargs:
            logger.info("Parameters:")
            for key, value in kwargs.items():
                # Don't log sensitive information
                sensitive_terms = ('password', 'token', 'key', 'secret')
                if any(t in key.lower() for t in sensitive_terms):
                    value = "***HIDDEN***"
                logger.info("  %s: %s", key, value)

        return logger

    @classmethod
    def log_function_end(cls, func_name: str, success: bool = True, **results) -> logging.Logger:
        """Helper to log function completion."""
        logger = cls.get_logger()

        if success:
            logger.info("=" * 50)
            logger.info("COMPLETED: %s", func_name)
        else:
            logger.error("=" * 50)
            logger.error("FAILED: %s", func_name)

        if results:
            logger.info("Results:")
            for key, value in results.items():
                logger.info("  %s: %s", key, value)

        logger.info("=" * 50)
        return logger

# Convenience functions for easy importing
def get_logger(name: str = None) -> logging.Logger:
    """Get a project logger instance."""
    return ProjectLogger.get_logger(name)

def setup_logging(**kwargs):
    """Set up project-wide logging configuration."""
    return ProjectLogger.setup_project_logging(**kwargs)

def log_function_start(func_name: str, **kwargs) -> logging.Logger:
    """Log function start with parameters."""
    return ProjectLogger.log_function_start(func_name, **kwargs)

def log_function_end(func_name: str, success: bool = True, **results) -> logging.Logger:
    """Log function completion."""
    return ProjectLogger.log_function_end(func_name, success, **results)
