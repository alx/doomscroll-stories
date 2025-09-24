"""
Logging configuration for the Doom-scrolling Storyteller application.
Implements Python logging best practices with structured logging for LLM requests/responses.
"""

import os
import json
import logging
import logging.config
from datetime import datetime
from typing import Dict, Any


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        # Create base log entry
        log_entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }

        # Add extra fields if present
        if hasattr(record, 'request_id'):
            log_entry['request_id'] = record.request_id
        if hasattr(record, 'session_id'):
            log_entry['session_id'] = record.session_id
        if hasattr(record, 'model'):
            log_entry['model'] = record.model
        if hasattr(record, 'prompt_length'):
            log_entry['prompt_length'] = record.prompt_length
        if hasattr(record, 'response_length'):
            log_entry['response_length'] = record.response_length
        if hasattr(record, 'duration_ms'):
            log_entry['duration_ms'] = record.duration_ms
        if hasattr(record, 'active_badges'):
            log_entry['active_badges'] = record.active_badges
        if hasattr(record, 'active_characters'):
            log_entry['active_characters'] = record.active_characters
        if hasattr(record, 'prompt_preview'):
            log_entry['prompt_preview'] = record.prompt_preview
        if hasattr(record, 'response_preview'):
            log_entry['response_preview'] = record.response_preview
        if hasattr(record, 'error'):
            log_entry['error'] = record.error
        if hasattr(record, 'status_code'):
            log_entry['status_code'] = record.status_code

        # Add session debugging fields
        if hasattr(record, 'current_segments_count'):
            log_entry['current_segments_count'] = record.current_segments_count
        if hasattr(record, 'segments_count_before'):
            log_entry['segments_count_before'] = record.segments_count_before
        if hasattr(record, 'segments_count_after'):
            log_entry['segments_count_after'] = record.segments_count_after
        if hasattr(record, 'segment_added_successfully'):
            log_entry['segment_added_successfully'] = record.segment_added_successfully
        if hasattr(record, 'story_segments_preview'):
            log_entry['story_segments_preview'] = record.story_segments_preview
        if hasattr(record, 'has_story_summary'):
            log_entry['has_story_summary'] = record.has_story_summary
        if hasattr(record, 'session_contents'):
            log_entry['session_contents'] = record.session_contents
        if hasattr(record, 'session_keys'):
            log_entry['session_keys'] = record.session_keys
        if hasattr(record, 'segment_number'):
            log_entry['segment_number'] = record.segment_number
        if hasattr(record, 'segment_length'):
            log_entry['segment_length'] = record.segment_length
        if hasattr(record, 'total_segments'):
            log_entry['total_segments'] = record.total_segments
        if hasattr(record, 'new_segment_preview'):
            log_entry['new_segment_preview'] = record.new_segment_preview
        if hasattr(record, 'session_size_bytes'):
            log_entry['session_size_bytes'] = record.session_size_bytes
        if hasattr(record, 'session_size_kb'):
            log_entry['session_size_kb'] = record.session_size_kb

        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)

        return json.dumps(log_entry, ensure_ascii=False)


def get_logging_config() -> Dict[str, Any]:
    """Get logging configuration dictionary based on environment variables."""

    # Environment variables with defaults
    log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
    log_to_file = os.getenv('LOG_TO_FILE', 'true').lower() == 'true'
    log_directory = os.getenv('LOG_DIRECTORY', 'logs')
    log_max_bytes = int(os.getenv('LOG_MAX_BYTES', '10485760'))  # 10MB default
    log_backup_count = int(os.getenv('LOG_BACKUP_COUNT', '5'))

    # Ensure log directory exists
    if log_to_file and not os.path.exists(log_directory):
        os.makedirs(log_directory, exist_ok=True)

    # Base configuration
    config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'json': {
                '()': JSONFormatter,
            },
            'console': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
            'colored': {
                '()': 'colorlog.ColoredFormatter',
                'format': '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S',
                'log_colors': {
                    'DEBUG': 'cyan',
                    'INFO': 'green',
                    'WARNING': 'yellow',
                    'ERROR': 'red',
                    'CRITICAL': 'red,bg_white',
                }
            }
        },
        'handlers': {
            'console': {
                'level': log_level,
                'class': 'logging.StreamHandler',
                'formatter': 'colored' if 'colorlog' in globals() else 'console',
                'stream': 'ext://sys.stdout'
            }
        },
        'loggers': {
            'llm_requests': {
                'level': log_level,
                'handlers': ['console'],
                'propagate': False
            },
            'app': {
                'level': log_level,
                'handlers': ['console'],
                'propagate': False
            }
        },
        'root': {
            'level': log_level,
            'handlers': ['console']
        }
    }

    # Add file handlers if enabled
    if log_to_file:
        # Main application log
        config['handlers']['file_app'] = {
            'level': log_level,
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': os.path.join(log_directory, 'app.log'),
            'maxBytes': log_max_bytes,
            'backupCount': log_backup_count,
            'formatter': 'json',
            'encoding': 'utf-8'
        }

        # LLM requests log (structured JSON)
        config['handlers']['file_llm'] = {
            'level': log_level,
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': os.path.join(log_directory, 'llm_requests.log'),
            'maxBytes': log_max_bytes,
            'backupCount': log_backup_count,
            'formatter': 'json',
            'encoding': 'utf-8'
        }

        # Error log (errors only)
        config['handlers']['file_error'] = {
            'level': 'ERROR',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': os.path.join(log_directory, 'errors.log'),
            'maxBytes': log_max_bytes,
            'backupCount': log_backup_count,
            'formatter': 'json',
            'encoding': 'utf-8'
        }

        # Add file handlers to loggers
        config['loggers']['llm_requests']['handlers'].extend(['file_llm', 'file_error'])
        config['loggers']['app']['handlers'].extend(['file_app', 'file_error'])
        config['root']['handlers'].extend(['file_app', 'file_error'])

    return config


def setup_logging():
    """Initialize logging configuration."""
    try:
        # Try to import colorlog for colored console output
        global colorlog
        import colorlog
    except ImportError:
        colorlog = None

    # Get configuration and apply it
    config = get_logging_config()
    logging.config.dictConfig(config)

    # Log setup completion
    logger = logging.getLogger('app')
    logger.info("Logging configuration initialized", extra={
        'log_level': os.getenv('LOG_LEVEL', 'INFO'),
        'log_to_file': os.getenv('LOG_TO_FILE', 'true'),
        'log_directory': os.getenv('LOG_DIRECTORY', 'logs')
    })


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the specified name."""
    return logging.getLogger(name)


def truncate_text(text: str, max_length: int = None) -> str:
    """Truncate text for logging to prevent oversized log entries."""
    if max_length is None:
        max_length = int(os.getenv('LOG_TRUNCATE_LENGTH', '500'))

    if len(text) <= max_length:
        return text

    return text[:max_length-3] + '...'


def sanitize_for_logging(data: Any) -> Any:
    """Sanitize data for safe logging (remove/mask sensitive information)."""
    if isinstance(data, str):
        # Basic sanitization - could be expanded based on needs
        return truncate_text(data)
    elif isinstance(data, dict):
        return {k: sanitize_for_logging(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_for_logging(item) for item in data]
    else:
        return data