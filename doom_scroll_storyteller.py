#!/usr/bin/env python3
"""
Doom-scrolling storytelling web app with Bootstrap badge-based user interaction.
Uses OpenRouter API for story generation.
"""

import os
import sys
import json
import requests
import time
import uuid
import re
import hashlib
from typing import List, Dict, Optional, Tuple
from flask import Flask, render_template, request, jsonify, session
from flask_session import Session
from dotenv import load_dotenv

from logging_config import setup_logging, get_logger, truncate_text, sanitize_for_logging

# Load environment variables from .env file
load_dotenv()

# Initialize logging
setup_logging()

# Try to import SD WebUI API (optional dependency)
try:
    from webuiapi import WebUIApi
    SDWEBUI_AVAILABLE = True
except ImportError:
    temp_logger = get_logger('app')
    temp_logger.warning("SD WebUI API not available - image generation will be disabled")
    SDWEBUI_AVAILABLE = False

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'your-secret-key-change-this')

# Configure Flask-Session for server-side sessions
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = os.path.join(os.path.dirname(__file__), 'sessions')
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = True
app.config['SESSION_KEY_PREFIX'] = 'doomscroll:'

# Initialize Flask-Session
Session(app)

# Get application logger
app_logger = get_logger('app')

# OpenRouter configuration
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY', 'your-openrouter-api-key')
OPENROUTER_BASE_URL = 'https://openrouter.ai/api/v1'

def load_model_configuration() -> Dict[str, str]:
    """Load and validate model configuration from environment variables"""

    # Legacy support - if old OPENROUTER_MODEL is set, use it as default
    legacy_model = os.getenv('OPENROUTER_MODEL', 'openai/gpt-4o-mini')

    # Load model configuration with fallbacks
    default_model = os.getenv('OPENROUTER_MODEL_DEFAULT', legacy_model)

    model_config = {
        'story_generation': os.getenv('OPENROUTER_MODEL_STORY_GENERATION', default_model),
        'story_summary': os.getenv('OPENROUTER_MODEL_STORY_SUMMARY', default_model),
        'storyline_options': os.getenv('OPENROUTER_MODEL_STORYLINE_OPTIONS', default_model),
        'default': default_model
    }

    # Log model configuration
    app_logger = get_logger('app')
    app_logger.info("Model configuration loaded", extra={
        'event': 'model_config_loaded',
        'models': model_config,
        'using_legacy_fallback': legacy_model != default_model
    })

    return model_config

# Load model configuration
MODEL_CONFIG = load_model_configuration()

# Story summarization configuration
SUMMARY_WINDOW_SIZE = int(os.getenv('SUMMARY_WINDOW_SIZE', '5'))  # Summarize after this many segments
RECENT_SEGMENTS_COUNT = int(os.getenv('RECENT_SEGMENTS_COUNT', '2'))  # Keep this many recent segments with summary

# Session optimization configuration
MAX_SESSION_SIZE_KB = int(os.getenv('MAX_SESSION_SIZE_KB', '50'))  # Max session size in KB before optimization
MAX_SEGMENTS_IN_SESSION = int(os.getenv('MAX_SEGMENTS_IN_SESSION', '10'))  # Max segments to keep in session

# SD WebUI configuration
SDWEBUI_ENABLED = os.getenv('SDWEBUI_ENABLED', 'true').lower() == 'true'
SDWEBUI_HOST = os.getenv('SDWEBUI_HOST', '127.0.0.1')
SDWEBUI_PORT = int(os.getenv('SDWEBUI_PORT', '7860'))
SDWEBUI_USE_HTTPS = os.getenv('SDWEBUI_USE_HTTPS', 'false').lower() == 'true'
SDWEBUI_TIMEOUT = int(os.getenv('SDWEBUI_TIMEOUT', '60'))

# Image generation parameters
IMAGE_STYLE_PROMPT = os.getenv('IMAGE_STYLE_PROMPT', 'ornamental border, decorative chapter heading, elegant manuscript illumination, medieval manuscript style, intricate detailed border design, fantasy art, high quality')
IMAGE_WIDTH = int(os.getenv('IMAGE_WIDTH', '512'))
IMAGE_HEIGHT = int(os.getenv('IMAGE_HEIGHT', '256'))
IMAGE_STEPS = int(os.getenv('IMAGE_STEPS', '4'))  # Updated for fast model: 3-5 steps
IMAGE_CFG_SCALE = float(os.getenv('IMAGE_CFG_SCALE', '1.5'))  # Updated for fast model: 1-2.25
IMAGE_MODEL = os.getenv('IMAGE_MODEL', 'turbovisionxlSuperFastXLBasedOnNew_tvxlV431Bakedvae.safetensors')
IMAGE_SAMPLER = os.getenv('IMAGE_SAMPLER', 'DPM++ SDE Karras')  # DPM++ SDE or DPM++ SDE Karras

def load_badge_categories_from_env() -> Dict[str, List[str]]:
    """Load badge categories from environment variables"""
    categories = {}

    # Define category mappings from env var names to category keys
    env_mappings = {
        'BADGE_CATEGORIES_MOOD': 'mood',
        'BADGE_CATEGORIES_GENRE': 'genre',
        'BADGE_CATEGORIES_INTENSITY': 'intensity',
        'BADGE_CATEGORIES_SPICY': 'spicy'
    }

    # Load each category from environment
    for env_var, category_key in env_mappings.items():
        env_value = os.getenv(env_var, '')
        if env_value:
            # Split by comma and clean up whitespace
            categories[category_key] = [badge.strip() for badge in env_value.split(',') if badge.strip()]
        else:
            # Fallback defaults for each category
            if category_key == 'mood':
                categories[category_key] = ['mysterious', 'romantic', 'funny', 'dark', 'heartwarming']
            elif category_key == 'genre':
                categories[category_key] = ['sci-fi', 'fantasy', 'horror', 'thriller', 'western']
            elif category_key == 'intensity':
                categories[category_key] = ['action-packed', 'slow-burn', 'explosive', 'intimate', 'epic']
            elif category_key == 'spicy':
                categories[category_key] = ['passionate', 'seductive', 'dangerous', 'forbidden']

    return categories

def load_character_categories_from_env() -> Dict[str, List[str]]:
    """Load character categories from environment variables"""
    categories = {}

    # Define category mappings from env var names to category keys
    env_mappings = {
        'CHARACTER_CATEGORIES_ARCHETYPES': 'archetypes',
        'CHARACTER_CATEGORIES_TRAITS': 'traits',
        'CHARACTER_CATEGORIES_RELATIONSHIPS': 'relationships'
    }

    # Load each category from environment
    for env_var, category_key in env_mappings.items():
        env_value = os.getenv(env_var, '')
        if env_value:
            # Split by comma and clean up whitespace
            categories[category_key] = [character.strip() for character in env_value.split(',') if character.strip()]
        else:
            # Fallback defaults for each category
            if category_key == 'archetypes':
                categories[category_key] = ['hero', 'villain', 'mentor', 'trickster', 'innocent']
            elif category_key == 'traits':
                categories[category_key] = ['mysterious stranger', 'reluctant hero', 'femme fatale', 'comic relief']
            elif category_key == 'relationships':
                categories[category_key] = ['enemies-to-lovers', 'mentor-student', 'rivals', 'siblings']

    return categories

def optimize_session_size(session_dict: dict, app_logger) -> dict:
    """Optimize session size by removing old segments when limits are exceeded"""

    # Calculate current session size
    session_size_bytes = len(str(session_dict).encode('utf-8'))
    session_size_kb = session_size_bytes / 1024

    story_segments = session_dict.get('story_segments', [])
    segments_count = len(story_segments)

    # Check if optimization is needed
    needs_optimization = (
        session_size_kb > MAX_SESSION_SIZE_KB or
        segments_count > MAX_SEGMENTS_IN_SESSION
    )

    if not needs_optimization:
        return session_dict

    app_logger.info("Session optimization triggered", extra={
        'event': 'session_optimization_start',
        'session_id': session_dict.get('session_id', 'unknown'),
        'current_size_kb': round(session_size_kb, 2),
        'current_segments': segments_count,
        'max_size_kb': MAX_SESSION_SIZE_KB,
        'max_segments': MAX_SEGMENTS_IN_SESSION
    })

    # Create a new optimized session dict
    optimized_session = session_dict.copy()

    if segments_count > RECENT_SEGMENTS_COUNT:
        # Keep only the most recent segments
        optimized_session['story_segments'] = story_segments[-RECENT_SEGMENTS_COUNT:]

        # Generate/update summary if needed - avoid circular import by using global
        if not session_dict.get('story_summary') and segments_count >= SUMMARY_WINDOW_SIZE:
            # We'll generate the summary later in the flow to avoid circular imports
            pass

    # Calculate new size
    new_size_bytes = len(str(optimized_session).encode('utf-8'))
    new_size_kb = new_size_bytes / 1024

    app_logger.info("Session optimization completed", extra={
        'event': 'session_optimization_complete',
        'session_id': session_dict.get('session_id', 'unknown'),
        'old_size_kb': round(session_size_kb, 2),
        'new_size_kb': round(new_size_kb, 2),
        'old_segments': segments_count,
        'new_segments': len(optimized_session.get('story_segments', [])),
        'size_reduction_kb': round(session_size_kb - new_size_kb, 2)
    })

    return optimized_session

class StoryGenerator:
    def __init__(self, api_key: str, model_config: Dict[str, str] = None):
        self.api_key = api_key
        self.model_config = model_config or MODEL_CONFIG
        self.base_url = OPENROUTER_BASE_URL
        self.logger = get_logger('llm_requests')
        self.include_responses = os.getenv('LOG_INCLUDE_RESPONSES', 'true').lower() == 'true'

    def _get_model_for_request_type(self, request_type: str) -> str:
        """Get the appropriate model for a specific request type"""
        return self.model_config.get(request_type, self.model_config.get('default', 'openai/gpt-4o-mini'))

    def _validate_story_completion(self, content: str) -> bool:
        """Check if story content ends with proper sentence completion"""
        if not content or len(content.strip()) == 0:
            return False

        content = content.strip()
        # Check if it ends with proper punctuation
        proper_endings = ['.', '!', '?', '"', "'", '"']

        return any(content.endswith(ending) for ending in proper_endings)

    def generate_story_segment(self, story_context: str, active_badges: List[str], active_characters: List[str] = None) -> str:
        """Generate next story segment influenced by active badges and characters"""

        # Generate unique request ID for tracking
        request_id = str(uuid.uuid4())
        session_id = session.get('session_id', 'unknown')
        start_time = time.time()
        model = self._get_model_for_request_type('story_generation')

        # Structure the prompt with sections
        sections = []

        if active_badges:
            sections.append(f"STORY INFLUENCES: {', '.join(active_badges)}")

        if active_characters:
            sections.append(f"CHARACTER ELEMENTS: {', '.join(active_characters)}")

        influences_text = "\n".join(sections) if sections else "Continue naturally with creative freedom"

        prompt = f"""Continue this story with a new segment of 200-300 words.

CURRENT STORY CONTEXT:
{story_context}

CREATIVE INFLUENCES:
{influences_text}

INSTRUCTIONS:
- Write the next part of the story incorporating the influences naturally
- Maintain narrative flow and character consistency
- Keep the reader engaged and wanting more
- Focus on vivid descriptions and compelling dialogue when appropriate
- Always end with a complete sentence - never cut off mid-thought
- Ensure the segment is exactly 200-300 words with proper conclusion"""

        # Log request start
        self.logger.info("Starting LLM request", extra={
            'event': 'llm_request_start',
            'request_type': 'story_generation',
            'request_id': request_id,
            'session_id': session_id,
            'model': model,
            'prompt_length': len(prompt),
            'active_badges': sanitize_for_logging(active_badges or []),
            'active_characters': sanitize_for_logging(active_characters or []),
            'prompt_preview': truncate_text(prompt, 200) if self.include_responses else None
        })

        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }

        data = {
            'model': model,
            'messages': [
                {
                    'role': 'system',
                    'content': 'You are a creative storyteller who writes engaging, continuous narratives. Each segment should be 200-300 words and flow naturally from the previous content.'
                },
                {'role': 'user', 'content': prompt}
            ],
            'max_tokens': 600,
            'temperature': 0.8
        }

        try:
            response = requests.post(
                f'{self.base_url}/chat/completions',
                headers=headers,
                json=data,
                timeout=30
            )

            duration_ms = int((time.time() - start_time) * 1000)

            response.raise_for_status()
            result = response.json()
            generated_content = result['choices'][0]['message']['content'].strip()

            # Validate response completion
            is_complete = self._validate_story_completion(generated_content)
            if not is_complete:
                self.logger.warning("Generated story segment may be incomplete", extra={
                    'event': 'story_segment_incomplete',
                    'request_id': request_id,
                    'session_id': session_id,
                    'content_preview': truncate_text(generated_content, 100),
                    'ends_with': generated_content[-20:] if len(generated_content) > 20 else generated_content
                })

            # Process think tags before returning
            processed_content, think_sections = self.process_think_tags(generated_content)

            # Log successful response
            self.logger.info("LLM request completed successfully", extra={
                'event': 'llm_request_success',
                'request_type': 'story_generation',
                'request_id': request_id,
                'session_id': session_id,
                'model': model,
                'duration_ms': duration_ms,
                'status_code': response.status_code,
                'response_length': len(generated_content),
                'processed_length': len(processed_content),
                'think_sections_count': len(think_sections),
                'response_preview': truncate_text(generated_content, 200) if self.include_responses else None,
                'is_complete': is_complete
            })

            return processed_content, think_sections

        except requests.exceptions.RequestException as e:
            duration_ms = int((time.time() - start_time) * 1000)

            # Log request error
            self.logger.error("LLM request failed", extra={
                'event': 'llm_request_error',
                'request_type': 'story_generation',
                'request_id': request_id,
                'session_id': session_id,
                'model': model,
                'duration_ms': duration_ms,
                'error': str(e),
                'status_code': getattr(e.response, 'status_code', None) if hasattr(e, 'response') else None
            })

            return f"Error generating story: {str(e)}. Please check your OpenRouter API key and try again.", []

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)

            # Log unexpected error
            self.logger.error("Unexpected error during LLM request", extra={
                'event': 'llm_request_unexpected_error',
                'request_type': 'story_generation',
                'request_id': request_id,
                'session_id': session_id,
                'model': model,
                'duration_ms': duration_ms,
                'error': str(e)
            }, exc_info=True)

            return f"Error generating story: {str(e)}. Please check your OpenRouter API key and try again.", []

    def generate_story_summary(self, story_segments: List[str]) -> str:
        """Generate a concise summary of the story so far"""

        # Generate unique request ID for tracking
        request_id = str(uuid.uuid4())
        session_id = session.get('session_id', 'unknown')
        start_time = time.time()
        model = self._get_model_for_request_type('story_summary')

        # Join all segments for summarization
        full_story = "\n\n".join(story_segments)

        prompt = f"""Please create a concise summary of this story that captures:
1. Main characters and their key traits/relationships
2. Current setting and situation
3. Major plot developments and conflicts
4. Current story momentum and direction

Keep the summary focused and under 300 words, emphasizing elements that are crucial for continuing the narrative coherently.

STORY TO SUMMARIZE:
{full_story}

SUMMARY:"""

        # Log request start
        self.logger.info("Starting story summary LLM request", extra={
            'event': 'llm_summary_start',
            'request_type': 'story_summary',
            'request_id': request_id,
            'session_id': session_id,
            'model': model,
            'segments_count': len(story_segments),
            'total_length': len(full_story)
        })

        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }

        data = {
            'model': model,
            'messages': [
                {
                    'role': 'system',
                    'content': 'You are an expert at creating concise, comprehensive story summaries that preserve narrative continuity and character development.'
                },
                {'role': 'user', 'content': prompt}
            ],
            'max_tokens': 350,
            'temperature': 0.3  # Lower temperature for consistent summaries
        }

        try:
            response = requests.post(
                f'{self.base_url}/chat/completions',
                headers=headers,
                json=data,
                timeout=30
            )

            duration_ms = int((time.time() - start_time) * 1000)

            response.raise_for_status()
            result = response.json()
            generated_summary = result['choices'][0]['message']['content'].strip()

            # Log successful response
            self.logger.info("Story summary LLM request completed successfully", extra={
                'event': 'llm_summary_success',
                'request_type': 'story_summary',
                'request_id': request_id,
                'session_id': session_id,
                'model': model,
                'duration_ms': duration_ms,
                'status_code': response.status_code,
                'summary_length': len(generated_summary)
            })

            return generated_summary

        except requests.exceptions.RequestException as e:
            duration_ms = int((time.time() - start_time) * 1000)

            # Log request error
            self.logger.error("Story summary LLM request failed", extra={
                'event': 'llm_summary_error',
                'request_type': 'story_summary',
                'request_id': request_id,
                'session_id': session_id,
                'model': model,
                'duration_ms': duration_ms,
                'error': str(e),
                'status_code': getattr(e.response, 'status_code', None) if hasattr(e, 'response') else None
            })

            return "Error generating story summary. Using recent segments for context."

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)

            # Log unexpected error
            self.logger.error("Unexpected error during story summary LLM request", extra={
                'event': 'llm_summary_unexpected_error',
                'request_type': 'story_summary',
                'request_id': request_id,
                'session_id': session_id,
                'model': model,
                'duration_ms': duration_ms,
                'error': str(e)
            }, exc_info=True)

            return "Error generating story summary. Using recent segments for context."

    def generate_initial_story(self, active_badges: List[str], active_characters: List[str] = None) -> str:
        """Generate the first story segment with structured influences"""

        # Generate unique request ID for tracking
        request_id = str(uuid.uuid4())
        session_id = session.get('session_id', 'unknown')
        start_time = time.time()
        model = self._get_model_for_request_type('story_generation')

        # Structure the prompt with sections
        sections = []

        if active_badges:
            sections.append(f"STORY INFLUENCES: {', '.join(active_badges)}")

        if active_characters:
            sections.append(f"CHARACTER ELEMENTS: {', '.join(active_characters)}")

        influences_text = "\n".join(sections) if sections else "STORY INFLUENCES: mystery and adventure"

        prompt = f"""Write the beginning of an engaging story (200-300 words).

CREATIVE INFLUENCES:
{influences_text}

INSTRUCTIONS:
- Create an intriguing opening that hooks readers immediately
- Set up compelling characters and setting
- Establish initial conflict, mystery, or tension
- Incorporate the influences naturally into the narrative
- End with a moment that makes readers want to continue
- Use vivid descriptions and strong character voice
- Always end with a complete sentence - never cut off mid-thought
- Ensure the segment is exactly 200-300 words with proper conclusion"""

        # Log request start
        self.logger.info("Starting initial story LLM request", extra={
            'event': 'llm_initial_story_start',
            'request_type': 'story_generation',
            'request_id': request_id,
            'session_id': session_id,
            'model': model,
            'prompt_length': len(prompt),
            'active_badges': sanitize_for_logging(active_badges or []),
            'active_characters': sanitize_for_logging(active_characters or []),
            'prompt_preview': truncate_text(prompt, 200) if self.include_responses else None
        })

        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }

        data = {
            'model': model,
            'messages': [
                {
                    'role': 'system',
                    'content': 'You are a creative storyteller who writes engaging story openings. Create compelling beginnings that hook readers immediately.'
                },
                {'role': 'user', 'content': prompt}
            ],
            'max_tokens': 800,
            'temperature': 0.8
        }

        # Validate API key before making request
        if not self.api_key or self.api_key == 'your-openrouter-api-key':
            self.logger.error("Invalid or missing API key", extra={
                'event': 'invalid_api_key',
                'request_type': 'story_generation',
                'request_id': request_id,
                'session_id': session_id,
                'api_key_placeholder': self.api_key == 'your-openrouter-api-key'
            })
            return "Invalid or missing OpenRouter API key. Please set your OPENROUTER_API_KEY environment variable.", []

        self.logger.info("About to send HTTP request to OpenRouter", extra={
            'event': 'http_request_about_to_start',
            'request_type': 'story_generation',
            'request_id': request_id,
            'session_id': session_id,
            'url': f'{self.base_url}/chat/completions',
            'model': model,
            'max_tokens': data['max_tokens'],
            'temperature': data['temperature'],
            'api_key_length': len(self.api_key) if self.api_key else 0,
            'api_key_starts_with': self.api_key[:10] + '...' if self.api_key and len(self.api_key) > 10 else 'N/A'
        })

        try:
            self.logger.info("Making HTTP request to OpenRouter API", extra={
                'event': 'http_request_start',
                'request_type': 'story_generation',
                'request_id': request_id,
                'session_id': session_id,
                'timeout': 30
            })

            response = requests.post(
                f'{self.base_url}/chat/completions',
                headers=headers,
                json=data,
                timeout=30
            )

            self.logger.info("HTTP request completed", extra={
                'event': 'http_request_complete',
                'request_type': 'story_generation',
                'request_id': request_id,
                'session_id': session_id,
                'status_code': response.status_code
            })

            duration_ms = int((time.time() - start_time) * 1000)

            self.logger.info("Processing HTTP response", extra={
                'event': 'response_processing_start',
                'request_type': 'story_generation',
                'request_id': request_id,
                'session_id': session_id,
                'status_code': response.status_code,
                'content_length': len(response.content) if response.content else 0
            })

            response.raise_for_status()

            self.logger.info("Parsing JSON response", extra={
                'event': 'json_parsing_start',
                'request_type': 'story_generation',
                'request_id': request_id,
                'session_id': session_id
            })

            result = response.json()

            self.logger.info("Extracting generated content", extra={
                'event': 'content_extraction_start',
                'request_type': 'story_generation',
                'request_id': request_id,
                'session_id': session_id,
                'choices_count': len(result.get('choices', [])),
                'has_usage_info': 'usage' in result
            })

            generated_content = result['choices'][0]['message']['content'].strip()

            # Validate response completion
            is_complete = self._validate_story_completion(generated_content)
            if not is_complete:
                self.logger.warning("Generated initial story may be incomplete", extra={
                    'event': 'initial_story_incomplete',
                    'request_id': request_id,
                    'session_id': session_id,
                    'content_preview': truncate_text(generated_content, 100),
                    'ends_with': generated_content[-20:] if len(generated_content) > 20 else generated_content
                })

            # Process think tags before returning
            processed_content, think_sections = self.process_think_tags(generated_content)

            # Log successful response
            self.logger.info("Initial story LLM request completed successfully", extra={
                'event': 'llm_initial_story_success',
                'request_type': 'story_generation',
                'request_id': request_id,
                'session_id': session_id,
                'model': model,
                'duration_ms': duration_ms,
                'status_code': response.status_code,
                'response_length': len(generated_content),
                'processed_length': len(processed_content),
                'think_sections_count': len(think_sections),
                'response_preview': truncate_text(generated_content, 200) if self.include_responses else None,
                'is_complete': is_complete
            })

            return processed_content, think_sections

        except requests.exceptions.Timeout as e:
            duration_ms = int((time.time() - start_time) * 1000)

            # Log timeout error specifically
            self.logger.error("Initial story LLM request timed out", extra={
                'event': 'llm_initial_story_timeout',
                'request_type': 'story_generation',
                'request_id': request_id,
                'session_id': session_id,
                'model': model,
                'duration_ms': duration_ms,
                'timeout_seconds': 30,
                'error': str(e)
            })

            return f"Request timed out after 30 seconds. The model '{model}' may be slow or unavailable. Please try again.", []

        except requests.exceptions.ConnectionError as e:
            duration_ms = int((time.time() - start_time) * 1000)

            # Log connection error specifically
            self.logger.error("Initial story LLM request connection failed", extra={
                'event': 'llm_initial_story_connection_error',
                'request_type': 'story_generation',
                'request_id': request_id,
                'session_id': session_id,
                'model': model,
                'duration_ms': duration_ms,
                'error': str(e),
                'url': f'{self.base_url}/chat/completions'
            })

            return f"Connection failed to OpenRouter API. Please check your internet connection and try again. Error: {str(e)}", []

        except requests.exceptions.RequestException as e:
            duration_ms = int((time.time() - start_time) * 1000)

            # Log general request error
            self.logger.error("Initial story LLM request failed", extra={
                'event': 'llm_initial_story_error',
                'request_type': 'story_generation',
                'request_id': request_id,
                'session_id': session_id,
                'model': model,
                'duration_ms': duration_ms,
                'error': str(e),
                'error_type': type(e).__name__,
                'status_code': getattr(e.response, 'status_code', None) if hasattr(e, 'response') else None,
                'response_text': e.response.text if hasattr(e, 'response') and e.response else None
            })

            return f"Error generating story: {str(e)}. Please check your OpenRouter API key and try again.", []

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)

            # Log unexpected error
            self.logger.error("Unexpected error during initial story LLM request", extra={
                'event': 'llm_initial_story_unexpected_error',
                'request_type': 'story_generation',
                'request_id': request_id,
                'session_id': session_id,
                'model': model,
                'duration_ms': duration_ms,
                'error': str(e)
            }, exc_info=True)

            return f"Error generating story: {str(e)}. Please check your OpenRouter API key and try again.", []

    def process_think_tags(self, content: str) -> Tuple[str, List[Dict[str, str]]]:
        """Extract think tags from content and replace with placeholders"""

        think_sections = []
        think_counter = 0

        # Debug: Log raw content for think tag analysis
        self.logger.info("Starting think tag processing", extra={
            'event': 'think_tag_processing_start',
            'session_id': session.get('session_id', 'unknown'),
            'content_length': len(content),
            'content_preview': content[:200] + '...' if len(content) > 200 else content,
            'has_unicode_think': '\u25c1think\u25b7' in content,
            'has_html_think': '<think>' in content
        })

        def replace_think_tag(match):
            nonlocal think_counter, think_sections
            think_counter += 1
            think_id = f"think_{think_counter}"
            think_content = match.group(1).strip()

            # Store the think section
            think_sections.append({
                'id': think_id,
                'content': think_content
            })

            # Replace with placeholder
            return f"__THINK_PLACEHOLDER_{think_id}__"

        # Use regex to find and replace think tags (handles both HTML and Unicode formats)
        # This pattern matches both <think></think> and ◁think▷◁/think▷ formats
        think_pattern = r'(?:(?:\u25c1|◁)think(?:\u25b7|▷)|<think>)(.*?)(?:(?:\u25c1|◁)/think(?:\u25b7|▷)|</think>)'

        # Debug: Test if pattern matches before substitution
        matches = re.findall(think_pattern, content, flags=re.DOTALL | re.IGNORECASE)
        self.logger.info("Think tag regex analysis", extra={
            'event': 'think_tag_regex_analysis',
            'session_id': session.get('session_id', 'unknown'),
            'pattern': think_pattern,
            'matches_found': len(matches),
            'match_previews': [match[:100] + '...' if len(match) > 100 else match for match in matches[:3]]
        })

        processed_content = re.sub(think_pattern, replace_think_tag, content, flags=re.DOTALL | re.IGNORECASE)

        # Log think tag processing results
        self.logger.info("Think tag processing completed", extra={
            'event': 'think_tags_processed',
            'session_id': session.get('session_id', 'unknown'),
            'think_sections_count': len(think_sections),
            'think_ids': [section['id'] for section in think_sections] if think_sections else [],
            'content_changed': len(processed_content) != len(content),
            'original_length': len(content),
            'processed_length': len(processed_content)
        })

        return processed_content, think_sections

    def extract_json_from_response(self, response_content: str) -> Optional[List[Dict[str, str]]]:
        """Extract and parse JSON array from LLM response with multiple fallback strategies"""

        session_id = session.get('session_id', 'unknown')

        # Strategy 0: Pre-process to remove think tags (NEW - handles the main issue)
        try:
            self.logger.info("Attempting Strategy 0: Remove think tags", extra={
                'event': 'json_extraction_strategy_0',
                'session_id': session_id,
                'original_length': len(response_content),
                'has_think_tags': '◁think▷' in response_content or '<think>' in response_content
            })

            # Remove think tags using the same pattern as process_think_tags
            think_pattern = r'(?:(?:\u25c1|◁)think(?:\u25b7|▷)|<think>)(.*?)(?:(?:\u25c1|◁)/think(?:\u25b7|▷)|</think>)'
            cleaned_content = re.sub(think_pattern, '', response_content, flags=re.DOTALL | re.IGNORECASE)
            cleaned_content = cleaned_content.strip()

            if cleaned_content != response_content:
                self.logger.info("Think tags removed", extra={
                    'event': 'think_tags_removed',
                    'session_id': session_id,
                    'cleaned_length': len(cleaned_content),
                    'size_reduction': len(response_content) - len(cleaned_content)
                })

                parsed = json.loads(cleaned_content)
                if isinstance(parsed, list) and len(parsed) >= 3:
                    self.logger.info("Strategy 0 successful", extra={
                        'event': 'json_extraction_success',
                        'strategy': 0,
                        'session_id': session_id
                    })
                    return parsed[:3]

        except (json.JSONDecodeError, ValueError) as e:
            self.logger.info("Strategy 0 failed", extra={
                'event': 'json_extraction_strategy_failed',
                'strategy': 0,
                'session_id': session_id,
                'error': str(e)
            })

        # Strategy 1: Try parsing as-is (clean JSON response)
        try:
            self.logger.info("Attempting Strategy 1: Parse as-is", extra={
                'event': 'json_extraction_strategy_1',
                'session_id': session_id
            })
            parsed = json.loads(response_content.strip())
            if isinstance(parsed, list) and len(parsed) >= 3:
                self.logger.info("Strategy 1 successful", extra={
                    'event': 'json_extraction_success',
                    'strategy': 1,
                    'session_id': session_id
                })
                return parsed[:3]
        except (json.JSONDecodeError, ValueError) as e:
            self.logger.info("Strategy 1 failed", extra={
                'event': 'json_extraction_strategy_failed',
                'strategy': 1,
                'session_id': session_id,
                'error': str(e)
            })

        # Strategy 2: Strip markdown code blocks
        try:
            self.logger.info("Attempting Strategy 2: Remove markdown", extra={
                'event': 'json_extraction_strategy_2',
                'session_id': session_id
            })
            # Remove ```json and ``` markers
            cleaned = re.sub(r'```(?:json)?\s*', '', response_content, flags=re.IGNORECASE)
            cleaned = re.sub(r'```\s*$', '', cleaned, flags=re.MULTILINE)
            parsed = json.loads(cleaned.strip())
            if isinstance(parsed, list) and len(parsed) >= 3:
                self.logger.info("Strategy 2 successful", extra={
                    'event': 'json_extraction_success',
                    'strategy': 2,
                    'session_id': session_id
                })
                return parsed[:3]
        except (json.JSONDecodeError, ValueError) as e:
            self.logger.info("Strategy 2 failed", extra={
                'event': 'json_extraction_strategy_failed',
                'strategy': 2,
                'session_id': session_id,
                'error': str(e)
            })

        # Strategy 3: Extract JSON array using regex (IMPROVED)
        try:
            self.logger.info("Attempting Strategy 3: Regex extraction", extra={
                'event': 'json_extraction_strategy_3',
                'session_id': session_id
            })
            # More robust pattern that handles nested structures and thinks tags
            json_pattern = r'\[(?:[^[\]{}]*\{[^{}]*\}[^[\]{}]*,?\s*){2,}\]'
            matches = re.findall(json_pattern, response_content, re.DOTALL)

            self.logger.info("Regex matches found", extra={
                'event': 'json_regex_matches',
                'session_id': session_id,
                'matches_count': len(matches),
                'matches_preview': matches[:2] if matches else []
            })

            for i, match in enumerate(matches):
                try:
                    # Clean the match first
                    cleaned_match = re.sub(think_pattern, '', match, flags=re.DOTALL | re.IGNORECASE)
                    parsed = json.loads(cleaned_match)
                    if isinstance(parsed, list) and len(parsed) >= 3:
                        self.logger.info("Strategy 3 successful", extra={
                            'event': 'json_extraction_success',
                            'strategy': 3,
                            'session_id': session_id,
                            'match_index': i
                        })
                        return parsed[:3]
                except (json.JSONDecodeError, ValueError):
                    continue
        except Exception as e:
            self.logger.info("Strategy 3 failed", extra={
                'event': 'json_extraction_strategy_failed',
                'strategy': 3,
                'session_id': session_id,
                'error': str(e)
            })

        # Strategy 4: Fix common JSON issues and retry
        try:
            self.logger.info("Attempting Strategy 4: Fix common issues", extra={
                'event': 'json_extraction_strategy_4',
                'session_id': session_id
            })
            # Replace single quotes with double quotes
            fixed_content = response_content.replace("'", '"')
            # Remove trailing commas
            fixed_content = re.sub(r',\s*}', '}', fixed_content)
            fixed_content = re.sub(r',\s*]', ']', fixed_content)
            # Remove markdown
            fixed_content = re.sub(r'```(?:json)?\s*', '', fixed_content, flags=re.IGNORECASE)
            fixed_content = re.sub(r'```\s*$', '', fixed_content, flags=re.MULTILINE)
            # Remove think tags
            fixed_content = re.sub(think_pattern, '', fixed_content, flags=re.DOTALL | re.IGNORECASE)

            parsed = json.loads(fixed_content.strip())
            if isinstance(parsed, list) and len(parsed) >= 3:
                self.logger.info("Strategy 4 successful", extra={
                    'event': 'json_extraction_success',
                    'strategy': 4,
                    'session_id': session_id
                })
                return parsed[:3]
        except (json.JSONDecodeError, ValueError) as e:
            self.logger.info("Strategy 4 failed", extra={
                'event': 'json_extraction_strategy_failed',
                'strategy': 4,
                'session_id': session_id,
                'error': str(e)
            })

        # Strategy 5: Extract individual objects and reconstruct array
        try:
            self.logger.info("Attempting Strategy 5: Extract individual objects", extra={
                'event': 'json_extraction_strategy_5',
                'session_id': session_id
            })
            # Find all JSON objects in the text (improved pattern)
            object_pattern = r'\{\s*["\']title["\'].*?\}'
            matches = re.findall(object_pattern, response_content, re.DOTALL | re.IGNORECASE)

            parsed_objects = []
            for match in matches:
                try:
                    # Fix quotes and parse individual object
                    fixed_obj = match.replace("'", '"')
                    fixed_obj = re.sub(r',\s*}', '}', fixed_obj)
                    # Remove think tags from object
                    fixed_obj = re.sub(think_pattern, '', fixed_obj, flags=re.DOTALL | re.IGNORECASE)
                    obj = json.loads(fixed_obj)
                    if 'title' in obj and 'description' in obj:
                        parsed_objects.append(obj)
                except (json.JSONDecodeError, ValueError):
                    continue

            if len(parsed_objects) >= 3:
                self.logger.info("Strategy 5 successful", extra={
                    'event': 'json_extraction_success',
                    'strategy': 5,
                    'session_id': session_id,
                    'objects_found': len(parsed_objects)
                })
                return parsed_objects[:3]  # Take first 3 valid objects

        except Exception as e:
            self.logger.info("Strategy 5 failed", extra={
                'event': 'json_extraction_strategy_failed',
                'strategy': 5,
                'session_id': session_id,
                'error': str(e)
            })

        # Strategy 6: Handle JSON embedded within think tags (NEW)
        try:
            self.logger.info("Attempting Strategy 6: Extract JSON from think tags", extra={
                'event': 'json_extraction_strategy_6',
                'session_id': session_id
            })

            # Find content within think tags that might contain JSON
            think_content_pattern = r'(?:(?:\u25c1|◁)think(?:\u25b7|▷)|<think>)(.*?)(?:(?:\u25c1|◁)/think(?:\u25b7|▷)|</think>)'
            think_matches = re.findall(think_content_pattern, response_content, re.DOTALL | re.IGNORECASE)

            for think_content in think_matches:
                # Try to find JSON array in think content
                json_pattern = r'\[(?:[^[\]{}]*\{[^{}]*\}[^[\]{}]*,?\s*){2,}\]'
                json_matches = re.findall(json_pattern, think_content, re.DOTALL)

                for json_match in json_matches:
                    try:
                        parsed = json.loads(json_match)
                        if isinstance(parsed, list) and len(parsed) >= 3:
                            self.logger.info("Strategy 6 successful", extra={
                                'event': 'json_extraction_success',
                                'strategy': 6,
                                'session_id': session_id
                            })
                            return parsed[:3]
                    except (json.JSONDecodeError, ValueError):
                        continue

        except Exception as e:
            self.logger.info("Strategy 6 failed", extra={
                'event': 'json_extraction_strategy_failed',
                'strategy': 6,
                'session_id': session_id,
                'error': str(e)
            })

        # All strategies failed - log comprehensive failure
        self.logger.error("All JSON extraction strategies failed", extra={
            'event': 'all_json_strategies_failed',
            'session_id': session_id,
            'total_strategies_tried': 7,
            'response_sample': response_content[:200] + '...' if len(response_content) > 200 else response_content
        })

        return None

    def generate_storyline_options(self, story_context: str, active_badges: List[str], active_characters: List[str] = None) -> List[Dict[str, str]]:
        """Generate 3 follow-up storyline options for user selection"""

        # Generate unique request ID for tracking
        request_id = str(uuid.uuid4())
        session_id = session.get('session_id', 'unknown')
        start_time = time.time()
        model = self._get_model_for_request_type('storyline_options')

        # Structure the prompt with sections
        sections = []

        if active_badges:
            sections.append(f"STORY INFLUENCES: {', '.join(active_badges)}")

        if active_characters:
            sections.append(f"CHARACTER ELEMENTS: {', '.join(active_characters)}")

        influences_text = "\n".join(sections) if sections else "Continue naturally with creative freedom"

        prompt = f"""Based on this story context, generate 3 distinct follow-up storyline options for the reader to choose from.

CURRENT STORY CONTEXT:
{story_context}

CREATIVE INFLUENCES:
{influences_text}

CRITICAL INSTRUCTIONS:
- Create exactly 3 compelling and distinct storyline directions
- Each option must be 50-75 words describing what happens next
- Make each path significantly different (different conflicts, focuses, or directions)
- Incorporate the influences naturally into each option
- Maintain narrative coherence with the existing story
- Make each option enticing and leave the reader wanting more

RESPONSE FORMAT (CRITICAL):
You MUST respond with ONLY a valid JSON array. NO markdown, NO explanations, NO extra text.
Use this EXACT structure:
[
    {{
        "title": "Brief engaging title (3-6 words)",
        "description": "Detailed description of what happens next (50-75 words)"
    }},
    {{
        "title": "Brief engaging title (3-6 words)",
        "description": "Detailed description of what happens next (50-75 words)"
    }},
    {{
        "title": "Brief engaging title (3-6 words)",
        "description": "Detailed description of what happens next (50-75 words)"
    }}
]"""

        # Log request start
        self.logger.info("Starting storyline options LLM request", extra={
            'event': 'llm_storyline_options_start',
            'request_type': 'storyline_options',
            'request_id': request_id,
            'session_id': session_id,
            'model': model,
            'prompt_length': len(prompt),
            'active_badges': sanitize_for_logging(active_badges or []),
            'active_characters': sanitize_for_logging(active_characters or [])
        })

        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }

        data = {
            'model': model,
            'messages': [
                {
                    'role': 'system',
                    'content': 'You are a creative storyteller who generates storyline options. CRITICAL: You must respond with ONLY a valid JSON array. No markdown blocks, no explanations, no extra text. Start your response directly with [ and end with ]. Each option must have "title" and "description" fields. Example format: [{"title": "Dark Discovery", "description": "The protagonist uncovers a hidden truth that changes everything they believed about their world and forces them to make a difficult choice."}]'
                },
                {'role': 'user', 'content': prompt}
            ],
            'max_tokens': 800,
            'temperature': 0.8
        }

        try:
            response = requests.post(
                f'{self.base_url}/chat/completions',
                headers=headers,
                json=data,
                timeout=30
            )

            duration_ms = int((time.time() - start_time) * 1000)

            response.raise_for_status()
            result = response.json()
            generated_content = result['choices'][0]['message']['content'].strip()

            # Parse JSON response using enhanced extraction
            storyline_options = self.extract_json_from_response(generated_content)

            if storyline_options:
                # Validate and clean up the extracted options
                validated_options = []
                for i, option in enumerate(storyline_options):
                    if isinstance(option, dict) and 'title' in option and 'description' in option:
                        # Clean up and validate title and description
                        title = str(option['title']).strip()
                        description = str(option['description']).strip()

                        # Ensure title is reasonable length
                        if len(title) > 50:
                            title = title[:47] + "..."

                        # Ensure description meets word count (approximately)
                        word_count = len(description.split())
                        if word_count < 30:
                            description += " This storyline opens up new possibilities for character development and plot advancement."
                        elif word_count > 100:
                            # Truncate to approximately 75 words
                            words = description.split()
                            description = " ".join(words[:75]) + "..."

                        validated_options.append({
                            "title": title,
                            "description": description
                        })

                if len(validated_options) >= 3:
                    # Log successful response
                    self.logger.info("Storyline options LLM request completed successfully", extra={
                        'event': 'llm_storyline_options_success',
                        'request_type': 'storyline_options',
                        'request_id': request_id,
                        'session_id': session_id,
                        'model': model,
                        'duration_ms': duration_ms,
                        'status_code': response.status_code,
                        'options_count': len(validated_options),
                        'extraction_successful': True
                    })

                    return validated_options[:3]

            # Enhanced error logging with detailed analysis
            response_length = len(generated_content)
            response_preview = generated_content[:500] if len(generated_content) <= 500 else generated_content[:500] + "... [truncated]"

            # Detailed content analysis for debugging
            has_think_tags = '◁think▷' in generated_content or '<think>' in generated_content
            has_json_brackets = '[' in generated_content and ']' in generated_content
            has_json_braces = '{' in generated_content and '}' in generated_content
            think_tag_positions = []

            # Find think tag positions
            import re
            think_pattern = r'(?:(?:\u25c1|◁)think(?:\u25b7|▷)|<think>)'
            think_matches = list(re.finditer(think_pattern, generated_content, re.IGNORECASE))
            think_tag_positions = [(match.start(), match.end(), match.group()) for match in think_matches]

            # Count special characters that might interfere with JSON
            quote_counts = {
                'double_quotes': generated_content.count('"'),
                'single_quotes': generated_content.count("'"),
                'backticks': generated_content.count('`'),
                'brackets': generated_content.count('[') + generated_content.count(']'),
                'braces': generated_content.count('{') + generated_content.count('}')
            }

            self.logger.error("Failed to parse storyline options JSON after all extraction strategies", extra={
                'event': 'llm_storyline_options_parse_error',
                'request_type': 'storyline_options',
                'request_id': request_id,
                'session_id': session_id,
                'response_length': response_length,
                'response_preview': response_preview,
                'response_full_content': generated_content if response_length < 2000 else generated_content[:2000] + '... [truncated for logging]',
                'extraction_strategies_tried': 7,
                'valid_options_found': len(validated_options) if 'validated_options' in locals() else 0,
                'has_think_tags': has_think_tags,
                'has_json_brackets': has_json_brackets,
                'has_json_braces': has_json_braces,
                'think_tag_positions': think_tag_positions[:5],  # First 5 positions only
                'quote_counts': quote_counts,
                'starts_with': generated_content[:50] if generated_content else '',
                'ends_with': generated_content[-50:] if len(generated_content) >= 50 else generated_content
            })

            # Return fallback options
            return [
                    {
                        "title": "Continue Forward",
                        "description": "The story continues naturally, following the established narrative direction with new developments and character interactions."
                    },
                    {
                        "title": "Unexpected Twist",
                        "description": "A surprising revelation or unexpected event changes the course of the story, introducing new mysteries and challenges."
                    },
                    {
                        "title": "Character Focus",
                        "description": "The narrative shifts to explore character relationships and internal conflicts in greater depth."
                    }
                ]

        except requests.exceptions.RequestException as e:
            duration_ms = int((time.time() - start_time) * 1000)

            # Log request error
            self.logger.error("Storyline options LLM request failed", extra={
                'event': 'llm_storyline_options_error',
                'request_type': 'storyline_options',
                'request_id': request_id,
                'session_id': session_id,
                'model': model,
                'duration_ms': duration_ms,
                'error': str(e),
                'status_code': getattr(e.response, 'status_code', None) if hasattr(e, 'response') else None
            })

            # Return fallback options
            return [
                {
                    "title": "Continue Forward",
                    "description": "The story continues naturally, following the established narrative direction."
                },
                {
                    "title": "New Challenge",
                    "description": "A new obstacle or challenge appears to test the characters."
                },
                {
                    "title": "Explore Deeper",
                    "description": "Dive deeper into the current situation and character motivations."
                }
            ]

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)

            # Log unexpected error
            self.logger.error("Unexpected error during storyline options LLM request", extra={
                'event': 'llm_storyline_options_unexpected_error',
                'request_type': 'storyline_options',
                'request_id': request_id,
                'session_id': session_id,
                'model': model,
                'duration_ms': duration_ms,
                'error': str(e)
            }, exc_info=True)

            # Return fallback options
            return [
                {
                    "title": "Continue Forward",
                    "description": "The story continues with new developments."
                },
                {
                    "title": "New Direction",
                    "description": "The narrative takes an unexpected turn."
                },
                {
                    "title": "Character Development",
                    "description": "Focus on character growth and relationships."
                }
            ]

class ImageGenerator:
    def __init__(self):
        self.enabled = SDWEBUI_ENABLED and SDWEBUI_AVAILABLE
        self.logger = get_logger('image_generation')
        self.api = None
        self.api_host = SDWEBUI_HOST
        self.api_port = SDWEBUI_PORT
        self.target_model = IMAGE_MODEL

        if self.enabled:
            self.setup_api()
        else:
            self.logger.info("Image generation disabled", extra={
                'event': 'image_generation_disabled',
                'sdwebui_enabled': SDWEBUI_ENABLED,
                'sdwebui_available': SDWEBUI_AVAILABLE
            })

    def setup_api(self) -> bool:
        """Initialize connection to SD WebUI API and set the correct model"""
        try:
            self.api = WebUIApi(
                host=self.api_host,
                port=self.api_port,
                use_https=SDWEBUI_USE_HTTPS
            )

            # Test connection by getting current model
            current_model = self.api.util_get_current_model()
            print(f"✓ Connected to SD WebUI API at {self.api_host}:{self.api_port}")
            print(f"Current model: {current_model}")

            self.logger.info("SD WebUI API connected successfully", extra={
                'event': 'sdwebui_connected',
                'host': self.api_host,
                'port': self.api_port,
                'current_model': current_model,
                'target_model': self.target_model
            })

            # Switch to target model if not already active
            if self.target_model not in current_model:
                print(f"Switching to {self.target_model}...")
                try:
                    self.api.util_set_model(self.target_model)
                    print(f"✓ Model switched to {self.target_model}")

                    self.logger.info("SD WebUI model switched successfully", extra={
                        'event': 'sdwebui_model_switched',
                        'from_model': current_model,
                        'to_model': self.target_model
                    })

                    return True

                except Exception as e:
                    print(f"⚠ Warning: Could not switch model: {e}")
                    print("Continuing with current model...")

                    self.logger.warning("Could not switch SD WebUI model, using current", extra={
                        'event': 'sdwebui_model_switch_failed',
                        'error': str(e),
                        'current_model': current_model,
                        'target_model': self.target_model
                    })

                    return True  # Still connected, just using different model

            else:
                print(f"✓ Already using target model: {self.target_model}")
                self.logger.info("SD WebUI already using target model", extra={
                    'event': 'sdwebui_model_already_active',
                    'model': self.target_model
                })
                return True

        except Exception as e:
            print(f"✗ Failed to connect to SD WebUI API: {e}")
            self.enabled = False

            self.logger.error("Failed to connect to SD WebUI API - disabling image generation", extra={
                'event': 'sdwebui_connection_failed',
                'error': str(e),
                'host': self.api_host,
                'port': self.api_port
            })

            return False

    def generate_chapter_ornament(self, story_segment: str, active_badges: List[str] = None, segment_number: int = 1) -> Optional[str]:
        """Generate a chapter ornament image for a story segment"""

        if not self.enabled:
            return None

        request_id = str(uuid.uuid4())
        session_id = session.get('session_id', 'unknown')
        start_time = time.time()

        try:
            # Create a descriptive prompt based on story content and badges
            themes = self._extract_themes_from_story(story_segment, active_badges)
            prompt = self._build_image_prompt(themes, story_segment)

            # Generate filename based on story content hash
            content_hash = hashlib.md5((story_segment + str(active_badges or [])).encode()).hexdigest()[:10]
            filename = f"chapter_{segment_number}_{content_hash}.png"
            filepath = os.path.join('static', 'images', filename)

            # Check if image already exists (caching)
            if os.path.exists(filepath):
                self.logger.info("Using cached chapter ornament", extra={
                    'event': 'image_cache_hit',
                    'request_id': request_id,
                    'session_id': session_id,
                    'image_filename': filename
                })
                return f"/static/images/{filename}"

            self.logger.info("Starting image generation", extra={
                'event': 'image_generation_start',
                'request_id': request_id,
                'session_id': session_id,
                'prompt_preview': prompt[:100] + "..." if len(prompt) > 100 else prompt,
                'image_filename': filename
            })

            # Generate image using SD WebUI API with optimized parameters for fast model
            result = self.api.txt2img(
                prompt=prompt,
                negative_prompt="text, letters, words, watermark, signature, blurry, low quality, distorted",
                width=IMAGE_WIDTH,
                height=IMAGE_HEIGHT,
                steps=IMAGE_STEPS,
                cfg_scale=IMAGE_CFG_SCALE,
                sampler_name=IMAGE_SAMPLER
            )

            if result.image:
                # Save image to static/images directory
                result.image.save(filepath, "PNG")

                duration_ms = int((time.time() - start_time) * 1000)

                self.logger.info("Image generation completed successfully", extra={
                    'event': 'image_generation_success',
                    'request_id': request_id,
                    'session_id': session_id,
                    'duration_ms': duration_ms,
                    'image_filename': filename,
                    'image_filepath': filepath
                })

                return f"/static/images/{filename}"
            else:
                self.logger.error("Image generation returned no result", extra={
                    'event': 'image_generation_no_result',
                    'request_id': request_id,
                    'session_id': session_id
                })
                return None

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            self.logger.error("Image generation failed", extra={
                'event': 'image_generation_error',
                'request_id': request_id,
                'session_id': session_id,
                'duration_ms': duration_ms,
                'error': str(e)
            })
            return None

    def _extract_themes_from_story(self, story_segment: str, active_badges: List[str] = None) -> List[str]:
        """Extract visual themes from story content and badges with enhanced analysis"""
        all_themes = []

        # Extract character-specific themes (highest priority for current segment)
        character_elements = self._extract_characters_from_segment(story_segment)
        all_themes.extend(character_elements)

        # Extract story-specific elements (high priority for current segment)
        story_elements = self._extract_story_elements(story_segment)
        all_themes.extend(story_elements)

        # Add badge-based themes (medium priority)
        if active_badges:
            badge_theme_mapping = {
                'mysterious': ['dark shadows', 'moonlight', 'mist'],
                'romantic': ['roses', 'candlelight', 'soft pastels'],
                'funny': ['bright colors', 'whimsical', 'cartoonish'],
                'dark': ['gothic', 'shadows', 'ravens'],
                'heartwarming': ['warm light', 'golden hour', 'cozy'],
                'sci-fi': ['futuristic', 'neon', 'technological'],
                'fantasy': ['magical', 'ethereal', 'mystical'],
                'horror': ['ominous', 'dark', 'twisted'],
                'thriller': ['tension', 'suspense', 'dramatic angles'],
                'western': ['desert', 'sunset', 'rustic'],
                'action-packed': ['dynamic', 'energetic', 'bold'],
                'slow-burn': ['gradual building', 'subtle elements'],
                'explosive': ['burst patterns', 'high energy', 'dramatic'],
                'intimate': ['soft', 'warm', 'gentle'],
                'epic': ['grand scale', 'monumental', 'sweeping'],
                'passionate': ['intense colors', 'flowing forms', 'emotional'],
                'seductive': ['elegant curves', 'alluring patterns', 'mysterious'],
                'dangerous': ['sharp edges', 'warning symbols', 'threatening'],
                'forbidden': ['hidden elements', 'secret symbols', 'concealed']
            }

            for badge in active_badges:
                if badge.lower() in badge_theme_mapping:
                    all_themes.extend(badge_theme_mapping[badge.lower()])

        # Extract environmental themes from story content (lower priority)
        story_lower = story_segment.lower()
        content_themes = {
            'forest': ['trees', 'nature', 'green'],
            'castle': ['medieval', 'stone', 'towers'],
            'ocean': ['waves', 'blue', 'nautical'],
            'city': ['urban', 'buildings', 'modern'],
            'night': ['stars', 'moonlight', 'dark blue'],
            'battle': ['weapons', 'conflict', 'dramatic'],
            'magic': ['mystical', 'ethereal', 'glowing'],
            'winter': ['snow', 'ice', 'cold colors'],
            'summer': ['sun', 'warm colors', 'bright'],
            'mountain': ['peaks', 'height', 'rugged terrain'],
            'desert': ['sand', 'heat', 'sparse elements'],
            'river': ['flowing water', 'meandering', 'life source'],
            'cave': ['dark depths', 'hidden spaces', 'mystery'],
            'garden': ['cultivated beauty', 'growth', 'harmony'],
            'library': ['knowledge symbols', 'books', 'scholarly'],
            'temple': ['sacred geometry', 'spiritual symbols', 'reverence'],
            'tavern': ['warm gathering', 'rustic charm', 'community'],
            'ship': ['nautical elements', 'adventure', 'journey']
        }

        for keyword, theme_list in content_themes.items():
            if keyword in story_lower:
                all_themes.extend(theme_list)

        # Remove duplicates while preserving order (character/story elements first)
        seen = set()
        unique_themes = []
        for theme in all_themes:
            if theme.lower() not in seen:
                seen.add(theme.lower())
                unique_themes.append(theme)

        # Return prioritized themes - character and story elements get priority
        return unique_themes[:12]  # Increased limit to accommodate richer detail

    def _extract_characters_from_segment(self, story_segment: str) -> List[str]:
        """Extract character-related visual elements from the story segment"""
        character_elements = []
        story_lower = story_segment.lower()

        # Character appearance descriptors
        appearance_keywords = {
            'hair': ['golden hair', 'dark hair', 'silver hair', 'flowing locks'],
            'eyes': ['bright eyes', 'piercing gaze', 'emerald eyes', 'sapphire eyes'],
            'clothing': ['elegant robes', 'armor', 'cloak', 'dress', 'uniform'],
            'tall': ['towering figure', 'imposing stature'],
            'small': ['delicate figure', 'petite form'],
            'strong': ['muscular', 'powerful build'],
            'graceful': ['elegant movement', 'fluid grace'],
            'mysterious': ['shadowy figure', 'enigmatic presence'],
            'royal': ['crown', 'regalia', 'noble bearing'],
            'warrior': ['sword', 'shield', 'battle-worn'],
            'magic': ['glowing staff', 'mystical aura', 'arcane symbols'],
            'wise': ['ancient wisdom', 'knowing eyes', 'scholarly robes']
        }

        # Character emotions and actions (visual representations)
        emotion_visuals = {
            'angry': ['flames', 'storm clouds', 'lightning'],
            'sad': ['tears', 'rain', 'wilting flowers'],
            'happy': ['sunshine', 'bright colors', 'blooming flowers'],
            'afraid': ['shadows', 'dark corners', 'trembling'],
            'determined': ['firm stance', 'clenched fist', 'forward motion'],
            'love': ['hearts', 'roses', 'warm light'],
            'hope': ['dawn light', 'rising sun', 'bright horizon'],
            'despair': ['storm clouds', 'darkness', 'withered plants']
        }

        # Extract character names (simple proper noun detection)
        import re
        # Look for capitalized words that could be names (excluding common words)
        common_words = {'the', 'and', 'but', 'or', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an', 'in', 'on', 'as', 'is', 'was', 'are', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must', 'shall'}
        potential_names = re.findall(r'\b[A-Z][a-z]+\b', story_segment)
        character_names = [name for name in potential_names if name.lower() not in common_words and len(name) > 2]

        # Add character name elements to ornament
        if character_names:
            character_elements.extend(['character portrait', 'name inscriptions', 'personal emblems'])

        # Check for appearance descriptors
        for keyword, visuals in appearance_keywords.items():
            if keyword in story_lower:
                character_elements.extend(visuals)

        # Check for emotional visuals
        for emotion, visuals in emotion_visuals.items():
            if emotion in story_lower or f"{emotion}ly" in story_lower:
                character_elements.extend(visuals)

        # Character relationships (visual symbols)
        relationship_visuals = {
            'friend': ['intertwined symbols', 'clasped hands', 'unity emblems'],
            'enemy': ['crossed swords', 'opposing symbols', 'conflict motifs'],
            'lover': ['intertwined hearts', 'romantic flourishes', 'paired symbols'],
            'family': ['family crests', 'generational symbols', 'bloodline emblems'],
            'mentor': ['guiding light', 'wisdom symbols', 'teaching emblems'],
            'student': ['learning symbols', 'growth motifs', 'potential emblems']
        }

        for relationship, visuals in relationship_visuals.items():
            if relationship in story_lower:
                character_elements.extend(visuals)

        return character_elements[:8]  # Limit character elements

    def _extract_story_elements(self, story_segment: str) -> List[str]:
        """Extract specific objects, colors, and atmospheric details from the story segment"""
        story_elements = []
        story_lower = story_segment.lower()

        # Color extraction
        colors = {
            'red': ['crimson', 'scarlet', 'ruby'], 'blue': ['azure', 'sapphire', 'cobalt'],
            'green': ['emerald', 'jade', 'forest green'], 'gold': ['golden', 'amber', 'brass'],
            'silver': ['metallic', 'platinum', 'chrome'], 'purple': ['violet', 'amethyst', 'lavender'],
            'black': ['obsidian', 'ebony', 'midnight'], 'white': ['pearl', 'ivory', 'alabaster'],
            'yellow': ['citrine', 'sunny', 'lemon'], 'orange': ['copper', 'bronze', 'sunset']
        }

        for color, variants in colors.items():
            if any(variant in story_lower for variant in [color] + variants):
                story_elements.extend(variants[:1])  # Add one variant

        # Objects and artifacts
        objects = {
            'sword': ['ornate blade', 'crossed swords', 'weapon motifs'],
            'book': ['ancient tome', 'scrollwork', 'written knowledge'],
            'crown': ['royal circlet', 'noble regalia', 'sovereignty symbols'],
            'ring': ['enchanted band', 'circular motifs', 'binding symbols'],
            'staff': ['mystical rod', 'power conduit', 'authority symbol'],
            'crystal': ['glowing gem', 'prismatic light', 'magical focus'],
            'flower': ['botanical elements', 'natural beauty', 'growth symbols'],
            'tree': ['branching patterns', 'life symbols', 'natural frames'],
            'star': ['celestial bodies', 'guiding lights', 'cosmic elements'],
            'moon': ['lunar crescents', 'night symbols', 'celestial arcs'],
            'sun': ['solar rays', 'radiant patterns', 'daylight motifs'],
            'fire': ['flame patterns', 'elemental energy', 'burning motifs'],
            'water': ['flowing waves', 'liquid patterns', 'fluid elements'],
            'wind': ['swirling patterns', 'dynamic movement', 'air currents'],
            'stone': ['rocky textures', 'ancient patterns', 'enduring symbols'],
            'key': ['unlocking symbols', 'access motifs', 'opening patterns'],
            'door': ['gateway symbols', 'threshold patterns', 'portal designs'],
            'mirror': ['reflective surfaces', 'dual imagery', 'truth symbols'],
            'candle': ['flickering light', 'illumination symbols', 'warmth motifs'],
            'feather': ['delicate plumes', 'flight symbols', 'lightness motifs'],
            'shield': ['protective emblems', 'defensive patterns', 'guardian symbols'],
            'horse': ['noble steed', 'movement symbols', 'freedom motifs'],
            'dragon': ['serpentine forms', 'power symbols', 'mythic elements'],
            'tower': ['reaching heights', 'vertical elements', 'stronghold symbols']
        }

        for obj, visuals in objects.items():
            if obj in story_lower:
                story_elements.extend(visuals[:2])  # Add up to 2 visual elements

        # Atmospheric and environmental details
        atmosphere = {
            'mist': ['ethereal fog', 'mysterious veils', 'cloudy wisps'],
            'storm': ['turbulent patterns', 'chaotic energy', 'dramatic swirls'],
            'dawn': ['rising light', 'hopeful rays', 'new beginning motifs'],
            'dusk': ['fading light', 'transition symbols', 'evening hues'],
            'rain': ['water droplets', 'cleansing symbols', 'renewal patterns'],
            'snow': ['crystalline patterns', 'pure white elements', 'winter motifs'],
            'smoke': ['wispy tendrils', 'mysterious vapors', 'ethereal forms'],
            'shadow': ['dark silhouettes', 'contrasting elements', 'hidden forms'],
            'light': ['radiant beams', 'illuminating rays', 'brilliant patterns'],
            'darkness': ['deep voids', 'mysterious depths', 'shadow elements'],
            'warm': ['cozy elements', 'comfortable patterns', 'inviting motifs'],
            'cold': ['icy patterns', 'crystalline forms', 'frigid elements'],
            'ancient': ['weathered textures', 'time-worn patterns', 'historic motifs'],
            'new': ['fresh elements', 'pristine forms', 'beginning symbols'],
            'broken': ['fractured patterns', 'damaged elements', 'scattered pieces'],
            'whole': ['complete forms', 'unified patterns', 'integrated elements']
        }

        for atmo, visuals in atmosphere.items():
            if atmo in story_lower:
                story_elements.extend(visuals[:1])  # Add one atmospheric element

        # Extract quoted dialogue or important phrases for symbolic representation
        import re
        quotes = re.findall(r'"([^"]*)"', story_segment)
        if quotes:
            # Add symbolic elements for dialogue
            story_elements.extend(['speech scrolls', 'word banners', 'communication symbols'])

        # Time-related elements
        time_elements = {
            'morning': ['sunrise rays', 'dawn motifs'],
            'noon': ['zenith symbols', 'peak light'],
            'evening': ['sunset hues', 'twilight patterns'],
            'night': ['star fields', 'lunar elements'],
            'yesterday': ['past symbols', 'memory motifs'],
            'tomorrow': ['future symbols', 'hope patterns'],
            'forever': ['eternal symbols', 'infinity motifs'],
            'moment': ['fleeting patterns', 'instant symbols']
        }

        for time_word, visuals in time_elements.items():
            if time_word in story_lower:
                story_elements.extend(visuals)

        return story_elements[:10]  # Limit story elements

    def _build_image_prompt(self, themes: List[str], story_segment: str = "") -> str:
        """Build the final image generation prompt with enhanced story integration"""
        if not themes:
            theme_str = 'neutral fantasy'
        else:
            # Organize themes by priority and type for better prompt structure
            character_themes = []
            story_themes = []
            environmental_themes = []
            color_themes = []

            # Categorize themes for better organization
            character_keywords = ['portrait', 'hair', 'eyes', 'armor', 'robes', 'crown', 'emblems', 'inscriptions']
            story_keywords = ['sword', 'tome', 'crystal', 'flame', 'star', 'scroll', 'symbol', 'motif']
            color_keywords = ['crimson', 'azure', 'emerald', 'golden', 'silver', 'obsidian', 'pearl', 'amber']

            for theme in themes:
                theme_lower = theme.lower()
                if any(keyword in theme_lower for keyword in character_keywords):
                    character_themes.append(theme)
                elif any(keyword in theme_lower for keyword in story_keywords):
                    story_themes.append(theme)
                elif any(keyword in theme_lower for keyword in color_keywords):
                    color_themes.append(theme)
                else:
                    environmental_themes.append(theme)

            # Build structured theme string with priority ordering
            theme_parts = []

            # Character elements first (most specific to current segment)
            if character_themes:
                theme_parts.append(', '.join(character_themes[:4]))

            # Story-specific elements second
            if story_themes:
                theme_parts.append(', '.join(story_themes[:4]))

            # Color elements for visual richness
            if color_themes:
                theme_parts.append(', '.join(color_themes[:3]))

            # Environmental elements last
            if environmental_themes:
                theme_parts.append(', '.join(environmental_themes[:3]))

            theme_str = ', '.join(theme_parts) if theme_parts else 'fantasy elements'

        # Enhance the base prompt with story-aware language
        segment_preview = story_segment[:200] if story_segment else ""

        # Add contextual descriptors based on story content
        contextual_elements = []
        if segment_preview:
            segment_lower = segment_preview.lower()
            # Add contextual modifiers based on story tone
            if any(word in segment_lower for word in ['whisper', 'secret', 'hidden', 'shadow']):
                contextual_elements.append('subtle details')
            if any(word in segment_lower for word in ['bright', 'shining', 'radiant', 'brilliant']):
                contextual_elements.append('luminous accents')
            if any(word in segment_lower for word in ['ancient', 'old', 'weathered', 'worn']):
                contextual_elements.append('aged textures')
            if any(word in segment_lower for word in ['elegant', 'graceful', 'beautiful', 'refined']):
                contextual_elements.append('refined artistry')
            if any(word in segment_lower for word in ['power', 'strength', 'mighty', 'strong']):
                contextual_elements.append('bold elements')

        # Construct the final prompt with better structure
        prompt_parts = [IMAGE_STYLE_PROMPT]

        # Add story-specific themes
        prompt_parts.append(theme_str)

        # Add contextual elements if present
        if contextual_elements:
            prompt_parts.append(', '.join(contextual_elements))

        # Always end with essential requirements
        prompt_parts.extend(['no text', 'no letters', 'no words', 'ornamental chapter border', 'decorative frame design'])

        prompt = ', '.join(prompt_parts)

        return prompt

# Initialize story generator and image generator
story_generator = StoryGenerator(OPENROUTER_API_KEY, MODEL_CONFIG)
image_generator = ImageGenerator()

@app.route('/')
def index():
    """Main page"""
    # Initialize session if needed
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
        app_logger.info("New session created", extra={
            'event': 'session_created',
            'session_id': session['session_id']
        })

    if 'story_segments' not in session:
        session['story_segments'] = []
    if 'story_summary' not in session:
        session['story_summary'] = ""
    if 'characters' not in session:
        session['characters'] = load_character_categories_from_env()
    if 'badges' not in session:
        session['badges'] = load_badge_categories_from_env()
    if 'story_mode' not in session:
        session['story_mode'] = 'free'  # Default to free mode
    if 'pending_storylines' not in session:
        session['pending_storylines'] = []
    if 'selected_storyline' not in session:
        session['selected_storyline'] = ""

    return render_template('index.html')

@app.route('/api/start_story', methods=['POST'])
def start_story():
    """Start a new story with initial badges and characters"""
    data = request.get_json()
    active_badges = data.get('active_badges', [])
    active_characters = data.get('active_characters', [])

    app_logger.info("Starting new story", extra={
        'event': 'story_start',
        'session_id': session.get('session_id', 'unknown'),
        'active_badges': sanitize_for_logging(active_badges),
        'active_characters': sanitize_for_logging(active_characters)
    })

    # Generate initial story segment
    initial_segment, think_sections = story_generator.generate_initial_story(active_badges, active_characters)

    # Generate chapter ornament image
    image_url = image_generator.generate_chapter_ornament(
        initial_segment,
        active_badges,
        segment_number=1
    )

    # Reset and store in session
    session['story_segments'] = [initial_segment]
    session['story_summary'] = ""  # Reset summary for new story

    app_logger.info("Story started successfully", extra={
        'event': 'story_start_success',
        'session_id': session.get('session_id', 'unknown'),
        'segment_number': len(session['story_segments']),
        'segment_length': len(initial_segment),
        'think_sections_count': len(think_sections),
        'image_generated': image_url is not None
    })

    response_data = {
        'success': True,
        'segment': initial_segment,
        'think_sections': think_sections,
        'segment_number': len(session['story_segments'])
    }

    if image_url:
        response_data['image_url'] = image_url

    return jsonify(response_data)

@app.route('/api/continue_story', methods=['POST'])
def continue_story():
    """Generate next story segment using summary-based context"""
    data = request.get_json()
    active_badges = data.get('active_badges', [])
    active_characters = data.get('active_characters', [])

    # Log detailed session state before processing
    current_session_id = session.get('session_id', 'unknown')
    current_segments_count = len(session.get('story_segments', []))

    # Debug: log ALL session contents and size to identify issues
    session_dict = dict(session)
    session_size_bytes = len(str(session_dict).encode('utf-8'))

    app_logger.info("Session size and contents", extra={
        'event': 'session_debug_dump',
        'session_id': current_session_id,
        'session_size_bytes': session_size_bytes,
        'session_size_kb': round(session_size_bytes / 1024, 2),
        'session_contents': session_dict,
        'session_keys': list(session.keys())
    })

    app_logger.info("Continuing story - session state BEFORE", extra={
        'event': 'story_continue_session_before',
        'session_id': current_session_id,
        'current_segments_count': current_segments_count,
        'story_segments_preview': [seg[:100] + '...' if len(seg) > 100 else seg for seg in session.get('story_segments', [])][-2:],
        'has_story_summary': bool(session.get('story_summary')),
        'active_badges': sanitize_for_logging(active_badges),
        'active_characters': sanitize_for_logging(active_characters)
    })

    app_logger.info("Continuing story", extra={
        'event': 'story_continue',
        'session_id': current_session_id,
        'current_segments': current_segments_count,
        'active_badges': sanitize_for_logging(active_badges),
        'active_characters': sanitize_for_logging(active_characters)
    })

    # Initialize session if needed - but log if this happens unexpectedly
    if 'story_segments' not in session:
        app_logger.warning("Story segments not found in session during continue_story", extra={
            'event': 'story_segments_missing',
            'session_id': current_session_id,
            'session_keys': list(session.keys())
        })
        session['story_segments'] = []
    if 'story_summary' not in session:
        session['story_summary'] = ""

    story_segments = session['story_segments']

    # If no story segments exist, this indicates a session problem - cannot continue
    if len(story_segments) == 0:
        app_logger.error("Cannot continue story - no existing segments found", extra={
            'event': 'continue_story_no_segments',
            'session_id': current_session_id
        })
        return jsonify({
            'success': False,
            'error': 'No existing story found. Please start a new story first.',
            'segment_number': 0
        })

    # Determine if we need to generate/update summary
    should_generate_summary = (
        len(story_segments) >= SUMMARY_WINDOW_SIZE and
        (not session['story_summary'] or len(story_segments) % SUMMARY_WINDOW_SIZE == 0)
    )

    # Generate summary if needed
    if should_generate_summary:
        app_logger.info("Generating story summary", extra={
            'event': 'story_summary_generation',
            'session_id': session.get('session_id', 'unknown'),
            'segments_to_summarize': len(story_segments)
        })
        session['story_summary'] = story_generator.generate_story_summary(story_segments)

    # Build context for next segment generation
    context_parts = []

    # Add summary if available
    if session['story_summary']:
        context_parts.append(f"STORY SUMMARY SO FAR:\n{session['story_summary']}")

    # Add recent segments for immediate context
    recent_segments = story_segments[-RECENT_SEGMENTS_COUNT:] if len(story_segments) > 0 else []
    if recent_segments:
        context_parts.append(f"RECENT STORY SEGMENTS:\n" + "\n\n".join(recent_segments))

    # If no summary and few segments, use all segments (fallback for short stories)
    if not session['story_summary'] and len(story_segments) < SUMMARY_WINDOW_SIZE:
        story_context = "\n\n".join(story_segments)
    else:
        story_context = "\n\n---\n\n".join(context_parts) if context_parts else "Beginning of story"

    app_logger.info("Using context for story generation", extra={
        'event': 'story_context_built',
        'session_id': session.get('session_id', 'unknown'),
        'has_summary': bool(session['story_summary']),
        'recent_segments_count': len(recent_segments),
        'context_length': len(story_context)
    })

    # Generate next segment
    next_segment, think_sections = story_generator.generate_story_segment(story_context, active_badges, active_characters)

    # Generate chapter ornament image for the new segment
    image_url = image_generator.generate_chapter_ornament(
        next_segment,
        active_badges,
        segment_number=len(session['story_segments']) + 1
    )

    # Add to session
    session['story_segments'].append(next_segment)

    # Apply session optimization if needed
    session_dict_before_opt = dict(session)
    session_size_kb_before = len(str(session_dict_before_opt).encode('utf-8')) / 1024

    if session_size_kb_before > MAX_SESSION_SIZE_KB / 2:  # Start optimizing at half the limit
        optimized_session = optimize_session_size(session_dict_before_opt, app_logger)
        # Update the actual session with optimized data
        for key in ['story_segments', 'story_summary']:
            if key in optimized_session:
                session[key] = optimized_session[key]

    # Log session state after adding new segment
    final_segments_count = len(session['story_segments'])
    app_logger.info("Story continued successfully - session state AFTER", extra={
        'event': 'story_continue_session_after',
        'session_id': session.get('session_id', 'unknown'),
        'segments_count_before': current_segments_count,
        'segments_count_after': final_segments_count,
        'segment_added_successfully': final_segments_count == current_segments_count + 1,
        'new_segment_preview': next_segment[:100] + '...' if len(next_segment) > 100 else next_segment,
        'segment_length': len(next_segment),
        'think_sections_count': len(think_sections),
        'summary_active': bool(session['story_summary'])
    })

    app_logger.info("Story continued successfully", extra={
        'event': 'story_continue_success',
        'session_id': session.get('session_id', 'unknown'),
        'segment_number': len(session['story_segments']),
        'segment_length': len(next_segment),
        'think_sections_count': len(think_sections),
        'total_segments': len(session['story_segments']),
        'summary_active': bool(session['story_summary']),
        'image_generated': image_url is not None
    })

    response_data = {
        'success': True,
        'segment': next_segment,
        'think_sections': think_sections,
        'segment_number': len(session['story_segments'])
    }

    if image_url:
        response_data['image_url'] = image_url

    return jsonify(response_data)

@app.route('/api/add_badge', methods=['POST'])
def add_badge():
    """Add a new badge to a specific category"""
    data = request.get_json()
    new_badge = data.get('badge', '').strip().lower()
    category = data.get('category', 'mood')  # Default to mood category

    if 'badges' not in session:
        session['badges'] = load_badge_categories_from_env()

    # Ensure the category exists
    if category not in session['badges']:
        session['badges'][category] = []

    # Check if badge already exists in any category
    badge_exists = any(new_badge in badges for badges in session['badges'].values())

    if new_badge and not badge_exists:
        session['badges'][category].append(new_badge)
        return jsonify({'success': True, 'badge': new_badge, 'category': category})

    return jsonify({'success': False, 'error': 'Badge already exists or is empty'})

@app.route('/api/get_badges', methods=['GET'])
def get_badges():
    """Get current badges"""
    # Ensure badges are properly initialized
    if 'badges' not in session or not isinstance(session.get('badges'), dict):
        session['badges'] = load_badge_categories_from_env()

    return jsonify({
        'badges': session.get('badges', {})
    })

@app.route('/api/get_characters', methods=['GET'])
def get_characters():
    """Get current characters"""
    # Ensure characters are properly initialized
    if 'characters' not in session or not isinstance(session.get('characters'), dict):
        session['characters'] = load_character_categories_from_env()

    return jsonify({
        'characters': session.get('characters', {})
    })

@app.route('/api/add_character', methods=['POST'])
def add_character():
    """Add a new character element"""
    data = request.get_json()
    character_element = data.get('character', '').strip().lower()
    category = data.get('category', 'traits')

    if 'characters' not in session:
        session['characters'] = load_character_categories_from_env()

    # Ensure the category exists
    if category not in session['characters']:
        session['characters'][category] = []

    # Check if character already exists in any category
    character_exists = any(character_element in characters for characters in session['characters'].values())

    if character_element and not character_exists:
        session['characters'][category].append(character_element)
        return jsonify({'success': True, 'character': character_element, 'category': category})

    return jsonify({'success': False, 'error': 'Character element already exists or is empty'})

@app.route('/api/reset_story', methods=['POST'])
def reset_story():
    """Reset the current story"""
    app_logger.info("Resetting story", extra={
        'event': 'story_reset',
        'session_id': session.get('session_id', 'unknown'),
        'previous_segments': len(session.get('story_segments', []))
    })

    session['story_segments'] = []
    session['story_summary'] = ""

    app_logger.info("Story reset successfully", extra={
        'event': 'story_reset_success',
        'session_id': session.get('session_id', 'unknown')
    })

    return jsonify({'success': True})

@app.route('/api/get_storyline_options', methods=['POST'])
def get_storyline_options():
    """Generate 3 follow-up storyline options for the current story"""
    data = request.get_json()
    active_badges = data.get('active_badges', [])
    active_characters = data.get('active_characters', [])

    current_session_id = session.get('session_id', 'unknown')

    app_logger.info("Generating storyline options", extra={
        'event': 'storyline_options_request',
        'session_id': current_session_id,
        'active_badges': sanitize_for_logging(active_badges),
        'active_characters': sanitize_for_logging(active_characters),
        'current_segments': len(session.get('story_segments', []))
    })

    # Check if story exists
    story_segments = session.get('story_segments', [])
    if len(story_segments) == 0:
        app_logger.error("Cannot generate storyline options - no story segments", extra={
            'event': 'storyline_options_no_segments',
            'session_id': current_session_id
        })
        return jsonify({
            'success': False,
            'error': 'No existing story found. Please start a new story first.'
        })

    # Build context similar to continue_story
    context_parts = []

    if session.get('story_summary'):
        context_parts.append(f"STORY SUMMARY SO FAR:\n{session['story_summary']}")

    # Add recent segments for immediate context
    recent_segments = story_segments[-RECENT_SEGMENTS_COUNT:] if len(story_segments) > 0 else []
    if recent_segments:
        context_parts.append(f"RECENT STORY SEGMENTS:\n" + "\n\n".join(recent_segments))

    # If no summary and few segments, use all segments
    if not session.get('story_summary') and len(story_segments) < SUMMARY_WINDOW_SIZE:
        story_context = "\n\n".join(story_segments)
    else:
        story_context = "\n\n---\n\n".join(context_parts) if context_parts else "Beginning of story"

    # Generate storyline options
    storyline_options = story_generator.generate_storyline_options(story_context, active_badges, active_characters)

    # Store options in session for selection
    session['pending_storylines'] = storyline_options

    app_logger.info("Storyline options generated successfully", extra={
        'event': 'storyline_options_success',
        'session_id': current_session_id,
        'options_count': len(storyline_options)
    })

    return jsonify({
        'success': True,
        'storyline_options': storyline_options
    })

@app.route('/api/select_storyline', methods=['POST'])
def select_storyline():
    """Select a storyline option and generate the next segment"""
    data = request.get_json()
    selected_index = data.get('selected_index')
    active_badges = data.get('active_badges', [])
    active_characters = data.get('active_characters', [])

    current_session_id = session.get('session_id', 'unknown')

    app_logger.info("Processing storyline selection", extra={
        'event': 'storyline_selection',
        'session_id': current_session_id,
        'selected_index': selected_index,
        'active_badges': sanitize_for_logging(active_badges),
        'active_characters': sanitize_for_logging(active_characters)
    })

    # Validate selection
    pending_storylines = session.get('pending_storylines', [])
    if not pending_storylines or selected_index is None or selected_index < 0 or selected_index >= len(pending_storylines):
        app_logger.error("Invalid storyline selection", extra={
            'event': 'storyline_selection_invalid',
            'session_id': current_session_id,
            'selected_index': selected_index,
            'available_options': len(pending_storylines)
        })
        return jsonify({
            'success': False,
            'error': 'Invalid storyline selection'
        })

    selected_storyline = pending_storylines[selected_index]
    session['selected_storyline'] = selected_storyline

    # Clear pending storylines
    session['pending_storylines'] = []

    # Build enhanced context including the selected storyline
    context_parts = []

    if session.get('story_summary'):
        context_parts.append(f"STORY SUMMARY SO FAR:\n{session['story_summary']}")

    # Add recent segments for immediate context
    story_segments = session['story_segments']
    recent_segments = story_segments[-RECENT_SEGMENTS_COUNT:] if len(story_segments) > 0 else []
    if recent_segments:
        context_parts.append(f"RECENT STORY SEGMENTS:\n" + "\n\n".join(recent_segments))

    # Add selected storyline context
    context_parts.append(f"SELECTED STORYLINE DIRECTION:\n{selected_storyline['title']}: {selected_storyline['description']}")

    # If no summary and few segments, use all segments plus storyline
    if not session.get('story_summary') and len(story_segments) < SUMMARY_WINDOW_SIZE:
        story_context = "\n\n".join(story_segments) + f"\n\n---\n\nSELECTED STORYLINE DIRECTION:\n{selected_storyline['title']}: {selected_storyline['description']}"
    else:
        story_context = "\n\n---\n\n".join(context_parts)

    app_logger.info("Generating segment with selected storyline", extra={
        'event': 'storyline_segment_generation',
        'session_id': current_session_id,
        'selected_storyline_title': selected_storyline['title'],
        'context_length': len(story_context)
    })

    # Generate next segment with storyline context
    next_segment, think_sections = story_generator.generate_story_segment(story_context, active_badges, active_characters)

    # Generate chapter ornament image for the new segment
    image_url = image_generator.generate_chapter_ornament(
        next_segment,
        active_badges,
        segment_number=len(session['story_segments']) + 1
    )

    # Add to session
    session['story_segments'].append(next_segment)

    # Apply session optimization if needed (same logic as continue_story)
    session_dict_before_opt = dict(session)
    session_size_kb_before = len(str(session_dict_before_opt).encode('utf-8')) / 1024

    if session_size_kb_before > MAX_SESSION_SIZE_KB / 2:
        optimized_session = optimize_session_size(session_dict_before_opt, app_logger)
        for key in ['story_segments', 'story_summary']:
            if key in optimized_session:
                session[key] = optimized_session[key]

    app_logger.info("Storyline-based segment generated successfully", extra={
        'event': 'storyline_segment_success',
        'session_id': current_session_id,
        'segment_number': len(session['story_segments']),
        'segment_length': len(next_segment),
        'think_sections_count': len(think_sections),
        'selected_storyline_title': selected_storyline['title'],
        'image_generated': image_url is not None
    })

    response_data = {
        'success': True,
        'segment': next_segment,
        'think_sections': think_sections,
        'segment_number': len(session['story_segments']),
        'selected_storyline': selected_storyline
    }

    if image_url:
        response_data['image_url'] = image_url

    return jsonify(response_data)

@app.route('/api/get_story_mode', methods=['GET'])
def get_story_mode():
    """Get current story mode (free or guided)"""
    return jsonify({
        'story_mode': session.get('story_mode', 'free')
    })

@app.route('/api/set_story_mode', methods=['POST'])
def set_story_mode():
    """Set story mode (free or guided)"""
    data = request.get_json()
    new_mode = data.get('mode', 'free')

    # Validate mode
    if new_mode not in ['free', 'guided']:
        return jsonify({
            'success': False,
            'error': 'Invalid mode. Must be "free" or "guided"'
        })

    old_mode = session.get('story_mode', 'free')
    session['story_mode'] = new_mode

    # Clear pending storylines when switching modes
    if old_mode != new_mode:
        session['pending_storylines'] = []
        session['selected_storyline'] = ""

    app_logger.info("Story mode changed", extra={
        'event': 'story_mode_change',
        'session_id': session.get('session_id', 'unknown'),
        'old_mode': old_mode,
        'new_mode': new_mode
    })

    return jsonify({
        'success': True,
        'story_mode': new_mode
    })

if __name__ == '__main__':
    print("=== Doom-scrolling Storyteller ===")
    print("Make sure to set your OPENROUTER_API_KEY environment variable!")
    print("Example: export OPENROUTER_API_KEY='your-api-key-here'")
    print("Starting server at http://127.0.0.1:5000")

    app_logger.info("Application starting", extra={
        'event': 'app_start',
        'model_config': MODEL_CONFIG,
        'debug_mode': True,
        'host': '127.0.0.1',
        'port': 5000,
        'log_level': os.getenv('LOG_LEVEL', 'INFO'),
        'log_to_file': os.getenv('LOG_TO_FILE', 'true')
    })

    try:
        app.run(debug=True, host='127.0.0.1', port=5000)
    except KeyboardInterrupt:
        app_logger.info("Application shutdown requested", extra={
            'event': 'app_shutdown',
            'reason': 'keyboard_interrupt'
        })
    except Exception as e:
        app_logger.error("Application crashed", extra={
            'event': 'app_crash',
            'error': str(e)
        }, exc_info=True)
