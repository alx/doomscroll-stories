#!/usr/bin/env python3
"""
Doom-scrolling storytelling web app with Bootstrap badge-based user interaction.
Uses OpenRouter API for story generation.
"""

import os
import json
import requests
import time
import uuid
from typing import List, Dict, Optional
from flask import Flask, render_template, request, jsonify, session
from flask_session import Session
from dotenv import load_dotenv

from logging_config import setup_logging, get_logger, truncate_text, sanitize_for_logging

# Load environment variables from .env file
load_dotenv()

# Initialize logging
setup_logging()

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
OPENROUTER_MODEL = os.getenv('OPENROUTER_MODEL', 'openai/gpt-4o-mini')
OPENROUTER_BASE_URL = 'https://openrouter.ai/api/v1'

# Story summarization configuration
SUMMARY_WINDOW_SIZE = int(os.getenv('SUMMARY_WINDOW_SIZE', '5'))  # Summarize after this many segments
RECENT_SEGMENTS_COUNT = int(os.getenv('RECENT_SEGMENTS_COUNT', '2'))  # Keep this many recent segments with summary

# Session optimization configuration
MAX_SESSION_SIZE_KB = int(os.getenv('MAX_SESSION_SIZE_KB', '50'))  # Max session size in KB before optimization
MAX_SEGMENTS_IN_SESSION = int(os.getenv('MAX_SEGMENTS_IN_SESSION', '10'))  # Max segments to keep in session

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
    def __init__(self, api_key: str, model: str = None):
        self.api_key = api_key
        self.model = model or OPENROUTER_MODEL
        self.base_url = OPENROUTER_BASE_URL
        self.logger = get_logger('llm_requests')
        self.include_responses = os.getenv('LOG_INCLUDE_RESPONSES', 'true').lower() == 'true'

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
            'request_id': request_id,
            'session_id': session_id,
            'model': self.model,
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
            'model': self.model,
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

            # Log successful response
            self.logger.info("LLM request completed successfully", extra={
                'event': 'llm_request_success',
                'request_id': request_id,
                'session_id': session_id,
                'model': self.model,
                'duration_ms': duration_ms,
                'status_code': response.status_code,
                'response_length': len(generated_content),
                'response_preview': truncate_text(generated_content, 200) if self.include_responses else None,
                'is_complete': is_complete
            })

            return generated_content

        except requests.exceptions.RequestException as e:
            duration_ms = int((time.time() - start_time) * 1000)

            # Log request error
            self.logger.error("LLM request failed", extra={
                'event': 'llm_request_error',
                'request_id': request_id,
                'session_id': session_id,
                'model': self.model,
                'duration_ms': duration_ms,
                'error': str(e),
                'status_code': getattr(e.response, 'status_code', None) if hasattr(e, 'response') else None
            })

            return f"Error generating story: {str(e)}. Please check your OpenRouter API key and try again."

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)

            # Log unexpected error
            self.logger.error("Unexpected error during LLM request", extra={
                'event': 'llm_request_unexpected_error',
                'request_id': request_id,
                'session_id': session_id,
                'model': self.model,
                'duration_ms': duration_ms,
                'error': str(e)
            }, exc_info=True)

            return f"Error generating story: {str(e)}. Please check your OpenRouter API key and try again."

    def generate_story_summary(self, story_segments: List[str]) -> str:
        """Generate a concise summary of the story so far"""

        # Generate unique request ID for tracking
        request_id = str(uuid.uuid4())
        session_id = session.get('session_id', 'unknown')
        start_time = time.time()

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
            'request_id': request_id,
            'session_id': session_id,
            'model': self.model,
            'segments_count': len(story_segments),
            'total_length': len(full_story)
        })

        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }

        data = {
            'model': self.model,
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
                'request_id': request_id,
                'session_id': session_id,
                'model': self.model,
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
                'request_id': request_id,
                'session_id': session_id,
                'model': self.model,
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
                'request_id': request_id,
                'session_id': session_id,
                'model': self.model,
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
            'request_id': request_id,
            'session_id': session_id,
            'model': self.model,
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
            'model': self.model,
            'messages': [
                {
                    'role': 'system',
                    'content': 'You are a creative storyteller who writes engaging story openings. Create compelling beginnings that hook readers immediately.'
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
                self.logger.warning("Generated initial story may be incomplete", extra={
                    'event': 'initial_story_incomplete',
                    'request_id': request_id,
                    'session_id': session_id,
                    'content_preview': truncate_text(generated_content, 100),
                    'ends_with': generated_content[-20:] if len(generated_content) > 20 else generated_content
                })

            # Log successful response
            self.logger.info("Initial story LLM request completed successfully", extra={
                'event': 'llm_initial_story_success',
                'request_id': request_id,
                'session_id': session_id,
                'model': self.model,
                'duration_ms': duration_ms,
                'status_code': response.status_code,
                'response_length': len(generated_content),
                'response_preview': truncate_text(generated_content, 200) if self.include_responses else None,
                'is_complete': is_complete
            })

            return generated_content

        except requests.exceptions.RequestException as e:
            duration_ms = int((time.time() - start_time) * 1000)

            # Log request error
            self.logger.error("Initial story LLM request failed", extra={
                'event': 'llm_initial_story_error',
                'request_id': request_id,
                'session_id': session_id,
                'model': self.model,
                'duration_ms': duration_ms,
                'error': str(e),
                'status_code': getattr(e.response, 'status_code', None) if hasattr(e, 'response') else None
            })

            return f"Error generating story: {str(e)}. Please check your OpenRouter API key and try again."

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)

            # Log unexpected error
            self.logger.error("Unexpected error during initial story LLM request", extra={
                'event': 'llm_initial_story_unexpected_error',
                'request_id': request_id,
                'session_id': session_id,
                'model': self.model,
                'duration_ms': duration_ms,
                'error': str(e)
            }, exc_info=True)

            return f"Error generating story: {str(e)}. Please check your OpenRouter API key and try again."

# Initialize story generator
story_generator = StoryGenerator(OPENROUTER_API_KEY, OPENROUTER_MODEL)

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
    initial_segment = story_generator.generate_initial_story(active_badges, active_characters)

    # Reset and store in session
    session['story_segments'] = [initial_segment]
    session['story_summary'] = ""  # Reset summary for new story

    app_logger.info("Story started successfully", extra={
        'event': 'story_start_success',
        'session_id': session.get('session_id', 'unknown'),
        'segment_number': len(session['story_segments']),
        'segment_length': len(initial_segment)
    })

    return jsonify({
        'success': True,
        'segment': initial_segment,
        'segment_number': len(session['story_segments'])
    })

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
    next_segment = story_generator.generate_story_segment(story_context, active_badges, active_characters)

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
        'summary_active': bool(session['story_summary'])
    })

    app_logger.info("Story continued successfully", extra={
        'event': 'story_continue_success',
        'session_id': session.get('session_id', 'unknown'),
        'segment_number': len(session['story_segments']),
        'segment_length': len(next_segment),
        'total_segments': len(session['story_segments']),
        'summary_active': bool(session['story_summary'])
    })

    return jsonify({
        'success': True,
        'segment': next_segment,
        'segment_number': len(session['story_segments'])
    })

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

if __name__ == '__main__':
    print("=== Doom-scrolling Storyteller ===")
    print("Make sure to set your OPENROUTER_API_KEY environment variable!")
    print("Example: export OPENROUTER_API_KEY='your-api-key-here'")
    print("Starting server at http://127.0.0.1:5000")

    app_logger.info("Application starting", extra={
        'event': 'app_start',
        'model': OPENROUTER_MODEL,
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
