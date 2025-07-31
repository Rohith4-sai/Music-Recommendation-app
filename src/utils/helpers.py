"""
Utility functions for the music recommendation app.
Includes data processing, logging, and common operations.
"""

import logging
import json
import pickle
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from pathlib import Path

from .config import config

def setup_logging() -> logging.Logger:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('app.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def save_data(data: Any, filename: str, data_type: str = 'json') -> None:
    """Save data to file with specified format."""
    filepath = Path(config.DATA_PATH) / filename
    
    if data_type == 'json':
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    elif data_type == 'pickle':
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    elif data_type == 'csv' and isinstance(data, pd.DataFrame):
        data.to_csv(filepath, index=False)
    
    logging.info(f"Data saved to {filepath}")

def load_data(filename: str, data_type: str = 'json') -> Any:
    """Load data from file with specified format."""
    filepath = Path(config.DATA_PATH) / filename
    
    if not filepath.exists():
        logging.warning(f"File {filepath} does not exist")
        return None
    
    try:
        if data_type == 'json':
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        elif data_type == 'pickle':
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        elif data_type == 'csv':
            return pd.read_csv(filepath)
    except Exception as e:
        logging.error(f"Error loading {filepath}: {e}")
        return None

def calculate_popularity_score(track_data: Dict) -> float:
    """Calculate a normalized popularity score for a track."""
    # Spotify popularity is 0-100, normalize to 0-1
    popularity = track_data.get('popularity', 0) / 100.0
    
    # Apply popularity bias reduction
    return popularity ** config.POPULARITY_ALPHA

def calculate_novelty_score(track_data: Dict, user_history: List[str]) -> float:
    """Calculate novelty score based on user's listening history."""
    artist_id = track_data.get('artists', [{}])[0].get('id', '')
    track_id = track_data.get('id', '')
    
    # Check if artist is new to user
    artist_novelty = 1.0 if artist_id not in user_history else 0.3
    track_novelty = 1.0 if track_id not in user_history else 0.1
    
    # Combine novelty scores
    novelty_score = (artist_novelty * 0.7 + track_novelty * 0.3)
    
    return novelty_score * config.NOVELTY_BOOST

def calculate_diversity_score(recommendations: List[Dict]) -> float:
    """Calculate diversity score for a set of recommendations."""
    if not recommendations:
        return 0.0
    
    # Extract unique artists
    artists = set()
    for track in recommendations:
        for artist in track.get('artists', []):
            artists.add(artist.get('id', ''))
    
    # Diversity ratio: unique artists / total tracks
    diversity_ratio = len(artists) / len(recommendations)
    
    return diversity_ratio

def get_time_context() -> Dict[str, Any]:
    """Get current time context for recommendations."""
    now = datetime.now()
    
    # Time of day categories
    hour = now.hour
    if 6 <= hour < 12:
        time_category = 'morning'
    elif 12 <= hour < 17:
        time_category = 'afternoon'
    elif 17 <= hour < 22:
        time_category = 'evening'
    else:
        time_category = 'night'
    
    # Day of week
    day_of_week = now.strftime('%A').lower()
    
    # Weekend vs weekday
    is_weekend = day_of_week in ['saturday', 'sunday']
    
    return {
        'hour': hour,
        'time_category': time_category,
        'day_of_week': day_of_week,
        'is_weekend': is_weekend,
        'timestamp': now.isoformat()
    }

def normalize_audio_features(features: Dict) -> Dict[str, float]:
    """Normalize Spotify audio features to 0-1 range."""
    normalized = {}
    
    # Features that are already 0-1
    zero_one_features = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'speechiness', 'valence']
    
    for feature in zero_one_features:
        if feature in features:
            normalized[feature] = features[feature]
    
    # Features that need normalization
    if 'tempo' in features:
        # Normalize tempo (typically 50-200 BPM) to 0-1
        normalized['tempo'] = (features['tempo'] - 50) / 150
    
    if 'loudness' in features:
        # Normalize loudness (typically -60 to 0 dB) to 0-1
        normalized['loudness'] = (features['loudness'] + 60) / 60
    
    if 'key' in features:
        # Normalize key (0-11) to 0-1
        normalized['key'] = features['key'] / 11
    
    if 'mode' in features:
        # Mode is already 0-1 (0=minor, 1=major)
        normalized['mode'] = features['mode']
    
    return normalized

def create_track_embedding(track_data: Dict, audio_features: Dict) -> np.ndarray:
    """Create a feature embedding for a track."""
    features = []
    
    # Audio features
    normalized_features = normalize_audio_features(audio_features)
    features.extend([
        normalized_features.get('acousticness', 0),
        normalized_features.get('danceability', 0),
        normalized_features.get('energy', 0),
        normalized_features.get('instrumentalness', 0),
        normalized_features.get('liveness', 0),
        normalized_features.get('speechiness', 0),
        normalized_features.get('valence', 0),
        normalized_features.get('tempo', 0),
        normalized_features.get('loudness', 0),
        normalized_features.get('key', 0),
        normalized_features.get('mode', 0)
    ])
    
    # Popularity (normalized)
    features.append(calculate_popularity_score(track_data))
    
    # Duration (normalized to 0-1, assuming max 10 minutes)
    duration = track_data.get('duration_ms', 0) / (10 * 60 * 1000)
    features.append(min(duration, 1.0))
    
    return np.array(features, dtype=np.float32)

def format_duration(ms: int) -> str:
    """Format duration from milliseconds to MM:SS format."""
    seconds = ms // 1000
    minutes = seconds // 60
    seconds = seconds % 60
    return f"{minutes:02d}:{seconds:02d}"

def get_mood_suggestions() -> List[str]:
    """Get predefined mood categories for user selection."""
    return [
        'energetic', 'chill', 'happy', 'melancholic', 'focused', 
        'romantic', 'nostalgic', 'adventurous', 'relaxed', 'motivated'
    ]

def get_activity_suggestions() -> List[str]:
    """Get predefined activity categories for user selection."""
    return [
        'workout', 'study', 'commute', 'cooking', 'cleaning',
        'party', 'sleep', 'meditation', 'social', 'creative'
    ]

def validate_track_data(track_data: Dict) -> bool:
    """Validate that track data contains required fields."""
    required_fields = ['id', 'name', 'artists', 'duration_ms']
    return all(field in track_data for field in required_fields)

def create_session_id() -> str:
    """Create a unique session ID for user tracking."""
    return datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3] 