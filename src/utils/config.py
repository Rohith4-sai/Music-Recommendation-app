"""
Configuration management for the music recommendation app.
Handles environment variables, API credentials, and app settings.
"""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for the music recommendation app."""
    
    # Spotify API Configuration
    SPOTIFY_CLIENT_ID: str = os.getenv('SPOTIFY_CLIENT_ID', '')
    SPOTIFY_CLIENT_SECRET: str = os.getenv('SPOTIFY_CLIENT_SECRET', '')
    SPOTIFY_REDIRECT_URI: str = 'http://localhost:8501/callback'
    
    # App Configuration
    DEBUG: bool = os.getenv('DEBUG', 'True').lower() == 'true'
    LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')
    
    # Paths
    MODEL_SAVE_PATH: str = os.getenv('MODEL_SAVE_PATH', './models')
    DATA_PATH: str = os.getenv('DATA_PATH', './data')
    
    # Recommendation Settings
    DEFAULT_RECOMMENDATION_COUNT: int = int(os.getenv('DEFAULT_RECOMMENDATION_COUNT', '20'))
    EXPLORATION_RATE: float = float(os.getenv('EXPLORATION_RATE', '0.3'))
    DIVERSITY_WEIGHT: float = float(os.getenv('DIVERSITY_WEIGHT', '0.4'))
    NOVELTY_WEIGHT: float = float(os.getenv('NOVELTY_WEIGHT', '0.3'))
    
    # ML Model Parameters
    EMBEDDING_DIM: int = 64
    HIDDEN_LAYERS: list = [128, 64, 32]
    LEARNING_RATE: float = 0.001
    BATCH_SIZE: int = 32
    EPOCHS: int = 50
    
    # Debiasing Parameters
    POPULARITY_ALPHA: float = 0.7  # Controls popularity bias reduction
    DIVERSITY_THRESHOLD: float = 0.3  # Minimum diversity ratio
    NOVELTY_BOOST: float = 1.5  # Multiplier for novel tracks
    
    # Spotify API Scopes
    SPOTIFY_SCOPES: list = [
        'user-read-private',
        'user-read-email',
        'user-top-read',
        'user-read-recently-played',
        'user-read-playback-state',
        'user-modify-playback-state',
        'playlist-read-private',
        'playlist-modify-public',
        'playlist-modify-private'
    ]
    
    @classmethod
    def validate_credentials(cls) -> bool:
        """Validate that Spotify credentials are properly configured."""
        return bool(cls.SPOTIFY_CLIENT_ID and cls.SPOTIFY_CLIENT_SECRET)
    
    @classmethod
    def get_spotify_scopes_string(cls) -> str:
        """Get Spotify scopes as a space-separated string."""
        return ' '.join(cls.SPOTIFY_SCOPES)
    
    @classmethod
    def create_directories(cls) -> None:
        """Create necessary directories if they don't exist."""
        os.makedirs(cls.MODEL_SAVE_PATH, exist_ok=True)
        os.makedirs(cls.DATA_PATH, exist_ok=True)

# Global config instance
config = Config() 