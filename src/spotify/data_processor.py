"""
Data processor for Spotify data.
Handles data preprocessing, feature extraction, and preparation for ML models.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime
import json

from .api_client import SpotifyClient
from ..utils.helpers import (
    normalize_audio_features, 
    create_track_embedding, 
    calculate_popularity_score,
    calculate_novelty_score,
    save_data,
    load_data
)
from ..utils.config import config

logger = logging.getLogger(__name__)

class SpotifyDataProcessor:
    """Processes and prepares Spotify data for machine learning models."""
    
    def __init__(self, spotify_client: SpotifyClient):
        """Initialize data processor with Spotify client."""
        self.spotify_client = spotify_client
        self.track_features_cache = {}
        self.user_history = []
    
    def process_user_data(self, user_id: str = None) -> Dict[str, Any]:
        """Process all user data for recommendations."""
        if not self.spotify_client.is_authenticated():
            logger.error("Spotify client not authenticated")
            return {}
        
        try:
            # Get user profile
            user_profile = self.spotify_client.get_user_profile()
            
            # Get user's listening history
            top_tracks = self.spotify_client.get_user_top_tracks(limit=50)
            top_artists = self.spotify_client.get_user_top_artists(limit=50)
            recently_played = self.spotify_client.get_recently_played(limit=50)
            user_playlists = self.spotify_client.get_user_playlists(limit=50)
            
            # Process tracks and get audio features
            all_tracks = top_tracks + recently_played
            track_ids = [track['id'] for track in all_tracks if track.get('id')]
            
            # Get audio features for all tracks
            audio_features = self.spotify_client.get_track_audio_features(track_ids)
            
            # Create track embeddings
            track_embeddings = {}
            for track, features in zip(all_tracks, audio_features):
                if features:
                    embedding = create_track_embedding(track, features)
                    track_embeddings[track['id']] = {
                        'embedding': embedding,
                        'features': features,
                        'track_data': track
                    }
            
            # Extract user preferences
            user_preferences = self._extract_user_preferences(
                top_tracks, top_artists, audio_features
            )
            
            # Create user profile
            user_data = {
                'user_id': user_profile['id'] if user_profile else user_id,
                'display_name': user_profile['display_name'] if user_profile else 'Unknown',
                'top_tracks': top_tracks,
                'top_artists': top_artists,
                'recently_played': recently_played,
                'user_playlists': user_playlists,
                'track_embeddings': track_embeddings,
                'user_preferences': user_preferences,
                'processed_at': datetime.now().isoformat()
            }
            
            # Save processed data
            save_data(user_data, f'user_data_{user_data["user_id"]}.json')
            
            return user_data
            
        except Exception as e:
            logger.error(f"Error processing user data: {e}")
            return {}
    
    def _extract_user_preferences(self, 
                                 top_tracks: List[Dict], 
                                 top_artists: List[Dict],
                                 audio_features: List[Dict]) -> Dict[str, Any]:
        """Extract user preferences from listening history."""
        preferences = {
            'favorite_artists': [],
            'favorite_genres': [],
            'audio_preferences': {},
            'popularity_preference': 0.0,
            'diversity_score': 0.0
        }
        
        # Extract favorite artists
        preferences['favorite_artists'] = [
            {
                'id': artist['id'],
                'name': artist['name'],
                'popularity': artist.get('popularity', 0)
            }
            for artist in top_artists
        ]
        
        # Extract genres from top artists
        all_genres = []
        for artist in top_artists:
            all_genres.extend(artist.get('genres', []))
        
        # Count genre frequency
        genre_counts = {}
        for genre in all_genres:
            genre_counts[genre] = genre_counts.get(genre, 0) + 1
        
        preferences['favorite_genres'] = [
            {'genre': genre, 'count': count}
            for genre, count in sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        ]
        
        # Calculate audio preferences
        if audio_features:
            audio_df = pd.DataFrame(audio_features)
            preferences['audio_preferences'] = {
                'acousticness': float(audio_df['acousticness'].mean()),
                'danceability': float(audio_df['danceability'].mean()),
                'energy': float(audio_df['energy'].mean()),
                'instrumentalness': float(audio_df['instrumentalness'].mean()),
                'liveness': float(audio_df['liveness'].mean()),
                'speechiness': float(audio_df['speechiness'].mean()),
                'valence': float(audio_df['valence'].mean()),
                'tempo': float(audio_df['tempo'].mean()),
                'loudness': float(audio_df['loudness'].mean())
            }
        
        # Calculate popularity preference
        if top_tracks:
            avg_popularity = np.mean([track.get('popularity', 0) for track in top_tracks])
            preferences['popularity_preference'] = avg_popularity / 100.0
        
        # Calculate diversity score
        unique_artists = len(set(track['artists'][0]['id'] for track in top_tracks if track.get('artists')))
        preferences['diversity_score'] = unique_artists / len(top_tracks) if top_tracks else 0.0
        
        return preferences
    
    def process_track_data(self, track_data: Dict, audio_features: Dict = None) -> Dict[str, Any]:
        """Process individual track data for recommendations."""
        processed_track = {
            'id': track_data['id'],
            'name': track_data['name'],
            'artists': [
                {
                    'id': artist['id'],
                    'name': artist['name']
                }
                for artist in track_data.get('artists', [])
            ],
            'album': {
                'id': track_data['album']['id'],
                'name': track_data['album']['name'],
                'release_date': track_data['album'].get('release_date', ''),
                'images': track_data['album'].get('images', [])
            },
            'duration_ms': track_data.get('duration_ms', 0),
            'popularity': track_data.get('popularity', 0),
            'explicit': track_data.get('explicit', False),
            'external_urls': track_data.get('external_urls', {}),
            'preview_url': track_data.get('preview_url', ''),
            'processed_at': datetime.now().isoformat()
        }
        
        # Add audio features if provided
        if audio_features:
            processed_track['audio_features'] = audio_features
            processed_track['embedding'] = create_track_embedding(track_data, audio_features)
        
        return processed_track
    
    def get_track_features(self, track_id: str) -> Optional[Dict]:
        """Get track features with caching."""
        if track_id in self.track_features_cache:
            return self.track_features_cache[track_id]
        
        try:
            track_data = self.spotify_client.get_track(track_id)
            audio_features = self.spotify_client.get_track_audio_features_single(track_id)
            
            if track_data and audio_features:
                features = {
                    'track_data': track_data,
                    'audio_features': audio_features,
                    'embedding': create_track_embedding(track_data, audio_features)
                }
                self.track_features_cache[track_id] = features
                return features
            
        except Exception as e:
            logger.error(f"Error getting track features for {track_id}: {e}")
        
        return None
    
    def create_training_data(self, user_data: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create training data for ML models."""
        track_embeddings = user_data.get('track_embeddings', {})
        
        if not track_embeddings:
            logger.warning("No track embeddings available for training")
            return np.array([]), np.array([]), np.array([])
        
        # Extract features and create training data
        features = []
        track_ids = []
        user_ratings = []
        
        for track_id, track_info in track_embeddings.items():
            features.append(track_info['embedding'])
            track_ids.append(track_id)
            
            # Create synthetic ratings based on track position in user's history
            # This is a simplified approach - in a real system, you'd use actual user ratings
            rating = np.random.uniform(0.7, 1.0)  # Simulate positive ratings for user's tracks
            user_ratings.append(rating)
        
        return np.array(features), np.array(track_ids), np.array(user_ratings)
    
    def prepare_recommendation_data(self, 
                                  user_data: Dict, 
                                  candidate_tracks: List[Dict]) -> Dict[str, Any]:
        """Prepare data for recommendation generation."""
        # Get user preferences
        user_preferences = user_data.get('user_preferences', {})
        user_history = [track['id'] for track in user_data.get('top_tracks', [])]
        
        # Process candidate tracks
        processed_candidates = []
        candidate_embeddings = []
        
        for track in candidate_tracks:
            # Get audio features
            audio_features = self.spotify_client.get_track_audio_features_single(track['id'])
            
            if audio_features:
                # Process track
                processed_track = self.process_track_data(track, audio_features)
                
                # Calculate scores
                popularity_score = calculate_popularity_score(track)
                novelty_score = calculate_novelty_score(track, user_history)
                
                processed_track['scores'] = {
                    'popularity': popularity_score,
                    'novelty': novelty_score,
                    'diversity_boost': 1.0  # Will be updated during recommendation
                }
                
                processed_candidates.append(processed_track)
                candidate_embeddings.append(processed_track['embedding'])
        
        return {
            'candidate_tracks': processed_candidates,
            'candidate_embeddings': np.array(candidate_embeddings),
            'user_preferences': user_preferences,
            'user_history': user_history
        }
    
    def update_user_history(self, track_id: str, rating: float = 1.0) -> None:
        """Update user history with new interaction."""
        self.user_history.append({
            'track_id': track_id,
            'rating': rating,
            'timestamp': datetime.now().isoformat()
        })
        
        # Keep only recent history (last 100 interactions)
        if len(self.user_history) > 100:
            self.user_history = self.user_history[-100:]
    
    def get_user_history_track_ids(self) -> List[str]:
        """Get list of track IDs from user history."""
        return [item['track_id'] for item in self.user_history]
    
    def calculate_contextual_features(self, 
                                    track_data: Dict, 
                                    context: Dict[str, Any]) -> np.ndarray:
        """Calculate contextual features for a track."""
        # Base audio features
        audio_features = track_data.get('audio_features', {})
        base_features = list(normalize_audio_features(audio_features).values())
        
        # Context features
        context_features = []
        
        # Time of day features
        time_category = context.get('time_category', 'afternoon')
        time_encoding = {
            'morning': [1, 0, 0, 0],
            'afternoon': [0, 1, 0, 0],
            'evening': [0, 0, 1, 0],
            'night': [0, 0, 0, 1]
        }
        context_features.extend(time_encoding.get(time_category, [0, 0, 0, 0]))
        
        # Day of week features
        day_of_week = context.get('day_of_week', 'monday')
        day_encoding = {
            'monday': [1, 0, 0, 0, 0, 0, 0],
            'tuesday': [0, 1, 0, 0, 0, 0, 0],
            'wednesday': [0, 0, 1, 0, 0, 0, 0],
            'thursday': [0, 0, 0, 1, 0, 0, 0],
            'friday': [0, 0, 0, 0, 1, 0, 0],
            'saturday': [0, 0, 0, 0, 0, 1, 0],
            'sunday': [0, 0, 0, 0, 0, 0, 1]
        }
        context_features.extend(day_encoding.get(day_of_week, [0, 0, 0, 0, 0, 0, 0]))
        
        # Weekend vs weekday
        context_features.append(1.0 if context.get('is_weekend', False) else 0.0)
        
        # Mood features (if provided)
        mood = context.get('mood', 'neutral')
        mood_encoding = {
            'energetic': [1, 0, 0, 0, 0],
            'chill': [0, 1, 0, 0, 0],
            'happy': [0, 0, 1, 0, 0],
            'melancholic': [0, 0, 0, 1, 0],
            'focused': [0, 0, 0, 0, 1]
        }
        context_features.extend(mood_encoding.get(mood, [0, 0, 0, 0, 0]))
        
        # Activity features (if provided)
        activity = context.get('activity', 'general')
        activity_encoding = {
            'workout': [1, 0, 0, 0, 0],
            'study': [0, 1, 0, 0, 0],
            'commute': [0, 0, 1, 0, 0],
            'party': [0, 0, 0, 1, 0],
            'sleep': [0, 0, 0, 0, 1]
        }
        context_features.extend(activity_encoding.get(activity, [0, 0, 0, 0, 0]))
        
        # Combine all features
        all_features = base_features + context_features
        
        return np.array(all_features, dtype=np.float32)
    
    def save_processed_data(self, data: Dict, filename: str) -> None:
        """Save processed data to file."""
        save_data(data, filename)
    
    def load_processed_data(self, filename: str) -> Optional[Dict]:
        """Load processed data from file."""
        return load_data(filename) 