"""
Spotify API client for the music recommendation app.
Handles authentication, data fetching, and playlist management.
"""

import spotipy
from spotipy.oauth2 import SpotifyOAuth
from typing import Dict, List, Optional, Any
import logging
import time
from urllib.parse import urlencode

from ..utils.config import config
from ..utils.helpers import setup_logging

logger = setup_logging()

class SpotifyClient:
    """Spotify API client with authentication and data fetching capabilities."""
    
    def __init__(self):
        """Initialize Spotify client with OAuth authentication."""
        self.sp = None
        self.user_id = None
        self._authenticate()
    
    def _authenticate(self) -> None:
        """Authenticate with Spotify using OAuth."""
        try:
            if not config.validate_credentials():
                logger.error("Spotify credentials not configured. Please set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET")
                return
            
            self.sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
                client_id=config.SPOTIFY_CLIENT_ID,
                client_secret=config.SPOTIFY_CLIENT_SECRET,
                redirect_uri=config.SPOTIFY_REDIRECT_URI,
                scope=config.get_spotify_scopes_string(),
                cache_handler=spotipy.cache_handler.CacheFileHandler(cache_path='.spotify_cache')
            ))
            
            # Get user info
            user_info = self.sp.current_user()
            self.user_id = user_info['id']
            logger.info(f"Authenticated as user: {user_info['display_name']}")
            
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            self.sp = None
    
    def is_authenticated(self) -> bool:
        """Check if client is properly authenticated."""
        return self.sp is not None and self.user_id is not None
    
    def get_user_profile(self) -> Optional[Dict]:
        """Get current user's profile information."""
        if not self.is_authenticated():
            return None
        
        try:
            return self.sp.current_user()
        except Exception as e:
            logger.error(f"Error fetching user profile: {e}")
            return None
    
    def get_user_top_tracks(self, limit: int = 50, time_range: str = 'medium_term') -> List[Dict]:
        """Get user's top tracks."""
        if not self.is_authenticated():
            return []
        
        try:
            results = self.sp.current_user_top_tracks(
                limit=limit,
                offset=0,
                time_range=time_range
            )
            return results.get('items', [])
        except Exception as e:
            logger.error(f"Error fetching top tracks: {e}")
            return []
    
    def get_user_top_artists(self, limit: int = 50, time_range: str = 'medium_term') -> List[Dict]:
        """Get user's top artists."""
        if not self.is_authenticated():
            return []
        
        try:
            results = self.sp.current_user_top_artists(
                limit=limit,
                offset=0,
                time_range=time_range
            )
            return results.get('items', [])
        except Exception as e:
            logger.error(f"Error fetching top artists: {e}")
            return []
    
    def get_recently_played(self, limit: int = 50) -> List[Dict]:
        """Get user's recently played tracks."""
        if not self.is_authenticated():
            return []
        
        try:
            results = self.sp.current_user_recently_played(limit=limit)
            return [item['track'] for item in results.get('items', [])]
        except Exception as e:
            logger.error(f"Error fetching recently played: {e}")
            return []
    
    def get_user_playlists(self, limit: int = 50) -> List[Dict]:
        """Get user's playlists."""
        if not self.is_authenticated():
            return []
        
        try:
            results = self.sp.current_user_playlists(limit=limit)
            return results.get('items', [])
        except Exception as e:
            logger.error(f"Error fetching playlists: {e}")
            return []
    
    def get_playlist_tracks(self, playlist_id: str, limit: int = 100) -> List[Dict]:
        """Get tracks from a specific playlist."""
        if not self.is_authenticated():
            return []
        
        try:
            results = self.sp.playlist_tracks(playlist_id, limit=limit)
            return [item['track'] for item in results.get('items', []) if item['track']]
        except Exception as e:
            logger.error(f"Error fetching playlist tracks: {e}")
            return []
    
    def search_tracks(self, query: str, limit: int = 20, market: str = 'US') -> List[Dict]:
        """Search for tracks."""
        if not self.is_authenticated():
            return []
        
        try:
            results = self.sp.search(
                q=query,
                type='track',
                limit=limit,
                market=market
            )
            return results.get('tracks', {}).get('items', [])
        except Exception as e:
            logger.error(f"Error searching tracks: {e}")
            return []
    
    def get_track_audio_features(self, track_ids: List[str]) -> List[Dict]:
        """Get audio features for multiple tracks."""
        if not self.is_authenticated() or not track_ids:
            return []
        
        try:
            # Spotify API allows max 100 tracks per request
            features = []
            for i in range(0, len(track_ids), 100):
                batch = track_ids[i:i+100]
                batch_features = self.sp.audio_features(batch)
                features.extend([f for f in batch_features if f])
            
            return features
        except Exception as e:
            logger.error(f"Error fetching audio features: {e}")
            return []
    
    def get_track_audio_features_single(self, track_id: str) -> Optional[Dict]:
        """Get audio features for a single track."""
        if not self.is_authenticated():
            return None
        
        try:
            features = self.sp.audio_features([track_id])
            return features[0] if features else None
        except Exception as e:
            logger.error(f"Error fetching audio features for {track_id}: {e}")
            return None
    
    def get_recommendations(self, 
                          seed_tracks: List[str] = None,
                          seed_artists: List[str] = None,
                          seed_genres: List[str] = None,
                          limit: int = 20,
                          **kwargs) -> List[Dict]:
        """Get track recommendations based on seeds."""
        if not self.is_authenticated():
            return []
        
        try:
            # Ensure we have at least one seed
            if not any([seed_tracks, seed_artists, seed_genres]):
                logger.warning("No seeds provided for recommendations")
                return []
            
            # Limit seeds to 5 total (Spotify API limit)
            total_seeds = 0
            if seed_tracks:
                seed_tracks = seed_tracks[:min(5, len(seed_tracks))]
                total_seeds += len(seed_tracks)
            
            if seed_artists and total_seeds < 5:
                remaining = 5 - total_seeds
                seed_artists = seed_artists[:remaining]
                total_seeds += len(seed_artists)
            
            if seed_genres and total_seeds < 5:
                remaining = 5 - total_seeds
                seed_genres = seed_genres[:remaining]
            
            results = self.sp.recommendations(
                seed_tracks=seed_tracks,
                seed_artists=seed_artists,
                seed_genres=seed_genres,
                limit=limit,
                **kwargs
            )
            
            return results.get('tracks', [])
        except Exception as e:
            logger.error(f"Error getting recommendations: {e}")
            return []
    
    def get_available_genres(self) -> List[str]:
        """Get list of available genres for recommendations."""
        if not self.is_authenticated():
            return []
        
        try:
            results = self.sp.recommendation_genres()
            return results.get('genres', [])
        except Exception as e:
            logger.error(f"Error fetching genres: {e}")
            return []
    
    def create_playlist(self, name: str, description: str = "", public: bool = False) -> Optional[str]:
        """Create a new playlist for the user."""
        if not self.is_authenticated():
            return None
        
        try:
            playlist = self.sp.user_playlist_create(
                user=self.user_id,
                name=name,
                description=description,
                public=public
            )
            return playlist['id']
        except Exception as e:
            logger.error(f"Error creating playlist: {e}")
            return None
    
    def add_tracks_to_playlist(self, playlist_id: str, track_ids: List[str]) -> bool:
        """Add tracks to a playlist."""
        if not self.is_authenticated():
            return False
        
        try:
            # Spotify API allows max 100 tracks per request
            for i in range(0, len(track_ids), 100):
                batch = track_ids[i:i+100]
                self.sp.playlist_add_items(playlist_id, batch)
            
            return True
        except Exception as e:
            logger.error(f"Error adding tracks to playlist: {e}")
            return False
    
    def get_track(self, track_id: str) -> Optional[Dict]:
        """Get detailed information about a specific track."""
        if not self.is_authenticated():
            return None
        
        try:
            return self.sp.track(track_id)
        except Exception as e:
            logger.error(f"Error fetching track {track_id}: {e}")
            return None
    
    def get_artist(self, artist_id: str) -> Optional[Dict]:
        """Get detailed information about a specific artist."""
        if not self.is_authenticated():
            return None
        
        try:
            return self.sp.artist(artist_id)
        except Exception as e:
            logger.error(f"Error fetching artist {artist_id}: {e}")
            return None
    
    def get_artist_top_tracks(self, artist_id: str, country: str = 'US') -> List[Dict]:
        """Get top tracks for a specific artist."""
        if not self.is_authenticated():
            return []
        
        try:
            results = self.sp.artist_top_tracks(artist_id, country=country)
            return results.get('tracks', [])
        except Exception as e:
            logger.error(f"Error fetching artist top tracks: {e}")
            return []
    
    def get_artist_albums(self, artist_id: str, limit: int = 20) -> List[Dict]:
        """Get albums for a specific artist."""
        if not self.is_authenticated():
            return []
        
        try:
            results = self.sp.artist_albums(artist_id, limit=limit)
            return results.get('items', [])
        except Exception as e:
            logger.error(f"Error fetching artist albums: {e}")
            return []
    
    def get_album_tracks(self, album_id: str, limit: int = 50) -> List[Dict]:
        """Get tracks from a specific album."""
        if not self.is_authenticated():
            return []
        
        try:
            results = self.sp.album_tracks(album_id, limit=limit)
            return results.get('items', [])
        except Exception as e:
            logger.error(f"Error fetching album tracks: {e}")
            return []
    
    def get_new_releases(self, country: str = 'US', limit: int = 20) -> List[Dict]:
        """Get new releases."""
        if not self.is_authenticated():
            return []
        
        try:
            results = self.sp.new_releases(country=country, limit=limit)
            return results.get('albums', {}).get('items', [])
        except Exception as e:
            logger.error(f"Error fetching new releases: {e}")
            return []
    
    def get_featured_playlists(self, limit: int = 20, country: str = 'US') -> List[Dict]:
        """Get featured playlists."""
        if not self.is_authenticated():
            return []
        
        try:
            results = self.sp.featured_playlists(
                limit=limit,
                country=country
            )
            return results.get('playlists', {}).get('items', [])
        except Exception as e:
            logger.error(f"Error fetching featured playlists: {e}")
            return []
    
    def get_categories(self, country: str = 'US', limit: int = 20) -> List[Dict]:
        """Get available categories."""
        if not self.is_authenticated():
            return []
        
        try:
            results = self.sp.categories(country=country, limit=limit)
            return results.get('categories', {}).get('items', [])
        except Exception as e:
            logger.error(f"Error fetching categories: {e}")
            return []
    
    def get_category_playlists(self, category_id: str, country: str = 'US', limit: int = 20) -> List[Dict]:
        """Get playlists for a specific category."""
        if not self.is_authenticated():
            return []
        
        try:
            results = self.sp.category_playlists(
                category_id=category_id,
                country=country,
                limit=limit
            )
            return results.get('playlists', {}).get('items', [])
        except Exception as e:
            logger.error(f"Error fetching category playlists: {e}")
            return [] 