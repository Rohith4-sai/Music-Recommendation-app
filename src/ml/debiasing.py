"""
Debiasing mechanisms for the music recommendation system.
Implements techniques to reduce popularity bias, ensure diversity, and promote fairness.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import random

from ..utils.config import config
from ..utils.helpers import calculate_diversity_score, calculate_novelty_score

logger = logging.getLogger(__name__)

class PopularityDebiasing:
    """Popularity debiasing techniques to reduce bias towards popular tracks."""
    
    def __init__(self, alpha: float = 0.7):
        """Initialize popularity debiasing with alpha parameter."""
        self.alpha = alpha
        self.popularity_thresholds = {}
    
    def apply_popularity_normalization(self, tracks: List[Dict]) -> List[Dict]:
        """Apply popularity normalization to reduce bias."""
        try:
            # Extract popularity scores
            popularities = [track.get('popularity', 50) for track in tracks]
            
            # Calculate normalized popularity scores
            normalized_popularities = []
            for popularity in popularities:
                # Apply power law normalization
                normalized = (popularity / 100.0) ** self.alpha
                normalized_popularities.append(normalized)
            
            # Update tracks with normalized scores
            for i, track in enumerate(tracks):
                track['normalized_popularity'] = normalized_popularities[i]
                track['popularity_bias_reduction'] = 1.0 - normalized_popularities[i]
            
            return tracks
            
        except Exception as e:
            logger.error(f"Error in popularity normalization: {e}")
            return tracks
    
    def apply_popularity_penalty(self, tracks: List[Dict], penalty_strength: float = 0.3) -> List[Dict]:
        """Apply popularity penalty to highly popular tracks."""
        try:
            for track in tracks:
                popularity = track.get('popularity', 50) / 100.0
                
                # Calculate penalty based on popularity
                if popularity > 0.8:  # Very popular tracks
                    penalty = penalty_strength * (popularity - 0.8) * 5
                elif popularity > 0.6:  # Moderately popular tracks
                    penalty = penalty_strength * (popularity - 0.6) * 2.5
                else:
                    penalty = 0.0
                
                track['popularity_penalty'] = penalty
                track['popularity_adjusted_score'] = track.get('recommendation_score', 0) * (1 - penalty)
            
            return tracks
            
        except Exception as e:
            logger.error(f"Error in popularity penalty: {e}")
            return tracks
    
    def apply_popularity_aware_ranking(self, tracks: List[Dict]) -> List[Dict]:
        """Apply popularity-aware ranking that balances popularity with quality."""
        try:
            for track in tracks:
                popularity = track.get('popularity', 50) / 100.0
                base_score = track.get('recommendation_score', 0)
                
                # Calculate popularity-aware score
                # Higher weight for less popular tracks with good scores
                popularity_weight = 1.0 - (popularity * 0.5)
                adjusted_score = base_score * (1 + popularity_weight * 0.3)
                
                track['popularity_aware_score'] = adjusted_score
            
            return tracks
            
        except Exception as e:
            logger.error(f"Error in popularity-aware ranking: {e}")
            return tracks

class DiversityPromotion:
    """Techniques to promote diversity in recommendations."""
    
    def __init__(self, diversity_weight: float = 0.4):
        """Initialize diversity promotion with weight parameter."""
        self.diversity_weight = diversity_weight
        self.artist_clusters = {}
    
    def apply_diversity_boost(self, tracks: List[Dict]) -> List[Dict]:
        """Apply diversity boost to encourage variety in recommendations."""
        try:
            # Group tracks by artist
            artist_tracks = {}
            for track in tracks:
                artist_id = track.get('artists', [{}])[0].get('id', 'unknown')
                if artist_id not in artist_tracks:
                    artist_tracks[artist_id] = []
                artist_tracks[artist_id].append(track)
            
            # Calculate diversity boost for each track
            for track in tracks:
                artist_id = track.get('artists', [{}])[0].get('id', 'unknown')
                artist_count = len(artist_tracks[artist_id])
                
                # Higher boost for tracks from artists with fewer tracks
                diversity_boost = 1.0 / max(artist_count, 1)
                track['diversity_boost'] = diversity_boost
                
                # Apply diversity boost to recommendation score
                base_score = track.get('recommendation_score', 0)
                track['diversity_adjusted_score'] = base_score * (1 + diversity_boost * self.diversity_weight)
            
            return tracks
            
        except Exception as e:
            logger.error(f"Error in diversity boost: {e}")
            return tracks
    
    def apply_genre_diversity(self, tracks: List[Dict]) -> List[Dict]:
        """Apply genre diversity to ensure variety across musical genres."""
        try:
            # Extract genres from tracks
            track_genres = []
            for track in tracks:
                # Get genres from artist data (simplified)
                genres = track.get('genres', [])
                track_genres.append(genres)
            
            # Calculate genre diversity boost
            for i, track in enumerate(tracks):
                track_genre = track_genres[i]
                
                # Count how many other tracks have the same genre
                genre_overlap = 0
                for j, other_genres in enumerate(track_genres):
                    if i != j and any(g in other_genres for g in track_genre):
                        genre_overlap += 1
                
                # Higher boost for unique genres
                genre_diversity_boost = 1.0 / max(genre_overlap + 1, 1)
                track['genre_diversity_boost'] = genre_diversity_boost
            
            return tracks
            
        except Exception as e:
            logger.error(f"Error in genre diversity: {e}")
            return tracks
    
    def apply_temporal_diversity(self, tracks: List[Dict]) -> List[Dict]:
        """Apply temporal diversity to include tracks from different time periods."""
        try:
            # Extract release years
            release_years = []
            for track in tracks:
                release_date = track.get('album', {}).get('release_date', '')
                if release_date:
                    try:
                        year = int(release_date[:4])
                        release_years.append(year)
                    except:
                        release_years.append(2020)  # Default year
                else:
                    release_years.append(2020)
            
            # Calculate temporal diversity boost
            current_year = 2024
            for i, track in enumerate(tracks):
                year = release_years[i]
                years_old = current_year - year
                
                # Boost for older tracks (vintage appeal)
                if years_old > 20:
                    temporal_boost = 1.3
                elif years_old > 10:
                    temporal_boost = 1.2
                elif years_old > 5:
                    temporal_boost = 1.1
                else:
                    temporal_boost = 1.0
                
                track['temporal_diversity_boost'] = temporal_boost
            
            return tracks
            
        except Exception as e:
            logger.error(f"Error in temporal diversity: {e}")
            return tracks

class NoveltyPromotion:
    """Techniques to promote novelty and discovery in recommendations."""
    
    def __init__(self, novelty_weight: float = 0.3):
        """Initialize novelty promotion with weight parameter."""
        self.novelty_weight = novelty_weight
        self.user_history = []
    
    def set_user_history(self, user_history: List[str]):
        """Set user's listening history for novelty calculation."""
        self.user_history = user_history
    
    def apply_novelty_boost(self, tracks: List[Dict]) -> List[Dict]:
        """Apply novelty boost to encourage discovery of new artists/tracks."""
        try:
            for track in tracks:
                # Calculate novelty score
                novelty_score = calculate_novelty_score(track, self.user_history)
                
                # Apply novelty boost
                base_score = track.get('recommendation_score', 0)
                novelty_adjusted_score = base_score * (1 + novelty_score * self.novelty_weight)
                
                track['novelty_score'] = novelty_score
                track['novelty_adjusted_score'] = novelty_adjusted_score
            
            return tracks
            
        except Exception as e:
            logger.error(f"Error in novelty boost: {e}")
            return tracks
    
    def apply_artist_novelty(self, tracks: List[Dict]) -> List[Dict]:
        """Apply artist novelty to promote discovery of new artists."""
        try:
            # Count artist appearances
            artist_counts = {}
            for track in tracks:
                artist_id = track.get('artists', [{}])[0].get('id', 'unknown')
                artist_counts[artist_id] = artist_counts.get(artist_id, 0) + 1
            
            # Apply artist novelty boost
            for track in tracks:
                artist_id = track.get('artists', [{}])[0].get('id', 'unknown')
                artist_count = artist_counts[artist_id]
                
                # Higher boost for tracks from artists with fewer appearances
                artist_novelty_boost = 1.0 / max(artist_count, 1)
                track['artist_novelty_boost'] = artist_novelty_boost
            
            return tracks
            
        except Exception as e:
            logger.error(f"Error in artist novelty: {e}")
            return tracks

class FairnessConstraints:
    """Fairness constraints to ensure balanced representation."""
    
    def __init__(self, min_diversity_ratio: float = 0.3):
        """Initialize fairness constraints."""
        self.min_diversity_ratio = min_diversity_ratio
        self.artist_representation = {}
    
    def apply_fairness_constraints(self, tracks: List[Dict], max_recommendations: int = 20) -> List[Dict]:
        """Apply fairness constraints to ensure balanced representation."""
        try:
            # Group tracks by artist
            artist_tracks = {}
            for track in tracks:
                artist_id = track.get('artists', [{}])[0].get('id', 'unknown')
                if artist_id not in artist_tracks:
                    artist_tracks[artist_id] = []
                artist_tracks[artist_id].append(track)
            
            # Calculate fair distribution
            num_artists = len(artist_tracks)
            max_tracks_per_artist = max(1, max_recommendations // num_artists)
            
            # Select tracks ensuring fair representation
            selected_tracks = []
            artist_counts = {}
            
            # Sort tracks by score within each artist
            for artist_id, artist_track_list in artist_tracks.items():
                artist_track_list.sort(key=lambda x: x.get('recommendation_score', 0), reverse=True)
            
            # Select tracks ensuring diversity
            while len(selected_tracks) < max_recommendations:
                # Find artist with lowest representation
                min_count = float('inf')
                selected_artist = None
                
                for artist_id in artist_tracks:
                    current_count = artist_counts.get(artist_id, 0)
                    if current_count < max_tracks_per_artist and current_count < min_count:
                        min_count = current_count
                        selected_artist = artist_id
                
                if selected_artist is None:
                    break
                
                # Add track from selected artist
                artist_tracks_list = artist_tracks[selected_artist]
                if artist_tracks_list:
                    track = artist_tracks_list.pop(0)
                    selected_tracks.append(track)
                    artist_counts[selected_artist] = artist_counts.get(selected_artist, 0) + 1
            
            return selected_tracks
            
        except Exception as e:
            logger.error(f"Error in fairness constraints: {e}")
            return tracks[:max_recommendations]
    
    def apply_representation_quota(self, tracks: List[Dict], quota_percentages: Dict[str, float]) -> List[Dict]:
        """Apply representation quotas for different categories."""
        try:
            # Categorize tracks (simplified - could be based on genre, popularity, etc.)
            categorized_tracks = {'mainstream': [], 'indie': [], 'vintage': []}
            
            for track in tracks:
                popularity = track.get('popularity', 50)
                if popularity > 70:
                    categorized_tracks['mainstream'].append(track)
                elif popularity < 30:
                    categorized_tracks['indie'].append(track)
                else:
                    categorized_tracks['vintage'].append(track)
            
            # Apply quotas
            selected_tracks = []
            total_needed = len(tracks)
            
            for category, percentage in quota_percentages.items():
                category_tracks = categorized_tracks.get(category, [])
                num_needed = int(total_needed * percentage)
                
                # Sort by score and select
                category_tracks.sort(key=lambda x: x.get('recommendation_score', 0), reverse=True)
                selected_tracks.extend(category_tracks[:num_needed])
            
            return selected_tracks
            
        except Exception as e:
            logger.error(f"Error in representation quota: {e}")
            return tracks

class DebiasingPipeline:
    """Complete debiasing pipeline combining all techniques."""
    
    def __init__(self):
        """Initialize the debiasing pipeline."""
        self.popularity_debiasing = PopularityDebiasing()
        self.diversity_promotion = DiversityPromotion()
        self.novelty_promotion = NoveltyPromotion()
        self.fairness_constraints = FairnessConstraints()
    
    def apply_full_debiasing(self, 
                           tracks: List[Dict], 
                           user_history: List[str] = None,
                           context: Dict[str, Any] = None) -> List[Dict]:
        """Apply complete debiasing pipeline."""
        try:
            # Set user history for novelty calculation
            if user_history:
                self.novelty_promotion.set_user_history(user_history)
            
            # Apply popularity debiasing
            tracks = self.popularity_debiasing.apply_popularity_normalization(tracks)
            tracks = self.popularity_debiasing.apply_popularity_penalty(tracks)
            tracks = self.popularity_debiasing.apply_popularity_aware_ranking(tracks)
            
            # Apply diversity promotion
            tracks = self.diversity_promotion.apply_diversity_boost(tracks)
            tracks = self.diversity_promotion.apply_genre_diversity(tracks)
            tracks = self.diversity_promotion.apply_temporal_diversity(tracks)
            
            # Apply novelty promotion
            tracks = self.novelty_promotion.apply_novelty_boost(tracks)
            tracks = self.novelty_promotion.apply_artist_novelty(tracks)
            
            # Calculate final debiased scores
            for track in tracks:
                final_score = (
                    track.get('popularity_adjusted_score', track.get('recommendation_score', 0)) * 0.3 +
                    track.get('diversity_adjusted_score', track.get('recommendation_score', 0)) * 0.3 +
                    track.get('novelty_adjusted_score', track.get('recommendation_score', 0)) * 0.4
                )
                track['debiased_score'] = final_score
            
            # Sort by debiased score
            tracks.sort(key=lambda x: x.get('debiased_score', 0), reverse=True)
            
            return tracks
            
        except Exception as e:
            logger.error(f"Error in full debiasing pipeline: {e}")
            return tracks
    
    def get_debiasing_metrics(self, tracks: List[Dict]) -> Dict[str, float]:
        """Calculate debiasing metrics for evaluation."""
        try:
            metrics = {}
            
            # Diversity metrics
            unique_artists = len(set(track.get('artists', [{}])[0].get('id', '') for track in tracks))
            metrics['artist_diversity'] = unique_artists / len(tracks) if tracks else 0.0
            
            # Popularity metrics
            popularities = [track.get('popularity', 50) for track in tracks]
            metrics['avg_popularity'] = np.mean(popularities) if popularities else 0.0
            metrics['popularity_variance'] = np.var(popularities) if popularities else 0.0
            
            # Novelty metrics
            novelty_scores = [track.get('novelty_score', 0) for track in tracks]
            metrics['avg_novelty'] = np.mean(novelty_scores) if novelty_scores else 0.0
            
            # Overall diversity score
            metrics['overall_diversity'] = calculate_diversity_score(tracks)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating debiasing metrics: {e}")
            return {} 