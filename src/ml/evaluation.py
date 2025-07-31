"""
Evaluation metrics for the music recommendation system.
Tracks diversity, novelty, user satisfaction, and other performance indicators.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
import json
from collections import defaultdict, Counter

from ..utils.helpers import calculate_diversity_score, calculate_novelty_score, save_data, load_data

logger = logging.getLogger(__name__)

class RecommendationEvaluator:
    """Evaluates recommendation system performance and tracks metrics."""
    
    def __init__(self):
        """Initialize the recommendation evaluator."""
        self.metrics_history = []
        self.user_feedback = []
        self.diversity_tracker = DiversityTracker()
        self.novelty_tracker = NoveltyTracker()
        self.satisfaction_tracker = SatisfactionTracker()
    
    def evaluate_recommendations(self, 
                               recommendations: List[Dict], 
                               user_data: Dict = None,
                               context: Dict[str, Any] = None) -> Dict[str, float]:
        """Evaluate a set of recommendations and return metrics."""
        try:
            metrics = {}
            
            # Basic recommendation metrics
            metrics['num_recommendations'] = len(recommendations)
            metrics['avg_popularity'] = np.mean([track.get('popularity', 50) for track in recommendations])
            metrics['popularity_variance'] = np.var([track.get('popularity', 50) for track in recommendations])
            
            # Diversity metrics
            diversity_metrics = self.diversity_tracker.calculate_diversity_metrics(recommendations)
            metrics.update(diversity_metrics)
            
            # Novelty metrics
            if user_data:
                user_history = [track['id'] for track in user_data.get('top_tracks', [])]
                novelty_metrics = self.novelty_tracker.calculate_novelty_metrics(recommendations, user_history)
                metrics.update(novelty_metrics)
            
            # Context relevance metrics
            if context:
                context_metrics = self._calculate_context_relevance(recommendations, context)
                metrics.update(context_metrics)
            
            # Overall quality score
            metrics['overall_quality_score'] = self._calculate_overall_quality(metrics)
            
            # Store metrics with timestamp
            metrics['timestamp'] = datetime.now().isoformat()
            self.metrics_history.append(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating recommendations: {e}")
            return {}
    
    def _calculate_context_relevance(self, recommendations: List[Dict], context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate context relevance metrics."""
        try:
            metrics = {}
            
            # Time-based relevance
            time_category = context.get('time_category', 'afternoon')
            time_relevance_scores = []
            
            for track in recommendations:
                audio_features = track.get('audio_features', {})
                energy = audio_features.get('energy', 0.5)
                acousticness = audio_features.get('acousticness', 0.5)
                
                if time_category == 'morning':
                    # Prefer energetic tracks in morning
                    relevance = energy
                elif time_category == 'night':
                    # Prefer acoustic/chill tracks at night
                    relevance = acousticness
                else:
                    # Balanced for other times
                    relevance = 0.5
                
                time_relevance_scores.append(relevance)
            
            metrics['time_relevance'] = np.mean(time_relevance_scores)
            
            # Mood-based relevance
            mood = context.get('mood', 'neutral')
            mood_relevance_scores = []
            
            for track in recommendations:
                audio_features = track.get('audio_features', {})
                valence = audio_features.get('valence', 0.5)
                
                if mood == 'happy':
                    relevance = valence
                elif mood == 'melancholic':
                    relevance = 1.0 - valence
                else:
                    relevance = 0.5
                
                mood_relevance_scores.append(relevance)
            
            metrics['mood_relevance'] = np.mean(mood_relevance_scores)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating context relevance: {e}")
            return {}
    
    def _calculate_overall_quality(self, metrics: Dict[str, float]) -> float:
        """Calculate overall quality score from individual metrics."""
        try:
            # Weighted combination of key metrics
            weights = {
                'artist_diversity': 0.25,
                'genre_diversity': 0.20,
                'avg_novelty': 0.20,
                'time_relevance': 0.15,
                'mood_relevance': 0.10,
                'popularity_variance': 0.10
            }
            
            quality_score = 0.0
            total_weight = 0.0
            
            for metric, weight in weights.items():
                if metric in metrics:
                    quality_score += metrics[metric] * weight
                    total_weight += weight
            
            return quality_score / total_weight if total_weight > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating overall quality: {e}")
            return 0.0
    
    def record_user_feedback(self, feedback: Dict[str, Any]):
        """Record user feedback for evaluation."""
        try:
            feedback['timestamp'] = datetime.now().isoformat()
            self.user_feedback.append(feedback)
            
            # Update satisfaction tracker
            self.satisfaction_tracker.update_satisfaction(feedback)
            
        except Exception as e:
            logger.error(f"Error recording user feedback: {e}")
    
    def get_evaluation_summary(self, time_period: str = 'all') -> Dict[str, Any]:
        """Get evaluation summary for a time period."""
        try:
            # Filter metrics by time period
            if time_period == 'week':
                cutoff_date = datetime.now() - timedelta(days=7)
                recent_metrics = [m for m in self.metrics_history 
                                if datetime.fromisoformat(m['timestamp']) > cutoff_date]
            elif time_period == 'month':
                cutoff_date = datetime.now() - timedelta(days=30)
                recent_metrics = [m for m in self.metrics_history 
                                if datetime.fromisoformat(m['timestamp']) > cutoff_date]
            else:
                recent_metrics = self.metrics_history
            
            if not recent_metrics:
                return {}
            
            # Calculate summary statistics
            summary = {}
            
            # Average metrics
            for key in recent_metrics[0].keys():
                if key != 'timestamp' and isinstance(recent_metrics[0][key], (int, float)):
                    values = [m[key] for m in recent_metrics if key in m]
                    if values:
                        summary[f'avg_{key}'] = np.mean(values)
                        summary[f'std_{key}'] = np.std(values)
            
            # Satisfaction metrics
            satisfaction_summary = self.satisfaction_tracker.get_satisfaction_summary(time_period)
            summary.update(satisfaction_summary)
            
            # Diversity trends
            diversity_summary = self.diversity_tracker.get_diversity_summary(time_period)
            summary.update(diversity_summary)
            
            # Novelty trends
            novelty_summary = self.novelty_tracker.get_novelty_summary(time_period)
            summary.update(novelty_summary)
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting evaluation summary: {e}")
            return {}
    
    def save_evaluation_data(self, filepath: str):
        """Save evaluation data to file."""
        try:
            data = {
                'metrics_history': self.metrics_history,
                'user_feedback': self.user_feedback,
                'diversity_data': self.diversity_tracker.get_all_data(),
                'novelty_data': self.novelty_tracker.get_all_data(),
                'satisfaction_data': self.satisfaction_tracker.get_all_data()
            }
            
            save_data(data, filepath)
            logger.info(f"Evaluation data saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving evaluation data: {e}")
    
    def load_evaluation_data(self, filepath: str):
        """Load evaluation data from file."""
        try:
            data = load_data(filepath)
            if data:
                self.metrics_history = data.get('metrics_history', [])
                self.user_feedback = data.get('user_feedback', [])
                self.diversity_tracker.load_data(data.get('diversity_data', {}))
                self.novelty_tracker.load_data(data.get('novelty_data', {}))
                self.satisfaction_tracker.load_data(data.get('satisfaction_data', {}))
                
                logger.info(f"Evaluation data loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading evaluation data: {e}")

class DiversityTracker:
    """Tracks diversity metrics over time."""
    
    def __init__(self):
        """Initialize diversity tracker."""
        self.diversity_history = []
        self.artist_diversity_history = []
        self.genre_diversity_history = []
    
    def calculate_diversity_metrics(self, recommendations: List[Dict]) -> Dict[str, float]:
        """Calculate diversity metrics for recommendations."""
        try:
            metrics = {}
            
            # Artist diversity
            unique_artists = set()
            for track in recommendations:
                for artist in track.get('artists', []):
                    unique_artists.add(artist.get('id', ''))
            
            metrics['artist_diversity'] = len(unique_artists) / len(recommendations) if recommendations else 0.0
            
            # Genre diversity (simplified)
            genres = []
            for track in recommendations:
                track_genres = track.get('genres', [])
                genres.extend(track_genres)
            
            unique_genres = set(genres)
            metrics['genre_diversity'] = len(unique_genres) / len(recommendations) if recommendations else 0.0
            
            # Overall diversity score
            metrics['overall_diversity'] = calculate_diversity_score(recommendations)
            
            # Store in history
            self.diversity_history.append({
                'timestamp': datetime.now().isoformat(),
                'overall_diversity': metrics['overall_diversity'],
                'artist_diversity': metrics['artist_diversity'],
                'genre_diversity': metrics['genre_diversity']
            })
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating diversity metrics: {e}")
            return {}
    
    def get_diversity_summary(self, time_period: str = 'all') -> Dict[str, float]:
        """Get diversity summary for a time period."""
        try:
            if not self.diversity_history:
                return {}
            
            # Filter by time period
            if time_period == 'week':
                cutoff_date = datetime.now() - timedelta(days=7)
                recent_data = [d for d in self.diversity_history 
                              if datetime.fromisoformat(d['timestamp']) > cutoff_date]
            elif time_period == 'month':
                cutoff_date = datetime.now() - timedelta(days=30)
                recent_data = [d for d in self.diversity_history 
                              if datetime.fromisoformat(d['timestamp']) > cutoff_date]
            else:
                recent_data = self.diversity_history
            
            if not recent_data:
                return {}
            
            summary = {}
            for key in ['overall_diversity', 'artist_diversity', 'genre_diversity']:
                values = [d[key] for d in recent_data if key in d]
                if values:
                    summary[f'avg_{key}'] = np.mean(values)
                    summary[f'trend_{key}'] = self._calculate_trend(values)
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting diversity summary: {e}")
            return {}
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend (positive = increasing, negative = decreasing)."""
        if len(values) < 2:
            return 0.0
        
        # Simple linear trend
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        return slope
    
    def get_all_data(self) -> Dict[str, Any]:
        """Get all diversity tracking data."""
        return {
            'diversity_history': self.diversity_history,
            'artist_diversity_history': self.artist_diversity_history,
            'genre_diversity_history': self.genre_diversity_history
        }
    
    def load_data(self, data: Dict[str, Any]):
        """Load diversity tracking data."""
        self.diversity_history = data.get('diversity_history', [])
        self.artist_diversity_history = data.get('artist_diversity_history', [])
        self.genre_diversity_history = data.get('genre_diversity_history', [])

class NoveltyTracker:
    """Tracks novelty metrics over time."""
    
    def __init__(self):
        """Initialize novelty tracker."""
        self.novelty_history = []
        self.artist_novelty_history = []
        self.track_novelty_history = []
    
    def calculate_novelty_metrics(self, recommendations: List[Dict], user_history: List[str]) -> Dict[str, float]:
        """Calculate novelty metrics for recommendations."""
        try:
            metrics = {}
            
            # Track novelty scores
            track_novelty_scores = []
            artist_novelty_scores = []
            
            for track in recommendations:
                # Calculate novelty scores
                novelty_score = calculate_novelty_score(track, user_history)
                track_novelty_scores.append(novelty_score)
                
                # Artist novelty
                artist_id = track.get('artists', [{}])[0].get('id', '')
                artist_novelty = 1.0 if artist_id not in user_history else 0.3
                artist_novelty_scores.append(artist_novelty)
            
            metrics['avg_novelty'] = np.mean(track_novelty_scores) if track_novelty_scores else 0.0
            metrics['avg_artist_novelty'] = np.mean(artist_novelty_scores) if artist_novelty_scores else 0.0
            metrics['novelty_variance'] = np.var(track_novelty_scores) if track_novelty_scores else 0.0
            
            # Store in history
            self.novelty_history.append({
                'timestamp': datetime.now().isoformat(),
                'avg_novelty': metrics['avg_novelty'],
                'avg_artist_novelty': metrics['avg_artist_novelty'],
                'novelty_variance': metrics['novelty_variance']
            })
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating novelty metrics: {e}")
            return {}
    
    def get_novelty_summary(self, time_period: str = 'all') -> Dict[str, float]:
        """Get novelty summary for a time period."""
        try:
            if not self.novelty_history:
                return {}
            
            # Filter by time period
            if time_period == 'week':
                cutoff_date = datetime.now() - timedelta(days=7)
                recent_data = [n for n in self.novelty_history 
                              if datetime.fromisoformat(n['timestamp']) > cutoff_date]
            elif time_period == 'month':
                cutoff_date = datetime.now() - timedelta(days=30)
                recent_data = [n for n in self.novelty_history 
                              if datetime.fromisoformat(n['timestamp']) > cutoff_date]
            else:
                recent_data = self.novelty_history
            
            if not recent_data:
                return {}
            
            summary = {}
            for key in ['avg_novelty', 'avg_artist_novelty', 'novelty_variance']:
                values = [n[key] for n in recent_data if key in n]
                if values:
                    summary[f'avg_{key}'] = np.mean(values)
                    summary[f'trend_{key}'] = self._calculate_trend(values)
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting novelty summary: {e}")
            return {}
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend (positive = increasing, negative = decreasing)."""
        if len(values) < 2:
            return 0.0
        
        # Simple linear trend
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        return slope
    
    def get_all_data(self) -> Dict[str, Any]:
        """Get all novelty tracking data."""
        return {
            'novelty_history': self.novelty_history,
            'artist_novelty_history': self.artist_novelty_history,
            'track_novelty_history': self.track_novelty_history
        }
    
    def load_data(self, data: Dict[str, Any]):
        """Load novelty tracking data."""
        self.novelty_history = data.get('novelty_history', [])
        self.artist_novelty_history = data.get('artist_novelty_history', [])
        self.track_novelty_history = data.get('track_novelty_history', [])

class SatisfactionTracker:
    """Tracks user satisfaction metrics over time."""
    
    def __init__(self):
        """Initialize satisfaction tracker."""
        self.satisfaction_history = []
        self.rating_distribution = defaultdict(int)
        self.feedback_analysis = defaultdict(list)
    
    def update_satisfaction(self, feedback: Dict[str, Any]):
        """Update satisfaction metrics with new feedback."""
        try:
            rating = feedback.get('rating', 0.5)
            feedback_type = feedback.get('feedback_type', 'general')
            
            # Store satisfaction data
            satisfaction_data = {
                'timestamp': datetime.now().isoformat(),
                'rating': rating,
                'feedback_type': feedback_type,
                'track_id': feedback.get('track_id', ''),
                'context': feedback.get('context', {})
            }
            
            self.satisfaction_history.append(satisfaction_data)
            
            # Update rating distribution
            rating_bucket = int(rating * 10) / 10  # Round to nearest 0.1
            self.rating_distribution[rating_bucket] += 1
            
            # Store feedback analysis
            self.feedback_analysis[feedback_type].append(rating)
            
        except Exception as e:
            logger.error(f"Error updating satisfaction: {e}")
    
    def get_satisfaction_summary(self, time_period: str = 'all') -> Dict[str, float]:
        """Get satisfaction summary for a time period."""
        try:
            if not self.satisfaction_history:
                return {}
            
            # Filter by time period
            if time_period == 'week':
                cutoff_date = datetime.now() - timedelta(days=7)
                recent_data = [s for s in self.satisfaction_history 
                              if datetime.fromisoformat(s['timestamp']) > cutoff_date]
            elif time_period == 'month':
                cutoff_date = datetime.now() - timedelta(days=30)
                recent_data = [s for s in self.satisfaction_history 
                              if datetime.fromisoformat(s['timestamp']) > cutoff_date]
            else:
                recent_data = self.satisfaction_history
            
            if not recent_data:
                return {}
            
            # Calculate satisfaction metrics
            ratings = [s['rating'] for s in recent_data]
            
            summary = {
                'avg_satisfaction': np.mean(ratings),
                'satisfaction_variance': np.var(ratings),
                'total_feedback': len(ratings),
                'positive_feedback_rate': np.mean([1 for r in ratings if r > 0.7]),
                'negative_feedback_rate': np.mean([1 for r in ratings if r < 0.3])
            }
            
            # Feedback type analysis
            feedback_types = defaultdict(list)
            for data in recent_data:
                feedback_types[data['feedback_type']].append(data['rating'])
            
            for feedback_type, type_ratings in feedback_types.items():
                summary[f'avg_satisfaction_{feedback_type}'] = np.mean(type_ratings)
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting satisfaction summary: {e}")
            return {}
    
    def get_all_data(self) -> Dict[str, Any]:
        """Get all satisfaction tracking data."""
        return {
            'satisfaction_history': self.satisfaction_history,
            'rating_distribution': dict(self.rating_distribution),
            'feedback_analysis': dict(self.feedback_analysis)
        }
    
    def load_data(self, data: Dict[str, Any]):
        """Load satisfaction tracking data."""
        self.satisfaction_history = data.get('satisfaction_history', [])
        self.rating_distribution = defaultdict(int, data.get('rating_distribution', {}))
        self.feedback_analysis = defaultdict(list, data.get('feedback_analysis', {})) 