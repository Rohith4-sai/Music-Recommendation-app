"""
Machine learning models for the music recommendation system.
Includes hybrid recommendation, neural collaborative filtering, and content-based filtering.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
import joblib
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import torch
import torch.nn as nn
import torch.optim as optim

from ..utils.config import config
from ..utils.helpers import save_data, load_data

logger = logging.getLogger(__name__)

class HybridRecommendationModel:
    """Hybrid recommendation model combining multiple approaches."""
    
    def __init__(self):
        """Initialize the hybrid recommendation model."""
        self.collaborative_model = None
        self.content_model = None
        self.neural_model = None
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=32)
        self.is_trained = False
        
    def train(self, user_data: Dict, training_data: Tuple[np.ndarray, np.ndarray, np.ndarray]):
        """Train the hybrid recommendation model."""
        features, track_ids, ratings = training_data
        
        if len(features) == 0:
            logger.warning("No training data available")
            return
        
        try:
            # Initialize sub-models
            self.collaborative_model = CollaborativeFilteringModel()
            self.content_model = ContentBasedModel()
            self.neural_model = NeuralCollaborativeFilteringModel()
            
            # Train collaborative filtering
            self.collaborative_model.train(features, track_ids, ratings)
            
            # Train content-based model
            self.content_model.train(features, track_ids, ratings)
            
            # Train neural model
            self.neural_model.train(features, track_ids, ratings)
            
            # Fit scaler and PCA
            self.scaler.fit(features)
            self.pca.fit(self.scaler.transform(features))
            
            self.is_trained = True
            logger.info("Hybrid model training completed")
            
        except Exception as e:
            logger.error(f"Error training hybrid model: {e}")
    
    def predict(self, user_features: np.ndarray, candidate_tracks: List[Dict], 
                context: Dict[str, Any] = None) -> List[Dict]:
        """Generate hybrid recommendations."""
        if not self.is_trained:
            logger.warning("Model not trained yet")
            return []
        
        try:
            # Get predictions from each model
            collab_scores = self.collaborative_model.predict(user_features, candidate_tracks)
            content_scores = self.content_model.predict(user_features, candidate_tracks, context)
            neural_scores = self.neural_model.predict(user_features, candidate_tracks, context)
            
            # Combine predictions with weights
            combined_scores = []
            for i, track in enumerate(candidate_tracks):
                combined_score = (
                    0.3 * collab_scores[i] +
                    0.3 * content_scores[i] +
                    0.4 * neural_scores[i]
                )
                combined_scores.append(combined_score)
            
            # Apply debiasing
            debiased_scores = self._apply_debiasing(candidate_tracks, combined_scores)
            
            # Sort by score and return recommendations
            recommendations = []
            for i, track in enumerate(candidate_tracks):
                track['recommendation_score'] = debiased_scores[i]
                recommendations.append(track)
            
            recommendations.sort(key=lambda x: x['recommendation_score'], reverse=True)
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return []
    
    def _apply_debiasing(self, tracks: List[Dict], scores: List[float]) -> List[float]:
        """Apply debiasing to recommendation scores."""
        debiased_scores = []
        
        for i, track in enumerate(tracks):
            # Popularity debiasing
            popularity = track.get('popularity', 50) / 100.0
            popularity_penalty = popularity ** config.POPULARITY_ALPHA
            
            # Diversity boost
            diversity_boost = track.get('scores', {}).get('diversity_boost', 1.0)
            
            # Novelty boost
            novelty_boost = track.get('scores', {}).get('novelty', 1.0)
            
            # Apply debiasing
            debiased_score = (
                scores[i] * 
                popularity_penalty * 
                diversity_boost * 
                novelty_boost
            )
            
            debiased_scores.append(debiased_score)
        
        return debiased_scores
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        if not self.is_trained:
            logger.warning("No trained model to save")
            return
        
        try:
            model_data = {
                'collaborative_model': self.collaborative_model,
                'content_model': self.content_model,
                'neural_model': self.neural_model,
                'scaler': self.scaler,
                'pca': self.pca,
                'is_trained': self.is_trained
            }
            
            joblib.dump(model_data, filepath)
            logger.info(f"Model saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def load_model(self, filepath: str):
        """Load a trained model."""
        try:
            model_data = joblib.load(filepath)
            
            self.collaborative_model = model_data['collaborative_model']
            self.content_model = model_data['content_model']
            self.neural_model = model_data['neural_model']
            self.scaler = model_data['scaler']
            self.pca = model_data['pca']
            self.is_trained = model_data['is_trained']
            
            logger.info(f"Model loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")

class CollaborativeFilteringModel:
    """Collaborative filtering model using user-item similarity."""
    
    def __init__(self):
        """Initialize collaborative filtering model."""
        self.user_item_matrix = None
        self.user_similarities = None
        self.is_trained = False
    
    def train(self, features: np.ndarray, track_ids: np.ndarray, ratings: np.ndarray):
        """Train the collaborative filtering model."""
        try:
            # Create user-item matrix (simplified - in real system, you'd have multiple users)
            # For now, we'll use the features as a pseudo user-item matrix
            self.user_item_matrix = features
            
            # Calculate user similarities using cosine similarity
            self.user_similarities = cosine_similarity(self.user_item_matrix)
            
            self.is_trained = True
            logger.info("Collaborative filtering model trained")
            
        except Exception as e:
            logger.error(f"Error training collaborative filtering model: {e}")
    
    def predict(self, user_features: np.ndarray, candidate_tracks: List[Dict]) -> List[float]:
        """Generate collaborative filtering predictions."""
        if not self.is_trained:
            return [0.5] * len(candidate_tracks)
        
        try:
            # Calculate similarity between user and candidate tracks
            scores = []
            for track in candidate_tracks:
                track_embedding = track.get('embedding', np.zeros(user_features.shape))
                similarity = cosine_similarity([user_features], [track_embedding])[0][0]
                scores.append(similarity)
            
            return scores
            
        except Exception as e:
            logger.error(f"Error in collaborative filtering prediction: {e}")
            return [0.5] * len(candidate_tracks)

class ContentBasedModel:
    """Content-based filtering model using track features."""
    
    def __init__(self):
        """Initialize content-based model."""
        self.user_profile = None
        self.is_trained = False
    
    def train(self, features: np.ndarray, track_ids: np.ndarray, ratings: np.ndarray):
        """Train the content-based model."""
        try:
            # Create user profile as weighted average of liked tracks
            weights = ratings / np.sum(ratings)
            self.user_profile = np.average(features, axis=0, weights=weights)
            
            self.is_trained = True
            logger.info("Content-based model trained")
            
        except Exception as e:
            logger.error(f"Error training content-based model: {e}")
    
    def predict(self, user_features: np.ndarray, candidate_tracks: List[Dict], 
                context: Dict[str, Any] = None) -> List[float]:
        """Generate content-based predictions."""
        if not self.is_trained:
            return [0.5] * len(candidate_tracks)
        
        try:
            scores = []
            for track in candidate_tracks:
                track_embedding = track.get('embedding', np.zeros(user_features.shape))
                
                # Calculate similarity with user profile
                similarity = cosine_similarity([self.user_profile], [track_embedding])[0][0]
                
                # Apply context adjustment if provided
                if context:
                    context_adjustment = self._calculate_context_adjustment(track, context)
                    similarity *= context_adjustment
                
                scores.append(similarity)
            
            return scores
            
        except Exception as e:
            logger.error(f"Error in content-based prediction: {e}")
            return [0.5] * len(candidate_tracks)
    
    def _calculate_context_adjustment(self, track: Dict, context: Dict[str, Any]) -> float:
        """Calculate context-based adjustment for track similarity."""
        adjustment = 1.0
        
        # Time-based adjustments
        time_category = context.get('time_category', 'afternoon')
        audio_features = track.get('audio_features', {})
        
        if time_category == 'morning':
            # Prefer energetic tracks in the morning
            energy = audio_features.get('energy', 0.5)
            adjustment *= (0.8 + 0.4 * energy)
        elif time_category == 'night':
            # Prefer chill tracks at night
            acousticness = audio_features.get('acousticness', 0.5)
            adjustment *= (0.8 + 0.4 * acousticness)
        
        # Mood-based adjustments
        mood = context.get('mood', 'neutral')
        valence = audio_features.get('valence', 0.5)
        
        if mood == 'happy':
            adjustment *= (0.7 + 0.6 * valence)
        elif mood == 'melancholic':
            adjustment *= (0.7 + 0.6 * (1 - valence))
        
        return adjustment

class NeuralCollaborativeFilteringModel:
    """Neural collaborative filtering model using deep learning."""
    
    def __init__(self):
        """Initialize neural collaborative filtering model."""
        self.model = None
        self.is_trained = False
    
    def train(self, features: np.ndarray, track_ids: np.ndarray, ratings: np.ndarray):
        """Train the neural collaborative filtering model."""
        try:
            # Create neural network
            self.model = self._build_model(features.shape[1])
            
            # Prepare training data
            X = features
            y = ratings
            
            # Train the model
            self.model.fit(
                X, y,
                epochs=config.EPOCHS,
                batch_size=config.BATCH_SIZE,
                validation_split=0.2,
                verbose=0
            )
            
            self.is_trained = True
            logger.info("Neural collaborative filtering model trained")
            
        except Exception as e:
            logger.error(f"Error training neural model: {e}")
    
    def _build_model(self, input_dim: int) -> keras.Model:
        """Build the neural network architecture."""
        model = keras.Sequential([
            layers.Dense(config.HIDDEN_LAYERS[0], activation='relu', input_shape=(input_dim,)),
            layers.Dropout(0.3),
            layers.Dense(config.HIDDEN_LAYERS[1], activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(config.HIDDEN_LAYERS[2], activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def predict(self, user_features: np.ndarray, candidate_tracks: List[Dict], 
                context: Dict[str, Any] = None) -> List[float]:
        """Generate neural model predictions."""
        if not self.is_trained or self.model is None:
            return [0.5] * len(candidate_tracks)
        
        try:
            # Prepare candidate features
            candidate_features = []
            for track in candidate_tracks:
                track_embedding = track.get('embedding', np.zeros(user_features.shape))
                
                # Combine user and track features
                combined_features = np.concatenate([user_features, track_embedding])
                candidate_features.append(combined_features)
            
            candidate_features = np.array(candidate_features)
            
            # Generate predictions
            predictions = self.model.predict(candidate_features, verbose=0)
            scores = predictions.flatten().tolist()
            
            return scores
            
        except Exception as e:
            logger.error(f"Error in neural model prediction: {e}")
            return [0.5] * len(candidate_tracks)

class ExplorationExploitationModel:
    """Model for balancing exploration and exploitation in recommendations."""
    
    def __init__(self, exploration_rate: float = 0.3):
        """Initialize exploration-exploitation model."""
        self.exploration_rate = exploration_rate
        self.exploration_history = []
    
    def select_recommendations(self, 
                             exploitation_candidates: List[Dict],
                             exploration_candidates: List[Dict],
                             num_recommendations: int = 20) -> List[Dict]:
        """Select recommendations balancing exploration and exploitation."""
        try:
            # Determine number of exploration vs exploitation recommendations
            num_exploration = int(num_recommendations * self.exploration_rate)
            num_exploitation = num_recommendations - num_exploration
            
            # Select exploitation recommendations (highest scores)
            exploitation_selected = sorted(
                exploitation_candidates, 
                key=lambda x: x.get('recommendation_score', 0), 
                reverse=True
            )[:num_exploitation]
            
            # Select exploration recommendations (diverse, novel)
            exploration_selected = self._select_exploration_candidates(
                exploration_candidates, num_exploration
            )
            
            # Combine and shuffle
            all_recommendations = exploitation_selected + exploration_selected
            np.random.shuffle(all_recommendations)
            
            return all_recommendations
            
        except Exception as e:
            logger.error(f"Error in exploration-exploitation selection: {e}")
            return exploitation_candidates[:num_recommendations]
    
    def _select_exploration_candidates(self, candidates: List[Dict], num_select: int) -> List[Dict]:
        """Select diverse exploration candidates."""
        if len(candidates) <= num_select:
            return candidates
        
        # Score candidates for diversity
        diversity_scores = []
        for candidate in candidates:
            # Higher novelty score = better for exploration
            novelty_score = candidate.get('scores', {}).get('novelty', 0.5)
            
            # Lower popularity = better for exploration
            popularity = candidate.get('popularity', 50) / 100.0
            popularity_penalty = 1.0 - popularity
            
            # Combine scores
            diversity_score = novelty_score * 0.7 + popularity_penalty * 0.3
            diversity_scores.append(diversity_score)
        
        # Select top diversity candidates
        scored_candidates = list(zip(candidates, diversity_scores))
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        return [candidate for candidate, _ in scored_candidates[:num_select]]
    
    def update_exploration_rate(self, user_feedback: List[Dict]):
        """Update exploration rate based on user feedback."""
        if not user_feedback:
            return
        
        # Calculate average satisfaction with exploration recommendations
        exploration_satisfaction = []
        for feedback in user_feedback:
            if feedback.get('is_exploration', False):
                satisfaction = feedback.get('rating', 0.5)
                exploration_satisfaction.append(satisfaction)
        
        if exploration_satisfaction:
            avg_satisfaction = np.mean(exploration_satisfaction)
            
            # Adjust exploration rate based on satisfaction
            if avg_satisfaction > 0.7:
                # Increase exploration if users like it
                self.exploration_rate = min(0.5, self.exploration_rate + 0.05)
            elif avg_satisfaction < 0.3:
                # Decrease exploration if users don't like it
                self.exploration_rate = max(0.1, self.exploration_rate - 0.05)
    
    def get_exploration_rate(self) -> float:
        """Get current exploration rate."""
        return self.exploration_rate 