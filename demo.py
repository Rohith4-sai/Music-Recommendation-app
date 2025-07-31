#!/usr/bin/env python3
"""
Demo script for the Bias-Resistant Music Recommendation System
This script demonstrates the key features and capabilities of the application.
"""

import sys
import os
from pathlib import Path
import json
import time
from typing import Dict, List, Any

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.spotify.api_client import SpotifyClient
from src.spotify.data_processor import SpotifyDataProcessor
from src.ml.models import HybridRecommendationModel, ExplorationExploitationModel
from src.ml.debiasing import DebiasingPipeline
from src.ml.evaluation import RecommendationEvaluator
from src.utils.config import config
from src.utils.helpers import setup_logging, get_time_context, save_data, load_data

def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "="*60)
    print(f"🎵 {title}")
    print("="*60)

def print_section(title: str):
    """Print a formatted section."""
    print(f"\n📋 {title}")
    print("-" * 40)

def demo_authentication():
    """Demonstrate Spotify authentication."""
    print_header("Spotify Authentication Demo")
    
    print("🔐 Initializing Spotify client...")
    try:
        spotify_client = SpotifyClient()
        
        if spotify_client.is_authenticated():
            print("✅ Authentication successful!")
            user_profile = spotify_client.get_user_profile()
            if user_profile:
                print(f"👤 User: {user_profile.get('display_name', 'Unknown')}")
                print(f"📧 Email: {user_profile.get('email', 'Not provided')}")
                print(f"🌍 Country: {user_profile.get('country', 'Unknown')}")
            return spotify_client
        else:
            print("❌ Authentication failed. Please check your credentials.")
            return None
    except Exception as e:
        print(f"❌ Error during authentication: {e}")
        return None

def demo_data_loading(spotify_client: SpotifyClient):
    """Demonstrate data loading and processing."""
    print_header("Data Loading and Processing Demo")
    
    print("📊 Loading user data...")
    data_processor = SpotifyDataProcessor(spotify_client)
    
    # Load user data
    user_data = data_processor.process_user_data()
    
    if user_data:
        print("✅ User data loaded successfully!")
        print(f"🎵 Top tracks: {len(user_data.get('top_tracks', []))}")
        print(f"🎤 Top artists: {len(user_data.get('top_artists', []))}")
        print(f"📈 Recent plays: {len(user_data.get('recently_played', []))}")
        
        # Show sample preferences
        preferences = user_data.get('preferences', {})
        if preferences:
            print("\n🎯 User Preferences:")
            for key, value in list(preferences.items())[:5]:
                if isinstance(value, float):
                    print(f"  {key}: {value:.3f}")
                else:
                    print(f"  {key}: {value}")
        
        return data_processor, user_data
    else:
        print("❌ Failed to load user data.")
        return None, None

def demo_recommendation_generation(data_processor: SpotifyDataProcessor, user_data: Dict):
    """Demonstrate recommendation generation."""
    print_header("Recommendation Generation Demo")
    
    print("🤖 Initializing ML models...")
    
    # Initialize models
    hybrid_model = HybridRecommendationModel()
    exploration_model = ExplorationExploitationModel(exploration_rate=0.3)
    debiasing_pipeline = DebiasingPipeline()
    evaluator = RecommendationEvaluator()
    
    print("✅ Models initialized successfully!")
    
    # Generate training data
    print("\n📚 Preparing training data...")
    training_data = data_processor.create_training_data(user_data)
    print(f"✅ Training data prepared: {training_data[0].shape[0]} samples")
    
    # Train models
    print("\n🎓 Training models...")
    hybrid_model.train(user_data, training_data)
    print("✅ Models trained successfully!")
    
    # Generate recommendations
    print("\n🎯 Generating recommendations...")
    context = get_time_context()
    
    # Get candidate tracks for recommendations
    candidate_tracks = user_data.get('top_tracks', [])[:50]  # Use top tracks as candidates
    
    recommendations = hybrid_model.predict(
        user_features=training_data[0][0] if len(training_data[0]) > 0 else None,
        candidate_tracks=candidate_tracks,
        context=context
    )
    
    if recommendations:
        print(f"✅ Generated {len(recommendations)} recommendations!")
        
        # Apply debiasing
        print("\n⚖️ Applying debiasing techniques...")
        debiased_recommendations = debiasing_pipeline.apply_full_debiasing(
            recommendations, 
            user_history=data_processor.user_history,
            context=context
        )
        
        # Apply exploration/exploitation
        print("\n🎲 Applying exploration/exploitation...")
        final_recommendations = exploration_model.select_recommendations(
            exploitation_candidates=debiased_recommendations[:10],
            exploration_candidates=debiased_recommendations[10:],
            num_recommendations=10
        )
        
        print(f"✅ Final recommendations: {len(final_recommendations)} tracks")
        
        # Show sample recommendations
        print("\n🎵 Sample Recommendations:")
        for i, track in enumerate(final_recommendations[:5], 1):
            print(f"  {i}. {track.get('name', 'Unknown')} - {track.get('artists', [{}])[0].get('name', 'Unknown')}")
            print(f"     Popularity: {track.get('popularity', 0)} | Score: {track.get('score', 0):.3f}")
        
        return final_recommendations, evaluator
    else:
        print("❌ Failed to generate recommendations.")
        return None, None

def demo_analytics(recommendations: List[Dict], evaluator: RecommendationEvaluator):
    """Demonstrate analytics and evaluation."""
    print_header("Analytics and Evaluation Demo")
    
    if not recommendations:
        print("❌ No recommendations to analyze.")
        return
    
    print("📊 Calculating evaluation metrics...")
    
    # Evaluate recommendations
    metrics = evaluator.evaluate_recommendations(recommendations)
    
    print("✅ Evaluation completed!")
    print("\n📈 Key Metrics:")
    for metric, value in metrics.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.3f}")
        else:
            print(f"  {metric}: {value}")
    
    # Get diversity summary
    print("\n🎨 Diversity Analysis:")
    diversity_tracker = evaluator.diversity_tracker
    diversity_summary = diversity_tracker.get_diversity_summary()
    for metric, value in diversity_summary.items():
        print(f"  {metric}: {value:.3f}")
    
    # Get novelty summary
    print("\n🆕 Novelty Analysis:")
    novelty_tracker = evaluator.novelty_tracker
    novelty_summary = novelty_tracker.get_novelty_summary()
    for metric, value in novelty_summary.items():
        print(f"  {metric}: {value:.3f}")

def demo_search_functionality(spotify_client: SpotifyClient):
    """Demonstrate search functionality."""
    print_header("Search Functionality Demo")
    
    search_queries = ["rock", "jazz", "electronic", "classical", "hip hop"]
    
    for query in search_queries:
        print(f"\n🔍 Searching for: '{query}'")
        try:
            results = spotify_client.search_tracks(query, limit=5)
            if results:
                print(f"✅ Found {len(results)} tracks:")
                for i, track in enumerate(results[:3], 1):
                    print(f"  {i}. {track.get('name', 'Unknown')} - {track.get('artists', [{}])[0].get('name', 'Unknown')}")
            else:
                print("❌ No results found.")
        except Exception as e:
            print(f"❌ Search error: {e}")
        
        time.sleep(1)  # Rate limiting

def demo_context_awareness():
    """Demonstrate context awareness features."""
    print_header("Context Awareness Demo")
    
    print("🕐 Getting time context...")
    time_context = get_time_context()
    
    print("✅ Current context:")
    for key, value in time_context.items():
        print(f"  {key}: {value}")
    
    print("\n🎭 Mood and activity suggestions:")
    from src.utils.helpers import get_mood_suggestions, get_activity_suggestions
    
    moods = get_mood_suggestions()
    activities = get_activity_suggestions()
    
    print("  Moods:", ", ".join(moods[:5]))
    print("  Activities:", ", ".join(activities[:5]))

def demo_data_persistence():
    """Demonstrate data persistence features."""
    print_header("Data Persistence Demo")
    
    # Sample data
    sample_data = {
        "user_preferences": {"energy": 0.7, "valence": 0.6, "danceability": 0.8},
        "recommendations": [{"name": "Sample Track", "score": 0.85}],
        "evaluation_metrics": {"diversity": 0.75, "novelty": 0.6}
    }
    
    print("💾 Saving sample data...")
    try:
        save_data(sample_data, "demo_data.json", "json")
        print("✅ Data saved successfully!")
        
        print("📂 Loading saved data...")
        loaded_data = load_data("demo_data.json", "json")
        if loaded_data:
            print("✅ Data loaded successfully!")
            print(f"  Keys: {list(loaded_data.keys())}")
        else:
            print("❌ Failed to load data.")
    except Exception as e:
        print(f"❌ Data persistence error: {e}")

def main():
    """Run the complete demo."""
    print_header("Bias-Resistant Music Recommendation System Demo")
    print("This demo showcases the key features of the application.")
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting demo...")
    
    # Demo 1: Authentication
    spotify_client = demo_authentication()
    if not spotify_client:
        print("\n❌ Demo cannot continue without authentication.")
        print("Please ensure your Spotify credentials are properly configured.")
        return
    
    # Demo 2: Data Loading
    data_processor, user_data = demo_data_loading(spotify_client)
    if not data_processor or not user_data:
        print("\n❌ Demo cannot continue without user data.")
        return
    
    # Demo 3: Recommendation Generation
    recommendations, evaluator = demo_recommendation_generation(data_processor, user_data)
    
    # Demo 4: Analytics
    if recommendations and evaluator:
        demo_analytics(recommendations, evaluator)
    
    # Demo 5: Search
    demo_search_functionality(spotify_client)
    
    # Demo 6: Context Awareness
    demo_context_awareness()
    
    # Demo 7: Data Persistence
    demo_data_persistence()
    
    print_header("Demo Complete!")
    print("🎉 All demo features have been showcased successfully!")
    print("\n📝 Next steps:")
    print("  1. Run 'streamlit run app.py' to start the web application")
    print("  2. Configure your Spotify credentials in .env file")
    print("  3. Explore the interactive features in the web interface")
    print("  4. Check the analytics and settings pages for detailed insights")

if __name__ == "__main__":
    main() 