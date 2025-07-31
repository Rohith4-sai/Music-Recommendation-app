"""
Main Streamlit application for the Bias-Resistant Music Recommendation System.
Integrates all components to provide a complete music discovery experience.
"""

import streamlit as st
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.spotify.api_client import SpotifyClient
from src.spotify.data_processor import SpotifyDataProcessor
from src.ml.models import HybridRecommendationModel, ExplorationExploitationModel
from src.ml.debiasing import DebiasingPipeline
from src.ml.evaluation import RecommendationEvaluator
from src.ui.components import (
    create_track_card, create_music_player, create_context_selector,
    create_recommendation_section, create_search_bar, record_feedback
)
from src.utils.config import config
from src.utils.helpers import setup_logging, get_time_context

# Setup logging
logger = setup_logging()

# Page configuration
st.set_page_config(
    page_title="🎵 Bias-Resistant Music Recommendations",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1DB954;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1DB954;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1DB954;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables."""
    if 'spotify_client' not in st.session_state:
        st.session_state.spotify_client = None
    if 'data_processor' not in st.session_state:
        st.session_state.data_processor = None
    if 'recommendation_model' not in st.session_state:
        st.session_state.recommendation_model = None
    if 'debiasing_pipeline' not in st.session_state:
        st.session_state.debiasing_pipeline = None
    if 'evaluator' not in st.session_state:
        st.session_state.evaluator = None
    if 'user_data' not in st.session_state:
        st.session_state.user_data = None
    if 'current_track' not in st.session_state:
        st.session_state.current_track = None
    if 'is_playing' not in st.session_state:
        st.session_state.is_playing = False
    if 'current_context' not in st.session_state:
        st.session_state.current_context = {}
    if 'user_feedback' not in st.session_state:
        st.session_state.user_feedback = []

def initialize_spotify():
    """Initialize Spotify client and related components."""
    try:
        if st.session_state.spotify_client is None:
            st.session_state.spotify_client = SpotifyClient()
            
        if st.session_state.data_processor is None:
            st.session_state.data_processor = SpotifyDataProcessor(st.session_state.spotify_client)
            
        if st.session_state.recommendation_model is None:
            st.session_state.recommendation_model = HybridRecommendationModel()
            
        if st.session_state.debiasing_pipeline is None:
            st.session_state.debiasing_pipeline = DebiasingPipeline()
            
        if st.session_state.evaluator is None:
            st.session_state.evaluator = RecommendationEvaluator()
            
        return True
        
    except Exception as e:
        st.error(f"Error initializing Spotify: {e}")
        return False

def authenticate_user():
    """Handle user authentication with Spotify."""
    try:
        if not st.session_state.spotify_client.is_authenticated():
            st.error("⚠️ Spotify authentication required!")
            st.markdown("""
            **To use this app, you need to:**
            1. Create a Spotify Developer account at [developer.spotify.com](https://developer.spotify.com)
            2. Create a new app and get your Client ID and Client Secret
            3. Add `http://localhost:8501/callback` to your app's redirect URIs
            4. Create a `.env` file with your credentials:
            ```
            SPOTIFY_CLIENT_ID=your_client_id_here
            SPOTIFY_CLIENT_SECRET=your_client_secret_here
            ```
            5. Restart the app
            """)
            return False
        
        return True
        
    except Exception as e:
        st.error(f"Authentication error: {e}")
        return False

def load_user_data():
    """Load and process user data from Spotify."""
    try:
        if st.session_state.user_data is None:
            with st.spinner("Loading your music profile..."):
                st.session_state.user_data = st.session_state.data_processor.process_user_data()
                
                if st.session_state.user_data:
                    st.success("✅ Music profile loaded successfully!")
                else:
                    st.error("❌ Failed to load music profile")
                    return False
        
        return True
        
    except Exception as e:
        st.error(f"Error loading user data: {e}")
        return False

def generate_recommendations(context: dict = None):
    """Generate recommendations using the hybrid model."""
    try:
        if not st.session_state.user_data:
            return [], [], []
        
        # Get user's top tracks for seed recommendations
        top_tracks = st.session_state.user_data.get('top_tracks', [])
        if not top_tracks:
            return [], [], []
        
        # Get seed track IDs
        seed_track_ids = [track['id'] for track in top_tracks[:5]]
        
        # Generate base recommendations from Spotify
        base_recommendations = st.session_state.spotify_client.get_recommendations(
            seed_tracks=seed_track_ids,
            limit=50
        )
        
        if not base_recommendations:
            return [], [], []
        
        # Process recommendations
        processed_data = st.session_state.data_processor.prepare_recommendation_data(
            st.session_state.user_data, base_recommendations
        )
        
        # Apply debiasing
        debiased_recommendations = st.session_state.debiasing_pipeline.apply_full_debiasing(
            processed_data['candidate_tracks'],
            processed_data['user_history'],
            context
        )
        
        # Separate recommendations by type
        for_you = debiased_recommendations[:20]
        new_niche = [r for r in debiased_recommendations if r.get('scores', {}).get('novelty', 0) > 0.7][:15]
        experimental = [r for r in debiased_recommendations if r.get('popularity', 50) < 30][:15]
        
        return for_you, new_niche, experimental
        
    except Exception as e:
        st.error(f"Error generating recommendations: {e}")
        return [], [], []

def main():
    """Main application function."""
    st.markdown('<h1 class="main-header">🎵 Bias-Resistant Music Recommendations</h1>', unsafe_allow_html=True)
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar for navigation and settings
    with st.sidebar:
        st.markdown("### 🎛️ Navigation")
        page = st.selectbox(
            "Choose a page:",
            ["🏠 Home", "🎯 Recommendations", "🔍 Search", "📊 Analytics", "⚙️ Settings"]
        )
        
        st.markdown("---")
        
        # Context selector
        st.markdown("### 🎯 Context")
        context = create_context_selector()
        st.session_state.current_context = context
        
        st.markdown("---")
        
        # Music player
        create_music_player(st.session_state.current_track)
    
    # Main content area
    if page == "🏠 Home":
        show_home_page()
    elif page == "🎯 Recommendations":
        show_recommendations_page()
    elif page == "🔍 Search":
        show_search_page()
    elif page == "📊 Analytics":
        show_analytics_page()
    elif page == "⚙️ Settings":
        show_settings_page()

def show_home_page():
    """Display the home page."""
    st.markdown('<h2 class="section-header">Welcome to Your Music Discovery Hub</h2>', unsafe_allow_html=True)
    
    # Check authentication
    if not initialize_spotify():
        return
    
    if not authenticate_user():
        return
    
    # Load user data
    if not load_user_data():
        return
    
    # User profile section
    if st.session_state.user_data:
        user_profile = st.session_state.user_data
        st.markdown("### 👤 Your Music Profile")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Top Tracks", len(user_profile.get('top_tracks', [])))
        
        with col2:
            st.metric("Top Artists", len(user_profile.get('top_artists', [])))
        
        with col3:
            st.metric("Playlists", len(user_profile.get('user_playlists', [])))
        
        # Recent activity
        st.markdown("### 🎵 Recent Activity")
        recently_played = user_profile.get('recently_played', [])
        
        if recently_played:
            for i, track in enumerate(recently_played[:5]):
                create_track_card(track)
        else:
            st.info("No recent activity to display.")
    
    # Quick recommendations
    st.markdown("### 🚀 Quick Recommendations")
    
    if st.button("Generate Recommendations", key="quick_recs"):
        with st.spinner("Generating personalized recommendations..."):
            for_you, new_niche, experimental = generate_recommendations(st.session_state.current_context)
            
            if for_you:
                create_recommendation_section("For You", for_you[:5], "for_you")
            if new_niche:
                create_recommendation_section("New & Niche", new_niche[:5], "new_niche")
            if experimental:
                create_recommendation_section("Experimental", experimental[:5], "experimental")

def show_recommendations_page():
    """Display the recommendations page."""
    st.markdown('<h2 class="section-header">🎯 Personalized Recommendations</h2>', unsafe_allow_html=True)
    
    # Check authentication and data
    if not initialize_spotify() or not authenticate_user() or not load_user_data():
        st.warning("Please complete setup on the Home page first.")
        return
    
    # Generate recommendations
    with st.spinner("Generating your personalized recommendations..."):
        for_you, new_niche, experimental = generate_recommendations(st.session_state.current_context)
    
    # Display recommendations
    if for_you:
        create_recommendation_section("🎯 For You", for_you, "for_you")
    
    if new_niche:
        create_recommendation_section("🌟 New & Niche", new_niche, "new_niche")
    
    if experimental:
        create_recommendation_section("🚀 Experimental", experimental, "experimental")
    
    if not any([for_you, new_niche, experimental]):
        st.info("No recommendations available. Try adjusting your context or preferences.")

def show_search_page():
    """Display the search page."""
    st.markdown('<h2 class="section-header">🔍 Search Music</h2>', unsafe_allow_html=True)
    
    # Check authentication
    if not initialize_spotify() or not authenticate_user():
        st.warning("Please complete setup on the Home page first.")
        return
    
    # Search functionality
    search_query = create_search_bar()
    
    if search_query:
        with st.spinner(f"Searching for '{search_query}'..."):
            search_results = st.session_state.spotify_client.search_tracks(search_query, limit=20)
        
        if search_results:
            st.markdown(f"### Search Results for '{search_query}'")
            for track in search_results:
                create_track_card(track)
        else:
            st.info("No tracks found matching your search.")

def show_analytics_page():
    """Display the analytics page."""
    st.markdown('<h2 class="section-header">📊 Analytics & Insights</h2>', unsafe_allow_html=True)
    
    # Check if we have data
    if not st.session_state.user_data:
        st.warning("No user data available. Please complete setup on the Home page first.")
        return
    
    # User preferences analysis
    st.markdown("### 🎵 Your Music Preferences")
    
    user_prefs = st.session_state.user_data.get('user_preferences', {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Top Genres:**")
        favorite_genres = user_prefs.get('favorite_genres', [])
        for genre in favorite_genres[:5]:
            st.write(f"• {genre['genre']} ({genre['count']} artists)")
    
    with col2:
        st.markdown("**Audio Preferences:**")
        audio_prefs = user_prefs.get('audio_preferences', {})
        if audio_prefs:
            st.write(f"• Energy: {audio_prefs.get('energy', 0):.2f}")
            st.write(f"• Danceability: {audio_prefs.get('danceability', 0):.2f}")
            st.write(f"• Valence: {audio_prefs.get('valence', 0):.2f}")
    
    # Diversity metrics
    st.markdown("### 🌈 Diversity Analysis")
    
    diversity_score = user_prefs.get('diversity_score', 0)
    popularity_pref = user_prefs.get('popularity_preference', 0)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Artist Diversity", f"{diversity_score:.2f}")
    
    with col2:
        st.metric("Popularity Preference", f"{popularity_pref:.2f}")
    
    # Feedback analysis
    if st.session_state.user_feedback:
        st.markdown("### 📝 Your Feedback")
        
        feedback_df = pd.DataFrame(st.session_state.user_feedback)
        
        col1, col2 = st.columns(2)
        
        with col1:
            avg_rating = feedback_df['rating'].mean()
            st.metric("Average Rating", f"{avg_rating:.2f}")
        
        with col2:
            total_feedback = len(feedback_df)
            st.metric("Total Feedback", total_feedback)

def show_settings_page():
    """Display the settings page."""
    st.markdown('<h2 class="section-header">⚙️ Settings</h2>', unsafe_allow_html=True)
    
    st.markdown("### 🔧 App Configuration")
    
    # Exploration rate
    exploration_rate = st.slider(
        "Exploration Rate",
        min_value=0.1,
        max_value=0.5,
        value=0.3,
        step=0.05,
        help="How much the app should explore new music vs. stick to your preferences"
    )
    
    # Diversity weight
    diversity_weight = st.slider(
        "Diversity Weight",
        min_value=0.1,
        max_value=0.8,
        value=0.4,
        step=0.1,
        help="How much to prioritize diverse recommendations"
    )
    
    # Novelty weight
    novelty_weight = st.slider(
        "Novelty Weight",
        min_value=0.1,
        max_value=0.8,
        value=0.3,
        step=0.1,
        help="How much to prioritize novel/discovery recommendations"
    )
    
    # Save settings
    if st.button("Save Settings"):
        # Update configuration
        config.EXPLORATION_RATE = exploration_rate
        config.DIVERSITY_WEIGHT = diversity_weight
        config.NOVELTY_WEIGHT = novelty_weight
        st.success("Settings saved successfully!")
    
    st.markdown("---")
    
    # Data management
    st.markdown("### 📊 Data Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Clear User Data"):
            st.session_state.user_data = None
            st.success("User data cleared!")
    
    with col2:
        if st.button("Clear Feedback"):
            st.session_state.user_feedback = []
            st.success("Feedback cleared!")
    
    st.markdown("---")
    
    # About section
    st.markdown("### ℹ️ About This App")
    
    st.markdown("""
    **Bias-Resistant Music Recommendations** is designed to overcome three major problems in current streaming platforms:
    
    1. **Popularity Bias** - Deliberately surfaces new, independent, and niche artists
    2. **Overspecialization** - Proactively suggests diverse, novel, and unexpected tracks
    3. **Limited Personalization** - Goes beyond basic user-item matching to consider mood, context, and real-time taste evolution
    
    This app uses advanced machine learning techniques including:
    - Hybrid recommendation systems
    - Debiasing mechanisms
    - Exploration/exploitation algorithms
    - Contextual personalization
    """)

if __name__ == "__main__":
    main() 