"""
Reusable UI components for the Streamlit music recommendation app.
Includes track cards, music player, and interactive elements.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from datetime import datetime

from ..utils.helpers import format_duration, get_mood_suggestions, get_activity_suggestions

def create_track_card(track: Dict, show_audio_features: bool = False) -> None:
    """Create a track card component with track information and controls."""
    try:
        # Track header
        col1, col2, col3 = st.columns([1, 3, 1])
        
        with col1:
            # Album artwork
            album_images = track.get('album', {}).get('images', [])
            if album_images:
                st.image(album_images[0]['url'], width=80, use_column_width=False)
            else:
                st.image("https://via.placeholder.com/80x80?text=🎵", width=80)
        
        with col2:
            # Track info
            track_name = track.get('name', 'Unknown Track')
            artists = [artist.get('name', 'Unknown Artist') for artist in track.get('artists', [])]
            artist_names = ', '.join(artists)
            
            st.markdown(f"**{track_name}**")
            st.markdown(f"*{artist_names}*")
            
            # Album and duration
            album_name = track.get('album', {}).get('name', 'Unknown Album')
            duration = format_duration(track.get('duration_ms', 0))
            popularity = track.get('popularity', 0)
            
            st.caption(f"{album_name} • {duration} • Popularity: {popularity}")
        
        with col3:
            # Play button and controls
            if st.button("▶️", key=f"play_{track['id']}", help="Play track"):
                st.session_state['current_track'] = track
                st.session_state['is_playing'] = True
            
            # Rating buttons
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("👍", key=f"like_{track['id']}", help="Like"):
                    record_feedback(track, 1.0, 'like')
            with col_b:
                if st.button("👎", key=f"dislike_{track['id']}", help="Dislike"):
                    record_feedback(track, 0.0, 'dislike')
        
        # Recommendation score (if available)
        if 'recommendation_score' in track:
            score = track['recommendation_score']
            st.progress(score)
            st.caption(f"Recommendation Score: {score:.2f}")
        
        st.divider()
        
    except Exception as e:
        st.error(f"Error creating track card: {e}")

def create_music_player(current_track: Optional[Dict] = None) -> None:
    """Create a music player component."""
    try:
        st.markdown("### 🎵 Now Playing")
        
        if current_track and st.session_state.get('is_playing', False):
            # Player controls
            col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
            
            with col1:
                if st.button("⏮️", help="Previous"):
                    st.session_state['previous_track'] = current_track
            
            with col2:
                if st.button("⏸️", help="Pause"):
                    st.session_state['is_playing'] = False
            
            with col3:
                # Progress bar (simplified)
                progress = st.progress(0.0)
                st.caption("0:00 / " + format_duration(current_track.get('duration_ms', 0)))
            
            with col4:
                if st.button("⏭️", help="Next"):
                    st.session_state['next_track'] = current_track
            
            with col5:
                volume = st.slider("🔊", 0, 100, 50, key="volume")
            
            # Current track info
            st.markdown(f"**{current_track.get('name', 'Unknown')}**")
            artists = [artist.get('name', 'Unknown') for artist in current_track.get('artists', [])]
            st.markdown(f"*{', '.join(artists)}*")
            
        else:
            st.info("No track currently playing. Select a track to start listening!")
            
    except Exception as e:
        st.error(f"Error creating music player: {e}")

def create_context_selector() -> Dict[str, Any]:
    """Create context selection widgets for mood and activity."""
    try:
        context = {}
        
        st.markdown("### 🎯 Set Your Context")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**How are you feeling?**")
            mood = st.selectbox(
                "Mood",
                options=[''] + get_mood_suggestions(),
                key="mood_selector"
            )
            if mood:
                context['mood'] = mood
        
        with col2:
            st.markdown("**What are you doing?**")
            activity = st.selectbox(
                "Activity",
                options=[''] + get_activity_suggestions(),
                key="activity_selector"
            )
            if activity:
                context['activity'] = activity
        
        # Time context (automatic)
        from ..utils.helpers import get_time_context
        time_context = get_time_context()
        context.update(time_context)
        
        return context
        
    except Exception as e:
        st.error(f"Error creating context selector: {e}")
        return {}

def create_recommendation_section(title: str, recommendations: List[Dict], section_type: str = "general") -> None:
    """Create a recommendation section with tracks."""
    try:
        st.markdown(f"### {title}")
        
        if not recommendations:
            st.info("No recommendations available. Try adjusting your preferences or context.")
            return
        
        # Section-specific styling
        if section_type == "for_you":
            st.markdown("🎯 *Personalized picks based on your taste*")
        elif section_type == "new_niche":
            st.markdown("🌟 *Emerging artists and hidden gems*")
        elif section_type == "experimental":
            st.markdown("🚀 *Bold departures from your usual taste*")
        
        # Display tracks in a grid
        num_cols = 3
        for i in range(0, len(recommendations), num_cols):
            cols = st.columns(num_cols)
            for j, col in enumerate(cols):
                if i + j < len(recommendations):
                    with col:
                        create_track_card(recommendations[i + j])
        
    except Exception as e:
        st.error(f"Error creating recommendation section: {e}")

def create_analytics_dashboard(metrics: Dict[str, float]) -> None:
    """Create an analytics dashboard with charts and metrics."""
    try:
        st.markdown("### 📊 Analytics Dashboard")
        
        if not metrics:
            st.info("No analytics data available yet.")
            return
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Diversity Score",
                f"{metrics.get('artist_diversity', 0):.2f}",
                delta=f"{metrics.get('trend_artist_diversity', 0):.3f}"
            )
        
        with col2:
            st.metric(
                "Novelty Score",
                f"{metrics.get('avg_novelty', 0):.2f}",
                delta=f"{metrics.get('trend_avg_novelty', 0):.3f}"
            )
        
        with col3:
            st.metric(
                "Satisfaction",
                f"{metrics.get('avg_satisfaction', 0):.2f}",
                delta=f"{metrics.get('trend_avg_satisfaction', 0):.3f}"
            )
        
        with col4:
            st.metric(
                "Quality Score",
                f"{metrics.get('overall_quality_score', 0):.2f}"
            )
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            create_diversity_chart(metrics)
        
        with col2:
            create_satisfaction_chart(metrics)
        
    except Exception as e:
        st.error(f"Error creating analytics dashboard: {e}")

def create_diversity_chart(metrics: Dict[str, float]) -> None:
    """Create a diversity metrics chart."""
    try:
        diversity_data = {
            'Metric': ['Artist Diversity', 'Genre Diversity', 'Overall Diversity'],
            'Score': [
                metrics.get('artist_diversity', 0),
                metrics.get('genre_diversity', 0),
                metrics.get('overall_diversity', 0)
            ]
        }
        
        df = pd.DataFrame(diversity_data)
        
        fig = px.bar(
            df,
            x='Metric',
            y='Score',
            title='Diversity Metrics',
            color='Score',
            color_continuous_scale='viridis'
        )
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error creating diversity chart: {e}")

def create_satisfaction_chart(metrics: Dict[str, float]) -> None:
    """Create a satisfaction metrics chart."""
    try:
        satisfaction_data = {
            'Metric': ['Positive', 'Neutral', 'Negative'],
            'Rate': [
                metrics.get('positive_feedback_rate', 0),
                1 - metrics.get('positive_feedback_rate', 0) - metrics.get('negative_feedback_rate', 0),
                metrics.get('negative_feedback_rate', 0)
            ]
        }
        
        df = pd.DataFrame(satisfaction_data)
        
        fig = px.pie(
            df,
            values='Rate',
            names='Metric',
            title='Feedback Distribution',
            color_discrete_sequence=['#00ff00', '#ffff00', '#ff0000']
        )
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error creating satisfaction chart: {e}")

def create_search_bar() -> Optional[str]:
    """Create a search bar for finding tracks."""
    try:
        st.markdown("### 🔍 Search Music")
        
        search_query = st.text_input(
            "Search for tracks, artists, or albums...",
            placeholder="Enter your search query",
            key="search_input"
        )
        
        if search_query:
            return search_query
        
        return None
        
    except Exception as e:
        st.error(f"Error creating search bar: {e}")
        return None

def create_playlist_creator() -> None:
    """Create a playlist creation interface."""
    try:
        st.markdown("### 📝 Create Playlist")
        
        playlist_name = st.text_input("Playlist Name", key="playlist_name")
        playlist_description = st.text_area("Description (optional)", key="playlist_description")
        is_public = st.checkbox("Make playlist public", value=False, key="playlist_public")
        
        if st.button("Create Playlist", key="create_playlist"):
            if playlist_name:
                # This would integrate with Spotify API to create playlist
                st.success(f"Playlist '{playlist_name}' created successfully!")
            else:
                st.error("Please enter a playlist name.")
        
    except Exception as e:
        st.error(f"Error creating playlist creator: {e}")

def show_track_details(track: Dict) -> None:
    """Show detailed track information in an expander."""
    try:
        with st.expander("Track Details", expanded=True):
            st.markdown(f"**Track:** {track.get('name', 'Unknown')}")
            st.markdown(f"**Artists:** {', '.join([a.get('name', 'Unknown') for a in track.get('artists', [])])}")
            st.markdown(f"**Album:** {track.get('album', {}).get('name', 'Unknown')}")
            st.markdown(f"**Duration:** {format_duration(track.get('duration_ms', 0))}")
            st.markdown(f"**Popularity:** {track.get('popularity', 0)}/100")
            
            # Audio features
            audio_features = track.get('audio_features', {})
            if audio_features:
                st.markdown("**Audio Features:**")
                features_col1, features_col2 = st.columns(2)
                
                with features_col1:
                    st.write(f"Danceability: {audio_features.get('danceability', 0):.2f}")
                    st.write(f"Energy: {audio_features.get('energy', 0):.2f}")
                    st.write(f"Valence: {audio_features.get('valence', 0):.2f}")
                
                with features_col2:
                    st.write(f"Acousticness: {audio_features.get('acousticness', 0):.2f}")
                    st.write(f"Instrumentalness: {audio_features.get('instrumentalness', 0):.2f}")
                    st.write(f"Tempo: {audio_features.get('tempo', 0):.0f} BPM")
            
            # External links
            external_urls = track.get('external_urls', {})
            if 'spotify' in external_urls:
                st.markdown(f"[Open in Spotify]({external_urls['spotify']})")
            
    except Exception as e:
        st.error(f"Error showing track details: {e}")

def record_feedback(track: Dict, rating: float, feedback_type: str) -> None:
    """Record user feedback for a track."""
    try:
        feedback = {
            'track_id': track.get('id', ''),
            'track_name': track.get('name', ''),
            'rating': rating,
            'feedback_type': feedback_type,
            'timestamp': datetime.now().isoformat(),
            'context': st.session_state.get('current_context', {})
        }
        
        # Store feedback in session state
        if 'user_feedback' not in st.session_state:
            st.session_state['user_feedback'] = []
        
        st.session_state['user_feedback'].append(feedback)
        
        # Show feedback confirmation
        if rating > 0.5:
            st.success(f"👍 Liked: {track.get('name', 'Unknown')}")
        else:
            st.info(f"👎 Disliked: {track.get('name', 'Unknown')}")
        
    except Exception as e:
        st.error(f"Error recording feedback: {e}")

def create_loading_spinner(message: str = "Loading..."):
    """Create a loading spinner with custom message."""
    return st.spinner(message)

def create_error_message(error: str, title: str = "Error"):
    """Create a styled error message."""
    st.error(f"**{title}:** {error}")

def create_success_message(message: str, title: str = "Success"):
    """Create a styled success message."""
    st.success(f"**{title}:** {message}")

def create_info_message(message: str, title: str = "Info"):
    """Create a styled info message."""
    st.info(f"**{title}:** {message}") 