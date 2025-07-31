# 🎵 Bias-Resistant, Deeply Personalized Music Recommendation App

## 🎯 Project Overview

This advanced music recommendation web application addresses three critical problems in current streaming platforms:

1. **Popularity Bias** - Deliberately surfaces new, independent, and niche artists instead of just mainstream hits
2. **Overspecialization (Glass Ceiling Effect)** - Proactively suggests diverse, novel, and unexpected tracks to break taste echo chambers
3. **Limited Personalization** - Goes beyond basic user-item matching to consider mood, context, and real-time taste evolution

## 🚀 Key Features

### 🧠 Advanced ML Pipeline
- **Hybrid Recommendation System**: Combines collaborative filtering, content-based filtering, and deep learning
- **Debiasing Mechanisms**: Popularity normalization, fairness constraints, and adversarial debiasing
- **Exploration/Exploitation**: Epsilon-greedy and Thompson sampling for continuous discovery
- **Contextual Personalization**: Mood, activity, and time-of-day awareness

### 🎨 Modern Streamlit Interface
- **Premium Music Player**: Search, play/pause, and seamless exploration
- **Smart Categories**: "For You", "New/Niche", "Experimental" sections
- **Interactive Feedback**: Real-time rating and preference adjustment
- **Context Selection**: Mood and activity pickers for personalized recommendations

### 📊 Analytics & Evaluation
- **Diversity Tracking**: Measures non-mainstream content exposure
- **Novelty Metrics**: Tracks "new artist" discovery rates
- **Visual Reporting**: Real-time charts showing musical exploration patterns

## 🛠️ Setup Instructions

### Prerequisites
- Python 3.8+
- Spotify Developer Account
- Spotify Premium (for full playback features)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd music-recommendation-app
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Spotify API credentials**
   - Go to [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
   - Create a new app
   - Copy your `CLIENT_ID` and `CLIENT_SECRET`
   - Add `http://localhost:8501/callback` to your app's redirect URIs

4. **Configure environment variables**
   ```bash
   cp .env.example .env
   ```
   Edit `.env` and add your Spotify credentials:
   ```
   SPOTIFY_CLIENT_ID=your_client_id_here
   SPOTIFY_CLIENT_SECRET=your_client_secret_here
   ```

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

## 🎵 How It Works

### 1. Bias-Resistant Recommendations
The system uses multiple debiasing techniques:
- **Popularity Normalization**: Adjusts recommendations based on track popularity
- **Fairness Constraints**: Ensures diverse artist representation
- **Adversarial Training**: Trains models to be robust against popularity bias

### 2. Breaking Overspecialization
- **Diversity Injection**: Deliberately includes tracks outside user's comfort zone
- **Novelty Scoring**: Prioritizes tracks from new artists and genres
- **Exploration Algorithms**: Balances familiar favorites with discovery

### 3. Contextual Personalization
- **Mood Detection**: Considers user's emotional state
- **Activity Context**: Adapts to current activity (workout, study, relaxation)
- **Temporal Awareness**: Adjusts recommendations based on time of day

## 📁 Project Structure

```
music-recommendation-app/
├── app.py                          # Main Streamlit application
├── src/
│   ├── ml/
│   │   ├── models.py              # ML model implementations
│   │   ├── debiasing.py           # Bias reduction algorithms
│   │   └── evaluation.py          # Diversity and novelty metrics
│   ├── spotify/
│   │   ├── api_client.py          # Spotify API integration
│   │   └── data_processor.py      # Data preprocessing
│   ├── ui/
│   │   ├── components.py          # Reusable UI components
│   │   └── pages.py               # Streamlit page layouts
│   └── utils/
│       ├── config.py              # Configuration management
│       └── helpers.py             # Utility functions
├── data/                          # Local data storage
├── models/                        # Trained model storage
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🎯 Unique Features

### For You Section
Personalized recommendations that balance familiarity with discovery, using advanced collaborative filtering with debiasing.

### New/Niche Section
Curated selection of emerging artists and independent music, prioritized by novelty and diversity metrics.

### Experimental Section
Bold recommendations that intentionally break from user's established patterns, using exploration algorithms.

### Interactive Feedback
Real-time rating system that immediately adjusts future recommendations based on user preferences.

## 🔬 Technical Innovation

### Multi-Model Architecture
- **Neural Collaborative Filtering**: Deep learning for user-item interactions
- **Content-Based Filtering**: Audio feature analysis for similarity
- **Hybrid Ensemble**: Combines multiple approaches for robust recommendations

### Advanced Debiasing
- **Popularity-Aware Training**: Models learn to reduce popularity bias
- **Fairness Metrics**: Ensures balanced representation across artists
- **Dynamic Re-ranking**: Adjusts recommendations based on diversity goals

### Real-Time Adaptation
- **Incremental Learning**: Models update with each user interaction
- **Context Integration**: Seamlessly incorporates mood and activity data
- **Exploration Strategies**: Maintains discovery while respecting preferences

## 📈 Evaluation Metrics

The app tracks several key metrics to ensure it's meeting its goals:

- **Diversity Ratio**: Percentage of recommendations from non-mainstream artists
- **Novelty Score**: Average "newness" of recommended tracks
- **User Satisfaction**: Feedback-based satisfaction metrics
- **Exploration Rate**: How often users discover new artists

## 🚀 Future Enhancements

- **Emotion Detection**: Audio feature analysis for mood inference
- **Group Playlists**: Collaborative discovery sessions
- **Advanced Analytics**: Detailed user behavior insights
- **Mobile Optimization**: Responsive design for mobile devices

## 🤝 Contributing

This project is designed to be extensible. Key areas for contribution:
- New debiasing algorithms
- Additional ML models
- UI/UX improvements
- Performance optimizations

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Spotify Web API for music data
- Streamlit for the web framework
- Open-source ML community for algorithms and techniques

---

**Ready to discover music beyond the algorithm? Start exploring with our bias-resistant recommendation system!** 🎵 "# Music-Recommendation-app" 
"# Music-Recommendation-app" 
