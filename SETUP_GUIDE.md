# 🎵 Bias-Resistant Music Recommendation System - Setup Guide

## 📋 Prerequisites

Before setting up the application, ensure you have:

- **Python 3.8+** installed on your system
- **Spotify Account** (free or premium)
- **Spotify Developer Account** (for API access)

## 🚀 Quick Start

### 1. Clone/Download the Project

```bash
# If using git
git clone <repository-url>
cd music-recommendation-app

# Or download and extract the project files
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up Spotify API Credentials

#### Step 1: Create a Spotify App
1. Go to [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
2. Log in with your Spotify account
3. Click "Create App"
4. Fill in the app details:
   - **App name**: `Music Recommendation App` (or any name you prefer)
   - **App description**: `Advanced music recommendation system with bias reduction`
   - **Website**: `http://localhost:8501` (for development)
   - **Redirect URI**: `http://localhost:8501/callback`
5. Accept the terms and create the app

#### Step 2: Get Your Credentials
1. In your app dashboard, note down:
   - **Client ID**
   - **Client Secret**
2. Click "Edit Settings" and add `http://localhost:8501/callback` to the Redirect URIs

#### Step 3: Configure Environment Variables
1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
2. Edit `.env` and add your credentials:
   ```
   SPOTIFY_CLIENT_ID=your_client_id_here
   SPOTIFY_CLIENT_SECRET=your_client_secret_here
   ```

### 4. Run the Application

#### Option A: Run the Web App
```bash
streamlit run app.py
```
The app will open at `http://localhost:8501`

#### Option B: Run the Demo Script
```bash
python demo.py
```

## 🔧 Configuration Options

### Environment Variables (.env)

```bash
# Spotify API Credentials
SPOTIFY_CLIENT_ID=your_spotify_client_id_here
SPOTIFY_CLIENT_SECRET=your_spotify_client_secret_here

# App Configuration
DEBUG=True
LOG_LEVEL=INFO

# ML Model Settings
MODEL_SAVE_PATH=./models
DATA_PATH=./data

# Recommendation Settings
DEFAULT_RECOMMENDATION_COUNT=20
EXPLORATION_RATE=0.3
DIVERSITY_WEIGHT=0.4
NOVELTY_WEIGHT=0.3
```

### Key Parameters Explained

- **EXPLORATION_RATE**: Controls how often the system recommends new/niche music (0.0-1.0)
- **DIVERSITY_WEIGHT**: Importance of genre/artist diversity in recommendations (0.0-1.0)
- **NOVELTY_WEIGHT**: Importance of new artists/tracks in recommendations (0.0-1.0)

## 🎯 Features Overview

### Core Features
- **🎵 Personalized Recommendations**: AI-powered music suggestions based on your taste
- **⚖️ Bias Reduction**: Reduces popularity bias and promotes diverse music discovery
- **🆕 Novelty Promotion**: Introduces new artists and emerging music
- **🎭 Context Awareness**: Considers mood, time of day, and activity
- **🔍 Advanced Search**: Search across tracks, artists, and genres
- **📊 Analytics Dashboard**: Track your listening diversity and discovery patterns

### Recommendation Categories
1. **For You**: Personalized recommendations based on your taste
2. **New & Niche**: Emerging artists and independent music
3. **Experimental**: Bold departures from your usual preferences

## 🛠️ Troubleshooting

### Common Issues

#### 1. Authentication Errors
```
❌ Authentication failed. Please check your credentials.
```
**Solution:**
- Verify your Client ID and Secret in `.env`
- Ensure redirect URI matches exactly: `http://localhost:8501/callback`
- Check that your Spotify app is properly configured

#### 2. Import Errors
```
ModuleNotFoundError: No module named 'src'
```
**Solution:**
- Ensure you're running from the project root directory
- Check that all `__init__.py` files exist in the `src` folders
- Verify Python path includes the project directory

#### 3. API Rate Limits
```
Spotify API rate limit exceeded
```
**Solution:**
- Wait a few minutes before retrying
- Reduce the number of API calls in your session
- Consider implementing caching for frequently accessed data

#### 4. Model Training Issues
```
Error during model training
```
**Solution:**
- Ensure you have sufficient user data (top tracks, artists)
- Check that TensorFlow/PyTorch is properly installed
- Verify data preprocessing is working correctly

### Debug Mode

Enable debug mode for detailed logging:
```bash
# In .env file
DEBUG=True
LOG_LEVEL=DEBUG
```

## 📊 Understanding the Analytics

### Diversity Metrics
- **Artist Diversity**: Ratio of unique artists in recommendations
- **Genre Diversity**: Variety of musical genres represented
- **Temporal Diversity**: Mix of old and new music

### Novelty Metrics
- **New Artist Ratio**: Percentage of recommendations from new artists
- **Discovery Score**: How much the recommendations differ from your history
- **Popularity Distribution**: Balance between popular and niche music

### Satisfaction Tracking
- **User Feedback**: Track your reactions to recommendations
- **Engagement Metrics**: How often you interact with suggested tracks
- **Pattern Evolution**: How your taste preferences change over time

## 🔄 Advanced Configuration

### Customizing the ML Models

Edit `src/ml/models.py` to adjust:
- Model architectures
- Training parameters
- Feature engineering

### Adjusting Debiasing

Modify `src/ml/debiasing.py` to customize:
- Popularity penalty strength
- Diversity promotion weights
- Fairness constraints

### UI Customization

Edit `src/ui/components.py` to change:
- Visual styling
- Component layouts
- User interaction patterns

## 🚀 Deployment

### Local Development
```bash
streamlit run app.py --server.port 8501
```

### Production Deployment
1. Set up a production server
2. Configure environment variables
3. Use a process manager (PM2, systemd)
4. Set up SSL certificates for HTTPS
5. Configure proper redirect URIs

### Docker Deployment
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## 📚 API Documentation

### Spotify API Integration
- **Authentication**: OAuth 2.0 with PKCE
- **Scopes**: User profile, top tracks, recently played, playlists
- **Rate Limits**: Respects Spotify's API limits
- **Error Handling**: Graceful fallbacks for API failures

### ML Model APIs
- **HybridRecommendationModel**: Main recommendation engine
- **DebiasingPipeline**: Bias reduction algorithms
- **ExplorationExploitationModel**: Novelty vs. familiarity balance

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new features
5. Submit a pull request

### Code Style
- Follow PEP 8 guidelines
- Add type hints where appropriate
- Include docstrings for functions
- Write clear commit messages

## 📞 Support

### Getting Help
1. Check the troubleshooting section above
2. Review the demo script (`python demo.py`)
3. Examine the logs for error details
4. Verify your Spotify API configuration

### Reporting Issues
When reporting issues, include:
- Python version
- Operating system
- Error messages
- Steps to reproduce
- Spotify API configuration status

## 🎉 Success Indicators

You'll know the setup is successful when:
- ✅ Streamlit app loads without errors
- ✅ Spotify authentication completes successfully
- ✅ User data loads (top tracks, artists, etc.)
- ✅ Recommendations are generated
- ✅ Analytics dashboard shows metrics
- ✅ Search functionality works
- ✅ Context awareness responds to time/mood settings

## 🔮 Future Enhancements

### Planned Features
- **Audio Analysis**: Real-time audio feature extraction
- **Lyrics Analysis**: Sentiment analysis of song lyrics
- **Social Features**: Share playlists and recommendations
- **Mobile App**: Native mobile application
- **Voice Integration**: Voice-controlled music discovery

### Advanced ML Features
- **Transformer Models**: Attention-based recommendation systems
- **Multi-Modal Learning**: Combining audio, text, and visual features
- **Reinforcement Learning**: Adaptive recommendation strategies
- **Federated Learning**: Privacy-preserving collaborative learning

---

**Happy Music Discovery! 🎵**

For more information, check the main README.md file or run the demo script to see the system in action. 