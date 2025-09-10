# NBA Game Prediction System

A comprehensive machine learning system for predicting NBA game outcomes using advanced statistical analysis and news sentiment data.

## Overview

This project combines traditional basketball statistics with modern sentiment analysis to predict NBA game winners. The system processes 5+ years of NBA data (2020-2025), extracts meaningful features, and trains multiple machine learning models to achieve high prediction accuracy.

## Features

### **Comprehensive Data Collection**
- Real-time NBA game data from ESPN API
- Historical game statistics (2020-2025 seasons)
- Team performance metrics and roster information
- News articles for sentiment analysis

### **Advanced Feature Engineering**
- **Team Performance**: Win percentages, scoring averages, defensive stats
- **Situational Factors**: Home/away splits, rest days, recent form
- **Temporal Features**: Season progress, scheduling patterns
- **Sentiment Analysis**: News-based team confidence scores
- **Rolling Statistics**: 5-game and 10-game performance windows

### **Machine Learning Pipeline**
- **Multiple Algorithms**: Random Forest, XGBoost, LightGBM, Gradient Boosting
- **Ensemble Methods**: Voting classifier combining best performers
- **Time-Series Validation**: Chronological splits preventing data leakage
- **Hyperparameter Tuning**: Grid search with cross-validation
- **Model Persistence**: Save/load trained models for production use

### **Sentiment Integration**
- VADER and TextBlob sentiment analysis on NBA news
- Team-specific sentiment scoring
- Rolling sentiment averages (3-day, 7-day windows)
- News volume indicators
- Sentiment-weighted confidence adjustments

## Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Run Complete Pipeline
```bash
# Run the entire system (data collection → feature engineering → model training)
python run_full_pipeline.py
```

### Individual Components
```bash
# Collect NBA data
python data_collector.py

# Process sentiment analysis
python sentiment_analyzer.py

# Feature engineering
python feature_engineer.py

# Train models
python advanced_nba_model.py
```

## Project Structure

```
├── run_full_pipeline.py          # Main execution script
├── data_collector.py             # ESPN API data collection
├── sentiment_analyzer.py         # News sentiment processing
├── feature_engineer.py           # Feature creation and data prep
├── advanced_nba_model.py         # ML model training and evaluation
├── baseline_model.py             # Simple baseline for comparison
├── enhanced_nba_predictor.py     # Enhanced prediction methods
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Configuration

### Data Collection
- Modify `seasons_config` in `data_collector.py` to adjust date ranges
- ESPN API endpoints are configurable for different sports/leagues

### Model Training
- Algorithm selection in `advanced_nba_model.py`
- Hyperparameter grids can be customized
- Cross-validation settings adjustable

### Sentiment Analysis
- Supports VADER and TextBlob analyzers
- Team keyword matching customizable
- Rolling window sizes configurable

## Model Performance

### Training Data
- **Training Set**: 2020-2023 seasons (~3,700 games)
- **Validation Set**: 2023-24 season (~1,200 games)  
- **Test Set**: 2024-25 season (~1,200 games)

### Key Features (40+ total)
- Team win percentages and recent form
- Scoring/defensive averages and trends
- Home/away performance splits
- Rest days and scheduling factors
- Season timing and progression
- News sentiment scores

### Expected Performance
- **Baseline Accuracy**: ~60% (home team advantage)
- **Statistical Model**: ~67-70%
- **With Sentiment**: ~70-73%

## Learning Objectives

This project demonstrates:

### **Data Engineering**
- API integration and data collection
- JSON/CSV data processing and cleaning
- Feature engineering and selection
- Handling missing data and outliers

### **Machine Learning**
- Supervised classification problems
- Time series considerations and validation
- Multiple algorithm comparison
- Ensemble methods and model selection
- Hyperparameter optimization

### **Natural Language Processing**
- Sentiment analysis techniques
- Text preprocessing and feature extraction
- Multi-source data integration

### **Software Engineering**
- Modular code design and organization
- Error handling and logging
- Pipeline orchestration
- Model persistence and deployment

## Predictions

### Load Trained Model
```python
from advanced_nba_model import AdvancedNBAPredictor

# Load saved model
predictor = AdvancedNBAPredictor()
predictor.load_model()

# Make predictions (requires feature vector)
result = predictor.predict_game("LAL", "BOS", game_features)
print(f"Predicted winner: {result['predicted_winner']}")
print(f"Confidence: {result['home_win_probability']:.3f}")
```

## Future Enhancements

- **Deep Learning**: Neural networks for complex pattern recognition
- **Real-time API**: Live prediction service
- **Player Impact**: Individual player statistics and injuries
- **Advanced Metrics**: Plus/minus, efficiency ratings
- **Betting Integration**: Point spreads and over/under predictions
- **Mobile App**: User-friendly prediction interface

## Contributing

This project serves as an educational example of end-to-end machine learning systems. Suggestions for improvements:

1. **Data Sources**: Additional APIs or data providers
2. **Feature Engineering**: New statistical or contextual features  
3. **Model Architecture**: Advanced algorithms or neural networks
4. **Evaluation Metrics**: Additional performance measures
5. **Visualization**: Data exploration and model interpretation tools

## License

This project is for educational purposes. NBA data is used under fair use for non-commercial analysis.

## Acknowledgments

- **ESPN API**: Game data and statistics
- **scikit-learn**: Machine learning framework
- **VADER/TextBlob**: Sentiment analysis libraries
- **XGBoost/LightGBM**: Gradient boosting implementations

---

**Built for learning machine learning and sports analytics**