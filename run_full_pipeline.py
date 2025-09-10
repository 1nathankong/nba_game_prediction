"""
Complete NBA Prediction Pipeline Runner

This script runs the entire pipeline:
1. Process news for sentiment analysis
2. Combine game data with sentiment features
3. Train advanced ML models
4. Evaluate and save the best model

Usage: python run_full_pipeline.py
"""

import os
import sys
from datetime import datetime

def check_dependencies():
    """Check if required libraries are installed"""
    required_libs = [
        ('pandas', 'pandas'),
        ('numpy', 'numpy'), 
        ('scikit-learn', 'sklearn'),
        ('xgboost', 'xgboost'),
        ('lightgbm', 'lightgbm')
    ]
    
    optional_libs = [
        ('vaderSentiment', 'vaderSentiment'),
        ('textblob', 'textblob')
    ]
    
    missing_required = []
    missing_optional = []
    
    for display_name, import_name in required_libs:
        try:
            __import__(import_name)
        except ImportError:
            missing_required.append(display_name)
    
    for display_name, import_name in optional_libs:
        try:
            __import__(import_name)
        except ImportError:
            missing_optional.append(display_name)
    
    if missing_required:
        print(f"ERROR: Missing required libraries: {missing_required}")
        print("Install with: pip install " + " ".join(missing_required))
        return False
    
    if missing_optional:
        print(f"WARNING: Missing optional libraries for sentiment analysis: {missing_optional}")
        print("Install with: pip install " + " ".join(missing_optional))
        print("Sentiment features will be disabled.\n")
    
    return True

def run_pipeline():
    """Run the complete pipeline"""
    print("="*80)
    print("NBA PREDICTION PIPELINE")
    print("="*80)
    print(f"Started at: {datetime.now()}")
    print()
    
    # Check dependencies
    if not check_dependencies():
        return False
    
    # Step 1: Process sentiment data
    print("STEP 1: Processing news for sentiment analysis...")
    print("-" * 50)
    
    try:
        from sentiment_analyzer import NBASentimentAnalyzer
        
        analyzer = NBASentimentAnalyzer()
        news_df = analyzer.process_all_news()
        
        if not news_df.empty:
            team_sentiment_df = analyzer.create_team_sentiment_scores(news_df)
            analyzer.save_sentiment_data(news_df, team_sentiment_df)
            print(" Sentiment analysis completed successfully")
        else:
            print(" No news data found - sentiment features will be empty")
        
    except Exception as e:
        print(f" Error in sentiment analysis: {e}")
        print("Continuing without sentiment features...")
    
    print()
    
    # Step 2: Feature engineering
    print("STEP 2: Feature engineering and data preparation...")
    print("-" * 50)
    
    try:
        from feature_engineer import NBAFeatureEngineer
        
        engineer = NBAFeatureEngineer()
        ml_dataset = engineer.create_ml_dataset()
        
        if ml_dataset.empty:
            print("✗ No data available for feature engineering")
            return False
        
        # Save dataset
        output_file = engineer.save_ml_dataset(ml_dataset)
        print(f" Feature engineering completed: {output_file}")
        print(f"  Dataset shape: {ml_dataset.shape}")
        print(f"  Date range: {ml_dataset['date'].min()} to {ml_dataset['date'].max()}")
        
    except Exception as e:
        print(f" Error in feature engineering: {e}")
        return False
    
    print()
    
    # Step 3: Model training
    print("STEP 3: Training advanced ML models...")
    print("-" * 50)
    
    try:
        from advanced_nba_model import AdvancedNBAPredictor
        
        predictor = AdvancedNBAPredictor()
        predictor.train_complete_pipeline()
        
        print("Model training completed successfully")
        
    except Exception as e:
        print(f"Error in model training: {e}")
        return False
    
    print()
    print("="*80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"Finished at: {datetime.now()}")
    print()
    print("Generated files:")
    print("  - nba_data/processed_news_sentiment.csv (news sentiment analysis)")
    print("  - nba_data/team_sentiment_scores.csv (team-level sentiment)")
    print("  - nba_data/nba_ml_dataset.csv (complete ML dataset)")
    print("  - nba_data/best_nba_model.joblib (trained model)")
    print("  - nba_data/best_nba_model_scaler.joblib (feature scaler)")
    print("  - nba_data/best_nba_model_features.txt (feature list)")
    print()
    print("You can now use the trained model for predictions!")
    
    return True

def quick_test():
    """Run a quick test of the trained model"""
    print("\nRunning quick model test...")
    
    try:
        from advanced_nba_model import AdvancedNBAPredictor
        import pandas as pd
        
        # Load model
        predictor = AdvancedNBAPredictor()
        predictor.load_model()
        
        # Load test data
        df = predictor.load_data()
        
        # Get a recent game for testing
        recent_games = df.tail(5)
        
        print("\nRecent game predictions:")
        for _, game in recent_games.iterrows():
            # Prepare features for this game
            X_test = predictor.prepare_features(game.to_frame().T, fit_scaler=False)
            
            # Make prediction
            pred = predictor.best_model.predict(X_test)[0]
            proba = predictor.best_model.predict_proba(X_test)[0]
            
            actual_winner = "Home" if game['home_win'] == 1 else "Away"
            predicted_winner = "Home" if pred == 1 else "Away"
            confidence = max(proba)
            
            print(f"  {game['home_team_abbrev']} vs {game['away_team_abbrev']}")
            print(f"    Actual: {actual_winner}, Predicted: {predicted_winner} ({confidence:.3f})")
        
        print("\n✓ Model test completed successfully")
        
    except Exception as e:
        print(f"✗ Error in model test: {e}")

if __name__ == "__main__":
    success = run_pipeline()
    
    if success:
        # Run quick test
        quick_test()
    else:
        print("Pipeline failed. Check error messages above.")
        sys.exit(1)