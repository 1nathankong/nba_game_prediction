import pandas as pd
import numpy as np
from baseline_model import NBAGamePredictor

def load_and_prepare_data(csv_path):
    """Load and prepare real NBA data for the baseline model"""
    print(f"Loading data from {csv_path}...")
    
    # Load the consolidated games data
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} games")
    
    # Filter only completed games with scores
    df = df[df['status'] == 'Final'].copy()
    df = df.dropna(subset=['home_score', 'away_score'])
    
    print(f"After filtering completed games: {len(df)} games")
    
    # Create the target variable: home_wins (1 if home team wins, 0 if away team wins)
    df['home_wins'] = (df['home_score'] > df['away_score']).astype(int)
    
    # Ensure required columns match baseline model expectations
    required_columns = [
        'game_id', 'date', 'home_team_id', 'away_team_id', 
        'home_team_abbrev', 'away_team_abbrev', 'home_wins'
    ]
    
    # Check if all required columns exist
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        print(f"Warning: Missing columns: {missing_cols}")
    
    # Convert date to proper format
    df['date'] = pd.to_datetime(df['date'])
    
    # Rename columns to match baseline model expectations if needed
    if 'home_team_name' in df.columns:
        df['home_team_name'] = df['home_team_name']
    if 'away_team_name' in df.columns:
        df['away_team_name'] = df['away_team_name']
    
    print(f"Data preparation complete. Shape: {df.shape}")
    print(f"Home team win rate: {df['home_wins'].mean():.3f}")
    
    return df

def train_with_real_data():
    """Train the baseline model with real 2023-2024 season data"""
    
    print("="*60)
    print("TRAINING NBA PREDICTION MODEL WITH REAL DATA")
    print("="*60)
    
    # Load the data
    data_path = "nba_data/consolidated_games.csv"
    games_df = load_and_prepare_data(data_path)
    
    # Initialize predictor
    predictor = NBAGamePredictor()
    
    # Train model with real data
    print("\n1. TRAINING MODEL WITH REAL DATA")
    print("-" * 40)
    model = predictor.train_model(training_data=games_df)
    
    # Show feature importance
    print("\n2. FEATURE IMPORTANCE")
    print("-" * 40)
    predictor.get_feature_importance()
    
    # Create sample predictions with real teams
    print("\n3. SAMPLE PREDICTIONS")
    print("-" * 40)
    
    # Get some real teams from the data for sample predictions
    unique_teams = games_df[['home_team_id', 'home_team_abbrev', 'home_team_name']].drop_duplicates()
    
    # Create sample games
    sample_games = pd.DataFrame([
        {
            'game_id': 'sample_1',
            'home_team_id': '12',  # LA Clippers
            'away_team_id': '13',  # Lakers
            'home_team_name': 'LA Clippers',
            'away_team_name': 'Los Angeles Lakers',
            'home_team_abbrev': 'LAC',
            'away_team_abbrev': 'LAL',
            'date': '2024-03-15'
        },
        {
            'game_id': 'sample_2',
            'home_team_id': '2',   # Boston
            'away_team_id': '14',  # Miami
            'home_team_name': 'Boston Celtics',
            'away_team_name': 'Miami Heat',
            'home_team_abbrev': 'BOS',
            'away_team_abbrev': 'MIA',
            'date': '2024-03-15'
        }
    ])
    
    sample_predictions = predictor.predict_games(sample_games)
    predictor.display_predictions(sample_predictions)
    
    # Show model performance summary
    print("\n4. MODEL SUMMARY")
    print("-" * 40)
    print(f"Training data: {len(games_df)} games from 2023-2024 season")
    print(f"Home court advantage in data: {games_df['home_wins'].mean():.1%}")
    print(f"Model type: {predictor.best_model_name}")
    
    return predictor, games_df

if __name__ == "__main__":
    predictor, data = train_with_real_data()