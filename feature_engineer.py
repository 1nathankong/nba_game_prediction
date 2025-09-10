import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from collections import defaultdict

class NBAFeatureEngineer:
    def __init__(self, data_dir="nba_data"):
        self.data_dir = data_dir
        
    def load_game_data(self):
        """Load existing game CSV data"""
        # Try to load the consolidated data first
        consolidated_file = os.path.join(self.data_dir, "all_seasons_consolidated.csv")
        
        if os.path.exists(consolidated_file):
            print(f"Loading consolidated game data from {consolidated_file}")
            df = pd.read_csv(consolidated_file)
        else:
            # Fall back to individual season files
            print("Loading individual season files...")
            season_files = [
                "games_2020_21.csv",
                "games_2021_22.csv", 
                "games_2022_23.csv",
                "games_2023_24.csv",
                "games_2024_25.csv"
            ]
            
            dfs = []
            for file in season_files:
                file_path = os.path.join(self.data_dir, file)
                if os.path.exists(file_path):
                    season_df = pd.read_csv(file_path)
                    dfs.append(season_df)
                    print(f"Loaded {len(season_df)} games from {file}")
            
            if dfs:
                df = pd.concat(dfs, ignore_index=True)
            else:
                raise FileNotFoundError("No game CSV files found!")
        
        # Clean and prepare data
        df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)  # Remove timezone info
        df = df.sort_values('date').reset_index(drop=True)
        
        print(f"Loaded {len(df)} total games from {df['date'].min()} to {df['date'].max()}")
        return df
    
    def load_sentiment_data(self):
        """Load sentiment analysis results"""
        sentiment_file = os.path.join(self.data_dir, "team_sentiment_scores.csv")
        
        if not os.path.exists(sentiment_file):
            print(f"Sentiment file not found: {sentiment_file}")
            print("Run sentiment_analyzer.py first to generate sentiment scores")
            return pd.DataFrame()
        
        df = pd.read_csv(sentiment_file)
        df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)  # Remove timezone info
        
        print(f"Loaded {len(df)} sentiment records for {df['team_abbr'].nunique()} teams")
        return df
    
    def calculate_advanced_team_stats(self, games_df):
        """Calculate advanced team statistics and rolling metrics"""
        print("Calculating advanced team statistics...")
        
        # Initialize team tracking
        team_stats = defaultdict(lambda: {
            'games': [],
            'last_game_date': None,
            'home_games': [],
            'away_games': []
        })
        
        enriched_games = []
        
        for idx, game in games_df.iterrows():
            home_team = game['home_team_abbrev']
            away_team = game['away_team_abbrev']
            game_date = game['date']
            
            # Get current team stats
            home_stats = team_stats[home_team]
            away_stats = team_stats[away_team]
            
            # Create enriched game record
            enriched_game = game.copy()
            
            # Calculate pre-game statistics for both teams
            # Home team stats
            home_games = home_stats['games']
            enriched_game['home_games_played'] = len(home_games)
            
            if home_games:
                recent_home = home_games[-10:]  # Last 10 games
                enriched_game['home_recent_wins'] = sum([g['win'] for g in recent_home])
                enriched_game['home_recent_games'] = len(recent_home)
                enriched_game['home_win_pct'] = sum([g['win'] for g in home_games]) / len(home_games)
                enriched_game['home_avg_score'] = np.mean([g['score'] for g in home_games])
                enriched_game['home_avg_allowed'] = np.mean([g['allowed'] for g in home_games])
                enriched_game['home_avg_margin'] = np.mean([g['margin'] for g in home_games])
                
                # Recent form (last 5 games)
                recent_5 = home_games[-5:]
                enriched_game['home_recent_5_wins'] = sum([g['win'] for g in recent_5])
                enriched_game['home_recent_5_avg_score'] = np.mean([g['score'] for g in recent_5]) if recent_5 else 0
                
                # Home/Away splits
                home_home_games = [g for g in home_games if g['home']]
                enriched_game['home_home_record'] = sum([g['win'] for g in home_home_games]) / max(1, len(home_home_games))
                
                # Days rest
                if home_stats['last_game_date']:
                    enriched_game['home_days_rest'] = (game_date - home_stats['last_game_date']).days
                else:
                    enriched_game['home_days_rest'] = 0
            else:
                # First game defaults
                for stat in ['home_recent_wins', 'home_recent_games', 'home_win_pct', 'home_avg_score', 
                           'home_avg_allowed', 'home_avg_margin', 'home_recent_5_wins', 'home_recent_5_avg_score',
                           'home_home_record', 'home_days_rest']:
                    enriched_game[stat] = 0
            
            # Away team stats (similar logic)
            away_games = away_stats['games']
            enriched_game['away_games_played'] = len(away_games)
            
            if away_games:
                recent_away = away_games[-10:]
                enriched_game['away_recent_wins'] = sum([g['win'] for g in recent_away])
                enriched_game['away_recent_games'] = len(recent_away)
                enriched_game['away_win_pct'] = sum([g['win'] for g in away_games]) / len(away_games)
                enriched_game['away_avg_score'] = np.mean([g['score'] for g in away_games])
                enriched_game['away_avg_allowed'] = np.mean([g['allowed'] for g in away_games])
                enriched_game['away_avg_margin'] = np.mean([g['margin'] for g in away_games])
                
                recent_5 = away_games[-5:]
                enriched_game['away_recent_5_wins'] = sum([g['win'] for g in recent_5])
                enriched_game['away_recent_5_avg_score'] = np.mean([g['score'] for g in recent_5]) if recent_5 else 0
                
                away_away_games = [g for g in away_games if not g['home']]
                enriched_game['away_away_record'] = sum([g['win'] for g in away_away_games]) / max(1, len(away_away_games))
                
                if away_stats['last_game_date']:
                    enriched_game['away_days_rest'] = (game_date - away_stats['last_game_date']).days
                else:
                    enriched_game['away_days_rest'] = 0
            else:
                for stat in ['away_recent_wins', 'away_recent_games', 'away_win_pct', 'away_avg_score',
                           'away_avg_allowed', 'away_avg_margin', 'away_recent_5_wins', 'away_recent_5_avg_score',
                           'away_away_record', 'away_days_rest']:
                    enriched_game[stat] = 0
            
            # Derived features
            enriched_game['win_pct_diff'] = enriched_game['home_win_pct'] - enriched_game['away_win_pct']
            enriched_game['avg_score_diff'] = enriched_game['home_avg_score'] - enriched_game['away_avg_score']
            enriched_game['rest_advantage'] = enriched_game['home_days_rest'] - enriched_game['away_days_rest']
            
            # Add to results
            enriched_games.append(enriched_game)
            
            # Update team stats after this game
            home_win = game['home_score'] > game['away_score']
            away_win = not home_win
            
            # Home team update
            home_game_record = {
                'date': game_date,
                'score': game['home_score'],
                'allowed': game['away_score'],
                'margin': game['home_score'] - game['away_score'],
                'win': home_win,
                'home': True
            }
            team_stats[home_team]['games'].append(home_game_record)
            team_stats[home_team]['last_game_date'] = game_date
            
            # Away team update  
            away_game_record = {
                'date': game_date,
                'score': game['away_score'],
                'allowed': game['home_score'],
                'margin': game['away_score'] - game['home_score'],
                'win': away_win,
                'home': False
            }
            team_stats[away_team]['games'].append(away_game_record)
            team_stats[away_team]['last_game_date'] = game_date
        
        return pd.DataFrame(enriched_games)
    
    def merge_sentiment_features(self, games_df, sentiment_df):
        """Merge sentiment features with game data"""
        print("Merging sentiment features with game data...")
        
        if sentiment_df.empty:
            print("No sentiment data available - skipping sentiment features")
            # Add placeholder sentiment columns
            games_df['home_sentiment'] = 0.0
            games_df['away_sentiment'] = 0.0
            games_df['sentiment_diff'] = 0.0
            games_df['home_news_volume'] = 0
            games_df['away_news_volume'] = 0
            return games_df
        
        # Prepare sentiment data for merging
        sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
        
        # Create sentiment lookup
        sentiment_lookup = {}
        for _, row in sentiment_df.iterrows():
            key = (row['team_abbr'], row['date'].date())
            sentiment_lookup[key] = {
                'sentiment_3day': row.get('sentiment_3day', 0),
                'news_volume_3day': row.get('news_volume_3day', 0),
                'avg_sentiment': row.get('avg_sentiment', 0)
            }
        
        # Add sentiment features to games
        games_with_sentiment = []
        
        for idx, game in games_df.iterrows():
            game_date = game['date'].date()
            home_team = game['home_team_abbrev']
            away_team = game['away_team_abbrev']
            
            # Look for sentiment data on game date and recent days
            home_sentiment = 0.0
            away_sentiment = 0.0
            home_news_volume = 0
            away_news_volume = 0
            
            # Check up to 7 days before game for sentiment data
            for days_back in range(8):
                check_date = game_date - timedelta(days=days_back)
                
                home_key = (home_team, check_date)
                away_key = (away_team, check_date)
                
                if home_key in sentiment_lookup and home_sentiment == 0.0:
                    home_sentiment = sentiment_lookup[home_key]['sentiment_3day']
                    home_news_volume = sentiment_lookup[home_key]['news_volume_3day']
                
                if away_key in sentiment_lookup and away_sentiment == 0.0:
                    away_sentiment = sentiment_lookup[away_key]['sentiment_3day']  
                    away_news_volume = sentiment_lookup[away_key]['news_volume_3day']
                
                # Stop if we found both
                if home_sentiment != 0.0 and away_sentiment != 0.0:
                    break
            
            # Add sentiment features
            game_enriched = game.copy()
            game_enriched['home_sentiment'] = home_sentiment
            game_enriched['away_sentiment'] = away_sentiment
            game_enriched['sentiment_diff'] = home_sentiment - away_sentiment
            game_enriched['home_news_volume'] = home_news_volume
            game_enriched['away_news_volume'] = away_news_volume
            
            # Sentiment-weighted features
            base_home_confidence = game_enriched.get('home_win_pct', 0)
            base_away_confidence = game_enriched.get('away_win_pct', 0)
            
            # Apply sentiment boost/penalty (small effect, 5% max)
            sentiment_weight = 0.05
            game_enriched['home_confidence_score'] = base_home_confidence + (home_sentiment * sentiment_weight)
            game_enriched['away_confidence_score'] = base_away_confidence + (away_sentiment * sentiment_weight)
            
            games_with_sentiment.append(game_enriched)
        
        result_df = pd.DataFrame(games_with_sentiment)
        
        # Print sentiment merge statistics
        sentiment_games = len(result_df[(result_df['home_sentiment'] != 0) | (result_df['away_sentiment'] != 0)])
        print(f"Games with sentiment data: {sentiment_games}/{len(result_df)} ({sentiment_games/len(result_df)*100:.1f}%)")
        
        if sentiment_games == 0:
            print("Note: No sentiment data matched with games. This is expected for historical data.")
            print("The model will use statistical features only, which is still very effective.")
        
        return result_df
    
    def add_temporal_features(self, df):
        """Add temporal and seasonal features"""
        print("Adding temporal features...")
        
        df = df.copy()
        
        # Basic temporal features
        df['month'] = df['date'].dt.month
        df['day_of_week'] = df['date'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Season timing
        # NBA seasons typically run Oct-Apr
        def get_season_progress(date):
            year = date.year
            month = date.month
            
            # Determine season year
            if month >= 10:  # Oct-Dec
                season_start = datetime(year, 10, 15)
                season_end = datetime(year + 1, 4, 15)
            else:  # Jan-Apr
                season_start = datetime(year - 1, 10, 15)
                season_end = datetime(year, 4, 15)
            
            # Calculate progress through season
            total_days = (season_end - season_start).days
            days_in = (date - season_start).days
            progress = max(0, min(1, days_in / total_days))
            
            return progress
        
        df['season_progress'] = df['date'].apply(get_season_progress)
        
        # Holiday effects (simplified)
        df['is_christmas_period'] = ((df['month'] == 12) & (df['date'].dt.day >= 20)).astype(int)
        df['is_new_year'] = ((df['month'] == 1) & (df['date'].dt.day <= 7)).astype(int)
        
        return df
    
    def create_target_variables(self, df):
        """Create various target variables for prediction"""
        print("Creating target variables...")
        
        df = df.copy()
        
        # Primary target: home team wins
        df['home_win'] = (df['home_score'] > df['away_score']).astype(int)
        
        # Point spread and totals
        df['point_spread'] = df['home_score'] - df['away_score']
        df['total_points'] = df['home_score'] + df['away_score']
        
        # Categorical outcomes
        df['margin_category'] = pd.cut(
            abs(df['point_spread']), 
            bins=[0, 3, 7, 15, float('inf')],
            labels=['close', 'narrow', 'comfortable', 'blowout']
        )
        
        # Over/under categories (approximate NBA totals around 220)
        df['total_category'] = pd.cut(
            df['total_points'],
            bins=[0, 200, 220, 240, float('inf')],
            labels=['low', 'under', 'over', 'high']
        )
        
        return df
    
    def select_features(self, df):
        """Select and organize features for ML models"""
        print("Selecting features for ML...")
        
        # Define feature categories
        basic_features = [
            'game_id', 'date', 'season', 'home_team_abbrev', 'away_team_abbrev'
        ]
        
        team_performance_features = [
            'home_games_played', 'away_games_played',
            'home_win_pct', 'away_win_pct', 'win_pct_diff',
            'home_avg_score', 'away_avg_score', 'avg_score_diff',
            'home_avg_allowed', 'away_avg_allowed',
            'home_avg_margin', 'away_avg_margin',
            'home_recent_wins', 'away_recent_wins',
            'home_recent_5_wins', 'away_recent_5_wins',
            'home_home_record', 'away_away_record'
        ]
        
        situational_features = [
            'home_days_rest', 'away_days_rest', 'rest_advantage',
            'month', 'day_of_week', 'is_weekend',
            'season_progress', 'is_christmas_period', 'is_new_year'
        ]
        
        sentiment_features = [
            'home_sentiment', 'away_sentiment', 'sentiment_diff',
            'home_news_volume', 'away_news_volume',
            'home_confidence_score', 'away_confidence_score'
        ]
        
        target_features = [
            'home_win', 'point_spread', 'total_points',
            'home_score', 'away_score'
        ]
        
        # Combine all features
        all_features = (basic_features + team_performance_features + 
                       situational_features + sentiment_features + target_features)
        
        # Keep only features that exist in the dataframe
        available_features = [col for col in all_features if col in df.columns]
        
        result_df = df[available_features].copy()
        
        print(f"Selected {len(available_features)} features:")
        print(f"  - Basic: {len([f for f in basic_features if f in available_features])}")
        print(f"  - Team Performance: {len([f for f in team_performance_features if f in available_features])}")
        print(f"  - Situational: {len([f for f in situational_features if f in available_features])}")
        print(f"  - Sentiment: {len([f for f in sentiment_features if f in available_features])}")
        print(f"  - Targets: {len([f for f in target_features if f in available_features])}")
        
        return result_df
    
    def create_ml_dataset(self):
        """Create the complete ML-ready dataset"""
        print("Creating ML-ready dataset...")
        print("="*60)
        
        # Load data
        games_df = self.load_game_data()
        sentiment_df = self.load_sentiment_data()
        
        # Calculate advanced stats
        games_with_stats = self.calculate_advanced_team_stats(games_df)
        
        # Merge sentiment features
        games_with_sentiment = self.merge_sentiment_features(games_with_stats, sentiment_df)
        
        # Add temporal features
        games_with_temporal = self.add_temporal_features(games_with_sentiment)
        
        # Create target variables
        games_with_targets = self.create_target_variables(games_with_temporal)
        
        # Select final features
        final_dataset = self.select_features(games_with_targets)
        
        # Remove games without scores (incomplete data)
        final_dataset = final_dataset.dropna(subset=['home_score', 'away_score'])
        
        print(f"\nFinal dataset shape: {final_dataset.shape}")
        print(f"Date range: {final_dataset['date'].min()} to {final_dataset['date'].max()}")
        
        return final_dataset
    
    def save_ml_dataset(self, df, filename="nba_ml_dataset.csv"):
        """Save the ML dataset"""
        output_path = os.path.join(self.data_dir, filename)
        df.to_csv(output_path, index=False)
        print(f"Saved ML dataset to {output_path}")
        return output_path

def main():
    engineer = NBAFeatureEngineer()
    
    # Create complete ML dataset
    ml_dataset = engineer.create_ml_dataset()
    
    # Save dataset
    output_file = engineer.save_ml_dataset(ml_dataset)
    
    # Print summary
    print("\n" + "="*60)
    print("FEATURE ENGINEERING COMPLETE")
    print("="*60)
    print(f"Final dataset: {output_file}")
    print(f"Total games: {len(ml_dataset)}")
    print(f"Features: {ml_dataset.shape[1]}")
    print(f"Date range: {ml_dataset['date'].min()} to {ml_dataset['date'].max()}")
    
    # Feature summary
    print(f"\nFeatures available:")
    for col in ml_dataset.columns:
        if col not in ['game_id', 'date', 'home_team_abbrev', 'away_team_abbrev']:
            print(f"  - {col}")
    
    # Sample data
    print(f"\nSample predictions data:")
    sample_cols = ['date', 'home_team_abbrev', 'away_team_abbrev', 'home_win', 
                   'home_win_pct', 'away_win_pct', 'sentiment_diff', 'home_confidence_score']
    available_sample_cols = [col for col in sample_cols if col in ml_dataset.columns]
    print(ml_dataset[available_sample_cols].head(3).to_string(index=False))
    
    # Data splits suggestion
    total_games = len(ml_dataset)
    train_end_date = pd.to_datetime('2023-04-15')  # End of 2022-23 season
    val_end_date = pd.to_datetime('2024-04-15')    # End of 2023-24 season
    
    train_games = len(ml_dataset[ml_dataset['date'] <= train_end_date])
    val_games = len(ml_dataset[(ml_dataset['date'] > train_end_date) & (ml_dataset['date'] <= val_end_date)])
    test_games = len(ml_dataset[ml_dataset['date'] > val_end_date])
    
    print(f"\nSuggested data splits:")
    print(f"  Training (2020-2023): {train_games} games ({train_games/total_games*100:.1f}%)")
    print(f"  Validation (2023-24): {val_games} games ({val_games/total_games*100:.1f}%)")
    print(f"  Test (2024-25): {test_games} games ({test_games/total_games*100:.1f}%)")

if __name__ == "__main__":
    main()