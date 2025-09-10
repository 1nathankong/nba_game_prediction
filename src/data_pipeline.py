import pandas as pd
import numpy as np
import json
import os
import glob
from datetime import datetime, timedelta
from collections import defaultdict
import re
from textblob import TextBlob  # For sentiment analysis on news

class NBAFeatureEngineer:
    def __init__(self, data_dir="nba_data"):
        self.data_dir = data_dir
        self.teams_data = {}
        self.games_df = None
        self.news_data = []
        self.team_mapping = {}
        
    def load_all_data(self):
        """Load and organize all NBA data"""
        print("Loading NBA data...")
        
        # Load team data
        self.load_teams_data()
        
        # Load games data
        self.load_games_data()
        
        # Load news data
        self.load_news_data()
        
        print(f"Loaded {len(self.teams_data)} teams, {len(self.games_df) if self.games_df is not None else 0} games, {len(self.news_data)} news articles")
    
    def load_teams_data(self):
        """Load team information and stats"""
        
        team_files = glob.glob(f"{self.data_dir}/teams/team_*.json")
        
        for file_path in team_files:
            try:
                with open(file_path, 'r') as f:
                    team_data = json.load(f)
                
                team_info = team_data.get('team', {})
                team_id = team_info.get('id')
                team_abbrev = team_info.get('abbreviation')
                
                if team_id and team_abbrev:
                    self.teams_data[team_id] = team_info
                    self.team_mapping[team_abbrev] = team_id
                    
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    
    def load_games_data(self):
        """Load historical games data"""
        try:
            csv_path = f"{self.data_dir}/consolidated_games.csv"
            if os.path.exists(csv_path):
                self.games_df = pd.read_csv(csv_path)
                if not self.games_df.empty:
                    self.games_df['date'] = pd.to_datetime(self.games_df['date'])
            else:
                print("No consolidated games CSV found")
        except Exception as e:
            print(f"Error loading games data: {e}")
    
    def load_news_data(self):
        """Load news articles for sentiment analysis"""
        
        news_files = glob.glob(f"{self.data_dir}/news/news_*.json")
        
        for file_path in news_files:
            try:
                with open(file_path, 'r') as f:
                    news_data = json.load(f)
                
                if 'articles' in news_data:
                    self.news_data.extend(news_data['articles'])
                    
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    
    def extract_team_features(self, team_id):
        """Extract features for a specific team"""
        if team_id not in self.teams_data:
            return {}
        
        team = self.teams_data[team_id]
        
        features = {
            'team_id': team_id,
            'team_name': team.get('displayName', ''),
            'team_abbrev': team.get('abbreviation', ''),
            'is_active': team.get('isActive', True),
            'conference': self.get_conference(team),
            'division': self.get_division(team),
            'standing_summary': team.get('standingSummary', ''),
        }
        
        # Extract next game info
        next_event = team.get('nextEvent', [])
        if next_event:
            next_game = next_event[0]
            features.update({
                'next_game_date': next_game.get('date'),
                'next_game_is_home': self.is_home_game(next_game, team_id),
                'next_opponent': self.get_opponent(next_game, team_id),
                'next_game_type': next_game.get('seasonType', {}).get('name', '')
            })
        
        return features
    
    def get_conference(self, team):
        """Determine team's conference from group info"""
        groups = team.get('groups', {})
        parent_id = groups.get('parent', {}).get('id')
        
        # NBA conference mapping (you may need to adjust these)
        if parent_id == "5":  # Eastern Conference
            return "Eastern"
        elif parent_id == "6":  # Western Conference  
            return "Western"
        return "Unknown"
    
    def get_division(self, team):
        """Extract division information"""
        groups = team.get('groups', {})
        division_id = groups.get('id')
        
        # Division mapping (adjust based on actual API structure)
        division_map = {
            "9": "Southeast",
            "10": "Atlantic", 
            "11": "Central",
            "12": "Northwest",
            "13": "Pacific",
            "14": "Southwest"
        }
        
        return division_map.get(division_id, "Unknown")
    
    def is_home_game(self, game, team_id):
        """Check if team is playing at home"""
        competitions = game.get('competitions', [])
        if competitions:
            competitors = competitions[0].get('competitors', [])
            for competitor in competitors:
                if competitor.get('team', {}).get('id') == team_id:
                    return competitor.get('homeAway') == 'home'
        return False
    
    def get_opponent(self, game, team_id):
        """Get opponent team ID"""
        competitions = game.get('competitions', [])
        if competitions:
            competitors = competitions[0].get('competitors', [])
            for competitor in competitors:
                comp_team_id = competitor.get('team', {}).get('id')
                if comp_team_id != team_id:
                    return comp_team_id
        return None
    
    def calculate_team_sentiment(self, team_id):
        """Calculate news sentiment for a team"""
        team_articles = []
        team_name = self.teams_data.get(team_id, {}).get('displayName', '')
        team_abbrev = self.teams_data.get(team_id, {}).get('abbreviation', '')
        
        # Find articles mentioning this team
        for article in self.news_data:
            headline = article.get('headline', '').lower()
            description = article.get('description', '').lower()
            
            if (team_name.lower() in headline or 
                team_name.lower() in description or
                team_abbrev.lower() in headline or
                team_abbrev.lower() in description):
                team_articles.append(article)
        
        if not team_articles:
            return {
                'news_sentiment': 0.0,
                'news_count': 0,
                'recent_news_sentiment': 0.0
            }
        
        # Calculate sentiment scores
        sentiments = []
        recent_sentiments = []
        recent_threshold = datetime.now() - timedelta(days=7)
        
        for article in team_articles:
            text = f"{article.get('headline', '')} {article.get('description', '')}"
            try:
                # Using TextBlob for sentiment (install with: pip install textblob)
                sentiment = TextBlob(text).sentiment.polarity
                sentiments.append(sentiment)
                
                # Check if article is recent
                pub_date = article.get('published', '')
                if pub_date:
                    try:
                        article_date = datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
                        if article_date >= recent_threshold:
                            recent_sentiments.append(sentiment)
                    except:
                        pass
                        
            except Exception as e:
                print(f"Error calculating sentiment: {e}")
                sentiments.append(0.0)
        
        return {
            'news_sentiment': np.mean(sentiments) if sentiments else 0.0,
            'news_count': len(team_articles),
            'recent_news_sentiment': np.mean(recent_sentiments) if recent_sentiments else 0.0
        }
    
    def create_team_features_df(self):
        """Create DataFrame with all team features"""
        all_features = []
        
        for team_id in self.teams_data.keys():
            features = self.extract_team_features(team_id)
            
            # Add sentiment features
            sentiment_features = self.calculate_team_sentiment(team_id)
            features.update(sentiment_features)
            
            all_features.append(features)
        
        return pd.DataFrame(all_features)
    
    def engineer_game_features(self, home_team_id, away_team_id, game_date=None):
        """Create features for a specific matchup"""
        if game_date is None:
            game_date = datetime.now()
        
        features = {}
        
        # Basic matchup info
        features['home_team_id'] = home_team_id
        features['away_team_id'] = away_team_id
        features['game_date'] = game_date
        
        # Team-specific features
        home_features = self.extract_team_features(home_team_id)
        away_features = self.extract_team_features(away_team_id)
        
        # Prefix features to distinguish home/away
        for key, value in home_features.items():
            if key != 'team_id':
                features[f'home_{key}'] = value
        
        for key, value in away_features.items():
            if key != 'team_id':
                features[f'away_{key}'] = value
        
        # Conference/Division matchup features
        features['same_conference'] = (home_features.get('conference') == 
                                     away_features.get('conference'))
        features['same_division'] = (home_features.get('division') == 
                                   away_features.get('division'))
        
        # Sentiment differential
        home_sentiment = self.calculate_team_sentiment(home_team_id)
        away_sentiment = self.calculate_team_sentiment(away_team_id)
        
        features['sentiment_differential'] = (home_sentiment['news_sentiment'] - 
                                            away_sentiment['news_sentiment'])
        
        return features
    
    def create_prediction_features(self, games_list):
        """Create features for multiple games for prediction"""
        prediction_features = []
        
        for game in games_list:
            home_team = game['home_team_id']
            away_team = game['away_team_id']
            game_date = game.get('date', datetime.now())
            
            features = self.engineer_game_features(home_team, away_team, game_date)
            prediction_features.append(features)
        
        return pd.DataFrame(prediction_features)
    
    def add_historical_performance(self, features_df):
        """Add historical performance metrics if games data available"""
        if self.games_df is None or self.games_df.empty:
            print("No historical games data available")
            return features_df
        
        # Calculate team performance metrics
        team_stats = {}
        
        for _, game in self.games_df.iterrows():
            home_id = game['home_team_id']
            away_id = game['away_team_id']
            home_score = game.get('home_score', 0)
            away_score = game.get('away_score', 0)
            
            # Initialize team stats
            for team_id in [home_id, away_id]:
                if team_id not in team_stats:
                    team_stats[team_id] = {
                        'games_played': 0,
                        'wins': 0,
                        'home_wins': 0,
                        'away_wins': 0,
                        'points_for': 0,
                        'points_against': 0
                    }
            
            # Update stats
            if pd.notna(home_score) and pd.notna(away_score):
                team_stats[home_id]['games_played'] += 1
                team_stats[away_id]['games_played'] += 1
                
                team_stats[home_id]['points_for'] += home_score
                team_stats[home_id]['points_against'] += away_score
                team_stats[away_id]['points_for'] += away_score
                team_stats[away_id]['points_against'] += home_score
                
                # Determine winner
                if home_score > away_score:
                    team_stats[home_id]['wins'] += 1
                    team_stats[home_id]['home_wins'] += 1
                else:
                    team_stats[away_id]['wins'] += 1
                    team_stats[away_id]['away_wins'] += 1
        
        # Add calculated metrics to features
        for team_id, stats in team_stats.items():
            if stats['games_played'] > 0:
                win_pct = stats['wins'] / stats['games_played']
                avg_points_for = stats['points_for'] / stats['games_played']
                avg_points_against = stats['points_against'] / stats['games_played']
                
                # Update features dataframe for this team
                mask = (features_df['home_team_id'] == team_id) | (features_df['away_team_id'] == team_id)
                features_df.loc[mask, f'team_{team_id}_win_pct'] = win_pct
                features_df.loc[mask, f'team_{team_id}_avg_points_for'] = avg_points_for
                features_df.loc[mask, f'team_{team_id}_avg_points_against'] = avg_points_against
        
        return features_df

# Example usage functions
def demo_feature_engineering():
    """Demonstrate the feature engineering pipeline"""
    
    # Initialize the feature engineer
    fe = NBAFeatureEngineer()
    
    # Load all data
    fe.load_all_data()
    
    # Create team features DataFrame
    print("Creating team features...")
    team_features = fe.create_team_features_df()
    print(f"Team features shape: {team_features.shape}")
    print("\nSample team features:")
    print(team_features.head())
    
    # Example: Create features for a hypothetical game
    if len(fe.teams_data) >= 2:
        team_ids = list(fe.teams_data.keys())[:2]
        print(f"\nExample game features for {team_ids[0]} vs {team_ids[1]}:")
        
        game_features = fe.engineer_game_features(team_ids[0], team_ids[1])
        for key, value in game_features.items():
            print(f"  {key}: {value}")
    
    return fe, team_features

if __name__ == "__main__":
    fe, team_features = demo_feature_engineering()