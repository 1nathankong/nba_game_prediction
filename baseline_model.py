import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import requests
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class NBAGamePredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        self.is_fitted = False
        
    def collect_todays_games(self):
        """Fetch today's NBA games from ESPN API"""
        today = datetime.now().strftime('%Y%m%d')
        url = f"http://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={today}"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            games = []
            if 'events' in data:
                for event in data['events']:
                    game_info = self.parse_game_from_api(event)
                    if game_info:
                        games.append(game_info)
            
            return games
        except Exception as e:
            print(f"Error fetching today's games: {e}")
            return []
    
    def parse_game_from_api(self, event):
        """Parse game information from ESPN API response"""
        try:
            competition = event['competitions'][0]
            competitors = competition['competitors']
            
            # Find home and away teams
            home_team = next(c for c in competitors if c['homeAway'] == 'home')
            away_team = next(c for c in competitors if c['homeAway'] == 'away')
            
            return {
                'game_id': event['id'],
                'date': event['date'],
                'home_team_id': home_team['team']['id'],
                'home_team_name': home_team['team']['displayName'],
                'home_team_abbrev': home_team['team']['abbreviation'],
                'away_team_id': away_team['team']['id'],
                'away_team_name': away_team['team']['displayName'],
                'away_team_abbrev': away_team['team']['abbreviation'],
                'venue': competition.get('venue', {}).get('fullName', ''),
                'status': event['status']['type']['description']
            }
        except Exception as e:
            print(f"Error parsing game: {e}")
            return None
    
    def create_basic_features(self, games_df):
        """Create basic features from game data"""
        features_df = games_df.copy()
        
        # Home court advantage (simple binary feature)
        features_df['home_advantage'] = 1
        
        # Team ID numerical features
        features_df['home_team_id_num'] = pd.to_numeric(features_df['home_team_id'], errors='coerce')
        features_df['away_team_id_num'] = pd.to_numeric(features_df['away_team_id'], errors='coerce')
        
        # Team strength proxy (based on team ID - this is a placeholder)
        # In reality, you'd use ELO ratings or win percentages
        features_df['team_strength_diff'] = (features_df['home_team_id_num'] - 
                                           features_df['away_team_id_num'])
        
        # Date features
        if 'date' in features_df.columns:
            features_df['date'] = pd.to_datetime(features_df['date'])
            features_df['month'] = features_df['date'].dt.month
            features_df['day_of_week'] = features_df['date'].dt.dayofweek
            features_df['day_of_year'] = features_df['date'].dt.dayofyear
        
        return features_df
    
    def encode_categorical_features(self, df, fit=True):
        """Encode categorical features"""
        df_encoded = df.copy()
        
        categorical_cols = ['home_team_abbrev', 'away_team_abbrev']
        
        for col in categorical_cols:
            if col in df_encoded.columns:
                if fit:
                    if col not in self.label_encoders:
                        self.label_encoders[col] = LabelEncoder()
                    df_encoded[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df_encoded[col].astype(str))
                else:
                    if col in self.label_encoders:
                        # Handle unseen categories
                        unique_vals = set(df_encoded[col].astype(str))
                        known_vals = set(self.label_encoders[col].classes_)
                        
                        if unique_vals.issubset(known_vals):
                            df_encoded[f'{col}_encoded'] = self.label_encoders[col].transform(df_encoded[col].astype(str))
                        else:
                            # Fill unknown categories with most common value
                            mode_val = self.label_encoders[col].classes_[0]
                            df_encoded[col] = df_encoded[col].astype(str).replace(
                                [v for v in unique_vals if v not in known_vals], mode_val
                            )
                            df_encoded[f'{col}_encoded'] = self.label_encoders[col].transform(df_encoded[col])
        
        return df_encoded
    
    def prepare_features(self, df, fit=True):
        """Prepare features for modeling"""
        # Create basic features
        df_features = self.create_basic_features(df)
        
        # Encode categorical features
        df_encoded = self.encode_categorical_features(df_features, fit=fit)
        
        # Select numerical features for modeling
        feature_cols = [
            'home_advantage',
            'home_team_id_num',
            'away_team_id_num', 
            'team_strength_diff'
        ]
        
        # Add encoded categorical features
        encoded_cols = [col for col in df_encoded.columns if col.endswith('_encoded')]
        feature_cols.extend(encoded_cols)
        
        # Add date features if available
        date_cols = ['month', 'day_of_week', 'day_of_year']
        for col in date_cols:
            if col in df_encoded.columns:
                feature_cols.append(col)
        
        # Store feature columns for later use
        if fit:
            self.feature_columns = feature_cols
        
        # Select only available features
        available_features = [col for col in self.feature_columns if col in df_encoded.columns]
        X = df_encoded[available_features].fillna(0)
        
        return X
    
    def create_synthetic_training_data(self, n_games=1000):
        """Create synthetic training data for demonstration"""
        print("Creating synthetic training data for model demonstration...")
        
        # Create synthetic game data
        np.random.seed(42)
        
        # Team IDs (using realistic NBA team count)
        team_ids = list(range(1, 31))  # 30 NBA teams
        
        games = []
        for i in range(n_games):
            home_team = np.random.choice(team_ids)
            away_team = np.random.choice([t for t in team_ids if t != home_team])
            
            # Create synthetic game with realistic patterns
            home_advantage = 0.55  # Home teams win ~55% of the time
            team_strength_factor = abs(home_team - away_team) / 30  # Team "strength" based on ID
            
            # Probability home team wins
            win_prob = home_advantage + (team_strength_factor * 0.2)
            win_prob = min(max(win_prob, 0.1), 0.9)  # Keep between 10% and 90%
            
            home_wins = np.random.random() < win_prob
            
            game = {
                'game_id': f'synthetic_{i}',
                'home_team_id': str(home_team),
                'away_team_id': str(away_team),
                'home_team_abbrev': f'T{home_team:02d}',
                'away_team_abbrev': f'T{away_team:02d}',
                'home_wins': int(home_wins),
                'date': datetime.now().strftime('%Y-%m-%d'),
                'month': np.random.randint(10, 13),  # Season months
                'day_of_week': np.random.randint(0, 7)
            }
            games.append(game)
        
        return pd.DataFrame(games)
    
    def train_model(self, training_data=None):
        """Train the prediction model"""
        if training_data is None:
            # Use synthetic data for demonstration
            training_data = self.create_synthetic_training_data()
        
        print(f"Training model with {len(training_data)} games...")
        
        # Prepare features
        X = self.prepare_features(training_data, fit=True)
        y = training_data['home_wins']  # Target: 1 if home team wins, 0 if away team wins
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Features: {self.feature_columns}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Try multiple models
        models = {
            'Logistic Regression': LogisticRegression(random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
        }
        
        best_score = 0
        best_model = None
        
        for name, model in models.items():
            # Cross-validation
            if name == 'Logistic Regression':
                scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
            else:
                scores = cross_val_score(model, X_train, y_train, cv=5)
            
            print(f"{name} CV Score: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
            
            if scores.mean() > best_score:
                best_score = scores.mean()
                best_model = model
                self.best_model_name = name
        
        # Train best model on full training set
        if self.best_model_name == 'Logistic Regression':
            best_model.fit(X_train_scaled, y_train)
            y_pred = best_model.predict(X_test_scaled)
        else:
            best_model.fit(X_train, y_train)
            y_pred = best_model.predict(X_test)
        
        # Store model
        self.model = best_model
        self.is_fitted = True
        
        # Print results
        print(f"\nBest Model: {self.best_model_name}")
        print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return self.model
    
    def predict_games(self, games_df):
        """Predict outcomes for new games"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        
        # Prepare features
        X = self.prepare_features(games_df, fit=False)
        
        # Make predictions
        if self.best_model_name == 'Logistic Regression':
            X_scaled = self.scaler.transform(X)
            predictions = self.model.predict(X_scaled)
            probabilities = self.model.predict_proba(X_scaled)
        else:
            predictions = self.model.predict(X)
            probabilities = self.model.predict_proba(X)
        
        # Create results DataFrame
        results = games_df.copy()
        results['predicted_home_wins'] = predictions
        results['home_win_probability'] = probabilities[:, 1]
        results['away_win_probability'] = probabilities[:, 0]
        results['confidence'] = np.max(probabilities, axis=1)
        
        return results
    
    def predict_todays_games(self):
        """Get predictions for today's games"""
        # Fetch today's games
        todays_games = self.collect_todays_games()
        
        if not todays_games:
            print("No games found for today")
            return None
        
        games_df = pd.DataFrame(todays_games)
        predictions = self.predict_games(games_df)
        
        return predictions
    
    def display_predictions(self, predictions_df):
        """Display predictions in a readable format"""
        if predictions_df is None or predictions_df.empty:
            print("No predictions to display")
            return
        
        print("\n" + "="*80)
        print("NBA GAME PREDICTIONS")
        print("="*80)
        
        for _, game in predictions_df.iterrows():
            home_team = game['home_team_name']
            away_team = game['away_team_name']
            home_prob = game['home_win_probability']
            away_prob = game['away_win_probability']
            confidence = game['confidence']
            predicted_winner = home_team if game['predicted_home_wins'] else away_team
            
            print(f"\n{away_team} @ {home_team}")
            print(f"Predicted Winner: {predicted_winner}")
            print(f"Home Win Probability: {home_prob:.1%}")
            print(f"Away Win Probability: {away_prob:.1%}")
            print(f"Confidence: {confidence:.1%}")
            print("-" * 50)
    
    def get_feature_importance(self):
        """Get feature importance from the trained model"""
        if not self.is_fitted:
            print("Model not trained yet")
            return None
        
        if hasattr(self.model, 'feature_importances_'):
            # Random Forest
            importance_df = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\nFeature Importance (Random Forest):")
            print(importance_df)
            return importance_df
            
        elif hasattr(self.model, 'coef_'):
            # Logistic Regression
            importance_df = pd.DataFrame({
                'feature': self.feature_columns,
                'coefficient': self.model.coef_[0]
            }).sort_values('coefficient', key=abs, ascending=False)
            
            print("\nFeature Coefficients (Logistic Regression):")
            print(importance_df)
            return importance_df
        
        return None

# Usage example and demo functions
def run_complete_demo():
    """Complete demonstration of the NBA prediction system"""
    
    print("="*60)
    print("NBA GAME PREDICTION SYSTEM DEMO")
    print("="*60)
    
    # Initialize predictor
    predictor = NBAGamePredictor()
    
    # Train model (using synthetic data for demo)
    print("\n1. TRAINING MODEL")
    print("-" * 30)
    predictor.train_model()
    
    # Show feature importance
    print("\n2. FEATURE IMPORTANCE")
    print("-" * 30)
    predictor.get_feature_importance()
    
    # Try to predict today's games
    print("\n3. TODAY'S PREDICTIONS")
    print("-" * 30)
    try:
        predictions = predictor.predict_todays_games()
        predictor.display_predictions(predictions)
    except Exception as e:
        print(f"Error getting today's predictions: {e}")
        print("Creating sample predictions instead...")
        
        # Create sample games for demo
        sample_games = pd.DataFrame([
            {
                'home_team_id': '1', 'away_team_id': '5',
                'home_team_name': 'Atlanta Hawks', 'away_team_name': 'Boston Celtics',
                'home_team_abbrev': 'ATL', 'away_team_abbrev': 'BOS',
                'date': '2025-01-15'
            },
            {
                'home_team_id': '13', 'away_team_id': '17',
                'home_team_name': 'Los Angeles Lakers', 'away_team_name': 'Miami Heat',
                'home_team_abbrev': 'LAL', 'away_team_abbrev': 'MIA',
                'date': '2025-01-15'
            }
        ])
        
        sample_predictions = predictor.predict_games(sample_games)
        predictor.display_predictions(sample_predictions)
    
    print("\n4. NEXT STEPS")
    print("-" * 30)
    print("To improve predictions:")
    print("• Collect historical game data (wins, losses, scores)")
    print("• Add advanced stats (offensive/defensive ratings)")
    print("• Include player injury information from news")
    print("• Calculate ELO ratings or team strength metrics")
    print("• Add rest days, travel distance, schedule difficulty")
    print("• Use betting lines as features")
    
    return predictor

def create_historical_data_collector():
    """Create a more sophisticated data collector for better features"""
    
    collector_code = '''
# Enhanced NBA Data Collector for Better Predictions

class AdvancedNBACollector(NBADataCollector):
    def __init__(self):
        super().__init__()
        self.team_stats = {}
        self.season_stats = {}
    
    def collect_season_data(self, season_year=2025):
        """Collect comprehensive season data"""
        print(f"Collecting {season_year} season data...")
        
        # Get all teams first
        teams_data = self.get_all_teams()
        
        # For each team, collect detailed stats
        for team in self.extract_team_list(teams_data):
            team_abbrev = team['abbreviation']
            
            # Get team details with current season stats
            team_details = self.get_team_details(team_abbrev)
            if team_details:
                self.process_team_stats(team_details)
        
        return self.team_stats
    
    def process_team_stats(self, team_data):
        """Extract detailed statistics from team data"""
        # Implementation would parse team statistics
        # This is where you'd extract:
        # - Win/loss record
        # - Points per game
        # - Field goal percentage
        # - Three-point percentage
        # - Rebounds, assists, etc.
        pass
    
    def get_injury_reports(self):
        """Parse news for injury information"""
        injury_keywords = ['injury', 'injured', 'out', 'questionable', 'doubtful']
        injuries = {}
        
        for article in self.news_data:
            headline = article.get('headline', '').lower()
            description = article.get('description', '').lower()
            
            for keyword in injury_keywords:
                if keyword in headline or keyword in description:
                    # Extract player and team information
                    # This would require more sophisticated NLP
                    pass
        
        return injuries
'''
    
    print("Advanced data collector template:")
    print(collector_code)

if __name__ == "__main__":
    # Run the complete demo
    predictor = run_complete_demo()
    
    # Show template for advanced data collection
    print("\n" + "="*60)
    print("ADVANCED FEATURES TEMPLATE")
    print("="*60)
    create_historical_data_collector()