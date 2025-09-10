import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from enhanced_feature_extractor import EnhancedNBAFeatureExtractor

class EnhancedNBAPredictor:
    """
    Enhanced NBA Game Predictor using comprehensive features
    
    Integrates with the enhanced feature extractor to provide
    significantly improved prediction accuracy over the baseline model.
    """
    
    def __init__(self, data_dir="nba_data"):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        self.numerical_features = []
        self.categorical_features = []
        self.is_fitted = False
        self.best_model_name = None
        
        # Initialize enhanced feature extractor
        self.feature_extractor = EnhancedNBAFeatureExtractor(data_dir)
        self.feature_extractor.load_all_data()
        
    def prepare_enhanced_features(self, df, fit=True):
        """Prepare enhanced features for modeling"""
        print(f"Preparing enhanced features for {len(df)} games...")
        
        # Separate numerical and categorical features
        feature_df = df.copy()
        
        # Identify feature types
        exclude_cols = ['home_wins', 'home_score', 'away_score', 'game_date', 'home_team_id', 'away_team_id']
        
        numerical_cols = []
        categorical_cols = []
        
        for col in feature_df.columns:
            if col in exclude_cols:
                continue
                
            if feature_df[col].dtype in ['object', 'string']:
                categorical_cols.append(col)
            else:
                numerical_cols.append(col)
        
        # Store feature lists if fitting
        if fit:
            self.numerical_features = numerical_cols
            self.categorical_features = categorical_cols
        
        # Handle numerical features
        for col in self.numerical_features:
            if col in feature_df.columns:
                feature_df[col] = pd.to_numeric(feature_df[col], errors='coerce').fillna(0)
        
        # Handle categorical features
        for col in self.categorical_features:
            if col in feature_df.columns:
                feature_df[col] = feature_df[col].fillna('Unknown').astype(str)
                
                if fit:
                    if col not in self.label_encoders:
                        self.label_encoders[col] = LabelEncoder()
                    feature_df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(feature_df[col])
                else:
                    if col in self.label_encoders:
                        # Handle unseen categories
                        unique_vals = set(feature_df[col].unique())
                        known_vals = set(self.label_encoders[col].classes_)
                        
                        if unique_vals.issubset(known_vals):
                            feature_df[f'{col}_encoded'] = self.label_encoders[col].transform(feature_df[col])
                        else:
                            # Replace unknown categories with most common
                            mode_val = self.label_encoders[col].classes_[0]
                            feature_df[col] = feature_df[col].replace(
                                [v for v in unique_vals if v not in known_vals], mode_val
                            )
                            feature_df[f'{col}_encoded'] = self.label_encoders[col].transform(feature_df[col])
        
        # Select final feature columns
        final_features = self.numerical_features.copy()
        
        # Add encoded categorical features
        for col in self.categorical_features:
            encoded_col = f'{col}_encoded'
            if encoded_col in feature_df.columns:
                final_features.append(encoded_col)
        
        # Store feature columns if fitting
        if fit:
            self.feature_columns = final_features
        
        # Select available features
        available_features = [col for col in self.feature_columns if col in feature_df.columns]
        X = feature_df[available_features].fillna(0)
        
        print(f"Final feature matrix shape: {X.shape}")
        print(f"Features: {len(available_features)} total")
        
        return X
    
    def train_enhanced_model(self, use_synthetic=False):
        """Train model using enhanced features"""
        print("=" * 80)
        print("TRAINING ENHANCED NBA PREDICTION MODEL")
        print("=" * 80)
        
        if use_synthetic:
            # Create synthetic data for testing
            training_data = self.create_synthetic_enhanced_data()
        else:
            # Use real enhanced features
            print("Creating enhanced training dataset...")
            training_data = self.feature_extractor.create_training_dataset()
        
        if training_data.empty:
            print("No training data available. Using synthetic data instead.")
            training_data = self.create_synthetic_enhanced_data()
        
        print(f"Training on {len(training_data)} games with enhanced features...")
        
        # Prepare features
        X = self.prepare_enhanced_features(training_data, fit=True)
        y = training_data['home_wins'].astype(int)
        
        print(f"Training features shape: {X.shape}")
        print(f"Home team win rate: {y.mean():.3f}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Test multiple models
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42, max_depth=10),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
        }
        
        best_score = 0
        best_model = None
        results = {}
        
        print("\nModel Performance Comparison:")
        print("-" * 50)
        
        for name, model in models.items():
            # Cross-validation
            if 'Logistic' in name:
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            test_score = accuracy_score(y_test, y_pred)
            
            results[name] = {
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'test_score': test_score,
                'model': model
            }
            
            print(f"{name}:")
            print(f"  CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            print(f"  Test Score: {test_score:.4f}")
            
            if cv_scores.mean() > best_score:
                best_score = cv_scores.mean()
                best_model = model
                self.best_model_name = name
        
        # Store best model
        self.model = best_model
        self.is_fitted = True
        
        # Detailed results for best model
        print(f"\nBest Model: {self.best_model_name}")
        print(f"Best CV Score: {best_score:.4f}")
        print(f"Test Accuracy: {results[self.best_model_name]['test_score']:.4f}")
        
        # Classification report
        if 'Logistic' in self.best_model_name:
            final_pred = self.model.predict(X_test_scaled)
        else:
            final_pred = self.model.predict(X_test)
        
        print(f"\nDetailed Classification Report:")
        print(classification_report(y_test, final_pred, target_names=['Away Win', 'Home Win']))
        
        # Feature importance
        self.display_feature_importance()
        
        return results
    
    def create_synthetic_enhanced_data(self, n_games=1000):
        """Create synthetic enhanced data for testing"""
        print("Creating synthetic enhanced training data...")
        
        np.random.seed(42)
        games = []
        
        # Get team IDs from feature extractor
        team_ids = list(self.feature_extractor.teams_data.keys())
        if len(team_ids) < 2:
            team_ids = [str(i) for i in range(1, 31)]  # Fallback
        
        for i in range(n_games):
            home_team = np.random.choice(team_ids)
            away_team = np.random.choice([t for t in team_ids if t != home_team])
            
            # Create realistic enhanced features
            home_win_pct = np.random.beta(5, 5)  # More realistic distribution
            away_win_pct = np.random.beta(5, 5)
            
            # Advanced features
            home_avg_points = np.random.normal(110, 10)
            away_avg_points = np.random.normal(110, 10)
            home_last_5 = np.random.randint(0, 6)
            away_last_5 = np.random.randint(0, 6)
            
            # Sentiment and injury factors
            home_sentiment = np.random.normal(0, 0.3)
            away_sentiment = np.random.normal(0, 0.3)
            home_injury_impact = np.random.exponential(0.5)
            away_injury_impact = np.random.exponential(0.5)
            
            # Contextual features
            same_conference = np.random.choice([True, False], p=[0.7, 0.3])
            home_court_advantage = np.random.beta(6, 4)  # Slight home advantage
            
            # Calculate win probability with multiple factors
            win_prob = 0.5  # Base probability
            win_prob += (home_win_pct - away_win_pct) * 0.3
            win_prob += (home_avg_points - away_avg_points) / 100 * 0.2
            win_prob += (home_last_5 - away_last_5) / 10 * 0.15
            win_prob += (home_sentiment - away_sentiment) * 0.1
            win_prob += (away_injury_impact - home_injury_impact) * 0.05
            win_prob += (home_court_advantage - 0.5) * 0.2
            
            # Ensure probability is between 0.1 and 0.9
            win_prob = max(0.1, min(0.9, win_prob))
            
            home_wins = int(np.random.random() < win_prob)
            
            # Create comprehensive game record
            game = {
                'home_team_id': home_team,
                'away_team_id': away_team,
                'home_wins': home_wins,
                'game_date': datetime.now(),
                
                # Team performance features
                'home_win_percentage': home_win_pct,
                'away_win_percentage': away_win_pct,
                'win_pct_differential': home_win_pct - away_win_pct,
                'home_avg_points_for': home_avg_points,
                'away_avg_points_for': away_avg_points,
                'offensive_differential': home_avg_points - (110 - away_avg_points),
                
                # Recent form
                'home_last_5_wins': home_last_5,
                'away_last_5_wins': away_last_5,
                'recent_form_differential': home_last_5 - away_last_5,
                
                # Advanced stats
                'home_avg_rebounds': np.random.normal(45, 5),
                'away_avg_rebounds': np.random.normal(45, 5),
                'home_avg_assists': np.random.normal(25, 3),
                'away_avg_assists': np.random.normal(25, 3),
                'home_avg_fg_percentage': np.random.normal(0.45, 0.05),
                'away_avg_fg_percentage': np.random.normal(0.45, 0.05),
                
                # News and sentiment
                'home_news_sentiment': home_sentiment,
                'away_news_sentiment': away_sentiment,
                'sentiment_differential': home_sentiment - away_sentiment,
                'home_injury_impact': home_injury_impact,
                'away_injury_impact': away_injury_impact,
                
                # Contextual
                'same_conference': same_conference,
                'home_court_advantage': home_court_advantage,
                'home_conference': np.random.choice(['Eastern', 'Western']),
                'away_conference': np.random.choice(['Eastern', 'Western']),
                'month': np.random.randint(10, 13),
                'day_of_week': np.random.randint(0, 7),
                'is_weekend': np.random.choice([True, False]),
                
                # Consistency metrics
                'home_consistency': np.random.exponential(8),
                'away_consistency': np.random.exponential(8),
                'home_current_streak': np.random.randint(-5, 8),
                'away_current_streak': np.random.randint(-5, 8)
            }
            
            games.append(game)
        
        return pd.DataFrame(games)
    
    def predict_games(self, games_df):
        """Predict outcomes for new games using enhanced features"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        
        print(f"Making predictions for {len(games_df)} games...")
        
        # If input is basic format, enhance it
        if 'home_win_percentage' not in games_df.columns:
            enhanced_games = []
            for _, game in games_df.iterrows():
                enhanced_features = self.feature_extractor.create_enhanced_features(
                    game['home_team_id'], 
                    game['away_team_id'],
                    game.get('game_date', datetime.now())
                )
                enhanced_games.append(enhanced_features)
            games_df = pd.DataFrame(enhanced_games)
        
        # Prepare features
        X = self.prepare_enhanced_features(games_df, fit=False)
        
        # Make predictions
        if 'Logistic' in self.best_model_name:
            X_scaled = self.scaler.transform(X)
            predictions = self.model.predict(X_scaled)
            probabilities = self.model.predict_proba(X_scaled)
        else:
            predictions = self.model.predict(X)
            probabilities = self.model.predict_proba(X)
        
        # Create results
        results = games_df.copy()
        results['predicted_home_wins'] = predictions
        results['home_win_probability'] = probabilities[:, 1]
        results['away_win_probability'] = probabilities[:, 0]
        results['confidence'] = np.max(probabilities, axis=1)
        
        return results
    
    def display_feature_importance(self):
        """Display feature importance from trained model"""
        if not self.is_fitted:
            print("Model not trained yet")
            return
        
        print(f"\nFeature Importance Analysis ({self.best_model_name}):")
        print("-" * 60)
        
        if hasattr(self.model, 'feature_importances_'):
            # Tree-based models
            importance_df = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("Top 15 Most Important Features:")
            for i, (_, row) in enumerate(importance_df.head(15).iterrows()):
                print(f"{i+1:2d}. {row['feature']:<35} {row['importance']:.4f}")
                
        elif hasattr(self.model, 'coef_'):
            # Linear models
            importance_df = pd.DataFrame({
                'feature': self.feature_columns,
                'coefficient': self.model.coef_[0],
                'abs_coefficient': np.abs(self.model.coef_[0])
            }).sort_values('abs_coefficient', ascending=False)
            
            print("Top 15 Most Influential Features:")
            for i, (_, row) in enumerate(importance_df.head(15).iterrows()):
                print(f"{i+1:2d}. {row['feature']:<35} {row['coefficient']:+.4f}")
    
    def compare_with_baseline(self, test_games=None):
        """Compare enhanced model performance with baseline"""
        print("\n" + "=" * 80)
        print("ENHANCED MODEL VS BASELINE COMPARISON")
        print("=" * 80)
        
        if test_games is None:
            # Create test data
            test_games = self.create_synthetic_enhanced_data(200)
        
        # Enhanced model predictions
        enhanced_results = self.predict_games(test_games)
        enhanced_accuracy = accuracy_score(test_games['home_wins'], enhanced_results['predicted_home_wins'])
        
        # Simulate baseline model (basic features only)
        baseline_features = test_games[['home_win_percentage', 'away_win_percentage', 'home_court_advantage']].copy()
        baseline_features['home_advantage'] = 1
        baseline_features['team_strength_diff'] = baseline_features['home_win_percentage'] - baseline_features['away_win_percentage']
        
        # Simple baseline prediction
        baseline_pred = (baseline_features['team_strength_diff'] + 
                        baseline_features['home_court_advantage'] * 0.1 > 0).astype(int)
        baseline_accuracy = accuracy_score(test_games['home_wins'], baseline_pred)
        
        print(f"Baseline Model Accuracy:  {baseline_accuracy:.4f} ({baseline_accuracy:.1%})")
        print(f"Enhanced Model Accuracy:  {enhanced_accuracy:.4f} ({enhanced_accuracy:.1%})")
        print(f"Improvement:              {enhanced_accuracy - baseline_accuracy:+.4f} ({(enhanced_accuracy - baseline_accuracy):.1%} points)")
        print(f"Relative Improvement:     {((enhanced_accuracy - baseline_accuracy) / baseline_accuracy * 100):+.1f}%")
        
        return {
            'baseline_accuracy': baseline_accuracy,
            'enhanced_accuracy': enhanced_accuracy,
            'improvement': enhanced_accuracy - baseline_accuracy
        }

def demo_enhanced_prediction():
    """Demonstrate the enhanced prediction system"""
    print("=" * 80)
    print("ENHANCED NBA PREDICTION SYSTEM DEMO")
    print("=" * 80)
    
    # Initialize enhanced predictor
    predictor = EnhancedNBAPredictor()
    
    # Train model
    print("\n1. TRAINING ENHANCED MODEL")
    print("-" * 40)
    training_results = predictor.train_enhanced_model(use_synthetic=True)
    
    # Compare with baseline
    print("\n2. COMPARING WITH BASELINE")
    print("-" * 40)
    comparison = predictor.compare_with_baseline()
    
    # Make sample predictions
    print("\n3. SAMPLE PREDICTIONS")
    print("-" * 40)
    
    # Create sample games
    sample_data = predictor.create_synthetic_enhanced_data(5)
    predictions = predictor.predict_games(sample_data)
    
    print("Sample Game Predictions:")
    for i, (_, game) in enumerate(predictions.head().iterrows()):
        home_team = game.get('home_team_id', f'Team_{game.get("home_team_id", "A")}')
        away_team = game.get('away_team_id', f'Team_{game.get("away_team_id", "B")}')
        home_prob = game['home_win_probability']
        confidence = game['confidence']
        predicted_winner = home_team if game['predicted_home_wins'] else away_team
        actual_winner = home_team if game['home_wins'] else away_team
        correct = game['predicted_home_wins'] == game['home_wins']
        
        print(f"\nGame {i+1}: {away_team} @ {home_team}")
        print(f"  Predicted Winner: {predicted_winner}")
        print(f"  Actual Winner:    {actual_winner}")
        print(f"  Home Win Prob:    {home_prob:.1%}")
        print(f"  Confidence:       {confidence:.1%}")
        print(f"  Correct:          {'✓' if correct else '✗'}")
    
    print(f"\n4. SUMMARY")
    print("-" * 40)
    print(f"Enhanced model achieves {comparison['enhanced_accuracy']:.1%} accuracy")
    print(f"Improvement over baseline: +{comparison['improvement']:.1%} points")
    print(f"Key enhanced features include:")
    print(f"  • Team performance metrics (win %, avg points, etc.)")
    print(f"  • Recent form and streaks")
    print(f"  • Advanced statistics (rebounds, assists, shooting %)")
    print(f"  • News sentiment analysis")
    print(f"  • Injury impact assessment")
    print(f"  • Contextual factors (conference matchups, home court)")
    
    return predictor

if __name__ == "__main__":
    predictor = demo_enhanced_prediction()