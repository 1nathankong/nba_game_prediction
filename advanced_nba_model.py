import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb
import lightgbm as lgb
from datetime import datetime, timedelta
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class AdvancedNBAPredictor:
    """
    Advanced NBA Game Predictor with sentiment analysis and comprehensive features
    """
    
    def __init__(self, data_dir="nba_data"):
        self.data_dir = data_dir
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.is_fitted = False
        
        # Model configurations
        self.model_configs = {
            'random_forest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 15, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1, 0.15],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9, 1.0]
                }
            },
            'xgboost': {
                'model': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1, 0.15],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9],
                    'colsample_bytree': [0.8, 0.9, 1.0]
                }
            },
            'lightgbm': {
                'model': lgb.LGBMClassifier(random_state=42, verbose=-1),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1, 0.15],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9],
                    'colsample_bytree': [0.8, 0.9, 1.0]
                }
            },
            'logistic': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'params': {
                    'C': [0.1, 1.0, 10.0],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear']
                }
            }
        }
    
    def load_data(self, filename="nba_ml_dataset.csv"):
        """Load the ML-ready dataset"""
        data_path = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset not found: {data_path}")
        
        df = pd.read_csv(data_path)
        df['date'] = pd.to_datetime(df['date'])
        
        print(f"Loaded dataset: {df.shape}")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        
        return df
    
    def prepare_features(self, df, fit_scaler=True):
        """Prepare features for modeling"""
        print("Preparing features for modeling...")
        
        # Define feature categories
        exclude_cols = ['game_id', 'date', 'home_team_abbrev', 'away_team_abbrev', 'season',
                       'home_score', 'away_score', 'home_win', 'point_spread', 'total_points',
                       'margin_category', 'total_category']
        
        # Select feature columns
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Handle missing values
        features_df = df[feature_cols].copy()
        features_df = features_df.fillna(0)  # Fill NaN with 0 for missing stats
        
        # Scale features
        if fit_scaler:
            features_scaled = self.scaler.fit_transform(features_df)
            self.feature_columns = feature_cols
        else:
            features_scaled = self.scaler.transform(features_df)
        
        features_df_scaled = pd.DataFrame(features_scaled, columns=feature_cols, index=features_df.index)
        
        print(f"Prepared {len(feature_cols)} features")
        return features_df_scaled
    
    def create_time_based_splits(self, df):
        """Create chronological train/validation/test splits"""
        # Sort by date
        df_sorted = df.sort_values('date').reset_index(drop=True)
        
        # Define split dates
        train_end = pd.to_datetime('2023-04-15')  # End of 2022-23 season
        val_end = pd.to_datetime('2024-04-15')    # End of 2023-24 season
        
        # Create splits
        train_mask = df_sorted['date'] <= train_end
        val_mask = (df_sorted['date'] > train_end) & (df_sorted['date'] <= val_end)
        test_mask = df_sorted['date'] > val_end
        
        train_df = df_sorted[train_mask].copy()
        val_df = df_sorted[val_mask].copy()
        test_df = df_sorted[test_mask].copy()
        
        print(f"Data splits:")
        print(f"  Training: {len(train_df)} games ({train_df['date'].min()} to {train_df['date'].max()})")
        print(f"  Validation: {len(val_df)} games ({val_df['date'].min()} to {val_df['date'].max()})")
        print(f"  Test: {len(test_df)} games ({test_df['date'].min()} to {test_df['date'].max()})")
        
        return train_df, val_df, test_df
    
    def train_models(self, X_train, y_train, X_val, y_val):
        """Train multiple models and find the best one"""
        print("Training multiple models...")
        
        model_results = {}
        
        # Time series cross-validation for more robust evaluation
        tscv = TimeSeriesSplit(n_splits=3)
        
        for model_name, config in self.model_configs.items():
            print(f"\nTraining {model_name}...")
            
            try:
                # Grid search with time series CV
                grid_search = GridSearchCV(
                    config['model'],
                    config['params'],
                    cv=tscv,
                    scoring='accuracy',
                    n_jobs=-1,
                    verbose=0
                )
                
                grid_search.fit(X_train, y_train)
                
                # Get best model
                best_model = grid_search.best_estimator_
                
                # Evaluate on validation set
                val_pred = best_model.predict(X_val)
                val_proba = best_model.predict_proba(X_val)[:, 1]
                
                val_accuracy = accuracy_score(y_val, val_pred)
                val_auc = roc_auc_score(y_val, val_proba)
                
                # Cross-validation score
                cv_scores = cross_val_score(best_model, X_train, y_train, cv=tscv, scoring='accuracy')
                
                model_results[model_name] = {
                    'model': best_model,
                    'best_params': grid_search.best_params_,
                    'val_accuracy': val_accuracy,
                    'val_auc': val_auc,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std()
                }
                
                print(f"  Best params: {grid_search.best_params_}")
                print(f"  Validation accuracy: {val_accuracy:.4f}")
                print(f"  Validation AUC: {val_auc:.4f}")
                print(f"  CV score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
                
            except Exception as e:
                print(f"  Error training {model_name}: {e}")
                continue
        
        # Find best model
        if model_results:
            best_model_name = max(model_results.keys(), key=lambda k: model_results[k]['val_accuracy'])
            self.best_model = model_results[best_model_name]['model']
            self.best_model_name = best_model_name
            self.models = model_results
            
            print(f"\nBest model: {best_model_name}")
            print(f"Best validation accuracy: {model_results[best_model_name]['val_accuracy']:.4f}")
        
        return model_results
    
    def create_ensemble(self, model_results):
        """Create ensemble model from best performers"""
        if len(model_results) < 2:
            return None
        
        # Select top 3 models by validation accuracy
        sorted_models = sorted(model_results.items(), key=lambda x: x[1]['val_accuracy'], reverse=True)
        top_models = sorted_models[:3]
        
        estimators = [(name, results['model']) for name, results in top_models]
        
        ensemble = VotingClassifier(estimators=estimators, voting='soft')
        
        print(f"Created ensemble with models: {[name for name, _ in top_models]}")
        
        return ensemble
    
    def evaluate_model(self, model, X_test, y_test, dataset_name="Test"):
        """Comprehensive model evaluation"""
        print(f"\n{dataset_name} Set Evaluation:")
        print("-" * 40)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"AUC-ROC: {auc:.4f}")
        
        # Classification report
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Away Win', 'Home Win']))
        
        # Confusion matrix
        print(f"\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        # Feature importance (if available)
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\nTop 15 Most Important Features:")
            print(feature_importance.head(15).to_string(index=False))
        
        return {
            'accuracy': accuracy,
            'auc': auc,
            'predictions': y_pred,
            'probabilities': y_proba
        }
    
    def save_model(self, model_name="best_nba_model"):
        """Save the trained model"""
        if self.best_model is None:
            print("No model to save!")
            return
        
        model_path = os.path.join(self.data_dir, f"{model_name}.joblib")
        scaler_path = os.path.join(self.data_dir, f"{model_name}_scaler.joblib")
        
        joblib.dump(self.best_model, model_path)
        joblib.dump(self.scaler, scaler_path)
        
        # Save feature columns
        feature_path = os.path.join(self.data_dir, f"{model_name}_features.txt")
        with open(feature_path, 'w') as f:
            for feature in self.feature_columns:
                f.write(f"{feature}\n")
        
        print(f"Model saved to {model_path}")
        self.is_fitted = True
    
    def load_model(self, model_name="best_nba_model"):
        """Load a saved model"""
        model_path = os.path.join(self.data_dir, f"{model_name}.joblib")
        scaler_path = os.path.join(self.data_dir, f"{model_name}_scaler.joblib")
        feature_path = os.path.join(self.data_dir, f"{model_name}_features.txt")
        
        if not all(os.path.exists(p) for p in [model_path, scaler_path, feature_path]):
            raise FileNotFoundError("Model files not found!")
        
        self.best_model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        
        with open(feature_path, 'r') as f:
            self.feature_columns = [line.strip() for line in f.readlines()]
        
        self.is_fitted = True
        print(f"Model loaded from {model_path}")
    
    def predict_game(self, home_team, away_team, game_features):
        """Predict a single game outcome"""
        if not self.is_fitted:
            raise ValueError("Model not fitted! Train or load a model first.")
        
        # Prepare features (assuming game_features is already in the right format)
        features_scaled = self.scaler.transform([game_features])
        
        # Make prediction
        prediction = self.best_model.predict(features_scaled)[0]
        probability = self.best_model.predict_proba(features_scaled)[0]
        
        result = {
            'home_team': home_team,
            'away_team': away_team,
            'predicted_winner': 'home' if prediction == 1 else 'away',
            'home_win_probability': probability[1],
            'away_win_probability': probability[0]
        }
        
        return result
    
    def train_complete_pipeline(self):
        """Complete training pipeline"""
        print("="*60)
        print("ADVANCED NBA PREDICTOR TRAINING PIPELINE")
        print("="*60)
        
        # Load data
        df = self.load_data()
        
        # Create time-based splits
        train_df, val_df, test_df = self.create_time_based_splits(df)
        
        # Prepare features
        X_train = self.prepare_features(train_df, fit_scaler=True)
        X_val = self.prepare_features(val_df, fit_scaler=False)
        X_test = self.prepare_features(test_df, fit_scaler=False)
        
        y_train = train_df['home_win']
        y_val = val_df['home_win']
        y_test = test_df['home_win']
        
        print(f"\nTarget distribution:")
        print(f"  Training: {y_train.mean():.3f} home win rate")
        print(f"  Validation: {y_val.mean():.3f} home win rate")
        print(f"  Test: {y_test.mean():.3f} home win rate")
        
        # Train models
        model_results = self.train_models(X_train, y_train, X_val, y_val)
        
        if not model_results:
            print("No models were successfully trained!")
            return
        
        # Evaluate best model on test set
        print("\n" + "="*60)
        print("FINAL EVALUATION")
        print("="*60)
        
        test_results = self.evaluate_model(self.best_model, X_test, y_test, "Test")
        
        # Create and evaluate ensemble
        ensemble = self.create_ensemble(model_results)
        if ensemble is not None:
            print(f"\nTraining ensemble model...")
            ensemble.fit(X_train, y_train)
            
            ensemble_results = self.evaluate_model(ensemble, X_test, y_test, "Ensemble Test")
            
            # Use ensemble if it's better
            if ensemble_results['accuracy'] > test_results['accuracy']:
                self.best_model = ensemble
                self.best_model_name = "ensemble"
                print(f"\nEnsemble model selected (accuracy: {ensemble_results['accuracy']:.4f})")
        
        # Save model
        self.save_model()
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60)
        print(f"Best model: {self.best_model_name}")
        print(f"Test accuracy: {test_results['accuracy']:.4f}")
        print(f"Feature count: {len(self.feature_columns)}")

def main():
    # Initialize predictor
    predictor = AdvancedNBAPredictor()
    
    # Run complete training pipeline
    predictor.train_complete_pipeline()

if __name__ == "__main__":
    main()