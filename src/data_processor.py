import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import glob
from collections import defaultdict

class NBADataProcessor:
    def __init__(self, data_dir="nba_data"):
        self.data_dir = data_dir
        self.team_mappings = self.load_team_mappings()
        
    def load_team_mappings(self):
        """Load team ID to abbreviation mappings"""
        teams_file = os.path.join(self.data_dir, "teams", "all_teams.json")
        mappings = {}
        
        if os.path.exists(teams_file):
            with open(teams_file, 'r') as f:
                teams_data = json.load(f)
                
            # Extract team mappings
            for sport in teams_data.get('sports', []):
                for league in sport.get('leagues', []):
                    for team in league.get('teams', []):
                        team_info = team.get('team', {})
                        team_id = team_info.get('id')
                        team_abbr = team_info.get('abbreviation')
                        team_name = team_info.get('displayName')
                        
                        if team_id and team_abbr:
                            mappings[team_id] = {
                                'abbreviation': team_abbr,
                                'name': team_name
                            }
        
        return mappings
    
    def extract_game_data(self, game_json):
        """Extract game data from JSON format"""
        games = []
        
        if 'events' not in game_json:
            return games
            
        for event in game_json['events']:
            try:
                game_id = event.get('id')
                date = event.get('date')
                status = event.get('status', {}).get('type', {}).get('description', 'Unknown')
                
                # Skip games that haven't been played
                if status in ['Scheduled', 'Postponed', 'Canceled']:
                    continue
                
                competitions = event.get('competitions', [])
                if not competitions:
                    continue
                    
                competition = competitions[0]
                competitors = competition.get('competitors', [])
                
                if len(competitors) != 2:
                    continue
                
                # Extract team data
                home_team = None
                away_team = None
                
                for competitor in competitors:
                    if competitor.get('homeAway') == 'home':
                        home_team = competitor
                    else:
                        away_team = competitor
                
                if not home_team or not away_team:
                    continue
                
                # Extract basic game info
                game_data = {
                    'game_id': game_id,
                    'date': date,
                    'status': status,
                    'venue': competition.get('venue', {}).get('fullName', ''),
                    'attendance': competition.get('attendance', 0),
                    'home_team_id': home_team.get('team', {}).get('id'),
                    'home_team_name': home_team.get('team', {}).get('displayName'),
                    'home_team_abbr': home_team.get('team', {}).get('abbreviation'),
                    'home_score': int(home_team.get('score', 0)),
                    'away_team_id': away_team.get('team', {}).get('id'),
                    'away_team_name': away_team.get('team', {}).get('displayName'),
                    'away_team_abbr': away_team.get('team', {}).get('abbreviation'),
                    'away_score': int(away_team.get('score', 0)),
                }
                
                # Determine winner
                if game_data['home_score'] > game_data['away_score']:
                    game_data['winner'] = 'home'
                    game_data['home_win'] = 1
                    game_data['away_win'] = 0
                else:
                    game_data['winner'] = 'away'
                    game_data['home_win'] = 0
                    game_data['away_win'] = 1
                
                # Calculate point differential
                game_data['point_differential'] = game_data['home_score'] - game_data['away_score']
                game_data['total_points'] = game_data['home_score'] + game_data['away_score']
                
                # Extract detailed statistics if available
                home_stats = home_team.get('statistics', [])
                away_stats = away_team.get('statistics', [])
                
                # Add basic stats
                for stat in home_stats:
                    stat_name = stat.get('name', '').lower().replace(' ', '_')
                    game_data[f'home_{stat_name}'] = float(stat.get('displayValue', 0))
                
                for stat in away_stats:
                    stat_name = stat.get('name', '').lower().replace(' ', '_')
                    game_data[f'away_{stat_name}'] = float(stat.get('displayValue', 0))
                
                games.append(game_data)
                
            except Exception as e:
                print(f"Error processing game {game_id}: {e}")
                continue
        
        return games
    
    def process_all_games(self):
        """Process all game JSON files and create master dataset"""
        print("Processing all game files...")
        
        all_games = []
        games_pattern = os.path.join(self.data_dir, "games", "scoreboard_*.json")
        game_files = glob.glob(games_pattern)
        
        print(f"Found {len(game_files)} game files")
        
        for file_path in sorted(game_files):
            try:
                with open(file_path, 'r') as f:
                    game_data = json.load(f)
                
                games = self.extract_game_data(game_data)
                all_games.extend(games)
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
        
        print(f"Extracted {len(all_games)} games total")
        
        if not all_games:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(all_games)
        
        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)
        
        return df
    
    def calculate_team_stats(self, games_df):
        """Calculate rolling team statistics"""
        print("Calculating team statistics...")
        
        # Create team stats dictionary
        team_stats = defaultdict(lambda: {
            'games_played': 0,
            'wins': 0,
            'losses': 0,
            'points_for': [],
            'points_against': [],
            'home_games': 0,
            'home_wins': 0,
            'away_games': 0,
            'away_wins': 0,
            'recent_games': [],  # Last 10 games
            'last_game_date': None
        })
        
        enriched_games = []
        
        for idx, game in games_df.iterrows():
            home_team = game['home_team_abbr']
            away_team = game['away_team_abbr']
            game_date = game['date']
            
            # Calculate stats before this game
            home_stats = team_stats[home_team].copy()
            away_stats = team_stats[away_team].copy()
            
            # Calculate form and averages
            game_enriched = game.copy()
            
            # Home team pre-game stats
            game_enriched['home_games_played'] = home_stats['games_played']
            game_enriched['home_wins_before'] = home_stats['wins']
            game_enriched['home_win_pct'] = home_stats['wins'] / max(1, home_stats['games_played'])
            game_enriched['home_avg_points'] = np.mean(home_stats['points_for']) if home_stats['points_for'] else 0
            game_enriched['home_avg_allowed'] = np.mean(home_stats['points_against']) if home_stats['points_against'] else 0
            game_enriched['home_home_win_pct'] = home_stats['home_wins'] / max(1, home_stats['home_games'])
            game_enriched['home_away_win_pct'] = home_stats['away_wins'] / max(1, home_stats['away_games'])
            
            # Recent form (last 5 games)
            recent_home = home_stats['recent_games'][-5:] if len(home_stats['recent_games']) >= 5 else home_stats['recent_games']
            game_enriched['home_recent_wins'] = sum(recent_home)
            game_enriched['home_recent_form'] = len(recent_home)
            
            # Days since last game
            if home_stats['last_game_date']:
                game_enriched['home_days_rest'] = (game_date - home_stats['last_game_date']).days
            else:
                game_enriched['home_days_rest'] = 0
            
            # Away team pre-game stats
            game_enriched['away_games_played'] = away_stats['games_played']
            game_enriched['away_wins_before'] = away_stats['wins']
            game_enriched['away_win_pct'] = away_stats['wins'] / max(1, away_stats['games_played'])
            game_enriched['away_avg_points'] = np.mean(away_stats['points_for']) if away_stats['points_for'] else 0
            game_enriched['away_avg_allowed'] = np.mean(away_stats['points_against']) if away_stats['points_against'] else 0
            game_enriched['away_home_win_pct'] = away_stats['home_wins'] / max(1, away_stats['home_games'])
            game_enriched['away_away_win_pct'] = away_stats['away_wins'] / max(1, away_stats['away_games'])
            
            # Recent form (last 5 games)
            recent_away = away_stats['recent_games'][-5:] if len(away_stats['recent_games']) >= 5 else away_stats['recent_games']
            game_enriched['away_recent_wins'] = sum(recent_away)
            game_enriched['away_recent_form'] = len(recent_away)
            
            # Days since last game
            if away_stats['last_game_date']:
                game_enriched['away_days_rest'] = (game_date - away_stats['last_game_date']).days
            else:
                game_enriched['away_days_rest'] = 0
            
            # Head-to-head record (simplified)
            h2h_key = f"{home_team}_vs_{away_team}"
            game_enriched['h2h_games'] = 0  # Placeholder for H2H history
            
            enriched_games.append(game_enriched)
            
            # Update team stats after this game
            # Home team updates
            team_stats[home_team]['games_played'] += 1
            team_stats[home_team]['points_for'].append(game['home_score'])
            team_stats[home_team]['points_against'].append(game['away_score'])
            team_stats[home_team]['home_games'] += 1
            team_stats[home_team]['last_game_date'] = game_date
            
            if game['home_win']:
                team_stats[home_team]['wins'] += 1
                team_stats[home_team]['home_wins'] += 1
                team_stats[home_team]['recent_games'].append(1)
            else:
                team_stats[home_team]['losses'] += 1
                team_stats[home_team]['recent_games'].append(0)
            
            # Away team updates
            team_stats[away_team]['games_played'] += 1
            team_stats[away_team]['points_for'].append(game['away_score'])
            team_stats[away_team]['points_against'].append(game['home_score'])
            team_stats[away_team]['away_games'] += 1
            team_stats[away_team]['last_game_date'] = game_date
            
            if game['away_win']:
                team_stats[away_team]['wins'] += 1
                team_stats[away_team]['away_wins'] += 1
                team_stats[away_team]['recent_games'].append(1)
            else:
                team_stats[away_team]['losses'] += 1
                team_stats[away_team]['recent_games'].append(0)
            
            # Keep only last 10 games for recent form
            if len(team_stats[home_team]['recent_games']) > 10:
                team_stats[home_team]['recent_games'].pop(0)
            if len(team_stats[away_team]['recent_games']) > 10:
                team_stats[away_team]['recent_games'].pop(0)
        
        return pd.DataFrame(enriched_games)
    
    def add_season_context(self, df):
        """Add season timing and context features"""
        print("Adding season context...")
        
        df = df.copy()
        
        # Extract season year
        df['season_year'] = df['date'].dt.year
        
        # Determine season start/end dates by year
        season_dates = {
            2020: {'start': datetime(2020, 12, 22), 'end': datetime(2021, 5, 16)},
            2021: {'start': datetime(2021, 10, 19), 'end': datetime(2022, 4, 10)},
            2022: {'start': datetime(2022, 10, 18), 'end': datetime(2023, 4, 9)},
            2023: {'start': datetime(2023, 10, 17), 'end': datetime(2024, 4, 14)},
            2024: {'start': datetime(2024, 10, 22), 'end': datetime(2025, 4, 13)},
        }
        
        # Calculate season progress
        for idx, row in df.iterrows():
            game_date = row['date']
            year = game_date.year
            
            # Determine which season this game belongs to
            season_year = year
            if game_date.month >= 10:  # October-December games belong to next season
                season_year = year
            elif game_date.month <= 7:  # January-July games belong to previous season
                season_year = year - 1
            
            if season_year in season_dates:
                season_start = season_dates[season_year]['start']
                season_end = season_dates[season_year]['end']
                
                # Calculate days into season
                days_into_season = (game_date - season_start).days
                total_season_days = (season_end - season_start).days
                
                df.loc[idx, 'days_into_season'] = max(0, days_into_season)
                df.loc[idx, 'season_progress'] = max(0, min(1, days_into_season / total_season_days))
                df.loc[idx, 'season'] = f"{season_year}-{str(season_year+1)[-2:]}"
            else:
                df.loc[idx, 'days_into_season'] = 0
                df.loc[idx, 'season_progress'] = 0
                df.loc[idx, 'season'] = f"{season_year}-{str(season_year+1)[-2:]}"
        
        # Add month and day of week
        df['month'] = df['date'].dt.month
        df['day_of_week'] = df['date'].dt.dayofweek  # 0=Monday, 6=Sunday
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        return df
    
    def create_training_dataset(self):
        """Create the complete training dataset"""
        print("Creating comprehensive training dataset...")
        
        # Process all games
        games_df = self.process_all_games()
        
        if games_df.empty:
            print("No game data found!")
            return pd.DataFrame()
        
        # Calculate team statistics
        enriched_df = self.calculate_team_stats(games_df)
        
        # Add season context
        final_df = self.add_season_context(enriched_df)
        
        # Select relevant features for ML
        feature_columns = [
            'game_id', 'date', 'season', 'home_team_abbr', 'away_team_abbr',
            'home_score', 'away_score', 'home_win', 'point_differential', 'total_points',
            'home_games_played', 'away_games_played',
            'home_win_pct', 'away_win_pct',
            'home_avg_points', 'away_avg_points',
            'home_avg_allowed', 'away_avg_allowed',
            'home_home_win_pct', 'away_away_win_pct',
            'home_recent_wins', 'away_recent_wins',
            'home_recent_form', 'away_recent_form',
            'home_days_rest', 'away_days_rest',
            'days_into_season', 'season_progress',
            'month', 'day_of_week', 'is_weekend',
            'attendance'
        ]
        
        # Keep only columns that exist
        available_columns = [col for col in feature_columns if col in final_df.columns]
        
        final_df = final_df[available_columns]
        
        print(f"Final dataset shape: {final_df.shape}")
        print(f"Date range: {final_df['date'].min()} to {final_df['date'].max()}")
        print(f"Seasons included: {final_df['season'].unique()}")
        
        return final_df
    
    def save_processed_data(self, df, filename="processed_nba_games.csv"):
        """Save processed data to CSV"""
        output_path = os.path.join(self.data_dir, filename)
        df.to_csv(output_path, index=False)
        print(f"Saved processed data to {output_path}")
        return output_path

def main():
    processor = NBADataProcessor()
    
    # Create comprehensive training dataset
    df = processor.create_training_dataset()
    
    if not df.empty:
        # Save to CSV
        processor.save_processed_data(df)
        
        # Print summary statistics
        print("\n" + "="*60)
        print("DATASET SUMMARY")
        print("="*60)
        print(f"Total games: {len(df)}")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"Seasons: {', '.join(sorted(df['season'].unique()))}")
        print(f"Teams: {len(set(df['home_team_abbr'].unique()) | set(df['away_team_abbr'].unique()))}")
        print(f"Features: {df.shape[1]}")
        print("\nFeature columns:")
        for col in df.columns:
            print(f"  - {col}")
        
        # Sample of data
        print(f"\nSample data (first 3 rows):")
        print(df.head(3)[['date', 'home_team_abbr', 'away_team_abbr', 'home_score', 'away_score', 'home_win']].to_string())
        
    else:
        print("No data processed!")

if __name__ == "__main__":
    main()