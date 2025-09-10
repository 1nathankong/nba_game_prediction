import requests
import json
import pandas as pd
from datetime import datetime, timedelta
import time
import os

class NBADataCollector:
    def __init__(self):
        self.base_url = "http://site.api.espn.com/apis/site/v2/sports/basketball/nba"
        self.data_dir = "nba_data"
        self.create_directories()
    
    def create_directories(self):
        """Create directories for storing data"""
        os.makedirs(f"{self.data_dir}/games", exist_ok=True)
        os.makedirs(f"{self.data_dir}/teams", exist_ok=True)
        os.makedirs(f"{self.data_dir}/news", exist_ok=True)
    
    def get_scoreboard(self, date=None):
        """
        Get games for a specific date
        date: YYYYMMDD format (e.g., '20241101')
        """
        url = f"{self.base_url}/scoreboard"
        if date:
            url += f"?dates={date}"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching scoreboard for {date}: {e}")
            return None
    
    def get_all_teams(self):
        """Get information for all NBA teams"""
        url = f"{self.base_url}/teams"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching teams: {e}")
            return None
    
    def get_team_details(self, team_abbreviation):
        """Get detailed info for a specific team"""
        url = f"{self.base_url}/teams/{team_abbreviation}"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching team {team_abbreviation}: {e}")
            return None
    
    def get_news(self, date=None):
        """Get NBA news for a specific date or latest"""
        url = f"{self.base_url}/news"
        if date:
            url += f"?dates={date}"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching news for {date}: {e}")
            return None
    
    def save_data(self, data, filename, subfolder=""):
        """Save data to JSON file"""
        if data:
            filepath = os.path.join(self.data_dir, subfolder, filename)
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Data saved to {filepath}")
    
    def collect_historical_games(self, start_date, end_date, season_name=""):
        """
        Collect game data for a date range
        start_date, end_date: datetime objects
        season_name: optional name for the season (e.g., "2020-21")
        """
        print(f"Collecting games from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        current_date = start_date
        games_data = []
        
        while current_date <= end_date:
            date_str = current_date.strftime('%Y%m%d')
            print(f"Fetching games for {date_str}")
            
            scoreboard = self.get_scoreboard(date_str)
            if scoreboard:
                # Save daily scoreboard
                self.save_data(scoreboard, f"scoreboard_{date_str}.json", "games")
                
                # Extract game summaries
                if 'events' in scoreboard:
                    for game in scoreboard['events']:
                        game_info = self.extract_game_features(game)
                        if game_info:
                            if season_name:
                                game_info['season'] = season_name
                            games_data.append(game_info)
            
            current_date += timedelta(days=1)
            time.sleep(0.5)  # Be respectful to the API
        
        # Save season-specific and consolidated games data
        df = pd.DataFrame(games_data)
        
        if season_name:
            season_filename = f"games_{season_name.replace('-', '_')}.csv"
            df.to_csv(f"{self.data_dir}/{season_filename}", index=False)
            print(f"Saved {len(games_data)} games to {season_filename}")
        
        df.to_csv(f"{self.data_dir}/consolidated_games.csv", index=False)
        print(f"Saved {len(games_data)} games to consolidated_games.csv")
        
        return df

    def collect_multiple_seasons(self, seasons_config):
        """
        Collect data for multiple NBA seasons
        seasons_config: list of dicts with season info
        Example: [{"name": "2020-21", "start": datetime(2020, 12, 22), "end": datetime(2021, 5, 16)}]
        """
        all_games_data = []
        
        for season in seasons_config:
            print(f"\n{'='*60}")
            print(f"COLLECTING {season['name']} SEASON DATA")
            print(f"{'='*60}")
            
            season_df = self.collect_historical_games(
                season['start'], 
                season['end'], 
                season['name']
            )
            
            if season_df is not None and not season_df.empty:
                all_games_data.append(season_df)
                print(f"✓ {season['name']}: {len(season_df)} games collected")
            else:
                print(f"✗ {season['name']}: No games collected")
        
        # Combine all seasons
        if all_games_data:
            combined_df = pd.concat(all_games_data, ignore_index=True)
            combined_df.to_csv(f"{self.data_dir}/all_seasons_consolidated.csv", index=False)
            print(f"\n{'='*60}")
            print(f"TOTAL GAMES COLLECTED: {len(combined_df)}")
            print(f"Seasons: {', '.join([s['name'] for s in seasons_config])}")
            print(f"Data saved to all_seasons_consolidated.csv")
            print(f"{'='*60}")
            
            return combined_df
        else:
            print("No data collected for any season")
            return None
    
    def extract_game_features(self, game):
        """Extract key features from a game object"""
        try:
            # Basic game info
            game_id = game.get('id')
            date = game.get('date')
            
            # Teams
            competitions = game.get('competitions', [{}])
            if not competitions:
                return None
            
            competition = competitions[0]
            competitors = competition.get('competitors', [])
            
            if len(competitors) != 2:
                return None
            
            # Determine home/away teams
            home_team = None
            away_team = None
            
            for competitor in competitors:
                if competitor.get('homeAway') == 'home':
                    home_team = competitor
                else:
                    away_team = competitor
            
            if not home_team or not away_team:
                return None
            
            # Extract team info and scores
            game_info = {
                'game_id': game_id,
                'date': date,
                'status': game.get('status', {}).get('type', {}).get('description'),
                'home_team_id': home_team.get('team', {}).get('id'),
                'home_team_name': home_team.get('team', {}).get('displayName'),
                'home_team_abbrev': home_team.get('team', {}).get('abbreviation'),
                'home_score': home_team.get('score'),
                'away_team_id': away_team.get('team', {}).get('id'),
                'away_team_name': away_team.get('team', {}).get('displayName'),
                'away_team_abbrev': away_team.get('team', {}).get('abbreviation'),
                'away_score': away_team.get('score'),
                'venue': competition.get('venue', {}).get('fullName'),
                'attendance': competition.get('attendance')
            }
            
            return game_info
            
        except Exception as e:
            print(f"Error extracting game features: {e}")
            return None
    
    def collect_team_data(self):
        """Collect detailed data for all teams"""
        print("Collecting team data...")
        
        # Get all teams first
        all_teams = self.get_all_teams()
        if not all_teams:
            return None
        
        self.save_data(all_teams, "all_teams.json", "teams")
        
        # Get detailed info for each team
        teams_detailed = []
        
        if 'sports' in all_teams:
            for sport in all_teams['sports']:
                if 'leagues' in sport:
                    for league in sport['leagues']:
                        if 'teams' in league:
                            for team in league['teams']:
                                team_abbrev = team.get('team', {}).get('abbreviation')
                                if team_abbrev:
                                    print(f"Fetching details for {team_abbrev}")
                                    
                                    team_details = self.get_team_details(team_abbrev)
                                    if team_details:
                                        teams_detailed.append(team_details)
                                        self.save_data(team_details, f"team_{team_abbrev}.json", "teams")
                                    
                                    time.sleep(0.5)  # Rate limiting
        
        return teams_detailed
    
    def collect_news_data(self):
        """Collect latest NBA news"""
        print("Collecting news data...")
        
        news = self.get_news()
        if news:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.save_data(news, f"news_{timestamp}.json", "news")
            return news
        
        return None
    
    def collect_historical_news(self, start_date, end_date):
        """
        Collect news data for a date range
        start_date, end_date: datetime objects
        """
        print(f"Collecting news from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        current_date = start_date
        all_news = []
        
        while current_date <= end_date:
            # Try weekly intervals to reduce API calls
            if (current_date.weekday() == 0):  # Only collect on Mondays
                date_str = current_date.strftime('%Y%m%d')
                print(f"Fetching news for week of {date_str}")
                
                news = self.get_news(date_str)
                if news:
                    self.save_data(news, f"news_{date_str}.json", "news")
                    if 'articles' in news:
                        all_news.extend(news['articles'])
                
                time.sleep(1)  # Be respectful to the API
            
            current_date += timedelta(days=1)
        
        # Save consolidated news
        if all_news:
            consolidated_news = {
                'articles': all_news,
                'total_articles': len(all_news),
                'date_range': f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
            }
            self.save_data(consolidated_news, "consolidated_historical_news.json", "news")
            print(f"Saved {len(all_news)} news articles to consolidated file")
        
        return all_news

# Example usage
def main():
    collector = NBADataCollector()
    
    # Collect team data (run once to get team info)
    print("=== Collecting Team Data ===")
    collector.collect_team_data()
    
    # Define NBA seasons 2020-2025 (regular season dates)
    seasons_config = [
        {
            "name": "2020-21",
            "start": datetime(2020, 12, 22),  # COVID-delayed start
            "end": datetime(2021, 5, 16)      # Regular season end
        },
        {
            "name": "2021-22", 
            "start": datetime(2021, 10, 19),  # Season start
            "end": datetime(2022, 4, 10)      # Regular season end
        },
        {
            "name": "2022-23",
            "start": datetime(2022, 10, 18),  # Season start  
            "end": datetime(2023, 4, 9)       # Regular season end
        },
        {
            "name": "2023-24",
            "start": datetime(2023, 10, 17),  # Season start
            "end": datetime(2024, 4, 14)      # Regular season end
        },
        {
            "name": "2024-25",
            "start": datetime(2024, 10, 22),  # Season start
            "end": datetime(2025, 4, 13)      # Expected regular season end
        }
    ]
    
    # Collect all seasons data
    print("\n=== Collecting Historical NBA Data (2020-2025) ===")
    combined_df = collector.collect_multiple_seasons(seasons_config)
    
    # Collect historical news data from 2020-2025
    print("\n=== Collecting Historical News Data (2020-2025) ===")
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2025, 12, 31)
    collector.collect_historical_news(start_date, end_date)
    
    print("\n=== Data Collection Complete ===")
    if combined_df is not None:
        print(f"Total games collected: {len(combined_df)}")
        print("Expected: ~6,150 games (5 seasons × ~1,230 games each)")
        print("Files created:")
        print("  - all_seasons_consolidated.csv (all 5 seasons)")
        print("  - games_2020_21.csv")
        print("  - games_2021_22.csv") 
        print("  - games_2022_23.csv")
        print("  - games_2023_24.csv")
        print("  - games_2024_25.csv")
        print("Check the 'nba_data' folder for all collected data")
    else:
        print("No data was collected")

if __name__ == "__main__":
    main()