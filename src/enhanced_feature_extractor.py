import pandas as pd
import numpy as np
import json
import glob
import os
from datetime import datetime, timedelta
from collections import defaultdict
import re
from textblob import TextBlob  # For sentiment analysis
import warnings
warnings.filterwarnings('ignore')

class EnhancedNBAFeatureExtractor:
    """
    Enhanced NBA Feature Extractor for improved game prediction
    
    Extracts comprehensive features from:
    - Game JSON files (244+ files with detailed statistics)
    - Team JSON files (31 team files with standings/info)  
    - News JSON files (for sentiment and injury analysis)
    - Historical performance data
    """
    
    def __init__(self, data_dir="nba_data"):
        self.data_dir = data_dir
        self.teams_data = {}
        self.games_data = []
        self.news_data = []
        self.team_mapping = {}  # abbreviation -> team_id
        self.player_stats = defaultdict(dict)
        self.team_season_stats = defaultdict(dict)
        self.injury_tracker = defaultdict(list)
        
    def load_all_data(self):
        """Load and organize all NBA data sources"""
        print("Loading comprehensive NBA data...")
        
        # Load team data first to establish mapping
        self.load_teams_data()
        
        # Load historical games data 
        self.load_games_data()
        
        # Load news data
        self.load_news_data()
        
        # Process loaded data to calculate advanced metrics
        self.calculate_team_season_stats()
        self.extract_player_statistics()
        self.analyze_injury_reports()
        
        print(f"Loaded: {len(self.teams_data)} teams, {len(self.games_data)} games, {len(self.news_data)} news articles")
        
    def load_teams_data(self):
        """Load team information and basic stats"""
        team_files = glob.glob(os.path.join(self.data_dir, "teams", "team_*.json"))
        
        for file_path in team_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
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
        """Load historical games data from JSON files"""
        game_files = glob.glob(os.path.join(self.data_dir, "games", "scoreboard_*.json"))
        
        for file_path in game_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    game_data = json.load(f)
                
                # Extract events (games) from the scoreboard data
                events = game_data.get('events', [])
                for event in events:
                    processed_game = self.process_game_event(event)
                    if processed_game:
                        self.games_data.append(processed_game)
                        
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                
    def process_game_event(self, event):
        """Process a single game event from JSON to extract comprehensive features"""
        try:
            game_info = {
                'game_id': event.get('id'),
                'date': event.get('date'),
                'name': event.get('name'),
                'season_type': event.get('season', {}).get('slug', 'regular'),
                'season_year': event.get('season', {}).get('year'),
            }
            
            # Process competition data
            competition = event.get('competitions', [{}])[0]
            game_info.update({
                'attendance': competition.get('attendance'),
                'neutral_site': competition.get('neutralSite', False),
                'conference_game': competition.get('conferenceCompetition', False),
            })
            
            # Extract venue information
            venue = competition.get('venue', {})
            game_info.update({
                'venue_id': venue.get('id'),
                'venue_name': venue.get('fullName'),
                'venue_city': venue.get('address', {}).get('city'),
                'venue_state': venue.get('address', {}).get('state'),
                'indoor_venue': venue.get('indoor', True)
            })
            
            # Process competitors (teams)
            competitors = competition.get('competitors', [])
            home_team = None
            away_team = None
            
            for competitor in competitors:
                if competitor.get('homeAway') == 'home':
                    home_team = competitor
                else:
                    away_team = competitor
            
            if home_team and away_team:
                # Extract team information
                game_info.update(self.extract_team_game_data('home', home_team))
                game_info.update(self.extract_team_game_data('away', away_team))
                
                # Determine winner
                home_score = float(home_team.get('score', 0) or 0)
                away_score = float(away_team.get('score', 0) or 0)
                
                game_info.update({
                    'home_score': home_score,
                    'away_score': away_score,
                    'home_wins': 1 if home_score > away_score else 0,
                    'total_points': home_score + away_score,
                    'point_differential': abs(home_score - away_score),
                    'game_completed': home_score > 0 and away_score > 0
                })
                
                return game_info
                
        except Exception as e:
            print(f"Error processing game event: {e}")
            return None
            
    def extract_team_game_data(self, home_away, competitor):
        """Extract comprehensive team data from a game competitor"""
        prefix = f"{home_away}_"
        team_data = {}
        
        # Basic team info
        team_info = competitor.get('team', {})
        team_data.update({
            f"{prefix}team_id": team_info.get('id'),
            f"{prefix}team_name": team_info.get('displayName'),
            f"{prefix}team_abbrev": team_info.get('abbreviation'),
            f"{prefix}winner": competitor.get('winner', False)
        })
        
        # Extract detailed statistics
        statistics = competitor.get('statistics', [])
        for stat in statistics:
            stat_name = stat.get('name', '').lower().replace(' ', '_')
            stat_value = stat.get('displayValue', '0')
            
            # Convert to numeric where possible
            try:
                if '.' in stat_value:
                    stat_value = float(stat_value)
                else:
                    stat_value = int(stat_value)
            except:
                pass  # Keep as string if conversion fails
                
            team_data[f"{prefix}{stat_name}"] = stat_value
        
        # Extract quarter-by-quarter scores
        linescores = competitor.get('linescores', [])
        for i, period_score in enumerate(linescores):
            period = period_score.get('period', i+1)
            score = float(period_score.get('value', 0))
            team_data[f"{prefix}q{period}_score"] = score
        
        # Extract player leaders
        leaders = competitor.get('leaders', [])
        for leader_cat in leaders:
            cat_name = leader_cat.get('name', '').lower()
            if leader_cat.get('leaders'):
                leader = leader_cat['leaders'][0]
                team_data[f"{prefix}leading_{cat_name}"] = float(leader.get('value', 0))
                
                # Store player info
                athlete = leader.get('athlete', {})
                team_data[f"{prefix}leading_{cat_name}_player"] = athlete.get('displayName')
                team_data[f"{prefix}leading_{cat_name}_position"] = athlete.get('position', {}).get('abbreviation')
        
        return team_data
        
    def load_news_data(self):
        """Load news articles for sentiment analysis and injury detection"""
        news_files = glob.glob(os.path.join(self.data_dir, "news", "news_*.json"))
        
        for file_path in news_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    news_data = json.load(f)
                
                articles = news_data.get('articles', [])
                self.news_data.extend(articles)
                        
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                
    def calculate_team_season_stats(self):
        """Calculate comprehensive season statistics for each team"""
        for team_id in self.teams_data.keys():
            self.team_season_stats[team_id] = {
                'games_played': 0,
                'wins': 0,
                'losses': 0,
                'home_wins': 0,
                'away_wins': 0,
                'home_games': 0,
                'away_games': 0,
                'points_for': 0,
                'points_against': 0,
                'total_rebounds': 0,
                'total_assists': 0,
                'field_goal_pct_sum': 0,
                'three_point_pct_sum': 0,
                'free_throw_pct_sum': 0,
                'games_with_stats': 0,
                'conference_wins': 0,
                'conference_games': 0,
                'last_5_games': [],  # Track recent performance
                'winning_streak': 0,
                'longest_winning_streak': 0,
                'point_differentials': []  # For calculating consistency
            }
        
        # Process each game to update team stats
        for game in self.games_data:
            if not game.get('game_completed', False):
                continue
                
            home_id = game.get('home_team_id')
            away_id = game.get('away_team_id')
            home_score = game.get('home_score', 0)
            away_score = game.get('away_score', 0)
            
            if home_id in self.team_season_stats and away_id in self.team_season_stats:
                self.update_team_game_stats(home_id, away_id, game, True)  # Home team
                self.update_team_game_stats(away_id, home_id, game, False)  # Away team
                
    def update_team_game_stats(self, team_id, opponent_id, game, is_home):
        """Update team statistics based on a single game"""
        stats = self.team_season_stats[team_id]
        opponent_stats = self.team_season_stats[opponent_id]
        
        # Determine if team won
        if is_home:
            team_score = game.get('home_score', 0)
            opponent_score = game.get('away_score', 0)
            won = game.get('home_wins', 0) == 1
            prefix = 'home_'
        else:
            team_score = game.get('away_score', 0)
            opponent_score = game.get('home_score', 0)
            won = game.get('home_wins', 0) == 0
            prefix = 'away_'
        
        # Update basic stats
        stats['games_played'] += 1
        if won:
            stats['wins'] += 1
            if is_home:
                stats['home_wins'] += 1
            else:
                stats['away_wins'] += 1
        else:
            stats['losses'] += 1
            
        if is_home:
            stats['home_games'] += 1
        else:
            stats['away_games'] += 1
            
        # Update scoring stats
        stats['points_for'] += team_score
        stats['points_against'] += opponent_score
        
        # Track point differential for this game
        point_diff = team_score - opponent_score
        stats['point_differentials'].append(point_diff)
        
        # Update advanced stats if available
        rebounds = game.get(f'{prefix}rebounds', 0)
        assists = game.get(f'{prefix}assists', 0)
        fg_pct = game.get(f'{prefix}fieldgoalpct', 0)
        three_pct = game.get(f'{prefix}threepointpct', 0) or game.get(f'{prefix}threepointfieldgoalpct', 0)
        ft_pct = game.get(f'{prefix}freethrowpct', 0)
        
        if any([rebounds, assists, fg_pct]):
            stats['games_with_stats'] += 1
            stats['total_rebounds'] += float(rebounds) if rebounds else 0
            stats['total_assists'] += float(assists) if assists else 0
            stats['field_goal_pct_sum'] += float(fg_pct) if fg_pct else 0
            stats['three_point_pct_sum'] += float(three_pct) if three_pct else 0
            stats['free_throw_pct_sum'] += float(ft_pct) if ft_pct else 0
        
        # Update last 5 games performance
        stats['last_5_games'].append(1 if won else 0)
        if len(stats['last_5_games']) > 5:
            stats['last_5_games'].pop(0)
            
        # Update winning streak
        if won:
            stats['winning_streak'] += 1
            stats['longest_winning_streak'] = max(stats['longest_winning_streak'], stats['winning_streak'])
        else:
            stats['winning_streak'] = 0
            
        # Conference game tracking
        if game.get('conference_game', False):
            stats['conference_games'] += 1
            if won:
                stats['conference_wins'] += 1
                
    def extract_player_statistics(self):
        """Extract and aggregate player performance statistics"""
        for game in self.games_data:
            for prefix in ['home_', 'away_']:
                team_id = game.get(f'{prefix}team_id')
                if not team_id:
                    continue
                    
                # Extract leading players for each category
                for category in ['points', 'rebounds', 'assists']:
                    player_name = game.get(f'{prefix}leading_{category}_player')
                    player_value = game.get(f'{prefix}leading_{category}', 0)
                    player_position = game.get(f'{prefix}leading_{category}_position')
                    
                    if player_name:
                        if player_name not in self.player_stats[team_id]:
                            self.player_stats[team_id][player_name] = {
                                'position': player_position,
                                'games_leading_points': 0,
                                'games_leading_rebounds': 0,
                                'games_leading_assists': 0,
                                'total_leading_points': 0,
                                'total_leading_rebounds': 0,
                                'total_leading_assists': 0
                            }
                        
                        self.player_stats[team_id][player_name][f'games_leading_{category}'] += 1
                        self.player_stats[team_id][player_name][f'total_leading_{category}'] += float(player_value)
                        
    def analyze_injury_reports(self):
        """Analyze news articles for injury information"""
        injury_keywords = [
            'injury', 'injured', 'hurt', 'out', 'questionable', 'doubtful', 
            'probable', 'game-time decision', 'ruled out', 'sidelined',
            'recovering', 'rehabilitation', 'surgery', 'sprain', 'strain'
        ]
        
        for article in self.news_data:
            headline = article.get('headline', '').lower()
            description = article.get('description', '').lower()
            text = f"{headline} {description}"
            
            # Check for injury-related keywords
            for keyword in injury_keywords:
                if keyword in text:
                    # Try to extract team and player information
                    injury_info = self.extract_injury_info(article, text)
                    if injury_info:
                        team_id = injury_info.get('team_id')
                        if team_id:
                            self.injury_tracker[team_id].append(injury_info)
                            
    def extract_injury_info(self, article, text):
        """Extract injury information from news article"""
        injury_info = {
            'date': article.get('published'),
            'headline': article.get('headline'),
            'severity': 'unknown',
            'team_id': None,
            'player_name': None
        }
        
        # Determine injury severity
        if any(word in text for word in ['out', 'ruled out', 'sidelined', 'surgery']):
            injury_info['severity'] = 'high'
        elif any(word in text for word in ['questionable', 'doubtful']):
            injury_info['severity'] = 'medium'
        elif any(word in text for word in ['probable', 'game-time decision']):
            injury_info['severity'] = 'low'
            
        # Try to identify team from article
        for team_abbrev, team_id in self.team_mapping.items():
            team_name = self.teams_data.get(team_id, {}).get('displayName', '').lower()
            if team_abbrev.lower() in text or team_name in text:
                injury_info['team_id'] = team_id
                break
                
        return injury_info
        
    def calculate_team_sentiment(self, team_id):
        """Calculate news sentiment for a team"""
        team_articles = []
        team_name = self.teams_data.get(team_id, {}).get('displayName', '').lower()
        team_abbrev = self.teams_data.get(team_id, {}).get('abbreviation', '').lower()
        
        # Find articles mentioning this team
        for article in self.news_data:
            headline = article.get('headline', '').lower()
            description = article.get('description', '').lower()
            
            if (team_name in headline or team_name in description or
                team_abbrev in headline or team_abbrev in description):
                team_articles.append(article)
        
        if not team_articles:
            return {
                'news_sentiment': 0.0,
                'news_count': 0,
                'recent_news_sentiment': 0.0,
                'injury_impact_score': 0.0
            }
        
        # Calculate sentiment scores
        sentiments = []
        recent_sentiments = []
        recent_threshold = datetime.now() - timedelta(days=7)
        
        for article in team_articles:
            text = f"{article.get('headline', '')} {article.get('description', '')}"
            try:
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
                        
            except Exception:
                sentiments.append(0.0)
        
        # Calculate injury impact
        injury_impact = self.calculate_injury_impact(team_id)
        
        return {
            'news_sentiment': np.mean(sentiments) if sentiments else 0.0,
            'news_count': len(team_articles),
            'recent_news_sentiment': np.mean(recent_sentiments) if recent_sentiments else 0.0,
            'injury_impact_score': injury_impact
        }
        
    def calculate_injury_impact(self, team_id):
        """Calculate injury impact score for a team"""
        injuries = self.injury_tracker.get(team_id, [])
        if not injuries:
            return 0.0
            
        impact_score = 0.0
        recent_threshold = datetime.now() - timedelta(days=14)
        
        for injury in injuries:
            # Weight recent injuries more heavily
            try:
                injury_date = datetime.fromisoformat(injury['date'].replace('Z', '+00:00'))
                days_ago = (datetime.now() - injury_date).days
                
                if days_ago <= 14:  # Recent injuries
                    severity_weights = {'high': 1.0, 'medium': 0.6, 'low': 0.3, 'unknown': 0.4}
                    time_weight = max(0.1, 1.0 - (days_ago / 14))  # Decay over time
                    impact_score += severity_weights.get(injury['severity'], 0.4) * time_weight
            except:
                pass
                
        return min(impact_score, 3.0)  # Cap at 3.0 for normalization
        
    def create_enhanced_features(self, home_team_id, away_team_id, game_date=None):
        """Create comprehensive feature set for a matchup"""
        if game_date is None:
            game_date = datetime.now()
            
        features = {}
        
        # Basic matchup information
        features.update({
            'home_team_id': home_team_id,
            'away_team_id': away_team_id,
            'game_date': game_date
        })
        
        # Team performance features
        for team_type, team_id in [('home', home_team_id), ('away', away_team_id)]:
            team_features = self.extract_team_performance_features(team_id, team_type)
            features.update(team_features)
            
        # Matchup-specific features
        matchup_features = self.calculate_matchup_features(home_team_id, away_team_id)
        features.update(matchup_features)
        
        # Contextual features
        context_features = self.calculate_contextual_features(home_team_id, away_team_id, game_date)
        features.update(context_features)
        
        return features
        
    def extract_team_performance_features(self, team_id, team_type):
        """Extract comprehensive performance features for a team"""
        prefix = f"{team_type}_"
        features = {}
        
        # Get team stats
        stats = self.team_season_stats.get(team_id, {})
        team_info = self.teams_data.get(team_id, {})
        
        # Basic team info
        features.update({
            f"{prefix}team_name": team_info.get('displayName', ''),
            f"{prefix}team_abbrev": team_info.get('abbreviation', ''),
            f"{prefix}conference": self.get_conference(team_info),
            f"{prefix}division": self.get_division(team_info)
        })
        
        # Performance metrics
        games_played = stats.get('games_played', 0)
        if games_played > 0:
            features.update({
                f"{prefix}win_percentage": stats['wins'] / games_played,
                f"{prefix}home_win_percentage": stats['home_wins'] / max(stats['home_games'], 1),
                f"{prefix}away_win_percentage": stats['away_wins'] / max(stats['away_games'], 1),
                f"{prefix}avg_points_for": stats['points_for'] / games_played,
                f"{prefix}avg_points_against": stats['points_against'] / games_played,
                f"{prefix}point_differential": (stats['points_for'] - stats['points_against']) / games_played,
                f"{prefix}games_played": games_played
            })
            
            # Advanced metrics
            if stats['games_with_stats'] > 0:
                features.update({
                    f"{prefix}avg_rebounds": stats['total_rebounds'] / stats['games_with_stats'],
                    f"{prefix}avg_assists": stats['total_assists'] / stats['games_with_stats'],
                    f"{prefix}avg_fg_percentage": stats['field_goal_pct_sum'] / stats['games_with_stats'],
                    f"{prefix}avg_three_point_percentage": stats['three_point_pct_sum'] / stats['games_with_stats'],
                    f"{prefix}avg_ft_percentage": stats['free_throw_pct_sum'] / stats['games_with_stats']
                })
            
            # Recent form
            last_5 = stats.get('last_5_games', [])
            features.update({
                f"{prefix}last_5_wins": sum(last_5),
                f"{prefix}current_streak": stats.get('winning_streak', 0),
                f"{prefix}longest_streak": stats.get('longest_winning_streak', 0)
            })
            
            # Consistency metrics
            point_diffs = stats.get('point_differentials', [])
            if point_diffs:
                features.update({
                    f"{prefix}consistency": np.std(point_diffs),  # Lower is more consistent
                    f"{prefix}avg_margin": np.mean(point_diffs)
                })
                
        # Conference performance
        conf_games = stats.get('conference_games', 0)
        if conf_games > 0:
            features[f"{prefix}conference_win_pct"] = stats['conference_wins'] / conf_games
            
        # News sentiment
        sentiment_data = self.calculate_team_sentiment(team_id)
        features.update({
            f"{prefix}news_sentiment": sentiment_data['news_sentiment'],
            f"{prefix}recent_news_sentiment": sentiment_data['recent_news_sentiment'],
            f"{prefix}injury_impact": sentiment_data['injury_impact_score'],
            f"{prefix}news_volume": sentiment_data['news_count']
        })
        
        # Key player analysis
        players = self.player_stats.get(team_id, {})
        if players:
            # Find most consistent performers
            top_scorer = max(players.items(), key=lambda x: x[1]['total_leading_points'], default=(None, {}))
            if top_scorer[0]:
                features.update({
                    f"{prefix}top_scorer": top_scorer[0],
                    f"{prefix}top_scorer_consistency": top_scorer[1]['games_leading_points']
                })
                
        return features
        
    def calculate_matchup_features(self, home_team_id, away_team_id):
        """Calculate head-to-head and matchup-specific features"""
        features = {}
        
        # Conference/Division matchup
        home_info = self.teams_data.get(home_team_id, {})
        away_info = self.teams_data.get(away_team_id, {})
        
        home_conf = self.get_conference(home_info)
        away_conf = self.get_conference(away_info)
        home_div = self.get_division(home_info)
        away_div = self.get_division(away_info)
        
        features.update({
            'same_conference': home_conf == away_conf,
            'same_division': home_div == away_div,
            'conference_matchup': f"{home_conf}_vs_{away_conf}",
            'inter_conference': home_conf != away_conf
        })
        
        # Performance differential features
        home_stats = self.team_season_stats.get(home_team_id, {})
        away_stats = self.team_season_stats.get(away_team_id, {})
        
        if home_stats.get('games_played', 0) > 0 and away_stats.get('games_played', 0) > 0:
            home_wpct = home_stats['wins'] / home_stats['games_played']
            away_wpct = away_stats['wins'] / away_stats['games_played']
            home_ppg = home_stats['points_for'] / home_stats['games_played']
            away_ppg = away_stats['points_for'] / away_stats['games_played']
            home_papg = home_stats['points_against'] / home_stats['games_played']
            away_papg = away_stats['points_against'] / away_stats['games_played']
            
            features.update({
                'win_pct_differential': home_wpct - away_wpct,
                'offensive_differential': home_ppg - away_papg,  # Home offense vs Away defense
                'defensive_differential': away_ppg - home_papg,  # Away offense vs Home defense
                'pace_differential': (home_ppg + home_papg) - (away_ppg + away_papg),
                'strength_of_schedule_diff': 0  # Placeholder for SOS calculation
            })
            
        # Recent form comparison
        home_last5 = sum(home_stats.get('last_5_games', []))
        away_last5 = sum(away_stats.get('last_5_games', []))
        features['recent_form_differential'] = home_last5 - away_last5
        
        # Streak comparison
        home_streak = home_stats.get('winning_streak', 0)
        away_streak = away_stats.get('winning_streak', 0)
        features['streak_differential'] = home_streak - away_streak
        
        return features
        
    def calculate_contextual_features(self, home_team_id, away_team_id, game_date):
        """Calculate contextual features like rest, travel, schedule"""
        features = {}
        
        # Date-based features
        features.update({
            'month': game_date.month,
            'day_of_week': game_date.weekday(),
            'day_of_year': game_date.timetuple().tm_yday,
            'is_weekend': game_date.weekday() >= 5,
            'is_back_to_back': False,  # Placeholder - would need schedule analysis
            'days_rest_home': 1,  # Placeholder
            'days_rest_away': 1   # Placeholder
        })
        
        # Home court advantage calculation
        home_stats = self.team_season_stats.get(home_team_id, {})
        home_games = home_stats.get('home_games', 0)
        if home_games > 0:
            home_advantage = home_stats.get('home_wins', 0) / home_games
        else:
            home_advantage = 0.55  # League average
            
        features['home_court_advantage'] = home_advantage
        
        # Season context
        if game_date.month >= 10 or game_date.month <= 4:  # NBA season
            if game_date.month >= 10:
                season_progress = (game_date.month - 10) / 7  # Oct-Apr = 7 months
            else:
                season_progress = (game_date.month + 2) / 7  # Jan-Apr
        else:
            season_progress = 0  # Off-season
            
        features['season_progress'] = min(season_progress, 1.0)
        
        return features
        
    def get_conference(self, team_info):
        """Determine team's conference"""
        groups = team_info.get('groups', {})
        parent_id = groups.get('parent', {}).get('id')
        
        if parent_id == "5":
            return "Eastern"
        elif parent_id == "6":
            return "Western"
        return "Unknown"
        
    def get_division(self, team_info):
        """Determine team's division"""
        groups = team_info.get('groups', {})
        division_id = groups.get('id')
        
        division_map = {
            "9": "Southeast", "10": "Atlantic", "11": "Central",
            "12": "Northwest", "13": "Pacific", "14": "Southwest"
        }
        
        return division_map.get(division_id, "Unknown")
        
    def create_training_dataset(self, output_path=None):
        """Create comprehensive training dataset from all available games"""
        training_data = []
        
        print(f"Creating training dataset from {len(self.games_data)} games...")
        
        for game in self.games_data:
            if not game.get('game_completed', False):
                continue
                
            home_id = game.get('home_team_id')
            away_id = game.get('away_team_id')
            game_date_str = game.get('date')
            
            if home_id and away_id and game_date_str:
                try:
                    # Parse game date
                    game_date = datetime.fromisoformat(game_date_str.replace('Z', '+00:00'))
                    
                    # Create features for this game
                    features = self.create_enhanced_features(home_id, away_id, game_date)
                    
                    # Add target variable
                    features['home_wins'] = game.get('home_wins', 0)
                    features['home_score'] = game.get('home_score', 0)
                    features['away_score'] = game.get('away_score', 0)
                    
                    training_data.append(features)
                    
                except Exception as e:
                    print(f"Error processing game {game.get('game_id')}: {e}")
                    continue
        
        # Convert to DataFrame
        df = pd.DataFrame(training_data)
        
        # Save if path provided
        if output_path:
            df.to_csv(output_path, index=False)
            print(f"Training dataset saved to {output_path}")
            
        print(f"Created training dataset with {len(df)} games and {len(df.columns)} features")
        return df
        
    def prepare_for_baseline_model(self, df):
        """Prepare features for integration with baseline_model.py"""
        # Select the most important numerical features
        numerical_features = []
        categorical_features = []
        
        for col in df.columns:
            if col in ['home_wins', 'home_score', 'away_score', 'game_date']:
                continue  # Skip target and non-feature columns
                
            if df[col].dtype in ['object', 'string']:
                categorical_features.append(col)
            else:
                numerical_features.append(col)
        
        # Create final feature set
        feature_df = df.copy()
        
        # Fill missing values
        for col in numerical_features:
            feature_df[col] = pd.to_numeric(feature_df[col], errors='coerce').fillna(0)
            
        for col in categorical_features:
            feature_df[col] = feature_df[col].fillna('Unknown')
        
        return feature_df, numerical_features, categorical_features


def demo_enhanced_features():
    """Demonstrate the enhanced feature extraction"""
    print("=" * 80)
    print("ENHANCED NBA FEATURE EXTRACTION DEMO")
    print("=" * 80)
    
    # Initialize extractor
    extractor = EnhancedNBAFeatureExtractor()
    
    # Load all data
    extractor.load_all_data()
    
    # Create sample matchup features
    if len(extractor.teams_data) >= 2:
        team_ids = list(extractor.teams_data.keys())[:2]
        team1_name = extractor.teams_data[team_ids[0]].get('displayName', 'Team 1')
        team2_name = extractor.teams_data[team_ids[1]].get('displayName', 'Team 2')
        
        print(f"\nSample matchup: {team2_name} @ {team1_name}")
        print("-" * 50)
        
        features = extractor.create_enhanced_features(team_ids[0], team_ids[1])
        
        # Display key features
        key_features = [
            'home_win_percentage', 'away_win_percentage', 'win_pct_differential',
            'home_avg_points_for', 'away_avg_points_for', 'offensive_differential',
            'home_last_5_wins', 'away_last_5_wins', 'recent_form_differential',
            'home_news_sentiment', 'away_news_sentiment', 'home_injury_impact',
            'same_conference', 'home_court_advantage'
        ]
        
        for feature in key_features:
            if feature in features:
                print(f"{feature}: {features[feature]}")
    
    # Create training dataset
    print(f"\nCreating training dataset...")
    training_df = extractor.create_training_dataset()
    
    if not training_df.empty:
        print(f"\nTraining dataset shape: {training_df.shape}")
        print(f"Features available: {len(training_df.columns)}")
        
        # Show feature correlation with target
        if 'home_wins' in training_df.columns:
            print(f"\nHome team win rate: {training_df['home_wins'].mean():.3f}")
            
            # Show most predictive features
            numerical_cols = training_df.select_dtypes(include=[np.number]).columns
            numerical_cols = [col for col in numerical_cols if col != 'home_wins']
            
            if numerical_cols:
                correlations = training_df[numerical_cols + ['home_wins']].corr()['home_wins'].sort_values(key=abs, ascending=False)
                print("\nTop 10 most correlated features with home wins:")
                print(correlations.head(10))
    
    return extractor, training_df


if __name__ == "__main__":
    extractor, df = demo_enhanced_features()