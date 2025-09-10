import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import glob
import re
from collections import defaultdict

# Sentiment analysis libraries
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    print("Warning: vaderSentiment not installed. Install with: pip install vaderSentiment")

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    print("Warning: textblob not installed. Install with: pip install textblob")

class NBASentimentAnalyzer:
    def __init__(self, data_dir="nba_data"):
        self.data_dir = data_dir
        self.team_keywords = self.load_team_keywords()
        
        # Initialize sentiment analyzers
        if VADER_AVAILABLE:
            self.vader_analyzer = SentimentIntensityAnalyzer()
        
    def load_team_keywords(self):
        """Load team names and variations for keyword matching"""
        teams_file = os.path.join(self.data_dir, "teams", "all_teams.json")
        keywords = {}
        
        if os.path.exists(teams_file):
            with open(teams_file, 'r') as f:
                teams_data = json.load(f)
                
            for sport in teams_data.get('sports', []):
                for league in sport.get('leagues', []):
                    for team in league.get('teams', []):
                        team_info = team.get('team', {})
                        abbr = team_info.get('abbreviation')
                        name = team_info.get('displayName', '')
                        location = team_info.get('location', '')
                        nickname = team_info.get('name', '')
                        
                        if abbr:
                            # Create keyword variations
                            variations = [abbr.lower()]
                            if name:
                                variations.append(name.lower())
                            if location:
                                variations.append(location.lower())
                            if nickname:
                                variations.append(nickname.lower())
                            
                            keywords[abbr] = {
                                'name': name,
                                'keywords': variations
                            }
        
        return keywords
    
    def analyze_text_sentiment(self, text):
        """Analyze sentiment of text using available libraries"""
        sentiments = {}
        
        if not text or not isinstance(text, str):
            return {'vader_compound': 0.0, 'textblob_polarity': 0.0, 'combined_sentiment': 0.0}
        
        # VADER sentiment (good for social media/news text)
        if VADER_AVAILABLE:
            vader_scores = self.vader_analyzer.polarity_scores(text)
            sentiments['vader_compound'] = vader_scores['compound']
            sentiments['vader_positive'] = vader_scores['pos']
            sentiments['vader_negative'] = vader_scores['neg']
            sentiments['vader_neutral'] = vader_scores['neu']
        else:
            sentiments['vader_compound'] = 0.0
        
        # TextBlob sentiment
        if TEXTBLOB_AVAILABLE:
            try:
                blob = TextBlob(text)
                sentiments['textblob_polarity'] = blob.sentiment.polarity
                sentiments['textblob_subjectivity'] = blob.sentiment.subjectivity
            except:
                sentiments['textblob_polarity'] = 0.0
                sentiments['textblob_subjectivity'] = 0.0
        else:
            sentiments['textblob_polarity'] = 0.0
        
        # Combined sentiment score (average of available methods)
        available_scores = []
        if 'vader_compound' in sentiments and sentiments['vader_compound'] != 0:
            available_scores.append(sentiments['vader_compound'])
        if 'textblob_polarity' in sentiments and sentiments['textblob_polarity'] != 0:
            available_scores.append(sentiments['textblob_polarity'])
        
        if available_scores:
            sentiments['combined_sentiment'] = np.mean(available_scores)
        else:
            sentiments['combined_sentiment'] = 0.0
        
        return sentiments
    
    def extract_team_mentions(self, text):
        """Extract which teams are mentioned in the text"""
        if not text or not isinstance(text, str):
            return []
        
        text_lower = text.lower()
        mentioned_teams = []
        
        for team_abbr, team_data in self.team_keywords.items():
            for keyword in team_data['keywords']:
                if keyword in text_lower:
                    mentioned_teams.append(team_abbr)
                    break
        
        return list(set(mentioned_teams))  # Remove duplicates
    
    def process_news_file(self, news_file):
        """Process a single news JSON file"""
        with open(news_file, 'r') as f:
            news_data = json.load(f)
        
        articles_data = []
        
        # Extract filename date if available
        filename = os.path.basename(news_file)
        file_date = None
        
        # Try to extract date from filename
        date_match = re.search(r'news_(\d{8})', filename)
        if date_match:
            try:
                file_date = datetime.strptime(date_match.group(1), '%Y%m%d')
            except:
                pass
        
        # Process articles
        articles = news_data.get('articles', [])
        
        for article in articles:
            try:
                article_data = {
                    'article_id': article.get('id'),
                    'headline': article.get('headline', ''),
                    'description': article.get('description', ''),
                    'published': article.get('published'),
                    'lastModified': article.get('lastModified'),
                    'type': article.get('type', ''),
                    'file_date': file_date,
                    'source_file': filename
                }
                
                # Combine headline and description for sentiment analysis
                full_text = f"{article_data['headline']} {article_data['description']}"
                
                # Analyze sentiment
                sentiment_scores = self.analyze_text_sentiment(full_text)
                article_data.update(sentiment_scores)
                
                # Extract team mentions
                mentioned_teams = self.extract_team_mentions(full_text)
                article_data['mentioned_teams'] = mentioned_teams
                article_data['num_teams_mentioned'] = len(mentioned_teams)
                
                # Parse published date
                if article_data['published']:
                    try:
                        article_data['published_date'] = pd.to_datetime(article_data['published'])
                    except:
                        article_data['published_date'] = file_date
                else:
                    article_data['published_date'] = file_date
                
                articles_data.append(article_data)
                
            except Exception as e:
                print(f"Error processing article {article.get('id', 'unknown')}: {e}")
                continue
        
        return articles_data
    
    def process_all_news(self):
        """Process all news files and create sentiment dataset"""
        print("Processing news files for sentiment analysis...")
        
        all_articles = []
        news_pattern = os.path.join(self.data_dir, "news", "news_*.json")
        news_files = glob.glob(news_pattern)
        
        print(f"Found {len(news_files)} news files")
        
        for news_file in sorted(news_files):
            try:
                articles = self.process_news_file(news_file)
                all_articles.extend(articles)
                print(f"Processed {len(articles)} articles from {os.path.basename(news_file)}")
            except Exception as e:
                print(f"Error processing {news_file}: {e}")
                continue
        
        if not all_articles:
            print("No news articles found!")
            return pd.DataFrame()
        
        df = pd.DataFrame(all_articles)
        print(f"Total articles processed: {len(df)}")
        
        return df
    
    def create_team_sentiment_scores(self, news_df):
        """Create team-specific sentiment scores by date"""
        print("Creating team-specific sentiment scores...")
        
        if news_df.empty:
            return pd.DataFrame()
        
        # Expand articles to team-level records
        team_news = []
        
        for idx, article in news_df.iterrows():
            for team in article['mentioned_teams']:
                team_record = {
                    'date': article['published_date'],
                    'team_abbr': team,
                    'article_id': article['article_id'],
                    'sentiment_score': article['combined_sentiment'],
                    'vader_score': article['vader_compound'],
                    'textblob_score': article['textblob_polarity'],
                    'headline': article['headline'],
                    'type': article['type']
                }
                team_news.append(team_record)
        
        if not team_news:
            print("No team mentions found in news!")
            return pd.DataFrame()
        
        team_df = pd.DataFrame(team_news)
        team_df['date'] = pd.to_datetime(team_df['date'])
        
        # Aggregate sentiment by team and date
        daily_sentiment = team_df.groupby(['team_abbr', team_df['date'].dt.date]).agg({
            'sentiment_score': ['mean', 'count', 'std'],
            'vader_score': 'mean',
            'textblob_score': 'mean'
        }).reset_index()
        
        # Flatten column names
        daily_sentiment.columns = [
            'team_abbr', 'date', 'avg_sentiment', 'news_count', 'sentiment_std',
            'avg_vader', 'avg_textblob'
        ]
        
        # Fill NaN std with 0
        daily_sentiment['sentiment_std'] = daily_sentiment['sentiment_std'].fillna(0)
        
        # Create rolling sentiment averages
        daily_sentiment = daily_sentiment.sort_values(['team_abbr', 'date'])
        
        for team in daily_sentiment['team_abbr'].unique():
            team_mask = daily_sentiment['team_abbr'] == team
            
            # 3-day rolling average
            daily_sentiment.loc[team_mask, 'sentiment_3day'] = (
                daily_sentiment.loc[team_mask, 'avg_sentiment']
                .rolling(window=3, min_periods=1).mean()
            )
            
            # 7-day rolling average
            daily_sentiment.loc[team_mask, 'sentiment_7day'] = (
                daily_sentiment.loc[team_mask, 'avg_sentiment']
                .rolling(window=7, min_periods=1).mean()
            )
            
            # News volume (3-day rolling sum)
            daily_sentiment.loc[team_mask, 'news_volume_3day'] = (
                daily_sentiment.loc[team_mask, 'news_count']
                .rolling(window=3, min_periods=1).sum()
            )
        
        daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date'])
        
        print(f"Created sentiment scores for {len(daily_sentiment)} team-date combinations")
        print(f"Date range: {daily_sentiment['date'].min()} to {daily_sentiment['date'].max()}")
        print(f"Teams covered: {daily_sentiment['team_abbr'].nunique()}")
        
        return daily_sentiment
    
    def save_sentiment_data(self, news_df, team_sentiment_df):
        """Save sentiment analysis results"""
        # Save individual articles with sentiment
        news_output = os.path.join(self.data_dir, "processed_news_sentiment.csv")
        news_df.to_csv(news_output, index=False)
        print(f"Saved news sentiment data to {news_output}")
        
        # Save team-level sentiment scores
        if not team_sentiment_df.empty:
            team_output = os.path.join(self.data_dir, "team_sentiment_scores.csv")
            team_sentiment_df.to_csv(team_output, index=False)
            print(f"Saved team sentiment scores to {team_output}")
        
        return news_output, team_output

def main():
    analyzer = NBASentimentAnalyzer()
    
    # Check if sentiment libraries are available
    if not VADER_AVAILABLE and not TEXTBLOB_AVAILABLE:
        print("No sentiment analysis libraries available!")
        print("Install with: pip install vaderSentiment textblob")
        return
    
    # Process all news
    news_df = analyzer.process_all_news()
    
    if news_df.empty:
        print("No news data to process!")
        return
    
    # Create team sentiment scores
    team_sentiment_df = analyzer.create_team_sentiment_scores(news_df)
    
    # Save results
    analyzer.save_sentiment_data(news_df, team_sentiment_df)
    
    # Print summary
    print("\n" + "="*60)
    print("SENTIMENT ANALYSIS SUMMARY")
    print("="*60)
    print(f"Total articles analyzed: {len(news_df)}")
    print(f"Articles with team mentions: {len(news_df[news_df['num_teams_mentioned'] > 0])}")
    print(f"Date range: {news_df['published_date'].min()} to {news_df['published_date'].max()}")
    
    if not team_sentiment_df.empty:
        print(f"Team sentiment records: {len(team_sentiment_df)}")
        print(f"Teams with sentiment data: {team_sentiment_df['team_abbr'].nunique()}")
        print(f"Average sentiment score: {team_sentiment_df['avg_sentiment'].mean():.3f}")
        print(f"Sentiment range: {team_sentiment_df['avg_sentiment'].min():.3f} to {team_sentiment_df['avg_sentiment'].max():.3f}")
    
    # Sample sentiment data
    print(f"\nSample sentiment scores:")
    if not team_sentiment_df.empty:
        sample = team_sentiment_df.head(5)[['date', 'team_abbr', 'avg_sentiment', 'news_count', 'sentiment_3day']]
        print(sample.to_string(index=False))

if __name__ == "__main__":
    main()