
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
import time
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')

class NewsAnalyzer:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Enhanced news sources with FXStreet
        self.news_sources = {
            'fxstreet': 'https://www.fxstreet.com/currencies',
            'fxstreet_calendar': 'https://www.fxstreet.com/economic-calendar',
            'forex_factory': 'https://www.forexfactory.com/calendar',
            'investing': 'https://www.investing.com/news/forex-news',
            'marketwatch': 'https://www.marketwatch.com/markets/currencies',
            'reuters': 'https://www.reuters.com/markets/currencies',
            'bloomberg': 'https://www.bloomberg.com/markets/currencies'
        }
        
        # Currency impact keywords
        self.currency_keywords = {
            'USD': ['dollar', 'fed', 'federal reserve', 'interest rate', 'inflation', 'employment', 'gdp', 'nonfarm payrolls', 'cpi', 'pce', 'fomc'],
            'EUR': ['euro', 'ecb', 'european central bank', 'eurozone', 'inflation', 'employment', 'draghi', 'lagarde'],
            'GBP': ['pound', 'sterling', 'boe', 'bank of england', 'brexit', 'uk', 'bailey'],
            'JPY': ['yen', 'boj', 'bank of japan', 'japan', 'inflation', 'kuroda', 'ueda'],
            'CHF': ['franc', 'snb', 'swiss national bank', 'switzerland', 'jordan'],
            'AUD': ['australian dollar', 'rba', 'reserve bank australia', 'australia', 'lowe'],
            'NZD': ['new zealand dollar', 'rbnz', 'reserve bank new zealand', 'orr'],
            'XAU': ['gold', 'precious metals', 'safe haven', 'inflation hedge', 'treasury yields']
        }
        
        # Economic indicators that affect currencies
        self.economic_indicators = {
            'high_impact': [
                'interest rate', 'gdp', 'inflation', 'cpi', 'pce', 'nonfarm payrolls', 'employment', 
                'retail sales', 'manufacturing pmi', 'services pmi', 'trade balance',
                'fomc meeting', 'ecb meeting', 'boe meeting', 'boj meeting'
            ],
            'medium_impact': [
                'housing starts', 'building permits', 'industrial production', 'capacity utilization',
                'consumer confidence', 'business confidence', 'jobless claims', 'existing home sales'
            ],
            'low_impact': [
                'factory orders', 'wholesale inventories', 'crude oil inventories', 'api crude oil'
            ]
        }
        
        # Sentiment keywords
        self.bullish_keywords = [
            'surge', 'rally', 'strengthen', 'boost', 'rise', 'gain', 'bullish', 'positive',
            'optimistic', 'confidence', 'growth', 'recovery', 'improvement', 'higher',
            'support', 'upgrade', 'beat expectations', 'strong data', 'resilient', 'outperform'
        ]
        
        self.bearish_keywords = [
            'fall', 'decline', 'weaken', 'drop', 'bearish', 'negative', 'pessimistic',
            'concern', 'worry', 'recession', 'slowdown', 'deterioration', 'lower',
            'pressure', 'downgrade', 'miss expectations', 'weak data', 'volatile', 'underperform'
        ]

    def get_fxstreet_news(self):
        """Scrape news from FXStreet currencies section"""
        try:
            print("📰 Fetching FXStreet currency news...")
            response = requests.get(self.news_sources['fxstreet'], 
                                  headers=self.headers, timeout=15)
            
            if response.status_code != 200:
                print(f"❌ FXStreet request failed with status {response.status_code}")
                return []
            
            soup = BeautifulSoup(response.text, 'html.parser')
            news_items = []
            
            # Find news articles - FXStreet structure
            articles = soup.find_all(['article', 'div'], class_=lambda x: x and any(term in x.lower() for term in ['news', 'article', 'story', 'item']))
            
            if not articles:
                # Fallback: try to find any links that look like news
                articles = soup.find_all('a', href=lambda x: x and ('/news/' in x or '/analysis/' in x))
            
            for article in articles[:25]:  # Limit to 25 articles
                try:
                    title_elem = article.find(['h1', 'h2', 'h3', 'h4', 'span', 'a'])
                    if not title_elem:
                        title_elem = article
                    
                    title = title_elem.get_text(strip=True) if title_elem else ""
                    
                    if title and len(title) > 10:  # Filter out very short titles
                        # Determine impact level based on title content
                        impact_level = self.determine_news_impact(title)
                        
                        news_items.append({
                            'source': 'FXStreet',
                            'title': title,
                            'impact': impact_level,
                            'time': datetime.now(),
                            'relevance': 'high' if impact_level == 'High' else 'medium'
                        })
                except Exception as e:
                    continue
            
            print(f"✅ Found {len(news_items)} FXStreet articles")
            return news_items
            
        except Exception as e:
            print(f"❌ Error fetching FXStreet news: {e}")
            return []

    def get_fxstreet_economic_calendar(self):
        """Get economic calendar data from FXStreet"""
        try:
            print("📅 Fetching FXStreet economic calendar...")
            response = requests.get(self.news_sources['fxstreet_calendar'], 
                                  headers=self.headers, timeout=15)
            
            if response.status_code != 200:
                return []
            
            soup = BeautifulSoup(response.text, 'html.parser')
            calendar_items = []
            
            # Look for economic events
            events = soup.find_all(['tr', 'div'], class_=lambda x: x and any(term in x.lower() for term in ['event', 'calendar', 'economic']))
            
            for event in events[:20]:
                try:
                    # Extract event details
                    event_text = event.get_text(strip=True)
                    
                    if event_text and len(event_text) > 5:
                        impact_level = self.determine_news_impact(event_text)
                        
                        calendar_items.append({
                            'source': 'FXStreet Calendar',
                            'title': event_text,
                            'impact': impact_level,
                            'time': datetime.now(),
                            'relevance': 'high' if impact_level == 'High' else 'medium'
                        })
                except:
                    continue
            
            print(f"✅ Found {len(calendar_items)} economic events")
            return calendar_items
            
        except Exception as e:
            print(f"❌ Error fetching FXStreet calendar: {e}")
            return []

    def determine_news_impact(self, text):
        """Determine impact level of news based on content"""
        text_lower = text.lower()
        
        # Check for high impact indicators
        for indicator in self.economic_indicators['high_impact']:
            if indicator in text_lower:
                return 'High'
        
        # Check for medium impact indicators
        for indicator in self.economic_indicators['medium_impact']:
            if indicator in text_lower:
                return 'Medium'
        
        # Check for currency-specific high impact terms
        high_impact_terms = ['central bank', 'interest rate', 'monetary policy', 'inflation', 'gdp', 'employment']
        for term in high_impact_terms:
            if term in text_lower:
                return 'High'
        
        return 'Medium'  # Default to medium impact

    def extract_fundamental_signals(self, news_items, currency_pair):
        """Extract fundamental buy/sell signals based on actual vs forecast logic"""
        fundamental_signals = []
        
        for news in news_items:
            title = news['title'].lower()
            
            # Look for actual vs forecast patterns
            actual_forecast_patterns = [
                r'actual[:\s]*([+-]?\d+\.?\d*)[%]?\s*[vs\.]*\s*forecast[:\s]*([+-]?\d+\.?\d*)[%]?',
                r'([+-]?\d+\.?\d*)[%]?\s*vs[\.]*\s*([+-]?\d+\.?\d*)[%]?\s*forecast',
                r'beats?\s*forecast', r'misses?\s*forecast', r'exceeds?\s*expectations',
                r'below\s*expectations', r'above\s*expectations'
            ]
            
            signal_strength = 0
            signal_direction = None
            
            # Check for actual vs forecast numerical patterns
            for pattern in actual_forecast_patterns[:2]:
                match = re.search(pattern, title)
                if match:
                    try:
                        if len(match.groups()) >= 2:
                            actual = float(match.group(1))
                            forecast = float(match.group(2))
                            
                            if actual > forecast:
                                signal_direction = 'bullish'
                                signal_strength = min(10, abs(actual - forecast))
                            elif actual < forecast:
                                signal_direction = 'bearish'
                                signal_strength = min(10, abs(actual - forecast))
                    except:
                        continue
            
            # Check for qualitative beats/misses
            if 'beats forecast' in title or 'exceeds expectations' in title or 'above expectations' in title:
                signal_direction = 'bullish'
                signal_strength = 7
            elif 'misses forecast' in title or 'below expectations' in title:
                signal_direction = 'bearish'
                signal_strength = 7
            
            # Currency-specific impact
            base_currency = currency_pair[:3]
            quote_currency = currency_pair[4:7] if len(currency_pair) > 6 else 'USD'
            
            # Check if news affects the currencies in the pair
            affects_base = any(keyword in title for keyword in self.currency_keywords.get(base_currency, []))
            affects_quote = any(keyword in title for keyword in self.currency_keywords.get(quote_currency, []))
            
            if signal_direction and (affects_base or affects_quote):
                # Adjust signal based on which currency is affected
                final_direction = signal_direction
                if affects_quote and not affects_base:
                    # If quote currency is affected, reverse the signal
                    final_direction = 'bearish' if signal_direction == 'bullish' else 'bullish'
                
                impact_multiplier = 3 if news['impact'] == 'High' else 2 if news['impact'] == 'Medium' else 1
                final_strength = signal_strength * impact_multiplier
                
                fundamental_signals.append({
                    'direction': final_direction,
                    'strength': final_strength,
                    'impact': news['impact'],
                    'reason': f"{news['source']}: {news['title'][:100]}...",
                    'affects_base': affects_base,
                    'affects_quote': affects_quote
                })
        
        return fundamental_signals

    def get_forex_factory_news(self):
        """Scrape news from Forex Factory"""
        try:
            print("📰 Fetching Forex Factory news...")
            response = requests.get(self.news_sources['forex_factory'], 
                                  headers=self.headers, timeout=10)
            
            if response.status_code != 200:
                return []
            
            soup = BeautifulSoup(response.text, 'html.parser')
            news_items = []
            
            # Find calendar events
            events = soup.find_all('tr', class_='calendar_row')
            
            for event in events[:20]:  # Limit to recent events
                try:
                    time_elem = event.find('td', class_='calendar__time')
                    event_elem = event.find('td', class_='calendar__event')
                    impact_elem = event.find('td', class_='calendar__impact')
                    
                    if event_elem and impact_elem:
                        event_text = event_elem.get_text(strip=True)
                        impact = impact_elem.find('span')
                        impact_level = impact.get('title', '') if impact else ''
                        
                        if 'High' in impact_level or 'Medium' in impact_level:
                            news_items.append({
                                'source': 'Forex Factory',
                                'title': event_text,
                                'impact': impact_level,
                                'time': datetime.now(),
                                'relevance': 'high' if 'High' in impact_level else 'medium'
                            })
                except:
                    continue
            
            print(f"✅ Found {len(news_items)} Forex Factory events")
            return news_items
            
        except Exception as e:
            print(f"❌ Error fetching Forex Factory news: {e}")
            return []

    def analyze_enhanced_news_sentiment(self, news_items, currency_pair):
        """Enhanced sentiment analysis with fundamental priority logic"""
        if not news_items:
            return {
                'score': 0, 'relevance': 'none', 'news_count': 0, 'sentiment': 'neutral',
                'fundamental_signals': [], 'high_impact_count': 0, 'priority': 'technical'
            }
        
        base_currency = currency_pair[:3]
        quote_currency = currency_pair[4:7] if len(currency_pair) > 6 else 'USD'
        
        # Extract fundamental signals using actual vs forecast logic
        fundamental_signals = self.extract_fundamental_signals(news_items, currency_pair)
        
        # Count high impact news
        high_impact_count = len([news for news in news_items if news['impact'] == 'High'])
        medium_impact_count = len([news for news in news_items if news['impact'] == 'Medium'])
        
        # Determine priority: fundamentals vs technicals
        priority = 'fundamental' if high_impact_count >= 2 or (high_impact_count >= 1 and medium_impact_count >= 2) else 'technical'
        
        relevant_news = []
        total_sentiment = 0
        
        # Enhanced sentiment analysis
        for news in news_items:
            title = news['title'].lower()
            relevance_score = 0
            
            # Check relevance to currencies
            base_keywords = self.currency_keywords.get(base_currency, [])
            quote_keywords = self.currency_keywords.get(quote_currency, [])
            
            for keyword in base_keywords:
                if keyword in title:
                    relevance_score += 3  # Increased relevance for base currency
            
            for keyword in quote_keywords:
                if keyword in title:
                    relevance_score += 2  # Relevance for quote currency
            
            # General forex relevance
            forex_terms = ['forex', 'currency', 'exchange rate', 'central bank', 'monetary policy', 'interest rate']
            for term in forex_terms:
                if term in title:
                    relevance_score += 1
            
            if relevance_score > 0:
                # Analyze sentiment using TextBlob
                blob = TextBlob(news['title'])
                polarity = blob.sentiment.polarity
                
                # Enhance with keyword analysis
                sentiment_score = polarity
                
                for keyword in self.bullish_keywords:
                    if keyword in title:
                        sentiment_score += 0.2
                
                for keyword in self.bearish_keywords:
                    if keyword in title:
                        sentiment_score -= 0.2
                
                # Apply fundamental signal if available
                fundamental_boost = 0
                for signal in fundamental_signals:
                    if signal['reason'].lower() in title:
                        if signal['direction'] == 'bullish':
                            fundamental_boost += signal['strength'] * 0.1
                        else:
                            fundamental_boost -= signal['strength'] * 0.1
                
                sentiment_score += fundamental_boost
                
                # Weight by impact and relevance
                impact_weight = 5 if news['impact'] == 'High' else 3 if news['impact'] == 'Medium' else 1
                weighted_sentiment = sentiment_score * relevance_score * impact_weight
                
                relevant_news.append({
                    'title': news['title'],
                    'sentiment': sentiment_score,
                    'relevance': relevance_score,
                    'weighted_sentiment': weighted_sentiment,
                    'impact': news['impact']
                })
                
                total_sentiment += weighted_sentiment
        
        if not relevant_news:
            return {
                'score': 0, 'relevance': 'none', 'news_count': 0, 'sentiment': 'neutral',
                'fundamental_signals': fundamental_signals, 'high_impact_count': high_impact_count,
                'priority': priority
            }
        
        # Calculate final sentiment score
        avg_sentiment = total_sentiment / len(relevant_news) if relevant_news else 0
        
        # Apply fundamental signals boost
        fundamental_boost = 0
        for signal in fundamental_signals:
            if signal['direction'] == 'bullish':
                fundamental_boost += signal['strength']
            else:
                fundamental_boost -= signal['strength']
        
        # Normalize to -10 to +10 scale with fundamental boost
        normalized_score = max(-10, min(10, (avg_sentiment * 5) + (fundamental_boost * 0.5)))
        
        # Determine sentiment category
        if normalized_score > 3:
            sentiment_category = 'bullish'
        elif normalized_score < -3:
            sentiment_category = 'bearish'
        else:
            sentiment_category = 'neutral'
        
        relevance_level = 'high' if high_impact_count >= 2 else 'medium' if len(relevant_news) > 3 else 'low'
        
        return {
            'score': round(normalized_score, 2),
            'relevance': relevance_level,
            'news_count': len(relevant_news),
            'sentiment': sentiment_category,
            'news_items': relevant_news[:5],  # Top 5 relevant news
            'fundamental_signals': fundamental_signals,
            'high_impact_count': high_impact_count,
            'medium_impact_count': medium_impact_count,
            'priority': priority,
            'fundamental_boost': round(fundamental_boost, 2)
        }

    def get_comprehensive_news_analysis(self, currency_pair):
        """Get comprehensive news analysis for currency pair with FXStreet integration"""
        try:
            print(f"📊 Enhanced news analysis for {currency_pair}...")
            
            all_news = []
            
            # Collect news from multiple sources including FXStreet
            fxstreet_news = self.get_fxstreet_news()
            fxstreet_calendar = self.get_fxstreet_economic_calendar()
            forex_factory_news = self.get_forex_factory_news()
            
            all_news.extend(fxstreet_news)
            all_news.extend(fxstreet_calendar)
            all_news.extend(forex_factory_news)
            
            # Analyze sentiment with enhanced fundamental logic
            sentiment_analysis = self.analyze_enhanced_news_sentiment(all_news, currency_pair)
            
            # Add additional context
            sentiment_analysis['total_news_sources'] = len([n for n in all_news if n])
            sentiment_analysis['analysis_time'] = datetime.now()
            sentiment_analysis['fxstreet_count'] = len(fxstreet_news) + len(fxstreet_calendar)
            
            print(f"✅ Enhanced news analysis complete for {currency_pair}")
            print(f"   Sentiment: {sentiment_analysis['sentiment']} ({sentiment_analysis['score']})")
            print(f"   Priority: {sentiment_analysis['priority'].upper()}")
            print(f"   High Impact News: {sentiment_analysis['high_impact_count']}")
            print(f"   FXStreet Articles: {sentiment_analysis['fxstreet_count']}")
            print(f"   Fundamental Signals: {len(sentiment_analysis['fundamental_signals'])}")
            
            return sentiment_analysis
            
        except Exception as e:
            print(f"❌ Error in enhanced news analysis for {currency_pair}: {e}")
            return {
                'score': 0, 'relevance': 'none', 'news_count': 0, 'sentiment': 'neutral',
                'fundamental_signals': [], 'high_impact_count': 0, 'priority': 'technical'
            }

    def get_gold_specific_news(self):
        """Get gold-specific news and analysis with FXStreet integration"""
        try:
            print("🥇 Enhanced gold news analysis...")
            
            gold_news = []
            
            # Get FXStreet news first
            fxstreet_news = self.get_fxstreet_news()
            forex_factory_news = self.get_forex_factory_news()
            
            all_news = fxstreet_news + forex_factory_news
            
            # Gold-specific keywords
            gold_keywords = ['gold', 'precious metals', 'safe haven', 'inflation', 'treasury yields', 'fed policy', 'xau']
            
            for news in all_news:
                title_lower = news['title'].lower()
                for keyword in gold_keywords:
                    if keyword in title_lower:
                        gold_news.append(news)
                        break
            
            # Analyze sentiment specifically for gold with fundamental signals
            sentiment_analysis = self.analyze_enhanced_news_sentiment(gold_news, 'XAU/USD')
            
            # Gold-specific sentiment modifiers
            if sentiment_analysis['score'] != 0:
                # Inflation fears = bullish for gold
                inflation_keywords = ['inflation', 'cpi', 'pce', 'price pressure']
                risk_off_keywords = ['risk off', 'safe haven', 'uncertainty', 'crisis']
                
                for news in gold_news:
                    title_lower = news['title'].lower()
                    
                    for keyword in inflation_keywords:
                        if keyword in title_lower:
                            sentiment_analysis['score'] += 1.5
                    
                    for keyword in risk_off_keywords:
                        if keyword in title_lower:
                            sentiment_analysis['score'] += 2
            
            # Normalize again
            sentiment_analysis['score'] = max(-10, min(10, sentiment_analysis['score']))
            
            print(f"✅ Enhanced gold news analysis complete")
            print(f"   Gold sentiment: {sentiment_analysis['sentiment']} ({sentiment_analysis['score']})")
            print(f"   Priority: {sentiment_analysis['priority'].upper()}")
            
            return sentiment_analysis
            
        except Exception as e:
            print(f"❌ Error in enhanced gold news analysis: {e}")
            return {
                'score': 0, 'relevance': 'none', 'news_count': 0, 'sentiment': 'neutral',
                'fundamental_signals': [], 'high_impact_count': 0, 'priority': 'technical'
            }
