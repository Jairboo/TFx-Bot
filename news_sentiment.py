
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
        """Enhanced scraping from FXStreet currencies section with multiple selectors"""
        try:
            print("📰 Fetching FXStreet currency news...")
            
            # Add session with retries
            session = requests.Session()
            session.headers.update(self.headers)
            
            # Multiple attempts with different approaches
            for attempt in range(3):
                try:
                    response = session.get(self.news_sources['fxstreet'], timeout=20)
                    
                    if response.status_code != 200:
                        print(f"❌ FXStreet request failed with status {response.status_code}")
                        continue
                    
                    soup = BeautifulSoup(response.text, 'html.parser')
                    news_items = []
                    
                    # Multiple selector strategies for better coverage
                    selectors = [
                        # Standard article tags
                        {'tag': 'article', 'class': None},
                        {'tag': 'div', 'class': lambda x: x and any(term in x.lower() for term in ['news', 'article', 'story', 'item', 'post'])},
                        # Title-based selectors
                        {'tag': ['h1', 'h2', 'h3', 'h4'], 'class': None},
                        # Link-based selectors
                        {'tag': 'a', 'href': lambda x: x and ('/news/' in x or '/analysis/' in x or '/currencies/' in x)},
                        # Generic content selectors
                        {'tag': 'div', 'class': lambda x: x and any(term in x.lower() for term in ['content', 'headline', 'title'])},
                    ]
                    
                    for selector in selectors:
                        if selector['tag'] == 'a' and 'href' in selector:
                            articles = soup.find_all(selector['tag'], href=selector['href'])
                        elif selector['class']:
                            articles = soup.find_all(selector['tag'], class_=selector['class'])
                        else:
                            articles = soup.find_all(selector['tag'])
                        
                        for article in articles[:30]:  # Increased limit
                            try:
                                # Multiple strategies to extract title
                                title = ""
                                
                                # Strategy 1: Direct text content
                                if article.name in ['h1', 'h2', 'h3', 'h4']:
                                    title = article.get_text(strip=True)
                                
                                # Strategy 2: Find title in children
                                if not title:
                                    title_elem = article.find(['h1', 'h2', 'h3', 'h4', 'span', 'a', 'p'])
                                    if title_elem:
                                        title = title_elem.get_text(strip=True)
                                
                                # Strategy 3: Use article text if reasonable length
                                if not title:
                                    text = article.get_text(strip=True)
                                    if 20 <= len(text) <= 200:  # Reasonable title length
                                        title = text
                                
                                # Strategy 4: Extract from href title or alt text
                                if not title and article.name == 'a':
                                    title = article.get('title', '') or article.get('alt', '')
                                
                                # Clean and validate title
                                if title:
                                    title = re.sub(r'\s+', ' ', title)  # Clean whitespace
                                    title = title[:200]  # Limit length
                                    
                                    # Filter out unwanted content
                                    skip_patterns = [
                                        r'^(cookie|privacy|terms|subscribe|login|register)',
                                        r'^(home|menu|navigation|footer|header)',
                                        r'^(advertisement|sponsored|ad\s)',
                                        r'^(share|tweet|facebook|linkedin)',
                                        r'^(read more|continue reading|view all)',
                                        r'^\d+$',  # Just numbers
                                        r'^[^a-zA-Z]*$'  # No letters
                                    ]
                                    
                                    if any(re.search(pattern, title.lower()) for pattern in skip_patterns):
                                        continue
                                    
                                    if len(title) >= 15:  # Minimum meaningful length
                                        # Extract additional context
                                        context = ""
                                        summary_elem = article.find(['p', 'div'], class_=lambda x: x and any(term in x.lower() for term in ['summary', 'excerpt', 'description']))
                                        if summary_elem:
                                            context = summary_elem.get_text(strip=True)[:300]
                                        
                                        # Determine impact level
                                        impact_level = self.determine_news_impact(title + " " + context)
                                        
                                        # Extract timestamp if available
                                        time_elem = article.find(['time', 'span'], class_=lambda x: x and any(term in x.lower() for term in ['time', 'date', 'published']))
                                        news_time = datetime.now()
                                        if time_elem:
                                            time_text = time_elem.get_text(strip=True)
                                            # Basic time parsing
                                            if 'ago' in time_text.lower():
                                                news_time = datetime.now() - timedelta(hours=1)
                                        
                                        news_items.append({
                                            'source': 'FXStreet',
                                            'title': title,
                                            'context': context,
                                            'impact': impact_level,
                                            'time': news_time,
                                            'relevance': 'high' if impact_level == 'High' else 'medium'
                                        })
                                        
                                        # Avoid duplicates
                                        if len(news_items) >= 40:
                                            break
                                            
                            except Exception as e:
                                continue
                        
                        if news_items:
                            break  # Success with this selector
                    
                    # Remove duplicates based on title similarity
                    unique_items = []
                    seen_titles = set()
                    
                    for item in news_items:
                        title_key = re.sub(r'[^a-zA-Z0-9\s]', '', item['title'].lower())
                        title_key = re.sub(r'\s+', ' ', title_key).strip()
                        
                        if title_key not in seen_titles and len(title_key) > 10:
                            seen_titles.add(title_key)
                            unique_items.append(item)
                    
                    if unique_items:
                        print(f"✅ Found {len(unique_items)} unique FXStreet articles")
                        return unique_items
                    
                except Exception as e:
                    print(f"❌ Attempt {attempt + 1} failed: {e}")
                    if attempt == 2:
                        break
                    time.sleep(2)  # Wait between attempts
            
            print("❌ All FXStreet scraping attempts failed")
            return []
            
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
        """Advanced fundamental signal extraction with enhanced pattern recognition"""
        fundamental_signals = []
        
        # Enhanced pattern matching for economic data
        for news in news_items:
            title = news['title'].lower()
            context = news.get('context', '').lower()
            full_text = f"{title} {context}"
            
            signal_strength = 0
            signal_direction = None
            signal_reason = ""
            
            # Advanced actual vs forecast patterns
            actual_forecast_patterns = [
                # Numerical patterns
                r'actual[:\s]*([+-]?\d+\.?\d*)[%]?\s*[vs\.]*\s*forecast[:\s]*([+-]?\d+\.?\d*)[%]?',
                r'([+-]?\d+\.?\d*)[%]?\s*vs[\.]*\s*([+-]?\d+\.?\d*)[%]?\s*forecast',
                r'([+-]?\d+\.?\d*)[%]?\s*actual[,\s]*([+-]?\d+\.?\d*)[%]?\s*expected',
                r'came\s+in\s+at\s+([+-]?\d+\.?\d*)[%]?\s*[vs\.]*\s*([+-]?\d+\.?\d*)[%]?\s*expected',
                # Qualitative patterns
                r'beats?\s*forecast', r'misses?\s*forecast', r'exceeds?\s*expectations',
                r'below\s*expectations', r'above\s*expectations', r'better\s*than\s*expected',
                r'worse\s*than\s*expected', r'disappointing', r'strong\s*data', r'weak\s*data'
            ]
            
            # Check for numerical actual vs forecast
            for pattern in actual_forecast_patterns[:4]:
                match = re.search(pattern, full_text)
                if match and len(match.groups()) >= 2:
                    try:
                        actual = float(match.group(1))
                        forecast = float(match.group(2))
                        
                        difference = abs(actual - forecast)
                        percentage_diff = (difference / abs(forecast)) * 100 if forecast != 0 else 0
                        
                        if actual > forecast:
                            signal_direction = 'bullish'
                            signal_strength = min(10, percentage_diff)
                            signal_reason = f"Actual {actual} > Forecast {forecast}"
                        elif actual < forecast:
                            signal_direction = 'bearish'
                            signal_strength = min(10, percentage_diff)
                            signal_reason = f"Actual {actual} < Forecast {forecast}"
                    except:
                        continue
            
            # Enhanced qualitative analysis
            if not signal_direction:
                qualitative_signals = {
                    'bullish': [
                        'beats forecast', 'exceeds expectations', 'above expectations',
                        'better than expected', 'strong data', 'surge', 'rally',
                        'positive surprise', 'upward revision', 'robust growth',
                        'exceeded estimates', 'outperformed', 'stronger than anticipated'
                    ],
                    'bearish': [
                        'misses forecast', 'below expectations', 'worse than expected',
                        'disappointing', 'weak data', 'decline', 'fall',
                        'negative surprise', 'downward revision', 'slow growth',
                        'missed estimates', 'underperformed', 'weaker than anticipated'
                    ]
                }
                
                for direction, keywords in qualitative_signals.items():
                    for keyword in keywords:
                        if keyword in full_text:
                            signal_direction = direction
                            signal_strength = 7 if 'strong' in keyword or 'surge' in keyword else 5
                            signal_reason = f"Qualitative signal: {keyword}"
                            break
                    if signal_direction:
                        break
            
            # Enhanced economic indicator analysis
            if signal_direction:
                # Boost strength for high-impact indicators
                high_impact_indicators = [
                    'interest rate', 'gdp', 'inflation', 'cpi', 'pce', 'nonfarm payrolls',
                    'employment', 'unemployment', 'fomc', 'ecb decision', 'boe decision'
                ]
                
                for indicator in high_impact_indicators:
                    if indicator in full_text:
                        signal_strength *= 1.5
                        signal_reason += f" (High impact: {indicator})"
                        break
                
                # Currency-specific impact analysis
                base_currency = currency_pair[:3]
                quote_currency = currency_pair[4:7] if len(currency_pair) > 6 else 'USD'
                
                # Enhanced currency relevance detection
                base_keywords = self.currency_keywords.get(base_currency, [])
                quote_keywords = self.currency_keywords.get(quote_currency, [])
                
                affects_base = any(keyword in full_text for keyword in base_keywords)
                affects_quote = any(keyword in full_text for keyword in quote_keywords)
                
                # Country-specific indicators
                country_indicators = {
                    'USD': ['us ', 'united states', 'american', 'federal reserve', 'fed '],
                    'EUR': ['european', 'eurozone', 'euro area', 'ecb ', 'eu '],
                    'GBP': ['uk ', 'british', 'england', 'britain', 'boe '],
                    'JPY': ['japan', 'japanese', 'boj ', 'tokyo'],
                    'CHF': ['swiss', 'switzerland', 'snb '],
                    'AUD': ['australia', 'australian', 'rba '],
                    'NZD': ['new zealand', 'kiwi', 'rbnz ']
                }
                
                if not affects_base:
                    affects_base = any(indicator in full_text for indicator in country_indicators.get(base_currency, []))
                if not affects_quote:
                    affects_quote = any(indicator in full_text for indicator in country_indicators.get(quote_currency, []))
                
                # Apply signal logic
                if affects_base or affects_quote:
                    final_direction = signal_direction
                    
                    # If only quote currency is affected, consider reversing signal
                    if affects_quote and not affects_base:
                        # For quote currency, positive news is bearish for the pair
                        final_direction = 'bearish' if signal_direction == 'bullish' else 'bullish'
                        signal_reason += f" (Quote currency {quote_currency} impact)"
                    elif affects_base:
                        signal_reason += f" (Base currency {base_currency} impact)"
                    
                    # Impact multiplier
                    impact_multiplier = 3 if news['impact'] == 'High' else 2 if news['impact'] == 'Medium' else 1
                    final_strength = min(10, signal_strength * impact_multiplier)
                    
                    # Time sensitivity - recent news gets higher weight
                    time_diff = datetime.now() - news['time']
                    if time_diff.total_seconds() < 3600:  # Less than 1 hour old
                        final_strength *= 1.2
                    
                    # ADDITIONAL VALIDATION for fundamental signals
                    if final_strength >= 3:  # Only keep signals with meaningful strength
                        fundamental_signals.append({
                            'direction': final_direction,
                            'strength': final_strength,
                            'impact': news['impact'],
                            'reason': f"{news['source']}: {signal_reason}",
                            'affects_base': affects_base,
                            'affects_quote': affects_quote,
                            'title': news['title'],
                            'time': news['time'],
                            'validated': True  # Mark as validated
                        })
        
        # Sort by strength and return top signals
        fundamental_signals.sort(key=lambda x: x['strength'], reverse=True)
        return fundamental_signals[:10]  # Return top 10 signals

    def get_forex_factory_news(self):
        """Enhanced Forex Factory news scraping with fallback methods"""
        try:
            print("📰 Fetching Forex Factory news...")
            
            # Try multiple approaches for robust scraping
            for attempt in range(2):
                try:
                    response = requests.get(self.news_sources['forex_factory'], 
                                          headers=self.headers, timeout=15)
                    
                    if response.status_code != 200:
                        continue
                    
                    soup = BeautifulSoup(response.text, 'html.parser')
                    news_items = []
                    
                    # Primary approach - calendar events
                    events = soup.find_all('tr', class_=['calendar_row', 'calendar__row'])
                    
                    for event in events[:25]:
                        try:
                            # Extract event details with multiple selectors
                            event_elem = event.find('td', class_=['calendar__event', 'calendar_event'])
                            impact_elem = event.find('td', class_=['calendar__impact', 'calendar_impact'])
                            actual_elem = event.find('td', class_=['calendar__actual', 'calendar_actual'])
                            forecast_elem = event.find('td', class_=['calendar__forecast', 'calendar_forecast'])
                            
                            if event_elem:
                                event_text = event_elem.get_text(strip=True)
                                
                                # Extract impact level
                                impact_level = 'Medium'
                                if impact_elem:
                                    impact_span = impact_elem.find('span')
                                    if impact_span:
                                        impact_level = impact_span.get('title', impact_span.get('class', ['Medium'])[0])
                                        if 'red' in str(impact_span) or 'high' in impact_level.lower():
                                            impact_level = 'High'
                                        elif 'yellow' in str(impact_span) or 'medium' in impact_level.lower():
                                            impact_level = 'Medium'
                                        else:
                                            impact_level = 'Low'
                                
                                # Extract actual vs forecast if available
                                context = ""
                                if actual_elem and forecast_elem:
                                    actual_text = actual_elem.get_text(strip=True)
                                    forecast_text = forecast_elem.get_text(strip=True)
                                    if actual_text and forecast_text:
                                        context = f"Actual: {actual_text}, Forecast: {forecast_text}"
                                
                                if event_text and len(event_text) > 5:
                                    news_items.append({
                                        'source': 'Forex Factory',
                                        'title': event_text,
                                        'context': context,
                                        'impact': impact_level,
                                        'time': datetime.now(),
                                        'relevance': 'high' if impact_level == 'High' else 'medium'
                                    })
                        except:
                            continue
                    
                    # Fallback approach - news sections
                    if not news_items:
                        news_sections = soup.find_all(['div', 'article'], class_=lambda x: x and 'news' in x.lower())
                        for section in news_sections[:15]:
                            try:
                                title_elem = section.find(['h1', 'h2', 'h3', 'h4', 'a'])
                                if title_elem:
                                    title = title_elem.get_text(strip=True)
                                    if title and len(title) > 10:
                                        impact_level = self.determine_news_impact(title)
                                        news_items.append({
                                            'source': 'Forex Factory',
                                            'title': title,
                                            'context': '',
                                            'impact': impact_level,
                                            'time': datetime.now(),
                                            'relevance': 'medium'
                                        })
                            except:
                                continue
                    
                    if news_items:
                        print(f"✅ Found {len(news_items)} Forex Factory events")
                        return news_items
                    
                except Exception as e:
                    print(f"❌ Forex Factory attempt {attempt + 1} failed: {e}")
                    continue
            
            print("⚠️ Forex Factory scraping failed, using fallback")
            return []
            
        except Exception as e:
            print(f"❌ Error fetching Forex Factory news: {e}")
            return []

    def get_backup_news_sources(self, currency_pair):
        """Get news from backup sources when primary sources fail"""
        backup_news = []
        
        # Simple RSS/API-based sources as fallback
        backup_sources = [
            'https://feeds.finance.yahoo.com/rss/2.0/headline?s=EURUSD=X&region=US&lang=en-US',
            'https://feeds.finance.yahoo.com/rss/2.0/headline?s=GBPUSD=X&region=US&lang=en-US',
        ]
        
        for source in backup_sources:
            try:
                response = requests.get(source, headers=self.headers, timeout=10)
                if response.status_code == 200:
                    # Basic RSS parsing
                    soup = BeautifulSoup(response.text, 'xml')
                    items = soup.find_all('item')
                    
                    for item in items[:10]:
                        try:
                            title = item.find('title')
                            if title:
                                title_text = title.get_text(strip=True)
                                if len(title_text) > 15:
                                    backup_news.append({
                                        'source': 'Yahoo Finance',
                                        'title': title_text,
                                        'context': '',
                                        'impact': self.determine_news_impact(title_text),
                                        'time': datetime.now(),
                                        'relevance': 'medium'
                                    })
                        except:
                            continue
            except:
                continue
        
        return backup_news

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
        """Enhanced comprehensive news analysis with robust error handling and fallbacks"""
        try:
            print(f"📊 Enhanced news analysis for {currency_pair}...")
            
            all_news = []
            source_counts = {}
            
            # Primary news sources with individual error handling
            news_sources = [
                ('FXStreet', self.get_fxstreet_news),
                ('FXStreet Calendar', self.get_fxstreet_economic_calendar),
                ('Forex Factory', self.get_forex_factory_news),
            ]
            
            for source_name, source_func in news_sources:
                try:
                    print(f"🔍 Fetching from {source_name}...")
                    news_items = source_func()
                    if news_items:
                        all_news.extend(news_items)
                        source_counts[source_name] = len(news_items)
                        print(f"✅ {source_name}: {len(news_items)} articles")
                    else:
                        print(f"⚠️ {source_name}: No articles found")
                        source_counts[source_name] = 0
                except Exception as e:
                    print(f"❌ {source_name} failed: {e}")
                    source_counts[source_name] = 0
            
            # If primary sources fail, try backup sources
            if len(all_news) < 5:
                print("🔄 Using backup news sources...")
                try:
                    backup_news = self.get_backup_news_sources(currency_pair)
                    if backup_news:
                        all_news.extend(backup_news)
                        source_counts['Backup Sources'] = len(backup_news)
                        print(f"✅ Backup sources: {len(backup_news)} articles")
                except Exception as e:
                    print(f"❌ Backup sources failed: {e}")
            
            # Filter and deduplicate news
            if all_news:
                print(f"🔍 Processing {len(all_news)} total news items...")
                
                # Remove duplicates and filter by relevance
                unique_news = []
                seen_titles = set()
                
                for news in all_news:
                    title_key = re.sub(r'[^a-zA-Z0-9\s]', '', news['title'].lower())
                    title_key = re.sub(r'\s+', ' ', title_key).strip()
                    
                    if title_key not in seen_titles and len(title_key) > 10:
                        # Check currency relevance
                        base_currency = currency_pair[:3]
                        quote_currency = currency_pair[4:7] if len(currency_pair) > 6 else 'USD'
                        
                        title_lower = news['title'].lower()
                        context_lower = news.get('context', '').lower()
                        full_text = f"{title_lower} {context_lower}"
                        
                        # Enhanced relevance check
                        is_relevant = False
                        
                        # Currency-specific keywords
                        for currency in [base_currency, quote_currency]:
                            if currency in self.currency_keywords:
                                if any(keyword in full_text for keyword in self.currency_keywords[currency]):
                                    is_relevant = True
                                    break
                        
                        # General forex relevance
                        if not is_relevant:
                            forex_terms = ['forex', 'currency', 'exchange rate', 'central bank', 'monetary policy']
                            is_relevant = any(term in full_text for term in forex_terms)
                        
                        if is_relevant or news['impact'] == 'High':
                            seen_titles.add(title_key)
                            unique_news.append(news)
                
                print(f"✅ Filtered to {len(unique_news)} relevant news items")
                all_news = unique_news
            
            # Analyze sentiment with enhanced fundamental logic
            sentiment_analysis = self.analyze_enhanced_news_sentiment(all_news, currency_pair)
            
            # Add comprehensive metadata
            sentiment_analysis.update({
                'total_news_sources': len([count for count in source_counts.values() if count > 0]),
                'analysis_time': datetime.now(),
                'source_breakdown': source_counts,
                'total_articles_fetched': sum(source_counts.values()),
                'total_relevant_articles': len(all_news),
                'currency_pair': currency_pair,
                'data_quality': 'high' if len(all_news) >= 10 else 'medium' if len(all_news) >= 5 else 'low'
            })
            
            # Enhanced logging
            print(f"✅ Enhanced news analysis complete for {currency_pair}")
            print(f"   📊 Data Quality: {sentiment_analysis['data_quality'].upper()}")
            print(f"   🎯 Sentiment: {sentiment_analysis['sentiment'].upper()} ({sentiment_analysis['score']})")
            print(f"   ⚖️ Priority: {sentiment_analysis['priority'].upper()}")
            print(f"   🔥 High Impact News: {sentiment_analysis['high_impact_count']}")
            print(f"   📈 Fundamental Signals: {len(sentiment_analysis['fundamental_signals'])}")
            print(f"   📰 Total Relevant Articles: {sentiment_analysis['total_relevant_articles']}")
            
            return sentiment_analysis
            
        except Exception as e:
            print(f"❌ Critical error in enhanced news analysis for {currency_pair}: {e}")
            import traceback
            traceback.print_exc()
            
            # Return safe fallback
            return {
                'score': 0, 'relevance': 'none', 'news_count': 0, 'sentiment': 'neutral',
                'fundamental_signals': [], 'high_impact_count': 0, 'priority': 'technical',
                'total_news_sources': 0, 'analysis_time': datetime.now(),
                'source_breakdown': {}, 'data_quality': 'low', 'currency_pair': currency_pair
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
