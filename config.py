
import os

# Telegram Bot Configuration
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "8018851274:AAHWZEhTpC5GMIdDmnqAUApEVS2bdxlIZUA")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "7104833764")

# Trading Configuration
CONFIDENCE_SCALE = 10
ECONOMIC_CALENDAR_API = "https://api.tradingeconomics.com/calendar"

# Signal Configuration
MIN_CONFIDENCE = 90  # Minimum confidence level for signals (%)
CONFIDENCE_LEVELS = {
    'VERY_HIGH': 90,
    'HIGH': 80,
    'MEDIUM': 70,
    'LOW': 60
}

# Market Configuration
MARKETS = [
    "XAU/USD",  # Gold
    "EUR/USD",  # Major forex pairs
    "USD/JPY",
    "GBP/USD",
    "USD/CHF",
    "AUD/USD",
    "NZD/USD",
    "EUR/CHF",
    "EUR/GBP",
    "BTC/USD",  # Crypto
    "ETH/USD"
]

# Technical Analysis Configuration
TECHNICAL_WEIGHT = 0.6  # Weight given to technical analysis
FUNDAMENTAL_WEIGHT = 0.4  # Weight given to fundamental analysis

# Network Configuration
REQUEST_TIMEOUT = 45  # Timeout for HTTP requests in seconds
MAX_RETRIES = 3  # Maximum number of retry attempts
RETRY_DELAY = 5  # Delay between retries in seconds

# Validation Configuration
MIN_RISK_REWARD = 1.5  # Reduced from 2.0 for more opportunities while maintaining safety
MIN_VALIDATION_SCORE = 50  # Reduced from 60 for more opportunities while maintaining safety
MAX_DISTANCE_FROM_SUPPORT = 2.5  # Percentage distance from support level
TIMEOUT_SECONDS = 20  # Increased timeout for data fetching
MAX_RETRIES = 3  # Number of retries for failed data fetches
RETRY_DELAY = 5  # Seconds between retries

# Markets Configuration
MARKETS = {
    "XAU/USD": "GC=F",  # Gold
    "EUR/USD": "EURUSD=X",  # Euro/USD
    "USD/JPY": "USDJPY=X",  # USD/Japanese Yen
    "GBP/USD": "GBPUSD=X",  # British Pound/USD
    "USD/CHF": "USDCHF=X",  # USD/Swiss Franc
    "AUD/USD": "AUDUSD=X",  # Australian Dollar/USD
    "NZD/USD": "NZDUSD=X",  # New Zealand Dollar/USD
    "EUR/CHF": "EURCHF=X",  # Euro/Swiss Franc
    "EUR/GBP": "EURGBP=X",  # Euro/British Pound
    "BTC/USD": "BTC-USD",  # Bitcoin
    "ETH/USD": "ETH-USD"  # Ethereum
}
