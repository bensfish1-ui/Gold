import os

"""
Configuration file for Gold Trading Bot
Live Railway version — complete & error-free
"""

# =============================================================================
# API KEYS FROM RAILWAY ENVIRONMENT
# =============================================================================
METAL_PRICE_API_KEY = os.getenv("METAL_PRICE_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

if not METAL_PRICE_API_KEY or not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
    raise ValueError("API keys not found in Railway environment variables.")

METAL_PRICE_API_BASE_URL = "https://api.metalpriceapi.com/v1"

# =============================================================================
# CORE SETTINGS
# =============================================================================
SCAN_INTERVAL = 300
LOOKBACK_CANDLES = 100
TIMEFRAME = "M5"

# =============================================================================
# INDICATORS
# =============================================================================
EMA_FAST = 20
EMA_SLOW = 50
RSI_PERIOD = 14
ATR_PERIOD = 14
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2.0

# =============================================================================
# RISK
# =============================================================================
ATR_MULTIPLIER_SL = 1.5
RISK_REWARD_RATIO = 2.0
MIN_STOP_DISTANCE = 3.0
MAX_STOP_DISTANCE = 25.0
SPREAD_ESTIMATE = 0.50
SLIPPAGE_ESTIMATE = 0.30

# =============================================================================
# STRATEGY ENABLED (EMA CROSS ONLY)
# =============================================================================
STRATEGY_EMA_CROSS_ENABLED = True
EMA_CROSS_RSI_BUY = 55
EMA_CROSS_RSI_SELL = 45
EMA_CROSS_ATR_SL = 2.0
EMA_CROSS_RR = 3.0

# =============================================================================
# DISABLED STRATEGIES (kept so bot doesn't error)
# =============================================================================
STRATEGY_SR_BOUNCE_ENABLED = False
STRATEGY_VOL_BREAKOUT_ENABLED = False
STRATEGY_TREND_PULLBACK_ENABLED = False
STRATEGY_BREAKOUT_ENABLED = False
STRATEGY_MEAN_REVERSION_ENABLED = False

# =============================================================================
# VOLATILITY FILTERS (THIS FIXES YOUR ERRORS)
# =============================================================================
ATR_VOLATILITY_MIN_PERCENTILE = 25
ATR_VOLATILITY_MAX_PERCENTILE = 95
ATR_LOOKBACK_VOLATILITY = 50

MIN_CANDLE_BODY_ATR = 0.1
MIN_SIGNAL_QUALITY = 0.6
COOLDOWN_MINUTES = 30

# =============================================================================
# SESSIONS (UTC)
# =============================================================================
SESSION_FILTER_ENABLED = True

LONDON_SESSION_ENABLED = True
LONDON_SESSION_START = "07:00"
LONDON_SESSION_END = "16:00"

NY_SESSION_ENABLED = True
NY_SESSION_START = "13:00"
NY_SESSION_END = "22:00"

# =============================================================================
# NEWS FILTER (needed so bot doesn’t crash)
# =============================================================================
NEWS_BLACKOUT_WINDOWS = []

# =============================================================================
# LOGGING
# =============================================================================
LOG_LEVEL = "INFO"
LOG_FILE = "logs/gold_bot.log"
TRADE_LOG_FILE = "logs/trade_signals.csv"
STATE_FILE = "state.json"

# =============================================================================
# BACKTEST SETTINGS
# =============================================================================
BACKTEST_FILL_TYPE = "next_open"
BACKTEST_SPREAD = 0.50
BACKTEST_SLIPPAGE = 0.30
BACKTEST_DATA_FILE = "sample_data/xauusd_m5.csv"

