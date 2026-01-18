import os

"""
Configuration file for Gold Trading Bot
Railway-ready version (reads keys from environment variables)
"""

# =============================================================================
# API CREDENTIALS (Pulled securely from Railway Variables)
# =============================================================================

METAL_PRICE_API_KEY = os.getenv("METAL_PRICE_API_KEY")
METAL_PRICE_API_BASE_URL = "https://api.metalpriceapi.com/v1"

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# =============================================================================
# CORE TRADING PARAMETERS
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
# ACTIVE STRATEGY
# =============================================================================

STRATEGY_EMA_CROSS_ENABLED = True
EMA_CROSS_RSI_BUY = 55
EMA_CROSS_RSI_SELL = 45
EMA_CROSS_ATR_SL = 2.0
EMA_CROSS_RR = 3.0

# =============================================================================
# FILTERS
# =============================================================================

MIN_SIGNAL_QUALITY = 0.6
COOLDOWN_MINUTES = 30

SESSION_FILTER_ENABLED = True
LONDON_SESSION_ENABLED = True
LONDON_SESSION_START = "07:00"
LONDON_SESSION_END = "16:00"

NY_SESSION_ENABLED = True
NY_SESSION_START = "13:00"
NY_SESSION_END = "22:00"
ATR_LOOKBACK_VOLATILITY = 50

# =============================================================================
# LOGGING
# =============================================================================

LOG_LEVEL = "INFO"
LOG_FILE = "logs/gold_bot.log"
STATE_FILE = "state.json"
