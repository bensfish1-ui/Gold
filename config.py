"""
Configuration file for Gold Trading Bot
Store your API keys and settings here
"""

# =============================================================================
# API CREDENTIALS (Replace with your own keys)
# =============================================================================

# Metal Price API Configuration
# Get free API key from: https://metalpriceapi.com/
METAL_PRICE_API_KEY = "YOUR_METAL_PRICE_API_KEY_HERE"
METAL_PRICE_API_BASE_URL = "https://api.metalpriceapi.com/v1"

# Telegram Bot Configuration
# Get bot token from @BotFather on Telegram
TELEGRAM_BOT_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN_HERE"
TELEGRAM_CHAT_ID = "YOUR_CHAT_ID_HERE"

# =============================================================================
# CORE TRADING PARAMETERS
# =============================================================================

SCAN_INTERVAL = 300  # 5 minutes in seconds
LOOKBACK_CANDLES = 100  # Number of historical candles to fetch
TIMEFRAME = "M5"  # 5-minute candles

# =============================================================================
# INDICATOR SETTINGS
# =============================================================================

EMA_FAST = 20
EMA_SLOW = 50
RSI_PERIOD = 14
ATR_PERIOD = 14
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2.0

# =============================================================================
# RISK MANAGEMENT
# =============================================================================

ATR_MULTIPLIER_SL = 1.5  # Stop loss = 1.5 * ATR
RISK_REWARD_RATIO = 2.0  # Take profit = 2 * stop loss distance
MIN_STOP_DISTANCE = 3.0  # Minimum stop loss distance in USD
MAX_STOP_DISTANCE = 25.0  # Maximum stop loss distance in USD
SPREAD_ESTIMATE = 0.50  # Estimated spread in USD per ounce
SLIPPAGE_ESTIMATE = 0.30  # Estimated slippage in USD per ounce

# =============================================================================
# STRATEGY A: TREND PULLBACK (Primary Strategy)
# =============================================================================

STRATEGY_TREND_PULLBACK_ENABLED = True

# Trend identification
# EMA20 above EMA50 = bullish, EMA20 below EMA50 = bearish

# Pullback parameters
PULLBACK_ATR_DISTANCE = 1.0  # Max distance from EMA20 in ATR multiples
PULLBACK_MIN_ATR_DISTANCE = 0.2  # Minimum pullback depth in ATR

# RSI zones for momentum confirmation
PULLBACK_BUY_RSI_MIN = 40  # RSI must be above this for buys
PULLBACK_BUY_RSI_MAX = 60  # RSI must be below this for buys
PULLBACK_SELL_RSI_MIN = 40  # RSI must be above this for sells
PULLBACK_SELL_RSI_MAX = 60  # RSI must be below this for sells

# Swing lookback for stop loss placement
SWING_LOOKBACK = 10  # Candles to look back for swing high/low

# Entry type: "close" = market at close, "limit" = limit at EMA20
PULLBACK_ENTRY_TYPE = "close"

# Partial take profit at 1R (0 to disable)
PARTIAL_TP_AT_1R = 0.5  # Take 50% off at 1R (set to 0 to disable)

# =============================================================================
# STRATEGY B: BREAKOUT + RETEST (Secondary Strategy)
# =============================================================================

STRATEGY_BREAKOUT_ENABLED = True

# Range definition
BREAKOUT_RANGE_CANDLES = 20  # Candles for range high/low definition

# Breakout confirmation
BREAKOUT_ATR_THRESHOLD = 0.5  # Close must exceed range by k*ATR

# Retest parameters
RETEST_CANDLES = 3  # Max candles to wait for retest
RETEST_ATR_TOLERANCE = 0.3  # How close price must return to breakout level

# Stop loss
BREAKOUT_SL_ATR_MULTIPLIER = 1.2  # SL = 1.2 * ATR or inside range

# Take profit: 2R or measured move (range height)
BREAKOUT_USE_MEASURED_MOVE = True

# =============================================================================
# STRATEGY C: MEAN REVERSION (Disabled by default)
# =============================================================================

STRATEGY_MEAN_REVERSION_ENABLED = False  # Enable via this flag

# Range detection (EMA20 flat + low ATR)
MEAN_REVERSION_EMA_FLAT_THRESHOLD = 0.001  # Max EMA slope percentage
MEAN_REVERSION_ATR_PERCENTILE_MAX = 40  # ATR must be below this percentile

# Bollinger Band settings (uses BOLLINGER_PERIOD and BOLLINGER_STD above)
# RSI extremes
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70

# Entry: fade outer band only if price snaps back inside on close
# Stop loss
MEAN_REVERSION_SL_ATR = 1.0  # SL outside band by this ATR multiple

# Take profit: midline or 1.5R max
MEAN_REVERSION_MAX_RR = 1.5

# =============================================================================
# QUALITY FILTERS & GUARDRAILS
# =============================================================================

# Signal quality threshold (0-1 scale)
MIN_SIGNAL_QUALITY = 0.6  # Only send alerts for quality >= this

# Cooldown: prevent spam signals
COOLDOWN_MINUTES = 30  # Minutes between signals in same direction

# ATR volatility sanity check (chop/chaos filter)
ATR_VOLATILITY_MIN_PERCENTILE = 25  # Skip if ATR below this (too quiet)
ATR_VOLATILITY_MAX_PERCENTILE = 95  # Skip if ATR above this (too chaotic)
ATR_LOOKBACK_VOLATILITY = 50  # Candles for ATR percentile calculation

# Minimum candle body size (avoid doji signals)
MIN_CANDLE_BODY_ATR = 0.1  # Body must be at least 10% of ATR

# =============================================================================
# SESSION FILTERS (times in UTC)
# =============================================================================

# Enable/disable session filtering
SESSION_FILTER_ENABLED = True

# London session
LONDON_SESSION_ENABLED = True
LONDON_SESSION_START = "07:00"
LONDON_SESSION_END = "16:00"

# New York session
NY_SESSION_ENABLED = True
NY_SESSION_START = "13:00"
NY_SESSION_END = "22:00"

# =============================================================================
# NEWS BLACKOUT WINDOWS (UTC time ranges)
# =============================================================================

# Format: list of tuples [(start_time, end_time, description), ...]
# No signals will be sent during these windows
NEWS_BLACKOUT_WINDOWS = [
    # NFP (First Friday of month)
    # ("12:30", "14:00", "NFP Release"),

    # FOMC meetings (check Fed calendar)
    # ("18:00", "20:00", "FOMC Statement"),

    # CPI releases
    # ("12:30", "14:00", "CPI Release"),

    # Add your scheduled high-impact news events here
    # Example:
    # ("13:30", "14:30", "US Economic Data Release"),
]

# =============================================================================
# LOGGING & FILES
# =============================================================================

LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
LOG_FILE = "logs/gold_bot.log"
TRADE_LOG_FILE = "logs/trade_signals.csv"
STATE_FILE = "state.json"

# =============================================================================
# BACKTEST SETTINGS
# =============================================================================

# Backtest fill assumptions
BACKTEST_FILL_TYPE = "next_open"  # "next_open" or "signal_close"
BACKTEST_SPREAD = 0.50  # Spread in USD
BACKTEST_SLIPPAGE = 0.30  # Slippage in USD

# Default data file for backtesting
BACKTEST_DATA_FILE = "sample_data/xauusd_m5.csv"
