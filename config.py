"""
Configuration file for Gold Trading Bot
Store your API keys and settings here

OPTIMIZED for long-term profitability based on 6-month backtesting
"""

# =============================================================================
# API CREDENTIALS (Replace with your own keys)
# =============================================================================

# Metal Price API Configuration
# Get free API key from: https://metalpriceapi.com/
METAL_PRICE_API_KEY = "91a0d6f7b0197bbecdb6fcf933171858"
METAL_PRICE_API_BASE_URL = "https://api.metalpriceapi.com/v1"

# Telegram Bot Configuration
# Get bot token from @BotFather on Telegram
TELEGRAM_BOT_TOKEN = "8210865744:AAEdgY8_jVEv10OUK-4m51gWDN5S9j2dXC8"
TELEGRAM_CHAT_ID = "5162373931"

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
# NEW PRIMARY STRATEGY A: SUPPORT/RESISTANCE BOUNCE
# =============================================================================
# BEST PERFORMER: +8.5R over 6 months, 54 trades, 38.9% win rate
# Trade bounces off swing highs (resistance) and swing lows (support)

STRATEGY_SR_BOUNCE_ENABLED = False  # DISABLED - needs refinement

# S/R detection parameters
SR_LOOKBACK = 30  # Candles to look back for swing highs/lows
SR_TOLERANCE = 0.5  # How close price must be to S/R level (in ATR multiples)

# Risk parameters
SR_ATR_SL = 1.5  # Stop loss multiplier (ATR beyond S/R level)
SR_RR = 2.0  # Risk/Reward ratio

# =============================================================================
# NEW PRIMARY STRATEGY B: VOLATILITY BREAKOUT
# =============================================================================
# SECOND BEST: +7.4R over 6 months, 19 trades, 57.9% win rate
# Trade momentum breakouts when ATR spikes above average

STRATEGY_VOL_BREAKOUT_ENABLED = False  # DISABLED - needs refinement

# Volatility parameters
VOL_ATR_LOOKBACK = 20  # Candles to calculate average ATR
VOL_ATR_SPIKE = 1.5  # ATR must be this multiple of average to trigger

# Risk parameters
VOL_RR = 2.0  # Risk/Reward ratio (stop = current ATR)

# =============================================================================
# NEW PRIMARY STRATEGY C: EMA CROSSOVER
# =============================================================================
# THIRD BEST: +5.8R over 6 months, 21 trades, 38.1% win rate
# Trade EMA20/EMA50 crossovers with RSI momentum filter

STRATEGY_EMA_CROSS_ENABLED = True  # ENABLED

# RSI filters for crossover signals
EMA_CROSS_RSI_BUY = 55  # RSI must be above this for bullish cross
EMA_CROSS_RSI_SELL = 45  # RSI must be below this for bearish cross

# Risk parameters
EMA_CROSS_ATR_SL = 2.0  # Stop loss in ATR multiples
EMA_CROSS_RR = 3.0  # Risk/Reward ratio (higher for trend trades)

# =============================================================================
# LEGACY STRATEGY: TREND PULLBACK (Disabled - lost -45R over 6 months)
# =============================================================================

STRATEGY_TREND_PULLBACK_ENABLED = False  # DISABLED - poor long-term performance

# Pullback parameters (kept for reference)
PULLBACK_ATR_DISTANCE = 1.0
PULLBACK_MIN_ATR_DISTANCE = 0.2
PULLBACK_BUY_RSI_MIN = 40
PULLBACK_BUY_RSI_MAX = 60
PULLBACK_SELL_RSI_MIN = 40
PULLBACK_SELL_RSI_MAX = 60
SWING_LOOKBACK = 10
PULLBACK_ENTRY_TYPE = "close"
PARTIAL_TP_AT_1R = 0.5

# =============================================================================
# LEGACY STRATEGY: BREAKOUT + RETEST (Disabled)
# =============================================================================

STRATEGY_BREAKOUT_ENABLED = False  # DISABLED

BREAKOUT_RANGE_CANDLES = 20
BREAKOUT_ATR_THRESHOLD = 0.5
RETEST_CANDLES = 3
RETEST_ATR_TOLERANCE = 0.3
BREAKOUT_SL_ATR_MULTIPLIER = 1.2
BREAKOUT_USE_MEASURED_MOVE = True

# =============================================================================
# LEGACY STRATEGY: MEAN REVERSION (Disabled - break-even over 6 months)
# =============================================================================

STRATEGY_MEAN_REVERSION_ENABLED = False  # DISABLED - not profitable long-term

# Mean reversion parameters (kept for reference)
MEAN_REVERSION_EMA_FLAT_THRESHOLD = 1.0
MEAN_REVERSION_ATR_PERCENTILE_MAX = 100
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
MEAN_REVERSION_SL_ATR = 1.5
MEAN_REVERSION_MAX_RR = 2.0

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

NEWS_BLACKOUT_WINDOWS = [
    # NFP (First Friday of month)
    # ("12:30", "14:00", "NFP Release"),

    # FOMC meetings (check Fed calendar)
    # ("18:00", "20:00", "FOMC Statement"),

    # CPI releases
    # ("12:30", "14:00", "CPI Release"),
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

BACKTEST_FILL_TYPE = "next_open"  # "next_open" or "signal_close"
BACKTEST_SPREAD = 0.50  # Spread in USD
BACKTEST_SLIPPAGE = 0.30  # Slippage in USD
BACKTEST_DATA_FILE = "sample_data/xauusd_m5.csv"
