"""
Technical indicators module for Gold Trading Bot
Calculates EMA, RSI, ATR, Bollinger Bands, and provides helper functions
for signal detection. All indicators use CLOSED candles only (no repainting).
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
import config


def calculate_ema(data: pd.Series, period: int) -> pd.Series:
    """
    Calculate Exponential Moving Average

    Args:
        data: pandas Series of prices
        period: EMA period

    Returns:
        pandas Series with EMA values
    """
    return data.ewm(span=period, adjust=False).mean()


def calculate_sma(data: pd.Series, period: int) -> pd.Series:
    """
    Calculate Simple Moving Average

    Args:
        data: pandas Series of prices
        period: SMA period

    Returns:
        pandas Series with SMA values
    """
    return data.rolling(window=period).mean()


def calculate_rsi(data: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index using Wilder's smoothing method

    Args:
        data: pandas Series of prices
        period: RSI period (default 14)

    Returns:
        pandas Series with RSI values (0-100)
    """
    delta = data.diff()

    gain = delta.where(delta > 0, 0.0)
    loss = (-delta.where(delta < 0, 0.0))

    # Use Wilder's smoothing (exponential)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series,
                  period: int = 14) -> pd.Series:
    """
    Calculate Average True Range

    Args:
        high: pandas Series of high prices
        low: pandas Series of low prices
        close: pandas Series of close prices
        period: ATR period (default 14)

    Returns:
        pandas Series with ATR values
    """
    high_low = high - low
    high_close = np.abs(high - close.shift(1))
    low_close = np.abs(low - close.shift(1))

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

    return atr


def calculate_bollinger_bands(data: pd.Series, period: int = 20,
                               std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Bollinger Bands

    Args:
        data: pandas Series of prices
        period: MA period (default 20)
        std_dev: standard deviation multiplier (default 2.0)

    Returns:
        Tuple of (middle_band, upper_band, lower_band)
    """
    middle = data.rolling(window=period).mean()
    std = data.rolling(window=period).std()

    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)

    return middle, upper, lower


def is_rsi_rising(rsi_series: pd.Series, lookback: int = 3) -> bool:
    """
    Check if RSI is rising (momentum building for longs)

    Args:
        rsi_series: pandas Series of RSI values
        lookback: number of periods to check

    Returns:
        bool - True if RSI showing upward momentum
    """
    if len(rsi_series) < lookback + 1:
        return False

    recent = rsi_series.iloc[-(lookback+1):].values
    if np.any(np.isnan(recent)):
        return False

    # Check for rising pattern (at least 2 increases in last 3 periods)
    increases = sum(1 for i in range(1, len(recent)) if recent[i] > recent[i-1])
    return increases >= 2


def is_rsi_falling(rsi_series: pd.Series, lookback: int = 3) -> bool:
    """
    Check if RSI is falling (momentum building for shorts)

    Args:
        rsi_series: pandas Series of RSI values
        lookback: number of periods to check

    Returns:
        bool - True if RSI showing downward momentum
    """
    if len(rsi_series) < lookback + 1:
        return False

    recent = rsi_series.iloc[-(lookback+1):].values
    if np.any(np.isnan(recent)):
        return False

    # Check for falling pattern (at least 2 decreases in last 3 periods)
    decreases = sum(1 for i in range(1, len(recent)) if recent[i] < recent[i-1])
    return decreases >= 2


def get_swing_high(df: pd.DataFrame, lookback: int = 10,
                   exclude_current: bool = True) -> Optional[float]:
    """
    Find the highest high in lookback period

    Args:
        df: DataFrame with 'high' column
        lookback: candles to look back
        exclude_current: if True, excludes the most recent candle

    Returns:
        float - swing high price or None
    """
    if len(df) < lookback:
        lookback = len(df)

    if exclude_current:
        data = df.iloc[-(lookback+1):-1]['high']
    else:
        data = df.iloc[-lookback:]['high']

    if data.empty:
        return None

    return data.max()


def get_swing_low(df: pd.DataFrame, lookback: int = 10,
                  exclude_current: bool = True) -> Optional[float]:
    """
    Find the lowest low in lookback period

    Args:
        df: DataFrame with 'low' column
        lookback: candles to look back
        exclude_current: if True, excludes the most recent candle

    Returns:
        float - swing low price or None
    """
    if len(df) < lookback:
        lookback = len(df)

    if exclude_current:
        data = df.iloc[-(lookback+1):-1]['low']
    else:
        data = df.iloc[-lookback:]['low']

    if data.empty:
        return None

    return data.min()


def get_range_high_low(df: pd.DataFrame, candles: int = 20) -> Tuple[float, float]:
    """
    Get range boundaries (highest high, lowest low) for breakout detection

    Args:
        df: DataFrame with OHLC data
        candles: lookback period for range (excluding current candle)

    Returns:
        Tuple of (range_high, range_low)
    """
    if len(df) < candles + 1:
        candles = len(df) - 1

    # Exclude current candle
    range_data = df.iloc[-(candles+1):-1]

    range_high = range_data['high'].max()
    range_low = range_data['low'].min()

    return range_high, range_low


def is_bullish_rejection_candle(candle: pd.Series, atr: float) -> bool:
    """
    Check if candle shows bullish rejection (hammer/pin bar)
    - Close above open (green candle)
    - Lower wick > 2x body
    - Upper wick < body

    Args:
        candle: pandas Series with OHLC data
        atr: current ATR for minimum body filter

    Returns:
        bool
    """
    o, h, l, c = candle['open'], candle['high'], candle['low'], candle['close']

    body = abs(c - o)
    lower_wick = min(o, c) - l
    upper_wick = h - max(o, c)

    # Must have some body (avoid pure doji)
    min_body = atr * config.MIN_CANDLE_BODY_ATR
    if body < min_body:
        return False

    # Bullish rejection: close > open, long lower wick
    if c > o:
        if lower_wick >= 2 * body and upper_wick <= body:
            return True

    return False


def is_bearish_rejection_candle(candle: pd.Series, atr: float) -> bool:
    """
    Check if candle shows bearish rejection (shooting star/inverted hammer)
    - Close below open (red candle)
    - Upper wick > 2x body
    - Lower wick < body

    Args:
        candle: pandas Series with OHLC data
        atr: current ATR for minimum body filter

    Returns:
        bool
    """
    o, h, l, c = candle['open'], candle['high'], candle['low'], candle['close']

    body = abs(c - o)
    lower_wick = min(o, c) - l
    upper_wick = h - max(o, c)

    # Must have some body (avoid pure doji)
    min_body = atr * config.MIN_CANDLE_BODY_ATR
    if body < min_body:
        return False

    # Bearish rejection: close < open, long upper wick
    if c < o:
        if upper_wick >= 2 * body and lower_wick <= body:
            return True

    return False


def is_bullish_candle(candle: pd.Series, atr: float = None) -> bool:
    """
    Check if candle closed bullish (close > open) with meaningful body

    Args:
        candle: pandas Series with OHLC data
        atr: optional ATR for minimum body check

    Returns:
        bool
    """
    body = candle['close'] - candle['open']

    if atr is not None:
        min_body = atr * config.MIN_CANDLE_BODY_ATR
        return body >= min_body

    return body > 0


def is_bearish_candle(candle: pd.Series, atr: float = None) -> bool:
    """
    Check if candle closed bearish (close < open) with meaningful body

    Args:
        candle: pandas Series with OHLC data
        atr: optional ATR for minimum body check

    Returns:
        bool
    """
    body = candle['open'] - candle['close']

    if atr is not None:
        min_body = atr * config.MIN_CANDLE_BODY_ATR
        return body >= min_body

    return body > 0


def calculate_atr_percentile(df: pd.DataFrame, lookback: int = 50) -> float:
    """
    Calculate current ATR percentile relative to recent history
    Used for volatility filtering (avoid too quiet or too chaotic)

    Args:
        df: DataFrame with 'atr' column
        lookback: periods for percentile calculation

    Returns:
        percentile (0-100)
    """
    if len(df) < lookback or 'atr' not in df.columns:
        return 50  # Default to middle if insufficient data

    recent_atr = df['atr'].iloc[-lookback:].dropna()
    current_atr = df['atr'].iloc[-1]

    if pd.isna(current_atr) or len(recent_atr) == 0:
        return 50

    percentile = (recent_atr < current_atr).sum() / len(recent_atr) * 100
    return percentile


def calculate_ema_slope(df: pd.DataFrame, ema_col: str = 'ema_20',
                        lookback: int = 5) -> float:
    """
    Calculate EMA slope as percentage change
    Used for detecting flat/ranging markets

    Args:
        df: DataFrame with EMA column
        ema_col: name of EMA column
        lookback: periods for slope calculation

    Returns:
        slope as percentage (e.g., 0.001 = 0.1%)
    """
    if len(df) < lookback + 1 or ema_col not in df.columns:
        return 0.0

    ema = df[ema_col].iloc[-lookback:].dropna()
    if len(ema) < 2:
        return 0.0

    start_val = ema.iloc[0]
    end_val = ema.iloc[-1]

    if start_val == 0:
        return 0.0

    slope = (end_val - start_val) / start_val
    return slope


def is_price_near_ema(close: float, ema: float, atr: float,
                       max_distance: float = 1.0) -> bool:
    """
    Check if price is within specified ATR distance of EMA

    Args:
        close: current close price
        ema: EMA value
        atr: current ATR
        max_distance: maximum distance in ATR multiples

    Returns:
        bool
    """
    if pd.isna(ema) or pd.isna(atr) or atr == 0:
        return False

    distance = abs(close - ema) / atr
    return distance <= max_distance


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all technical indicators to the DataFrame

    Args:
        df: DataFrame with OHLC data (timestamp, open, high, low, close)

    Returns:
        DataFrame with added indicator columns
    """
    df = df.copy()

    # EMAs
    df['ema_20'] = calculate_ema(df['close'], config.EMA_FAST)
    df['ema_50'] = calculate_ema(df['close'], config.EMA_SLOW)

    # RSI
    df['rsi'] = calculate_rsi(df['close'], config.RSI_PERIOD)

    # ATR
    df['atr'] = calculate_atr(df['high'], df['low'], df['close'], config.ATR_PERIOD)

    # Bollinger Bands
    df['bb_middle'], df['bb_upper'], df['bb_lower'] = calculate_bollinger_bands(
        df['close'], config.BOLLINGER_PERIOD, config.BOLLINGER_STD
    )

    # EMA slope for ranging detection
    df['ema_slope'] = df['ema_20'].pct_change(5)

    return df


def get_trend_bias(df: pd.DataFrame) -> str:
    """
    Determine overall trend bias from EMA relationship

    Args:
        df: DataFrame with indicator columns

    Returns:
        "BULLISH", "BEARISH", or "NEUTRAL"
    """
    if len(df) < 2:
        return "NEUTRAL"

    current = df.iloc[-1]
    ema_20 = current.get('ema_20')
    ema_50 = current.get('ema_50')

    if pd.isna(ema_20) or pd.isna(ema_50):
        return "NEUTRAL"

    if ema_20 > ema_50:
        return "BULLISH"
    elif ema_20 < ema_50:
        return "BEARISH"
    else:
        return "NEUTRAL"


def is_ranging_market(df: pd.DataFrame) -> bool:
    """
    Check if market is ranging (flat EMA + low volatility)

    Args:
        df: DataFrame with indicator columns

    Returns:
        bool - True if market appears to be ranging
    """
    if len(df) < 20:
        return False

    # Check EMA slope
    slope = calculate_ema_slope(df, 'ema_20', 5)
    ema_flat = abs(slope) < config.MEAN_REVERSION_EMA_FLAT_THRESHOLD

    # Check ATR percentile
    atr_percentile = calculate_atr_percentile(df, config.ATR_LOOKBACK_VOLATILITY)
    low_volatility = atr_percentile < config.MEAN_REVERSION_ATR_PERCENTILE_MAX

    return ema_flat and low_volatility


if __name__ == "__main__":
    print("Testing technical indicators...")

    # Create sample data
    np.random.seed(42)
    n = 100
    dates = pd.date_range(start='2024-01-01', periods=n, freq='5min')

    # Simulate price with trend
    base_price = 2000
    trend = np.linspace(0, 30, n)
    noise = np.random.randn(n) * 3
    prices = base_price + trend + noise

    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices - np.random.uniform(-1, 1, n),
        'high': prices + np.random.uniform(0, 3, n),
        'low': prices - np.random.uniform(0, 3, n),
        'close': prices
    })

    # Add indicators
    df = add_all_indicators(df)

    print("\nLatest values:")
    print(f"  Close:     ${df['close'].iloc[-1]:.2f}")
    print(f"  EMA(20):   ${df['ema_20'].iloc[-1]:.2f}")
    print(f"  EMA(50):   ${df['ema_50'].iloc[-1]:.2f}")
    print(f"  RSI(14):   {df['rsi'].iloc[-1]:.1f}")
    print(f"  ATR(14):   ${df['atr'].iloc[-1]:.2f}")
    print(f"  BB Upper:  ${df['bb_upper'].iloc[-1]:.2f}")
    print(f"  BB Lower:  ${df['bb_lower'].iloc[-1]:.2f}")

    print(f"\nTrend Bias: {get_trend_bias(df)}")
    print(f"Is Ranging: {is_ranging_market(df)}")
    print(f"ATR Percentile: {calculate_atr_percentile(df):.1f}%")

    print("\nIndicators calculated successfully!")
