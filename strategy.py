"""
Trading Strategy Engine for Gold (XAUUSD)
Implements three signal strategies:
  A) Trend Pullback (primary)
  B) Breakout + Retest (secondary)
  C) Mean Reversion (disabled by default)

All signals use CLOSED candles only to avoid repainting.
Returns structured signal dicts with entry, SL, TP, rationale, and quality score.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime
import config
from indicators import (
    get_trend_bias, get_swing_high, get_swing_low, get_range_high_low,
    is_rsi_rising, is_rsi_falling, is_bullish_rejection_candle,
    is_bearish_rejection_candle, is_bullish_candle, is_bearish_candle,
    is_price_near_ema, calculate_atr_percentile, is_ranging_market
)


class StrategyEngine:
    """
    Main strategy engine that evaluates all enabled strategies
    and returns high-quality trading setups.
    """

    def __init__(self):
        self.last_breakout_high = None
        self.last_breakout_low = None
        self.breakout_candle_idx = None
        self.pending_retest = None  # Track breakouts waiting for retest

    def evaluate(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Main entry point: evaluate all enabled strategies

        Args:
            df: DataFrame with OHLC and indicator columns

        Returns:
            Signal dict or None if no valid setup
        """
        if len(df) < 50:
            return None

        signals = []

        # Strategy A: Trend Pullback (primary)
        if config.STRATEGY_TREND_PULLBACK_ENABLED:
            pullback_signal = self._check_trend_pullback(df)
            if pullback_signal:
                signals.append(pullback_signal)

        # Strategy B: Breakout + Retest
        if config.STRATEGY_BREAKOUT_ENABLED:
            breakout_signal = self._check_breakout_retest(df)
            if breakout_signal:
                signals.append(breakout_signal)

        # Strategy C: Mean Reversion (only if enabled and market is ranging)
        if config.STRATEGY_MEAN_REVERSION_ENABLED:
            if is_ranging_market(df):
                mr_signal = self._check_mean_reversion(df)
                if mr_signal:
                    signals.append(mr_signal)

        # Return highest quality signal
        if signals:
            signals.sort(key=lambda x: x['quality'], reverse=True)
            return signals[0]

        return None

    def _check_trend_pullback(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Strategy A: Trend Pullback

        BUY Setup:
        - EMA20 > EMA50 (bullish trend)
        - Price pulls back to within PULLBACK_ATR_DISTANCE of EMA20
        - RSI in 40-60 zone and rising
        - Bullish rejection candle OR bullish close back in trend direction

        SELL Setup:
        - EMA20 < EMA50 (bearish trend)
        - Price pulls back to within PULLBACK_ATR_DISTANCE of EMA20
        - RSI in 40-60 zone and falling
        - Bearish rejection candle OR bearish close back in trend direction
        """
        current = df.iloc[-1]
        prev = df.iloc[-2]

        close = current['close']
        ema_20 = current['ema_20']
        ema_50 = current['ema_50']
        rsi = current['rsi']
        atr = current['atr']

        # Validate indicator values
        if any(pd.isna([close, ema_20, ema_50, rsi, atr])):
            return None

        trend = get_trend_bias(df)
        quality_factors = []

        # ========== BUY SETUP ==========
        if trend == "BULLISH":
            # Check pullback to EMA20
            if not is_price_near_ema(close, ema_20, atr, config.PULLBACK_ATR_DISTANCE):
                return None

            # Price must be above EMA50 (confirms uptrend)
            if close <= ema_50:
                return None

            # RSI in buy zone and rising
            if not (config.PULLBACK_BUY_RSI_MIN <= rsi <= config.PULLBACK_BUY_RSI_MAX):
                return None

            rsi_rising = is_rsi_rising(df['rsi'])
            if not rsi_rising:
                return None
            quality_factors.append("RSI rising from pullback zone")

            # Check for rejection candle or bullish close
            has_rejection = is_bullish_rejection_candle(current, atr)
            has_bullish_close = is_bullish_candle(current, atr)

            if not (has_rejection or has_bullish_close):
                return None

            if has_rejection:
                quality_factors.append("Bullish rejection candle formed")
            if has_bullish_close:
                quality_factors.append("Strong bullish close")

            # Calculate stop loss (swing low or ATR-based, whichever is tighter)
            swing_low = get_swing_low(df, config.SWING_LOOKBACK)
            atr_stop = close - (atr * config.ATR_MULTIPLIER_SL)

            if swing_low is not None:
                structure_stop = swing_low - (atr * 0.2)  # Buffer below swing
                stop_loss = max(structure_stop, atr_stop)  # Use tighter stop
            else:
                stop_loss = atr_stop

            # Enforce minimum stop distance
            if close - stop_loss < config.MIN_STOP_DISTANCE:
                stop_loss = close - config.MIN_STOP_DISTANCE

            # Enforce maximum stop distance
            if close - stop_loss > config.MAX_STOP_DISTANCE:
                stop_loss = close - config.MAX_STOP_DISTANCE

            # Calculate take profit
            risk = close - stop_loss
            take_profit = close + (risk * config.RISK_REWARD_RATIO)

            # Entry price
            if config.PULLBACK_ENTRY_TYPE == "limit":
                entry = ema_20
                entry_type = "Limit at EMA20"
            else:
                entry = close
                entry_type = "Market at close"

            # Quality score
            quality = self._calculate_quality(
                trend_aligned=True,
                rsi_confirmation=rsi_rising,
                candle_confirmation=has_rejection or has_bullish_close,
                near_structure=swing_low is not None
            )

            return self._build_signal(
                direction="BUY",
                setup_type="Trend Pullback",
                entry=entry,
                entry_type=entry_type,
                stop_loss=round(stop_loss, 2),
                take_profit=round(take_profit, 2),
                quality=quality,
                invalidation=f"Closes below ${stop_loss:.2f} OR EMA20 crosses below EMA50",
                rationale=[
                    f"Uptrend: EMA20 (${ema_20:.2f}) > EMA50 (${ema_50:.2f})",
                    f"Pullback to EMA20 support zone",
                    f"RSI at {rsi:.1f} - momentum building",
                    *quality_factors
                ],
                indicators={
                    'ema_20': ema_20, 'ema_50': ema_50, 'rsi': rsi, 'atr': atr
                }
            )

        # ========== SELL SETUP ==========
        elif trend == "BEARISH":
            # Check pullback to EMA20
            if not is_price_near_ema(close, ema_20, atr, config.PULLBACK_ATR_DISTANCE):
                return None

            # Price must be below EMA50 (confirms downtrend)
            if close >= ema_50:
                return None

            # RSI in sell zone and falling
            if not (config.PULLBACK_SELL_RSI_MIN <= rsi <= config.PULLBACK_SELL_RSI_MAX):
                return None

            rsi_falling = is_rsi_falling(df['rsi'])
            if not rsi_falling:
                return None
            quality_factors.append("RSI falling from pullback zone")

            # Check for rejection candle or bearish close
            has_rejection = is_bearish_rejection_candle(current, atr)
            has_bearish_close = is_bearish_candle(current, atr)

            if not (has_rejection or has_bearish_close):
                return None

            if has_rejection:
                quality_factors.append("Bearish rejection candle formed")
            if has_bearish_close:
                quality_factors.append("Strong bearish close")

            # Calculate stop loss
            swing_high = get_swing_high(df, config.SWING_LOOKBACK)
            atr_stop = close + (atr * config.ATR_MULTIPLIER_SL)

            if swing_high is not None:
                structure_stop = swing_high + (atr * 0.2)
                stop_loss = min(structure_stop, atr_stop)
            else:
                stop_loss = atr_stop

            # Enforce minimum stop distance
            if stop_loss - close < config.MIN_STOP_DISTANCE:
                stop_loss = close + config.MIN_STOP_DISTANCE

            # Enforce maximum stop distance
            if stop_loss - close > config.MAX_STOP_DISTANCE:
                stop_loss = close + config.MAX_STOP_DISTANCE

            # Calculate take profit
            risk = stop_loss - close
            take_profit = close - (risk * config.RISK_REWARD_RATIO)

            # Entry price
            if config.PULLBACK_ENTRY_TYPE == "limit":
                entry = ema_20
                entry_type = "Limit at EMA20"
            else:
                entry = close
                entry_type = "Market at close"

            # Quality score
            quality = self._calculate_quality(
                trend_aligned=True,
                rsi_confirmation=rsi_falling,
                candle_confirmation=has_rejection or has_bearish_close,
                near_structure=swing_high is not None
            )

            return self._build_signal(
                direction="SELL",
                setup_type="Trend Pullback",
                entry=entry,
                entry_type=entry_type,
                stop_loss=round(stop_loss, 2),
                take_profit=round(take_profit, 2),
                quality=quality,
                invalidation=f"Closes above ${stop_loss:.2f} OR EMA20 crosses above EMA50",
                rationale=[
                    f"Downtrend: EMA20 (${ema_20:.2f}) < EMA50 (${ema_50:.2f})",
                    f"Pullback to EMA20 resistance zone",
                    f"RSI at {rsi:.1f} - momentum weakening",
                    *quality_factors
                ],
                indicators={
                    'ema_20': ema_20, 'ema_50': ema_50, 'rsi': rsi, 'atr': atr
                }
            )

        return None

    def _check_breakout_retest(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Strategy B: Breakout + Retest

        BUY Setup:
        - Close breaks above N-candle range high by k*ATR
        - Within next 1-3 candles, price retests breakout level
        - Retest holds (no close back inside range)
        - Entry at retest confirmation

        SELL Setup:
        - Close breaks below N-candle range low by k*ATR
        - Within next 1-3 candles, price retests breakout level
        - Retest holds (no close back inside range)
        - Entry at retest confirmation
        """
        current = df.iloc[-1]
        close = current['close']
        atr = current['atr']

        if pd.isna(atr) or atr == 0:
            return None

        # Get range boundaries (excluding current candle)
        range_high, range_low = get_range_high_low(df, config.BREAKOUT_RANGE_CANDLES)
        range_height = range_high - range_low

        # Check for new breakout
        breakout_threshold = atr * config.BREAKOUT_ATR_THRESHOLD

        # ========== BULLISH BREAKOUT ==========
        if close > range_high + breakout_threshold:
            # First check if we already have a pending bullish breakout
            if self.pending_retest and self.pending_retest['direction'] == 'BUY':
                # Check for retest
                candles_since_breakout = len(df) - self.breakout_candle_idx - 1

                if candles_since_breakout <= config.RETEST_CANDLES:
                    # Check if price came back near breakout level
                    retest_level = self.pending_retest['level']
                    tolerance = atr * config.RETEST_ATR_TOLERANCE

                    low_touched_level = current['low'] <= retest_level + tolerance
                    close_above_level = close > retest_level

                    if low_touched_level and close_above_level:
                        # Retest confirmed!
                        stop_loss = min(
                            retest_level - (atr * 0.2),
                            close - (atr * config.BREAKOUT_SL_ATR_MULTIPLIER)
                        )

                        # Enforce limits
                        if close - stop_loss < config.MIN_STOP_DISTANCE:
                            stop_loss = close - config.MIN_STOP_DISTANCE
                        if close - stop_loss > config.MAX_STOP_DISTANCE:
                            stop_loss = close - config.MAX_STOP_DISTANCE

                        risk = close - stop_loss

                        # Take profit: 2R or measured move
                        if config.BREAKOUT_USE_MEASURED_MOVE:
                            take_profit = max(close + (risk * 2), retest_level + range_height)
                        else:
                            take_profit = close + (risk * config.RISK_REWARD_RATIO)

                        quality = self._calculate_quality(
                            trend_aligned=True,
                            rsi_confirmation=True,
                            candle_confirmation=True,
                            near_structure=True
                        )

                        signal = self._build_signal(
                            direction="BUY",
                            setup_type="Breakout Retest",
                            entry=close,
                            entry_type="Market at retest confirmation",
                            stop_loss=round(stop_loss, 2),
                            take_profit=round(take_profit, 2),
                            quality=quality,
                            invalidation=f"Closes back below ${retest_level:.2f} (breakout level)",
                            rationale=[
                                f"Breakout above ${range_high:.2f} range high",
                                f"Successful retest of breakout level ${retest_level:.2f}",
                                f"Buyers defending the breakout zone",
                                f"Range height target: ${range_height:.2f}"
                            ],
                            indicators={
                                'range_high': range_high, 'range_low': range_low,
                                'atr': atr, 'range_height': range_height
                            }
                        )

                        self.pending_retest = None
                        return signal

                elif candles_since_breakout > config.RETEST_CANDLES:
                    # Timeout - no valid retest
                    self.pending_retest = None

            else:
                # New breakout - record it and wait for retest
                self.pending_retest = {
                    'direction': 'BUY',
                    'level': range_high,
                    'timestamp': current.get('timestamp', datetime.now())
                }
                self.breakout_candle_idx = len(df) - 1

        # ========== BEARISH BREAKOUT ==========
        elif close < range_low - breakout_threshold:
            if self.pending_retest and self.pending_retest['direction'] == 'SELL':
                candles_since_breakout = len(df) - self.breakout_candle_idx - 1

                if candles_since_breakout <= config.RETEST_CANDLES:
                    retest_level = self.pending_retest['level']
                    tolerance = atr * config.RETEST_ATR_TOLERANCE

                    high_touched_level = current['high'] >= retest_level - tolerance
                    close_below_level = close < retest_level

                    if high_touched_level and close_below_level:
                        # Retest confirmed!
                        stop_loss = max(
                            retest_level + (atr * 0.2),
                            close + (atr * config.BREAKOUT_SL_ATR_MULTIPLIER)
                        )

                        if stop_loss - close < config.MIN_STOP_DISTANCE:
                            stop_loss = close + config.MIN_STOP_DISTANCE
                        if stop_loss - close > config.MAX_STOP_DISTANCE:
                            stop_loss = close + config.MAX_STOP_DISTANCE

                        risk = stop_loss - close

                        if config.BREAKOUT_USE_MEASURED_MOVE:
                            take_profit = min(close - (risk * 2), retest_level - range_height)
                        else:
                            take_profit = close - (risk * config.RISK_REWARD_RATIO)

                        quality = self._calculate_quality(
                            trend_aligned=True,
                            rsi_confirmation=True,
                            candle_confirmation=True,
                            near_structure=True
                        )

                        signal = self._build_signal(
                            direction="SELL",
                            setup_type="Breakout Retest",
                            entry=close,
                            entry_type="Market at retest confirmation",
                            stop_loss=round(stop_loss, 2),
                            take_profit=round(take_profit, 2),
                            quality=quality,
                            invalidation=f"Closes back above ${retest_level:.2f} (breakout level)",
                            rationale=[
                                f"Breakdown below ${range_low:.2f} range low",
                                f"Successful retest of breakdown level ${retest_level:.2f}",
                                f"Sellers defending the breakdown zone",
                                f"Range height target: ${range_height:.2f}"
                            ],
                            indicators={
                                'range_high': range_high, 'range_low': range_low,
                                'atr': atr, 'range_height': range_height
                            }
                        )

                        self.pending_retest = None
                        return signal

                elif candles_since_breakout > config.RETEST_CANDLES:
                    self.pending_retest = None

            else:
                self.pending_retest = {
                    'direction': 'SELL',
                    'level': range_low,
                    'timestamp': current.get('timestamp', datetime.now())
                }
                self.breakout_candle_idx = len(df) - 1

        return None

    def _check_mean_reversion(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Strategy C: Mean Reversion (ranging markets only)

        BUY Setup:
        - Previous candle touched/closed below lower BB
        - Current candle closes back INSIDE the bands
        - RSI was oversold or near oversold

        SELL Setup:
        - Previous candle touched/closed above upper BB
        - Current candle closes back INSIDE the bands
        - RSI was overbought or near overbought
        """
        if len(df) < 3:
            return None

        current = df.iloc[-1]
        prev = df.iloc[-2]

        close = current['close']
        atr = current['atr']
        rsi = current['rsi']
        bb_upper = current['bb_upper']
        bb_lower = current['bb_lower']
        bb_middle = current['bb_middle']

        if any(pd.isna([close, atr, rsi, bb_upper, bb_lower, bb_middle])):
            return None

        # ========== BUY SETUP (fade lower band) ==========
        prev_touched_lower = prev['low'] <= prev['bb_lower']
        current_inside = close > bb_lower

        if prev_touched_lower and current_inside:
            # RSI confirmation (was oversold or near)
            was_oversold = prev['rsi'] <= config.RSI_OVERSOLD + 10

            if was_oversold and is_bullish_candle(current, atr):
                stop_loss = bb_lower - (atr * config.MEAN_REVERSION_SL_ATR)

                if close - stop_loss < config.MIN_STOP_DISTANCE:
                    stop_loss = close - config.MIN_STOP_DISTANCE

                risk = close - stop_loss
                max_tp = close + (risk * config.MEAN_REVERSION_MAX_RR)
                take_profit = min(bb_middle, max_tp)

                quality = self._calculate_quality(
                    trend_aligned=False,
                    rsi_confirmation=was_oversold,
                    candle_confirmation=True,
                    near_structure=True
                )
                quality *= 0.9  # Slight penalty for counter-trend

                return self._build_signal(
                    direction="BUY",
                    setup_type="Mean Reversion",
                    entry=close,
                    entry_type="Market at close",
                    stop_loss=round(stop_loss, 2),
                    take_profit=round(take_profit, 2),
                    quality=quality,
                    invalidation=f"Closes below lower BB (${bb_lower:.2f})",
                    rationale=[
                        "Ranging market detected (flat EMA, low volatility)",
                        f"Price bounced off lower Bollinger Band (${bb_lower:.2f})",
                        f"RSI recovered from oversold ({prev['rsi']:.1f} -> {rsi:.1f})",
                        f"Target: BB middle line (${bb_middle:.2f})"
                    ],
                    indicators={
                        'bb_upper': bb_upper, 'bb_lower': bb_lower,
                        'bb_middle': bb_middle, 'rsi': rsi, 'atr': atr
                    }
                )

        # ========== SELL SETUP (fade upper band) ==========
        prev_touched_upper = prev['high'] >= prev['bb_upper']
        current_inside_upper = close < bb_upper

        if prev_touched_upper and current_inside_upper:
            was_overbought = prev['rsi'] >= config.RSI_OVERBOUGHT - 10

            if was_overbought and is_bearish_candle(current, atr):
                stop_loss = bb_upper + (atr * config.MEAN_REVERSION_SL_ATR)

                if stop_loss - close < config.MIN_STOP_DISTANCE:
                    stop_loss = close + config.MIN_STOP_DISTANCE

                risk = stop_loss - close
                max_tp = close - (risk * config.MEAN_REVERSION_MAX_RR)
                take_profit = max(bb_middle, max_tp)

                quality = self._calculate_quality(
                    trend_aligned=False,
                    rsi_confirmation=was_overbought,
                    candle_confirmation=True,
                    near_structure=True
                )
                quality *= 0.9

                return self._build_signal(
                    direction="SELL",
                    setup_type="Mean Reversion",
                    entry=close,
                    entry_type="Market at close",
                    stop_loss=round(stop_loss, 2),
                    take_profit=round(take_profit, 2),
                    quality=quality,
                    invalidation=f"Closes above upper BB (${bb_upper:.2f})",
                    rationale=[
                        "Ranging market detected (flat EMA, low volatility)",
                        f"Price rejected from upper Bollinger Band (${bb_upper:.2f})",
                        f"RSI reversed from overbought ({prev['rsi']:.1f} -> {rsi:.1f})",
                        f"Target: BB middle line (${bb_middle:.2f})"
                    ],
                    indicators={
                        'bb_upper': bb_upper, 'bb_lower': bb_lower,
                        'bb_middle': bb_middle, 'rsi': rsi, 'atr': atr
                    }
                )

        return None

    def _calculate_quality(self, trend_aligned: bool, rsi_confirmation: bool,
                           candle_confirmation: bool, near_structure: bool) -> float:
        """
        Calculate signal quality score (0-1)

        Factors:
        - Trend alignment: 0.3
        - RSI confirmation: 0.25
        - Candle pattern: 0.25
        - Structure (S/R): 0.2
        """
        score = 0.0

        if trend_aligned:
            score += 0.30
        if rsi_confirmation:
            score += 0.25
        if candle_confirmation:
            score += 0.25
        if near_structure:
            score += 0.20

        return min(score, 1.0)

    def _build_signal(self, direction: str, setup_type: str, entry: float,
                      entry_type: str, stop_loss: float, take_profit: float,
                      quality: float, invalidation: str, rationale: List[str],
                      indicators: Dict[str, float]) -> Dict[str, Any]:
        """
        Build standardized signal dictionary
        """
        risk = abs(entry - stop_loss)
        reward = abs(take_profit - entry)
        risk_reward = reward / risk if risk > 0 else 0

        return {
            'direction': direction,
            'setup_type': setup_type,
            'entry': round(entry, 2),
            'entry_type': entry_type,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_reward': round(risk_reward, 2),
            'quality': round(quality, 2),
            'invalidation': invalidation,
            'rationale': rationale,
            'timeframe': config.TIMEFRAME,
            'timestamp': datetime.utcnow().isoformat(),
            'indicators': indicators
        }


def check_volatility_filter(df: pd.DataFrame) -> tuple:
    """
    Check if current volatility is within acceptable range

    Returns:
        (is_valid: bool, reason: str or None)
    """
    atr_percentile = calculate_atr_percentile(df, config.ATR_LOOKBACK_VOLATILITY)

    if atr_percentile < config.ATR_VOLATILITY_MIN_PERCENTILE:
        return False, f"ATR too low ({atr_percentile:.0f}th percentile) - choppy market"

    if atr_percentile > config.ATR_VOLATILITY_MAX_PERCENTILE:
        return False, f"ATR too high ({atr_percentile:.0f}th percentile) - chaotic market"

    return True, None


if __name__ == "__main__":
    import numpy as np
    from indicators import add_all_indicators

    print("Testing Strategy Engine...")

    # Create sample uptrend data
    np.random.seed(42)
    n = 100
    dates = pd.date_range(start='2024-01-01', periods=n, freq='5min')

    # Simulate uptrend with pullback
    base_price = 2000
    trend = np.linspace(0, 40, n)
    noise = np.random.randn(n) * 2
    prices = base_price + trend + noise

    # Add a pullback
    pullback_start = 80
    pullback_end = 90
    for i in range(pullback_start, pullback_end):
        prices[i] -= (i - pullback_start) * 0.5

    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices - np.random.uniform(-1, 1, n),
        'high': prices + np.random.uniform(0, 3, n),
        'low': prices - np.random.uniform(0, 3, n),
        'close': prices
    })

    df = add_all_indicators(df)

    # Test strategy
    engine = StrategyEngine()
    signal = engine.evaluate(df)

    if signal:
        print(f"\nSignal Found!")
        print(f"  Direction: {signal['direction']}")
        print(f"  Setup: {signal['setup_type']}")
        print(f"  Entry: ${signal['entry']}")
        print(f"  Stop Loss: ${signal['stop_loss']}")
        print(f"  Take Profit: ${signal['take_profit']}")
        print(f"  R:R: {signal['risk_reward']}")
        print(f"  Quality: {signal['quality']:.0%}")
        print(f"\nRationale:")
        for reason in signal['rationale']:
            print(f"    - {reason}")
        print(f"\nInvalidation: {signal['invalidation']}")
    else:
        print("\nNo signal detected")

    # Test volatility filter
    is_valid, reason = check_volatility_filter(df)
    print(f"\nVolatility Check: {'PASS' if is_valid else 'FAIL'}")
    if reason:
        print(f"  Reason: {reason}")

    print("\nStrategy Engine test complete!")
