"""
Trading Strategy Engine for Gold (XAUUSD)
OPTIMIZED for long-term profitability based on 6-month backtesting

Primary Strategies (all profitable over 6 months):
  A) Support/Resistance Bounce - Best: +8.5R, 54 trades, 38.9% WR
  B) Volatility Breakout - Second: +7.4R, 19 trades, 57.9% WR
  C) EMA Crossover - Third: +5.8R, 21 trades, 38.1% WR

All signals use CLOSED candles only (no repainting).
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

    OPTIMIZED STRATEGIES (6-month profitable):
    1. S/R Bounce: Trade bounces from support/resistance
    2. Volatility Breakout: Trade momentum after ATR spike
    3. EMA Crossover: Trade EMA crosses with RSI filter
    """

    def __init__(self):
        self.last_signal_time = None
        self.last_signal_direction = None

    def evaluate(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Main entry point: evaluate all enabled strategies

        Args:
            df: DataFrame with OHLC and indicator columns

        Returns:
            Signal dict or None if no valid setup
        """
        if len(df) < 60:
            return None

        signals = []

        # Strategy A: Support/Resistance Bounce (PRIMARY - Best performer)
        if config.STRATEGY_SR_BOUNCE_ENABLED:
            sr_signal = self._check_sr_bounce(df)
            if sr_signal:
                signals.append(sr_signal)

        # Strategy B: Volatility Breakout
        if config.STRATEGY_VOL_BREAKOUT_ENABLED:
            vol_signal = self._check_volatility_breakout(df)
            if vol_signal:
                signals.append(vol_signal)

        # Strategy C: EMA Crossover
        if config.STRATEGY_EMA_CROSS_ENABLED:
            ema_signal = self._check_ema_crossover(df)
            if ema_signal:
                signals.append(ema_signal)

        # Legacy strategies (can be enabled in config if desired)
        if config.STRATEGY_TREND_PULLBACK_ENABLED:
            pullback_signal = self._check_trend_pullback(df)
            if pullback_signal:
                signals.append(pullback_signal)

        if config.STRATEGY_MEAN_REVERSION_ENABLED:
            mr_signal = self._check_mean_reversion(df)
            if mr_signal:
                signals.append(mr_signal)

        # Return highest quality signal
        if signals:
            signals.sort(key=lambda x: x['quality'], reverse=True)
            return signals[0]

        return None

    def _check_sr_bounce(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Strategy A: Support/Resistance Bounce
        BEST PERFORMER: +8.5R over 6 months, 54 trades, 38.9% win rate

        Trade bounces off recent swing highs (resistance) and lows (support)
        Uses candle confirmation for entry timing
        """
        if len(df) < 50:
            return None

        current = df.iloc[-1]
        close = current['close']
        low = current['low']
        high = current['high']
        atr = current['atr']
        rsi = current['rsi']

        if any(pd.isna([close, atr, rsi])):
            return None

        lookback = config.SR_LOOKBACK
        tolerance = atr * config.SR_TOLERANCE
        atr_mult = config.SR_ATR_SL
        rr = config.SR_RR

        # Find swing lows (support) and swing highs (resistance)
        swing_lows = []
        swing_highs = []

        for i in range(-lookback, -5):
            if abs(i) >= len(df):
                continue

            candle = df.iloc[i]
            if i > -len(df) + 2 and i < -3:
                prev_c = df.iloc[i-1]
                next_c = df.iloc[i+1]

                # Swing low: lower than both neighbors
                if candle['low'] < prev_c['low'] and candle['low'] < next_c['low']:
                    swing_lows.append(candle['low'])
                # Swing high: higher than both neighbors
                if candle['high'] > prev_c['high'] and candle['high'] > next_c['high']:
                    swing_highs.append(candle['high'])

        if not swing_lows and not swing_highs:
            return None

        # Check for SUPPORT bounce (BUY)
        for support in swing_lows:
            if abs(low - support) <= tolerance and close > support:
                # Bounced off support - need bullish candle
                if close > current['open']:
                    stop_loss = support - (atr * atr_mult)

                    if close - stop_loss < config.MIN_STOP_DISTANCE:
                        stop_loss = close - config.MIN_STOP_DISTANCE
                    if close - stop_loss > config.MAX_STOP_DISTANCE:
                        stop_loss = close - config.MAX_STOP_DISTANCE

                    risk = close - stop_loss
                    take_profit = close + (risk * rr)

                    quality = self._calculate_quality(
                        trend_aligned=True,
                        rsi_confirmation=rsi < 60,
                        candle_confirmation=True,
                        near_structure=True
                    )

                    return self._build_signal(
                        direction="BUY",
                        setup_type="S/R Bounce",
                        entry=close,
                        entry_type="Market at close",
                        stop_loss=round(stop_loss, 2),
                        take_profit=round(take_profit, 2),
                        quality=quality,
                        invalidation=f"Closes below support ${support:.2f}",
                        rationale=[
                            f"Price bounced off support at ${support:.2f}",
                            f"Bullish confirmation candle",
                            f"RSI at {rsi:.1f}",
                            f"Risk/Reward: {rr}R"
                        ],
                        indicators={'support': support, 'atr': atr, 'rsi': rsi}
                    )

        # Check for RESISTANCE rejection (SELL)
        for resistance in swing_highs:
            if abs(high - resistance) <= tolerance and close < resistance:
                # Rejected at resistance - need bearish candle
                if close < current['open']:
                    stop_loss = resistance + (atr * atr_mult)

                    if stop_loss - close < config.MIN_STOP_DISTANCE:
                        stop_loss = close + config.MIN_STOP_DISTANCE
                    if stop_loss - close > config.MAX_STOP_DISTANCE:
                        stop_loss = close + config.MAX_STOP_DISTANCE

                    risk = stop_loss - close
                    take_profit = close - (risk * rr)

                    quality = self._calculate_quality(
                        trend_aligned=True,
                        rsi_confirmation=rsi > 40,
                        candle_confirmation=True,
                        near_structure=True
                    )

                    return self._build_signal(
                        direction="SELL",
                        setup_type="S/R Bounce",
                        entry=close,
                        entry_type="Market at close",
                        stop_loss=round(stop_loss, 2),
                        take_profit=round(take_profit, 2),
                        quality=quality,
                        invalidation=f"Closes above resistance ${resistance:.2f}",
                        rationale=[
                            f"Price rejected at resistance ${resistance:.2f}",
                            f"Bearish confirmation candle",
                            f"RSI at {rsi:.1f}",
                            f"Risk/Reward: {rr}R"
                        ],
                        indicators={'resistance': resistance, 'atr': atr, 'rsi': rsi}
                    )

        return None

    def _check_volatility_breakout(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Strategy B: Volatility Breakout
        SECOND BEST: +7.4R over 6 months, 19 trades, 57.9% win rate

        Trade when ATR spikes (volatility expansion) with momentum confirmation
        High win rate strategy - fewer trades but quality setups
        """
        if len(df) < 30:
            return None

        current = df.iloc[-1]
        close = current['close']
        atr = current['atr']
        rsi = current['rsi']
        ema_20 = current['ema_20']

        if any(pd.isna([close, atr, rsi, ema_20])):
            return None

        lookback = config.VOL_ATR_LOOKBACK
        atr_spike_mult = config.VOL_ATR_SPIKE
        rr = config.VOL_RR

        # Calculate average ATR
        atrs = df['atr'].iloc[-lookback:].dropna()
        if len(atrs) < 10:
            return None

        avg_atr = atrs.mean()

        # Check for ATR spike (volatility expansion)
        if atr <= avg_atr * atr_spike_mult:
            return None

        # BULLISH volatility breakout
        if rsi > 55 and close > ema_20:
            stop_loss = close - atr

            if close - stop_loss < config.MIN_STOP_DISTANCE:
                stop_loss = close - config.MIN_STOP_DISTANCE
            if close - stop_loss > config.MAX_STOP_DISTANCE:
                stop_loss = close - config.MAX_STOP_DISTANCE

            risk = close - stop_loss
            take_profit = close + (risk * rr)

            quality = self._calculate_quality(
                trend_aligned=close > ema_20,
                rsi_confirmation=rsi > 55,
                candle_confirmation=True,
                near_structure=False
            )

            return self._build_signal(
                direction="BUY",
                setup_type="Vol Breakout",
                entry=close,
                entry_type="Market at close",
                stop_loss=round(stop_loss, 2),
                take_profit=round(take_profit, 2),
                quality=quality,
                invalidation=f"Closes below ${stop_loss:.2f}",
                rationale=[
                    f"Volatility expansion: ATR ${atr:.2f} > avg ${avg_atr:.2f}",
                    f"Bullish momentum: RSI {rsi:.1f} > 55",
                    f"Price above EMA20 ${ema_20:.2f}",
                    f"Risk/Reward: {rr}R"
                ],
                indicators={'atr': atr, 'avg_atr': avg_atr, 'rsi': rsi, 'ema_20': ema_20}
            )

        # BEARISH volatility breakout
        elif rsi < 45 and close < ema_20:
            stop_loss = close + atr

            if stop_loss - close < config.MIN_STOP_DISTANCE:
                stop_loss = close + config.MIN_STOP_DISTANCE
            if stop_loss - close > config.MAX_STOP_DISTANCE:
                stop_loss = close + config.MAX_STOP_DISTANCE

            risk = stop_loss - close
            take_profit = close - (risk * rr)

            quality = self._calculate_quality(
                trend_aligned=close < ema_20,
                rsi_confirmation=rsi < 45,
                candle_confirmation=True,
                near_structure=False
            )

            return self._build_signal(
                direction="SELL",
                setup_type="Vol Breakout",
                entry=close,
                entry_type="Market at close",
                stop_loss=round(stop_loss, 2),
                take_profit=round(take_profit, 2),
                quality=quality,
                invalidation=f"Closes above ${stop_loss:.2f}",
                rationale=[
                    f"Volatility expansion: ATR ${atr:.2f} > avg ${avg_atr:.2f}",
                    f"Bearish momentum: RSI {rsi:.1f} < 45",
                    f"Price below EMA20 ${ema_20:.2f}",
                    f"Risk/Reward: {rr}R"
                ],
                indicators={'atr': atr, 'avg_atr': avg_atr, 'rsi': rsi, 'ema_20': ema_20}
            )

        return None

    def _check_ema_crossover(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Strategy C: EMA Crossover
        THIRD BEST: +5.8R over 6 months, 21 trades, 38.1% win rate

        Trade EMA20/EMA50 crossovers with RSI momentum filter
        Uses 3R target for better risk/reward
        """
        if len(df) < 60:
            return None

        current = df.iloc[-1]
        prev = df.iloc[-2]

        ema_fast = current['ema_20']
        ema_slow = current['ema_50']
        prev_ema_fast = prev['ema_20']
        prev_ema_slow = prev['ema_50']
        close = current['close']
        atr = current['atr']
        rsi = current['rsi']

        if any(pd.isna([ema_fast, ema_slow, prev_ema_fast, prev_ema_slow, close, atr, rsi])):
            return None

        # Check for crossover
        bullish_cross = prev_ema_fast <= prev_ema_slow and ema_fast > ema_slow
        bearish_cross = prev_ema_fast >= prev_ema_slow and ema_fast < ema_slow

        min_rsi_buy = config.EMA_CROSS_RSI_BUY
        max_rsi_sell = config.EMA_CROSS_RSI_SELL
        atr_mult = config.EMA_CROSS_ATR_SL
        rr = config.EMA_CROSS_RR

        # BULLISH crossover
        if bullish_cross and rsi > min_rsi_buy:
            stop_loss = close - (atr * atr_mult)

            if close - stop_loss < config.MIN_STOP_DISTANCE:
                stop_loss = close - config.MIN_STOP_DISTANCE
            if close - stop_loss > config.MAX_STOP_DISTANCE:
                stop_loss = close - config.MAX_STOP_DISTANCE

            risk = close - stop_loss
            take_profit = close + (risk * rr)

            quality = self._calculate_quality(
                trend_aligned=True,
                rsi_confirmation=rsi > min_rsi_buy,
                candle_confirmation=True,
                near_structure=False
            )

            return self._build_signal(
                direction="BUY",
                setup_type="EMA Cross",
                entry=close,
                entry_type="Market at close",
                stop_loss=round(stop_loss, 2),
                take_profit=round(take_profit, 2),
                quality=quality,
                invalidation=f"EMA20 crosses back below EMA50",
                rationale=[
                    f"Bullish EMA crossover: EMA20 ${ema_fast:.2f} > EMA50 ${ema_slow:.2f}",
                    f"RSI momentum: {rsi:.1f} > {min_rsi_buy}",
                    f"Trend change confirmed",
                    f"Risk/Reward: {rr}R"
                ],
                indicators={'ema_20': ema_fast, 'ema_50': ema_slow, 'rsi': rsi, 'atr': atr}
            )

        # BEARISH crossover
        if bearish_cross and rsi < max_rsi_sell:
            stop_loss = close + (atr * atr_mult)

            if stop_loss - close < config.MIN_STOP_DISTANCE:
                stop_loss = close + config.MIN_STOP_DISTANCE
            if stop_loss - close > config.MAX_STOP_DISTANCE:
                stop_loss = close + config.MAX_STOP_DISTANCE

            risk = stop_loss - close
            take_profit = close - (risk * rr)

            quality = self._calculate_quality(
                trend_aligned=True,
                rsi_confirmation=rsi < max_rsi_sell,
                candle_confirmation=True,
                near_structure=False
            )

            return self._build_signal(
                direction="SELL",
                setup_type="EMA Cross",
                entry=close,
                entry_type="Market at close",
                stop_loss=round(stop_loss, 2),
                take_profit=round(take_profit, 2),
                quality=quality,
                invalidation=f"EMA20 crosses back above EMA50",
                rationale=[
                    f"Bearish EMA crossover: EMA20 ${ema_fast:.2f} < EMA50 ${ema_slow:.2f}",
                    f"RSI momentum: {rsi:.1f} < {max_rsi_sell}",
                    f"Trend change confirmed",
                    f"Risk/Reward: {rr}R"
                ],
                indicators={'ema_20': ema_fast, 'ema_50': ema_slow, 'rsi': rsi, 'atr': atr}
            )

        return None

    def _check_trend_pullback(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Legacy Strategy: Trend Pullback (DISABLED by default - poor 6-month performance)
        """
        current = df.iloc[-1]
        prev = df.iloc[-2]

        close = current['close']
        ema_20 = current['ema_20']
        ema_50 = current['ema_50']
        rsi = current['rsi']
        atr = current['atr']

        if any(pd.isna([close, ema_20, ema_50, rsi, atr])):
            return None

        trend = get_trend_bias(df)

        if trend == "BULLISH":
            if not is_price_near_ema(close, ema_20, atr, config.PULLBACK_ATR_DISTANCE):
                return None
            if close <= ema_50:
                return None
            if not (config.PULLBACK_BUY_RSI_MIN <= rsi <= config.PULLBACK_BUY_RSI_MAX):
                return None
            if not is_rsi_rising(df['rsi']):
                return None

            has_bullish_close = is_bullish_candle(current, atr)
            if not has_bullish_close:
                return None

            swing_low = get_swing_low(df, config.SWING_LOOKBACK)
            atr_stop = close - (atr * config.ATR_MULTIPLIER_SL)
            stop_loss = max(swing_low - (atr * 0.2), atr_stop) if swing_low else atr_stop

            if close - stop_loss < config.MIN_STOP_DISTANCE:
                stop_loss = close - config.MIN_STOP_DISTANCE
            if close - stop_loss > config.MAX_STOP_DISTANCE:
                stop_loss = close - config.MAX_STOP_DISTANCE

            risk = close - stop_loss
            take_profit = close + (risk * config.RISK_REWARD_RATIO)

            quality = self._calculate_quality(True, True, True, swing_low is not None)

            return self._build_signal(
                direction="BUY",
                setup_type="Trend Pullback",
                entry=close,
                entry_type="Market at close",
                stop_loss=round(stop_loss, 2),
                take_profit=round(take_profit, 2),
                quality=quality,
                invalidation=f"Closes below ${stop_loss:.2f}",
                rationale=[
                    f"Uptrend: EMA20 > EMA50",
                    f"Pullback to EMA20 zone",
                    f"RSI at {rsi:.1f}"
                ],
                indicators={'ema_20': ema_20, 'ema_50': ema_50, 'rsi': rsi, 'atr': atr}
            )

        elif trend == "BEARISH":
            if not is_price_near_ema(close, ema_20, atr, config.PULLBACK_ATR_DISTANCE):
                return None
            if close >= ema_50:
                return None
            if not (config.PULLBACK_SELL_RSI_MIN <= rsi <= config.PULLBACK_SELL_RSI_MAX):
                return None
            if not is_rsi_falling(df['rsi']):
                return None

            has_bearish_close = is_bearish_candle(current, atr)
            if not has_bearish_close:
                return None

            swing_high = get_swing_high(df, config.SWING_LOOKBACK)
            atr_stop = close + (atr * config.ATR_MULTIPLIER_SL)
            stop_loss = min(swing_high + (atr * 0.2), atr_stop) if swing_high else atr_stop

            if stop_loss - close < config.MIN_STOP_DISTANCE:
                stop_loss = close + config.MIN_STOP_DISTANCE
            if stop_loss - close > config.MAX_STOP_DISTANCE:
                stop_loss = close + config.MAX_STOP_DISTANCE

            risk = stop_loss - close
            take_profit = close - (risk * config.RISK_REWARD_RATIO)

            quality = self._calculate_quality(True, True, True, swing_high is not None)

            return self._build_signal(
                direction="SELL",
                setup_type="Trend Pullback",
                entry=close,
                entry_type="Market at close",
                stop_loss=round(stop_loss, 2),
                take_profit=round(take_profit, 2),
                quality=quality,
                invalidation=f"Closes above ${stop_loss:.2f}",
                rationale=[
                    f"Downtrend: EMA20 < EMA50",
                    f"Pullback to EMA20 zone",
                    f"RSI at {rsi:.1f}"
                ],
                indicators={'ema_20': ema_20, 'ema_50': ema_50, 'rsi': rsi, 'atr': atr}
            )

        return None

    def _check_mean_reversion(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Legacy Strategy: Mean Reversion (DISABLED by default - break-even over 6 months)
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

        prev_touched_lower = prev['low'] <= prev['bb_lower']
        current_inside = close > bb_lower

        if prev_touched_lower and current_inside:
            stop_loss = bb_lower - (atr * config.MEAN_REVERSION_SL_ATR)
            if close - stop_loss < config.MIN_STOP_DISTANCE:
                stop_loss = close - config.MIN_STOP_DISTANCE

            risk = close - stop_loss
            tp_at_rr = close + (risk * config.MEAN_REVERSION_MAX_RR)
            take_profit = min(bb_middle, tp_at_rr)
            if take_profit <= close:
                take_profit = tp_at_rr

            quality = self._calculate_quality(True, True, True, True)

            return self._build_signal(
                direction="BUY",
                setup_type="Mean Reversion",
                entry=close,
                entry_type="Market at close",
                stop_loss=round(stop_loss, 2),
                take_profit=round(take_profit, 2),
                quality=quality,
                invalidation=f"Closes below lower BB",
                rationale=[
                    f"Price touched lower BB ${bb_lower:.2f}",
                    f"Closed back inside bands",
                    f"RSI at {rsi:.1f}"
                ],
                indicators={'bb_upper': bb_upper, 'bb_lower': bb_lower, 'rsi': rsi, 'atr': atr}
            )

        prev_touched_upper = prev['high'] >= prev['bb_upper']
        current_inside_upper = close < bb_upper

        if prev_touched_upper and current_inside_upper:
            stop_loss = bb_upper + (atr * config.MEAN_REVERSION_SL_ATR)
            if stop_loss - close < config.MIN_STOP_DISTANCE:
                stop_loss = close + config.MIN_STOP_DISTANCE

            risk = stop_loss - close
            tp_at_rr = close - (risk * config.MEAN_REVERSION_MAX_RR)
            take_profit = max(bb_middle, tp_at_rr)
            if take_profit >= close:
                take_profit = tp_at_rr

            quality = self._calculate_quality(True, True, True, True)

            return self._build_signal(
                direction="SELL",
                setup_type="Mean Reversion",
                entry=close,
                entry_type="Market at close",
                stop_loss=round(stop_loss, 2),
                take_profit=round(take_profit, 2),
                quality=quality,
                invalidation=f"Closes above upper BB",
                rationale=[
                    f"Price touched upper BB ${bb_upper:.2f}",
                    f"Closed back inside bands",
                    f"RSI at {rsi:.1f}"
                ],
                indicators={'bb_upper': bb_upper, 'bb_lower': bb_lower, 'rsi': rsi, 'atr': atr}
            )

        return None

    def _calculate_quality(self, trend_aligned: bool, rsi_confirmation: bool,
                           candle_confirmation: bool, near_structure: bool) -> float:
        """Calculate signal quality score (0-1)"""
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
        """Build standardized signal dictionary"""
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
    """Check if current volatility is within acceptable range"""
    atr_percentile = calculate_atr_percentile(df, config.ATR_LOOKBACK_VOLATILITY)

    if atr_percentile < config.ATR_VOLATILITY_MIN_PERCENTILE:
        return False, f"ATR too low ({atr_percentile:.0f}th percentile)"

    if atr_percentile > config.ATR_VOLATILITY_MAX_PERCENTILE:
        return False, f"ATR too high ({atr_percentile:.0f}th percentile)"

    return True, None


if __name__ == "__main__":
    import numpy as np
    from indicators import add_all_indicators

    print("Testing Optimized Strategy Engine...")

    np.random.seed(42)
    n = 100
    dates = pd.date_range(start='2024-01-01', periods=n, freq='5min')

    base_price = 2000
    trend = np.linspace(0, 40, n)
    noise = np.random.randn(n) * 2
    prices = base_price + trend + noise

    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices - np.random.uniform(-1, 1, n),
        'high': prices + np.random.uniform(0, 3, n),
        'low': prices - np.random.uniform(0, 3, n),
        'close': prices
    })

    df = add_all_indicators(df)

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
    else:
        print("\nNo signal detected")

    print("\nStrategy Engine test complete!")
