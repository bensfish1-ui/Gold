"""
Strategy Optimizer V2 - Deep Parameter Search
Focus on finding profitable long-term strategy through:
1. Trend-following with proper filters
2. Mean reversion with strict conditions
3. Hybrid approaches
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from indicators import add_all_indicators


class Trade:
    def __init__(self, direction: str, entry: float, stop: float, tp: float,
                 entry_time: datetime, setup: str):
        self.direction = direction
        self.entry = entry
        self.stop = stop
        self.tp = tp
        self.entry_time = entry_time
        self.setup = setup
        self.exit_price = None
        self.exit_time = None
        self.pnl = 0.0
        self.r_multiple = 0.0

    @property
    def risk(self):
        return abs(self.entry - self.stop)

    def close(self, price: float, time: datetime, reason: str):
        self.exit_price = price
        self.exit_time = time
        if self.direction == "BUY":
            self.pnl = price - self.entry
        else:
            self.pnl = self.entry - price
        self.r_multiple = self.pnl / self.risk if self.risk > 0 else 0


def fetch_data(months: int = 6) -> pd.DataFrame:
    """Load cached data or fetch from Yahoo"""
    cache_file = f'sample_data/gold_{months}months_1h.csv'
    if os.path.exists(cache_file):
        df = pd.read_csv(cache_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df

    print("Data not found. Run strategy_optimizer.py first to fetch data.")
    sys.exit(1)


# ============================================================================
# STRATEGY 1: EMA CROSSOVER + MOMENTUM
# Trade with trend after EMA cross, confirmed by RSI momentum
# ============================================================================

def strategy_ema_crossover(df: pd.DataFrame, params: Dict) -> Optional[Dict]:
    """
    Trade EMA crossovers with momentum confirmation
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

    # BUY: Bullish crossover just happened
    bullish_cross = prev_ema_fast <= prev_ema_slow and ema_fast > ema_slow
    # SELL: Bearish crossover just happened
    bearish_cross = prev_ema_fast >= prev_ema_slow and ema_fast < ema_slow

    min_rsi = params.get('min_rsi_buy', 50)
    max_rsi = params.get('max_rsi_sell', 50)
    atr_mult = params.get('atr_sl', 2.0)
    rr = params.get('rr', 2.0)

    if bullish_cross and rsi > min_rsi:
        stop_loss = close - (atr * atr_mult)
        risk = close - stop_loss
        take_profit = close + (risk * rr)
        return {'direction': 'BUY', 'entry': close, 'stop': stop_loss,
                'tp': take_profit, 'setup': 'EMA Cross'}

    if bearish_cross and rsi < max_rsi:
        stop_loss = close + (atr * atr_mult)
        risk = stop_loss - close
        take_profit = close - (risk * rr)
        return {'direction': 'SELL', 'entry': close, 'stop': stop_loss,
                'tp': take_profit, 'setup': 'EMA Cross'}

    return None


# ============================================================================
# STRATEGY 2: RSI DIVERGENCE
# Look for RSI making higher lows while price makes lower lows (bullish)
# ============================================================================

def strategy_rsi_extreme(df: pd.DataFrame, params: Dict) -> Optional[Dict]:
    """
    Trade RSI extremes with reversal confirmation
    """
    if len(df) < 10:
        return None

    current = df.iloc[-1]
    prev = df.iloc[-2]

    close = current['close']
    atr = current['atr']
    rsi = current['rsi']
    prev_rsi = prev['rsi']

    if any(pd.isna([close, atr, rsi, prev_rsi])):
        return None

    oversold = params.get('oversold', 25)
    overbought = params.get('overbought', 75)
    atr_mult = params.get('atr_sl', 2.0)
    rr = params.get('rr', 2.0)

    # BUY: RSI was oversold and now turning up
    if prev_rsi < oversold and rsi > prev_rsi and rsi < 50:
        stop_loss = current['low'] - (atr * atr_mult)
        risk = close - stop_loss
        if risk < 3:
            stop_loss = close - 3
            risk = 3
        take_profit = close + (risk * rr)
        return {'direction': 'BUY', 'entry': close, 'stop': stop_loss,
                'tp': take_profit, 'setup': 'RSI Extreme'}

    # SELL: RSI was overbought and now turning down
    if prev_rsi > overbought and rsi < prev_rsi and rsi > 50:
        stop_loss = current['high'] + (atr * atr_mult)
        risk = stop_loss - close
        if risk < 3:
            stop_loss = close + 3
            risk = 3
        take_profit = close - (risk * rr)
        return {'direction': 'SELL', 'entry': close, 'stop': stop_loss,
                'tp': take_profit, 'setup': 'RSI Extreme'}

    return None


# ============================================================================
# STRATEGY 3: TREND CONTINUATION (Price above/below both EMAs)
# Only trade in strong trends, wait for pullback then continuation
# ============================================================================

def strategy_trend_continuation(df: pd.DataFrame, params: Dict) -> Optional[Dict]:
    """
    Trade continuation in established trends
    """
    if len(df) < 60:
        return None

    current = df.iloc[-1]
    prev = df.iloc[-2]
    prev2 = df.iloc[-3]

    close = current['close']
    ema_20 = current['ema_20']
    ema_50 = current['ema_50']
    atr = current['atr']
    rsi = current['rsi']

    if any(pd.isna([close, ema_20, ema_50, atr, rsi])):
        return None

    atr_mult = params.get('atr_sl', 1.5)
    rr = params.get('rr', 2.0)
    trend_strength = params.get('trend_strength', 0.5)  # EMA gap as % of ATR

    ema_gap = abs(ema_20 - ema_50) / atr if atr > 0 else 0

    # Strong uptrend: EMA20 > EMA50 with good separation
    if ema_20 > ema_50 and ema_gap >= trend_strength:
        # Price above both EMAs
        if close > ema_20:
            # Pullback happened (prev candle touched or went below EMA20)
            if prev['low'] <= ema_20 or prev2['low'] <= prev['ema_20']:
                # Current candle closes bullish above EMA20
                if close > current['open']:
                    stop_loss = min(ema_20, current['low']) - (atr * 0.5)
                    if close - stop_loss < 3:
                        stop_loss = close - 3
                    if close - stop_loss > 25:
                        stop_loss = close - 25

                    risk = close - stop_loss
                    take_profit = close + (risk * rr)

                    return {'direction': 'BUY', 'entry': close, 'stop': stop_loss,
                            'tp': take_profit, 'setup': 'Trend Continuation'}

    # Strong downtrend
    if ema_20 < ema_50 and ema_gap >= trend_strength:
        if close < ema_20:
            if prev['high'] >= ema_20 or prev2['high'] >= prev['ema_20']:
                if close < current['open']:
                    stop_loss = max(ema_20, current['high']) + (atr * 0.5)
                    if stop_loss - close < 3:
                        stop_loss = close + 3
                    if stop_loss - close > 25:
                        stop_loss = close + 25

                    risk = stop_loss - close
                    take_profit = close - (risk * rr)

                    return {'direction': 'SELL', 'entry': close, 'stop': stop_loss,
                            'tp': take_profit, 'setup': 'Trend Continuation'}

    return None


# ============================================================================
# STRATEGY 4: BOLLINGER SQUEEZE BREAKOUT
# Trade breakouts after low volatility periods (BB width contracts)
# ============================================================================

def strategy_bb_squeeze(df: pd.DataFrame, params: Dict) -> Optional[Dict]:
    """
    Trade breakouts from Bollinger Band squeezes
    """
    if len(df) < 30:
        return None

    current = df.iloc[-1]
    prev = df.iloc[-2]

    close = current['close']
    bb_upper = current['bb_upper']
    bb_lower = current['bb_lower']
    bb_middle = current['bb_middle']
    atr = current['atr']

    if any(pd.isna([close, bb_upper, bb_lower, bb_middle, atr])):
        return None

    # Calculate BB width
    bb_width = (bb_upper - bb_lower) / bb_middle if bb_middle > 0 else 0

    # Look back to see if we were in a squeeze (low BB width)
    lookback = params.get('squeeze_lookback', 20)
    squeeze_threshold = params.get('squeeze_threshold', 0.02)

    recent_widths = []
    for i in range(-lookback, -1):
        if i >= -len(df):
            candle = df.iloc[i]
            if not pd.isna(candle['bb_upper']):
                w = (candle['bb_upper'] - candle['bb_lower']) / candle['bb_middle'] if candle['bb_middle'] > 0 else 0
                recent_widths.append(w)

    if not recent_widths:
        return None

    min_width = min(recent_widths)
    was_squeezed = min_width < squeeze_threshold

    atr_mult = params.get('atr_sl', 2.0)
    rr = params.get('rr', 2.0)

    # Breakout from squeeze
    if was_squeezed:
        # Bullish breakout: close above upper band
        if close > bb_upper and prev['close'] <= prev['bb_upper']:
            stop_loss = bb_middle - (atr * 0.5)
            if close - stop_loss < 3:
                stop_loss = close - 3

            risk = close - stop_loss
            take_profit = close + (risk * rr)

            return {'direction': 'BUY', 'entry': close, 'stop': stop_loss,
                    'tp': take_profit, 'setup': 'BB Squeeze'}

        # Bearish breakout
        if close < bb_lower and prev['close'] >= prev['bb_lower']:
            stop_loss = bb_middle + (atr * 0.5)
            if stop_loss - close < 3:
                stop_loss = close + 3

            risk = stop_loss - close
            take_profit = close - (risk * rr)

            return {'direction': 'SELL', 'entry': close, 'stop': stop_loss,
                    'tp': take_profit, 'setup': 'BB Squeeze'}

    return None


# ============================================================================
# STRATEGY 5: SUPPORT/RESISTANCE BOUNCE
# Trade bounces off recent swing highs/lows
# ============================================================================

def strategy_sr_bounce(df: pd.DataFrame, params: Dict) -> Optional[Dict]:
    """
    Trade bounces from support/resistance levels
    """
    if len(df) < 50:
        return None

    current = df.iloc[-1]
    prev = df.iloc[-2]

    close = current['close']
    low = current['low']
    high = current['high']
    atr = current['atr']

    if any(pd.isna([close, atr])):
        return None

    lookback = params.get('sr_lookback', 30)
    tolerance = params.get('sr_tolerance', 0.5)  # ATR multiplier
    atr_mult = params.get('atr_sl', 1.5)
    rr = params.get('rr', 2.0)

    # Find swing lows (potential support)
    swing_lows = []
    swing_highs = []

    for i in range(-lookback, -5):
        if i >= -len(df):
            candle = df.iloc[i]
            # Simple swing detection: lower than neighbors
            if i > -len(df) + 2 and i < -3:
                prev_c = df.iloc[i-1]
                next_c = df.iloc[i+1]
                if candle['low'] < prev_c['low'] and candle['low'] < next_c['low']:
                    swing_lows.append(candle['low'])
                if candle['high'] > prev_c['high'] and candle['high'] > next_c['high']:
                    swing_highs.append(candle['high'])

    if not swing_lows and not swing_highs:
        return None

    tol = atr * tolerance

    # Check for support bounce
    for support in swing_lows:
        if abs(low - support) <= tol and close > support:
            # Bounced off support
            if close > current['open']:  # Bullish candle
                stop_loss = support - (atr * atr_mult)
                if close - stop_loss < 3:
                    stop_loss = close - 3

                risk = close - stop_loss
                take_profit = close + (risk * rr)

                return {'direction': 'BUY', 'entry': close, 'stop': stop_loss,
                        'tp': take_profit, 'setup': 'S/R Bounce'}

    # Check for resistance rejection
    for resistance in swing_highs:
        if abs(high - resistance) <= tol and close < resistance:
            if close < current['open']:  # Bearish candle
                stop_loss = resistance + (atr * atr_mult)
                if stop_loss - close < 3:
                    stop_loss = close + 3

                risk = stop_loss - close
                take_profit = close - (risk * rr)

                return {'direction': 'SELL', 'entry': close, 'stop': stop_loss,
                        'tp': take_profit, 'setup': 'S/R Bounce'}

    return None


# ============================================================================
# STRATEGY 6: VOLATILITY BREAKOUT
# Trade when ATR spikes (volatility expansion) in direction of momentum
# ============================================================================

def strategy_volatility_breakout(df: pd.DataFrame, params: Dict) -> Optional[Dict]:
    """
    Trade volatility expansion with momentum
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

    lookback = params.get('atr_lookback', 20)
    atr_mult = params.get('atr_spike', 1.5)  # ATR must be this multiple of avg
    rr = params.get('rr', 2.0)

    # Calculate average ATR
    atrs = df['atr'].iloc[-lookback:].dropna()
    if len(atrs) < 10:
        return None

    avg_atr = atrs.mean()

    # ATR spike
    if atr > avg_atr * atr_mult:
        # Direction from momentum
        if rsi > 55 and close > ema_20:
            # Bullish volatility expansion
            stop_loss = close - atr
            if close - stop_loss < 3:
                stop_loss = close - 3
            if close - stop_loss > 25:
                stop_loss = close - 25

            risk = close - stop_loss
            take_profit = close + (risk * rr)

            return {'direction': 'BUY', 'entry': close, 'stop': stop_loss,
                    'tp': take_profit, 'setup': 'Vol Breakout'}

        elif rsi < 45 and close < ema_20:
            stop_loss = close + atr
            if stop_loss - close < 3:
                stop_loss = close + 3
            if stop_loss - close > 25:
                stop_loss = close + 25

            risk = stop_loss - close
            take_profit = close - (risk * rr)

            return {'direction': 'SELL', 'entry': close, 'stop': stop_loss,
                    'tp': take_profit, 'setup': 'Vol Breakout'}

    return None


def run_backtest(df: pd.DataFrame, strategy_func, params: Dict,
                 spread: float = 0.50, slippage: float = 0.30) -> Dict:
    """Run backtest with given strategy and parameters"""

    df_calc = add_all_indicators(df.copy())

    trades = []
    current_trade = None

    for i in range(60, len(df_calc)):
        hist = df_calc.iloc[:i+1].copy()
        candle = df_calc.iloc[i]
        time = candle['timestamp']

        # Check exit
        if current_trade:
            if current_trade.direction == "BUY":
                if candle['low'] <= current_trade.stop:
                    current_trade.close(current_trade.stop, time, "SL")
                    trades.append(current_trade)
                    current_trade = None
                elif candle['high'] >= current_trade.tp:
                    current_trade.close(current_trade.tp, time, "TP")
                    trades.append(current_trade)
                    current_trade = None
            else:
                if candle['high'] >= current_trade.stop:
                    current_trade.close(current_trade.stop, time, "SL")
                    trades.append(current_trade)
                    current_trade = None
                elif candle['low'] <= current_trade.tp:
                    current_trade.close(current_trade.tp, time, "TP")
                    trades.append(current_trade)
                    current_trade = None

        # Check for new signal
        if not current_trade:
            signal = strategy_func(hist, params)

            if signal and i + 1 < len(df_calc):
                next_candle = df_calc.iloc[i + 1]
                cost = spread + slippage

                if signal['direction'] == 'BUY':
                    entry = next_candle['open'] + cost
                else:
                    entry = next_candle['open'] - cost

                current_trade = Trade(
                    signal['direction'], entry, signal['stop'], signal['tp'],
                    next_candle['timestamp'], signal['setup']
                )

    if current_trade:
        last = df_calc.iloc[-1]
        current_trade.close(last['close'], last['timestamp'], "End")
        trades.append(current_trade)

    if not trades:
        return {'total_r': 0, 'trades': 0, 'win_rate': 0, 'avg_r': 0, 'max_dd': 0, 'pf': 0}

    winners = [t for t in trades if t.pnl > 0]
    losers = [t for t in trades if t.pnl <= 0]
    r_values = [t.r_multiple for t in trades]

    equity = [0]
    for t in trades:
        equity.append(equity[-1] + t.r_multiple)
    peak = 0
    max_dd = 0
    for e in equity:
        peak = max(peak, e)
        max_dd = max(max_dd, peak - e)

    gross_win = sum(t.pnl for t in winners) if winners else 0
    gross_loss = abs(sum(t.pnl for t in losers)) if losers else 0.001

    return {
        'total_r': sum(r_values),
        'trades': len(trades),
        'win_rate': len(winners) / len(trades),
        'avg_r': np.mean(r_values),
        'max_dd': max_dd,
        'pf': gross_win / gross_loss
    }


def main():
    df = fetch_data(months=6)
    print(f"Testing on {len(df)} candles: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")

    strategies = [
        ("EMA Crossover", strategy_ema_crossover, [
            {'min_rsi_buy': 50, 'max_rsi_sell': 50, 'atr_sl': 2.0, 'rr': 2.0},
            {'min_rsi_buy': 45, 'max_rsi_sell': 55, 'atr_sl': 2.0, 'rr': 2.0},
            {'min_rsi_buy': 50, 'max_rsi_sell': 50, 'atr_sl': 2.5, 'rr': 2.5},
            {'min_rsi_buy': 55, 'max_rsi_sell': 45, 'atr_sl': 2.0, 'rr': 3.0},
        ]),
        ("RSI Extreme", strategy_rsi_extreme, [
            {'oversold': 25, 'overbought': 75, 'atr_sl': 2.0, 'rr': 2.0},
            {'oversold': 20, 'overbought': 80, 'atr_sl': 2.0, 'rr': 2.0},
            {'oversold': 30, 'overbought': 70, 'atr_sl': 1.5, 'rr': 1.5},
            {'oversold': 25, 'overbought': 75, 'atr_sl': 2.5, 'rr': 3.0},
        ]),
        ("Trend Continuation", strategy_trend_continuation, [
            {'trend_strength': 0.5, 'atr_sl': 1.5, 'rr': 2.0},
            {'trend_strength': 0.3, 'atr_sl': 1.5, 'rr': 2.0},
            {'trend_strength': 0.7, 'atr_sl': 2.0, 'rr': 2.5},
            {'trend_strength': 0.5, 'atr_sl': 1.0, 'rr': 3.0},
        ]),
        ("BB Squeeze", strategy_bb_squeeze, [
            {'squeeze_lookback': 20, 'squeeze_threshold': 0.02, 'atr_sl': 2.0, 'rr': 2.0},
            {'squeeze_lookback': 30, 'squeeze_threshold': 0.015, 'atr_sl': 2.0, 'rr': 2.5},
            {'squeeze_lookback': 20, 'squeeze_threshold': 0.025, 'atr_sl': 1.5, 'rr': 2.0},
        ]),
        ("S/R Bounce", strategy_sr_bounce, [
            {'sr_lookback': 30, 'sr_tolerance': 0.5, 'atr_sl': 1.5, 'rr': 2.0},
            {'sr_lookback': 40, 'sr_tolerance': 0.3, 'atr_sl': 1.5, 'rr': 2.0},
            {'sr_lookback': 30, 'sr_tolerance': 0.5, 'atr_sl': 2.0, 'rr': 3.0},
        ]),
        ("Vol Breakout", strategy_volatility_breakout, [
            {'atr_lookback': 20, 'atr_spike': 1.5, 'rr': 2.0},
            {'atr_lookback': 20, 'atr_spike': 1.3, 'rr': 2.0},
            {'atr_lookback': 30, 'atr_spike': 1.5, 'rr': 3.0},
        ]),
    ]

    print("\n" + "=" * 90)
    print("                         STRATEGY OPTIMIZATION V2")
    print("                         6-Month Backtest Results")
    print("=" * 90)

    all_results = []

    for name, func, param_sets in strategies:
        print(f"\n--- {name} ---")
        for params in param_sets:
            result = run_backtest(df, func, params)
            result['name'] = name
            result['params'] = params
            all_results.append(result)

            status = "✓" if result['total_r'] > 0 else "✗"
            print(f"  {status} {str(params)[:50]:50s} | "
                  f"{result['trades']:3d} tr | WR: {result['win_rate']:5.1%} | "
                  f"R: {result['total_r']:+6.1f} | DD: {result['max_dd']:4.1f}")

    # Sort by total R
    all_results.sort(key=lambda x: x['total_r'], reverse=True)

    print("\n" + "=" * 90)
    print("                         TOP 10 CONFIGURATIONS")
    print("=" * 90)

    for i, r in enumerate(all_results[:10], 1):
        profit_1k = r['total_r'] * 10
        print(f"\n{i}. {r['name']}")
        print(f"   Params: {r['params']}")
        print(f"   Trades: {r['trades']} | Win Rate: {r['win_rate']:.1%} | "
              f"Total R: {r['total_r']:+.1f} | Max DD: {r['max_dd']:.1f}R | PF: {r['pf']:.2f}")
        print(f"   $1,000 @ 1%: ${profit_1k:+,.0f}")

    # Best profitable strategy
    profitable = [r for r in all_results if r['total_r'] > 0 and r['trades'] >= 10]

    print("\n" + "=" * 90)
    print("                         RECOMMENDATION")
    print("=" * 90)

    if profitable:
        best = profitable[0]
        print(f"\nBest Long-Term Strategy: {best['name']}")
        print(f"Parameters: {best['params']}")
        print(f"\n6-Month Performance:")
        print(f"  Trades:       {best['trades']}")
        print(f"  Win Rate:     {best['win_rate']:.1%}")
        print(f"  Total R:      {best['total_r']:+.1f}R")
        print(f"  Max Drawdown: {best['max_dd']:.1f}R")
        print(f"  Profit Factor: {best['pf']:.2f}")
        print(f"\n  $1,000 @ 1% risk: ${best['total_r'] * 10:+,.0f}")
        print(f"  $5,000 @ 1% risk: ${best['total_r'] * 50:+,.0f}")
    else:
        print("\nNo strategy with 10+ trades was profitable over 6 months.")
        print("Consider:")
        print("  1. Using discretionary filters on top of signals")
        print("  2. Trading only during high-probability sessions")
        print("  3. Reducing position size and accepting lower returns")

    return all_results


if __name__ == "__main__":
    main()
