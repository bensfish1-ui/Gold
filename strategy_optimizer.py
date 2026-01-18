"""
Strategy Optimizer for Gold Trading Bot
Tests multiple strategy variations on 6-month historical data
to find the most profitable long-term configuration
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from indicators import add_all_indicators


@dataclass
class StrategyConfig:
    """Configuration for a strategy test"""
    name: str
    # Strategy enables
    trend_pullback: bool = False
    breakout_retest: bool = False
    mean_reversion: bool = False
    # Trend Pullback params
    pullback_atr_dist: float = 1.0
    pullback_rsi_min: int = 40
    pullback_rsi_max: int = 60
    # Mean Reversion params
    bb_period: int = 20
    bb_std: float = 2.0
    mr_sl_atr: float = 1.5
    mr_rr: float = 2.0
    rsi_oversold: int = 30
    rsi_overbought: int = 70
    require_rsi_extreme: bool = False
    # Breakout params
    breakout_range_candles: int = 20
    breakout_atr_thresh: float = 0.5
    # General
    atr_sl_mult: float = 1.5
    rr_ratio: float = 2.0
    min_stop: float = 3.0
    max_stop: float = 25.0


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
        self.exit_reason = ""
        self.pnl = 0.0
        self.r_multiple = 0.0

    @property
    def risk(self):
        return abs(self.entry - self.stop)

    def close(self, price: float, time: datetime, reason: str):
        self.exit_price = price
        self.exit_time = time
        self.exit_reason = reason
        if self.direction == "BUY":
            self.pnl = price - self.entry
        else:
            self.pnl = self.entry - price
        self.r_multiple = self.pnl / self.risk if self.risk > 0 else 0


def fetch_data(months: int = 6) -> pd.DataFrame:
    """Fetch historical data from Yahoo Finance"""
    print(f"Fetching {months} months of Gold data...")

    # Check if we already have the data
    cache_file = f'sample_data/gold_{months}months_1h.csv'
    if os.path.exists(cache_file):
        print(f"Loading cached data from {cache_file}")
        df = pd.read_csv(cache_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df

    symbol = 'GC=F'
    url = f'https://query1.finance.yahoo.com/v8/finance/chart/{symbol}'

    end_time = int(datetime.now().timestamp())
    start_time = int((datetime.now() - timedelta(days=months * 30)).timestamp())

    params = {'interval': '1h', 'period1': start_time, 'period2': end_time}
    headers = {'User-Agent': 'Mozilla/5.0'}

    response = requests.get(url, params=params, headers=headers, timeout=30)
    data = response.json()

    result = data['chart']['result'][0]
    timestamps = result['timestamp']
    quotes = result['indicators']['quote'][0]

    df = pd.DataFrame({
        'timestamp': [datetime.fromtimestamp(ts) for ts in timestamps],
        'open': quotes['open'],
        'high': quotes['high'],
        'low': quotes['low'],
        'close': quotes['close']
    })

    df = df.dropna().reset_index(drop=True)

    os.makedirs('sample_data', exist_ok=True)
    df.to_csv(cache_file, index=False)

    print(f"Fetched {len(df)} candles: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
    return df


def check_trend_pullback(df: pd.DataFrame, cfg: StrategyConfig) -> Optional[Dict]:
    """Check for trend pullback signal"""
    if len(df) < 60:
        return None

    current = df.iloc[-1]
    close = current['close']
    ema_20 = current['ema_20']
    ema_50 = current['ema_50']
    rsi = current['rsi']
    atr = current['atr']

    if any(pd.isna([close, ema_20, ema_50, rsi, atr])):
        return None

    # Determine trend
    if ema_20 > ema_50:
        trend = "BULLISH"
    elif ema_20 < ema_50:
        trend = "BEARISH"
    else:
        return None

    # Check pullback distance
    distance_to_ema = abs(close - ema_20) / atr if atr > 0 else 999
    if distance_to_ema > cfg.pullback_atr_dist:
        return None

    # BUY setup
    if trend == "BULLISH":
        if close <= ema_50:
            return None
        if not (cfg.pullback_rsi_min <= rsi <= cfg.pullback_rsi_max):
            return None

        # Check RSI rising
        if len(df) >= 3:
            if not (df.iloc[-1]['rsi'] > df.iloc[-3]['rsi']):
                return None

        # Check bullish candle
        body = current['close'] - current['open']
        if body <= 0:
            return None

        # Calculate levels
        stop_loss = close - (atr * cfg.atr_sl_mult)
        if close - stop_loss < cfg.min_stop:
            stop_loss = close - cfg.min_stop
        if close - stop_loss > cfg.max_stop:
            stop_loss = close - cfg.max_stop

        risk = close - stop_loss
        take_profit = close + (risk * cfg.rr_ratio)

        return {'direction': 'BUY', 'entry': close, 'stop': stop_loss,
                'tp': take_profit, 'setup': 'Trend Pullback'}

    # SELL setup
    elif trend == "BEARISH":
        if close >= ema_50:
            return None
        if not (cfg.pullback_rsi_min <= rsi <= cfg.pullback_rsi_max):
            return None

        if len(df) >= 3:
            if not (df.iloc[-1]['rsi'] < df.iloc[-3]['rsi']):
                return None

        body = current['close'] - current['open']
        if body >= 0:
            return None

        stop_loss = close + (atr * cfg.atr_sl_mult)
        if stop_loss - close < cfg.min_stop:
            stop_loss = close + cfg.min_stop
        if stop_loss - close > cfg.max_stop:
            stop_loss = close + cfg.max_stop

        risk = stop_loss - close
        take_profit = close - (risk * cfg.rr_ratio)

        return {'direction': 'SELL', 'entry': close, 'stop': stop_loss,
                'tp': take_profit, 'setup': 'Trend Pullback'}

    return None


def check_mean_reversion(df: pd.DataFrame, cfg: StrategyConfig) -> Optional[Dict]:
    """Check for mean reversion signal"""
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

    # BUY: Previous touched lower BB, current closes inside
    prev_touched_lower = prev['low'] <= prev['bb_lower']
    current_inside = close > bb_lower

    if prev_touched_lower and current_inside:
        # Optional RSI filter
        if cfg.require_rsi_extreme and rsi > cfg.rsi_oversold:
            return None

        stop_loss = bb_lower - (atr * cfg.mr_sl_atr)
        if close - stop_loss < cfg.min_stop:
            stop_loss = close - cfg.min_stop

        risk = close - stop_loss
        tp_rr = close + (risk * cfg.mr_rr)
        take_profit = min(bb_middle, tp_rr)
        if take_profit <= close:
            take_profit = tp_rr

        return {'direction': 'BUY', 'entry': close, 'stop': stop_loss,
                'tp': take_profit, 'setup': 'Mean Reversion'}

    # SELL: Previous touched upper BB, current closes inside
    prev_touched_upper = prev['high'] >= prev['bb_upper']
    current_inside_upper = close < bb_upper

    if prev_touched_upper and current_inside_upper:
        if cfg.require_rsi_extreme and rsi < cfg.rsi_overbought:
            return None

        stop_loss = bb_upper + (atr * cfg.mr_sl_atr)
        if stop_loss - close < cfg.min_stop:
            stop_loss = close + cfg.min_stop

        risk = stop_loss - close
        tp_rr = close - (risk * cfg.mr_rr)
        take_profit = max(bb_middle, tp_rr)
        if take_profit >= close:
            take_profit = tp_rr

        return {'direction': 'SELL', 'entry': close, 'stop': stop_loss,
                'tp': take_profit, 'setup': 'Mean Reversion'}

    return None


def check_breakout_retest(df: pd.DataFrame, cfg: StrategyConfig, state: Dict) -> Optional[Dict]:
    """Check for breakout + retest signal"""
    if len(df) < cfg.breakout_range_candles + 10:
        return None

    current = df.iloc[-1]
    close = current['close']
    atr = current['atr']

    if pd.isna(atr) or atr == 0:
        return None

    # Calculate range (excluding last candle)
    range_df = df.iloc[-(cfg.breakout_range_candles+1):-1]
    range_high = range_df['high'].max()
    range_low = range_df['low'].min()
    range_height = range_high - range_low

    threshold = atr * cfg.breakout_atr_thresh

    # Check for breakout
    if close > range_high + threshold:
        if state.get('pending') and state['pending']['dir'] == 'BUY':
            # Check retest
            candles_since = len(df) - state['pending']['idx'] - 1
            if candles_since <= 3:
                level = state['pending']['level']
                tolerance = atr * 0.3
                if current['low'] <= level + tolerance and close > level:
                    # Retest confirmed
                    stop_loss = level - (atr * 0.2)
                    if close - stop_loss < cfg.min_stop:
                        stop_loss = close - cfg.min_stop

                    risk = close - stop_loss
                    take_profit = close + (risk * cfg.rr_ratio)

                    state['pending'] = None
                    return {'direction': 'BUY', 'entry': close, 'stop': stop_loss,
                            'tp': take_profit, 'setup': 'Breakout Retest'}
            elif candles_since > 3:
                state['pending'] = None
        else:
            state['pending'] = {'dir': 'BUY', 'level': range_high, 'idx': len(df)-1}

    elif close < range_low - threshold:
        if state.get('pending') and state['pending']['dir'] == 'SELL':
            candles_since = len(df) - state['pending']['idx'] - 1
            if candles_since <= 3:
                level = state['pending']['level']
                tolerance = atr * 0.3
                if current['high'] >= level - tolerance and close < level:
                    stop_loss = level + (atr * 0.2)
                    if stop_loss - close < cfg.min_stop:
                        stop_loss = close + cfg.min_stop

                    risk = stop_loss - close
                    take_profit = close - (risk * cfg.rr_ratio)

                    state['pending'] = None
                    return {'direction': 'SELL', 'entry': close, 'stop': stop_loss,
                            'tp': take_profit, 'setup': 'Breakout Retest'}
            elif candles_since > 3:
                state['pending'] = None
        else:
            state['pending'] = {'dir': 'SELL', 'level': range_low, 'idx': len(df)-1}

    return None


def run_backtest(df: pd.DataFrame, cfg: StrategyConfig, spread: float = 0.50,
                 slippage: float = 0.30) -> Dict:
    """Run backtest with given configuration"""

    df = add_all_indicators(df.copy())

    trades = []
    current_trade = None
    breakout_state = {}

    for i in range(60, len(df)):
        hist = df.iloc[:i+1].copy()
        candle = df.iloc[i]
        time = candle['timestamp']

        # Check exit
        if current_trade:
            if current_trade.direction == "BUY":
                if candle['low'] <= current_trade.stop:
                    current_trade.close(current_trade.stop, time, "Stop Loss")
                    trades.append(current_trade)
                    current_trade = None
                elif candle['high'] >= current_trade.tp:
                    current_trade.close(current_trade.tp, time, "Take Profit")
                    trades.append(current_trade)
                    current_trade = None
            else:
                if candle['high'] >= current_trade.stop:
                    current_trade.close(current_trade.stop, time, "Stop Loss")
                    trades.append(current_trade)
                    current_trade = None
                elif candle['low'] <= current_trade.tp:
                    current_trade.close(current_trade.tp, time, "Take Profit")
                    trades.append(current_trade)
                    current_trade = None

        # Check for new signal
        if not current_trade:
            signal = None

            if cfg.trend_pullback:
                signal = check_trend_pullback(hist, cfg)

            if not signal and cfg.mean_reversion:
                signal = check_mean_reversion(hist, cfg)

            if not signal and cfg.breakout_retest:
                signal = check_breakout_retest(hist, cfg, breakout_state)

            if signal and i + 1 < len(df):
                next_candle = df.iloc[i + 1]
                cost = spread + slippage

                if signal['direction'] == 'BUY':
                    entry = next_candle['open'] + cost
                else:
                    entry = next_candle['open'] - cost

                current_trade = Trade(
                    signal['direction'], entry, signal['stop'], signal['tp'],
                    next_candle['timestamp'], signal['setup']
                )

    # Close any open trade
    if current_trade:
        last = df.iloc[-1]
        current_trade.close(last['close'], last['timestamp'], "End")
        trades.append(current_trade)

    # Calculate stats
    if not trades:
        return {'total_r': 0, 'trades': 0, 'win_rate': 0, 'avg_r': 0, 'max_dd': 0}

    winners = [t for t in trades if t.pnl > 0]
    r_values = [t.r_multiple for t in trades]

    # Max drawdown
    equity = [0]
    for t in trades:
        equity.append(equity[-1] + t.r_multiple)
    peak = 0
    max_dd = 0
    for e in equity:
        peak = max(peak, e)
        max_dd = max(max_dd, peak - e)

    return {
        'total_r': sum(r_values),
        'trades': len(trades),
        'win_rate': len(winners) / len(trades),
        'avg_r': np.mean(r_values),
        'max_dd': max_dd,
        'profit_factor': sum(t.pnl for t in winners) / abs(sum(t.pnl for t in trades if t.pnl < 0)) if any(t.pnl < 0 for t in trades) else float('inf'),
        'trades_list': trades
    }


def main():
    # Fetch data
    df = fetch_data(months=6)

    print(f"\nLoaded {len(df)} hourly candles")
    print(f"Period: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")

    # Define strategy configurations to test
    configs = [
        # Original Mean Reversion (current)
        StrategyConfig(
            name="MR Basic (Current)",
            mean_reversion=True,
            mr_sl_atr=1.5,
            mr_rr=2.0
        ),

        # Mean Reversion with RSI filter
        StrategyConfig(
            name="MR + RSI Filter",
            mean_reversion=True,
            mr_sl_atr=1.5,
            mr_rr=2.0,
            require_rsi_extreme=True,
            rsi_oversold=35,
            rsi_overbought=65
        ),

        # Mean Reversion with tighter stop
        StrategyConfig(
            name="MR Tight Stop (1.0 ATR)",
            mean_reversion=True,
            mr_sl_atr=1.0,
            mr_rr=2.0
        ),

        # Mean Reversion with wider stop
        StrategyConfig(
            name="MR Wide Stop (2.0 ATR)",
            mean_reversion=True,
            mr_sl_atr=2.0,
            mr_rr=2.0
        ),

        # Mean Reversion with 1.5R target
        StrategyConfig(
            name="MR 1.5R Target",
            mean_reversion=True,
            mr_sl_atr=1.5,
            mr_rr=1.5
        ),

        # Mean Reversion with 3R target
        StrategyConfig(
            name="MR 3R Target",
            mean_reversion=True,
            mr_sl_atr=1.5,
            mr_rr=3.0
        ),

        # Trend Pullback only
        StrategyConfig(
            name="Trend Pullback Only",
            trend_pullback=True,
            pullback_atr_dist=1.0,
            atr_sl_mult=1.5,
            rr_ratio=2.0
        ),

        # Trend Pullback with wider zone
        StrategyConfig(
            name="Trend Pullback Wide",
            trend_pullback=True,
            pullback_atr_dist=1.5,
            atr_sl_mult=1.5,
            rr_ratio=2.0
        ),

        # Trend Pullback with 3R
        StrategyConfig(
            name="Trend Pullback 3R",
            trend_pullback=True,
            pullback_atr_dist=1.0,
            atr_sl_mult=1.5,
            rr_ratio=3.0
        ),

        # Breakout + Retest
        StrategyConfig(
            name="Breakout Retest",
            breakout_retest=True,
            breakout_range_candles=20,
            breakout_atr_thresh=0.5,
            rr_ratio=2.0
        ),

        # Combined: Trend + MR
        StrategyConfig(
            name="Trend + MR Combined",
            trend_pullback=True,
            mean_reversion=True,
            mr_sl_atr=1.5,
            mr_rr=2.0,
            atr_sl_mult=1.5,
            rr_ratio=2.0
        ),

        # Trend Pullback with relaxed RSI
        StrategyConfig(
            name="Trend RSI 30-70",
            trend_pullback=True,
            pullback_atr_dist=1.0,
            pullback_rsi_min=30,
            pullback_rsi_max=70,
            atr_sl_mult=1.5,
            rr_ratio=2.0
        ),

        # Mean Reversion wider bands (2.5 std)
        StrategyConfig(
            name="MR Wide Bands (2.5σ)",
            mean_reversion=True,
            bb_std=2.5,
            mr_sl_atr=1.5,
            mr_rr=2.0
        ),

        # Mean Reversion with RSI + tight SL
        StrategyConfig(
            name="MR RSI + Tight SL",
            mean_reversion=True,
            mr_sl_atr=1.0,
            mr_rr=2.0,
            require_rsi_extreme=True,
            rsi_oversold=40,
            rsi_overbought=60
        ),

        # All strategies combined
        StrategyConfig(
            name="All Strategies",
            trend_pullback=True,
            mean_reversion=True,
            breakout_retest=True,
            mr_sl_atr=1.5,
            mr_rr=2.0,
            atr_sl_mult=1.5,
            rr_ratio=2.0
        ),
    ]

    print("\n" + "=" * 80)
    print("                    STRATEGY OPTIMIZATION RESULTS")
    print("                    6-Month Backtest (1-Hour Candles)")
    print("=" * 80)

    results = []

    for cfg in configs:
        result = run_backtest(df, cfg)
        result['name'] = cfg.name
        results.append(result)

        # Print progress
        status = "✓" if result['total_r'] > 0 else "✗"
        print(f"{status} {cfg.name:30s} | {result['trades']:3d} trades | "
              f"WR: {result['win_rate']:5.1%} | R: {result['total_r']:+7.1f}")

    # Sort by total R
    results.sort(key=lambda x: x['total_r'], reverse=True)

    print("\n" + "=" * 80)
    print("                    TOP PERFORMING STRATEGIES")
    print("=" * 80)

    for i, r in enumerate(results[:5], 1):
        print(f"\n{i}. {r['name']}")
        print(f"   Trades:        {r['trades']}")
        print(f"   Win Rate:      {r['win_rate']:.1%}")
        print(f"   Total R:       {r['total_r']:+.1f}R")
        print(f"   Avg R/Trade:   {r['avg_r']:+.2f}R")
        print(f"   Max Drawdown:  {r['max_dd']:.1f}R")
        print(f"   Profit Factor: {r['profit_factor']:.2f}")

        # Profit projection
        for bank in [1000, 5000]:
            profit = r['total_r'] * (bank * 0.01)
            print(f"   ${bank:,} @ 1%:   ${profit:+,.0f} ({profit/bank*100:+.1f}%)")

    # Find best config
    best = results[0]

    print("\n" + "=" * 80)
    print("                    RECOMMENDATION")
    print("=" * 80)

    if best['total_r'] > 0:
        print(f"\nBest Strategy: {best['name']}")
        print(f"Expected over 6 months: {best['total_r']:+.1f}R")
        print(f"On $1,000 bank at 1% risk: ${best['total_r'] * 10:+,.0f}")
    else:
        # Find any profitable one
        profitable = [r for r in results if r['total_r'] > 0]
        if profitable:
            print(f"\nMost Profitable: {profitable[0]['name']}")
        else:
            print("\nNo profitable strategy found in 6-month backtest.")
            print("Consider: reducing trade frequency, wider stops, or manual discretion.")

    print("\n" + "=" * 80)

    # Return best for further analysis
    return results


if __name__ == "__main__":
    results = main()
