"""
Test the optimized strategy configuration on 6-month data
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from indicators import add_all_indicators
from strategy import StrategyEngine


class Trade:
    def __init__(self, direction, entry, stop, tp, entry_time, setup):
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

    def close(self, price, time, reason):
        self.exit_price = price
        self.exit_time = time
        if self.direction == "BUY":
            self.pnl = price - self.entry
        else:
            self.pnl = self.entry - price
        self.r_multiple = self.pnl / self.risk if self.risk > 0 else 0


def main():
    # Load data
    cache_file = 'sample_data/gold_6months_1h.csv'
    if not os.path.exists(cache_file):
        print("Data file not found. Run strategy_optimizer.py first.")
        return

    df = pd.read_csv(cache_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    print("=" * 70)
    print("       OPTIMIZED STRATEGY BACKTEST")
    print("       Combined: S/R Bounce + Vol Breakout + EMA Cross")
    print("=" * 70)
    print(f"\nData: {len(df)} hourly candles")
    print(f"Period: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")

    # Add indicators
    df = add_all_indicators(df)

    # Run backtest
    engine = StrategyEngine()
    trades = []
    current_trade = None

    for i in range(60, len(df)):
        hist = df.iloc[:i+1].copy()
        candle = df.iloc[i]
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
            signal = engine.evaluate(hist)

            if signal and i + 1 < len(df):
                next_candle = df.iloc[i + 1]
                cost = 0.80  # spread + slippage

                if signal['direction'] == 'BUY':
                    entry = next_candle['open'] + cost
                else:
                    entry = next_candle['open'] - cost

                current_trade = Trade(
                    signal['direction'], entry, signal['stop_loss'],
                    signal['take_profit'], next_candle['timestamp'],
                    signal['setup_type']
                )

    if current_trade:
        last = df.iloc[-1]
        current_trade.close(last['close'], last['timestamp'], "End")
        trades.append(current_trade)

    # Calculate results
    if not trades:
        print("\nNo trades generated")
        return

    winners = [t for t in trades if t.pnl > 0]
    losers = [t for t in trades if t.pnl <= 0]
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

    # By setup
    setup_stats = {}
    for t in trades:
        if t.setup not in setup_stats:
            setup_stats[t.setup] = {'trades': 0, 'wins': 0, 'r': 0}
        setup_stats[t.setup]['trades'] += 1
        setup_stats[t.setup]['r'] += t.r_multiple
        if t.pnl > 0:
            setup_stats[t.setup]['wins'] += 1

    print(f"\n{'─' * 40}")
    print("OVERALL RESULTS")
    print(f"{'─' * 40}")
    print(f"  Total Trades:   {len(trades)}")
    print(f"  Winners:        {len(winners)}")
    print(f"  Losers:         {len(losers)}")
    print(f"  Win Rate:       {len(winners)/len(trades):.1%}")
    print(f"  Total R:        {sum(r_values):+.1f}R")
    print(f"  Average R:      {np.mean(r_values):+.2f}R")
    print(f"  Max Drawdown:   {max_dd:.1f}R")

    print(f"\n{'─' * 40}")
    print("BY STRATEGY")
    print(f"{'─' * 40}")
    for setup, stats in setup_stats.items():
        wr = stats['wins'] / stats['trades'] if stats['trades'] > 0 else 0
        print(f"  {setup:15s}: {stats['trades']:3d} trades, "
              f"{wr:5.1%} WR, {stats['r']:+6.1f}R")

    print(f"\n{'─' * 40}")
    print("PROFIT PROJECTIONS (1% risk per trade)")
    print(f"{'─' * 40}")
    total_r = sum(r_values)
    for bank in [1000, 5000, 10000]:
        profit = total_r * (bank * 0.01)
        pct = profit / bank * 100
        print(f"  ${bank:,} bank:    ${profit:+,.0f} ({pct:+.1f}%)")

    print("\n" + "=" * 70)
    print("DISCLAIMER: Past performance does not guarantee future results.")
    print("=" * 70)


if __name__ == "__main__":
    main()
