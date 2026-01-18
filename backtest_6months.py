"""
6-Month Backtest Script for Gold Trading Strategy
Uses 1-hour candles from Yahoo Finance (available for up to 2 years)
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from indicators import add_all_indicators
from strategy import StrategyEngine


def fetch_yahoo_historical(months: int = 6) -> pd.DataFrame:
    """
    Fetch historical 1-hour candles from Yahoo Finance

    Args:
        months: Number of months of data to fetch

    Returns:
        DataFrame with OHLC data
    """
    print(f"Fetching {months} months of Gold data from Yahoo Finance...")

    symbol = 'GC=F'  # Gold Futures
    url = f'https://query1.finance.yahoo.com/v8/finance/chart/{symbol}'

    # Calculate timestamps
    end_time = int(datetime.now().timestamp())
    start_time = int((datetime.now() - timedelta(days=months * 30)).timestamp())

    params = {
        'interval': '1h',
        'period1': start_time,
        'period2': end_time
    }

    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

    try:
        response = requests.get(url, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()

        if 'chart' not in data or not data['chart']['result']:
            print("Error: No data returned from Yahoo Finance")
            return None

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

        # Remove NaN rows
        df = df.dropna()
        df = df.reset_index(drop=True)

        print(f"Fetched {len(df)} hourly candles")
        print(f"Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")

        return df

    except Exception as e:
        print(f"Error fetching data: {e}")
        return None


class Trade:
    """Represents a single backtested trade"""

    def __init__(self, signal: dict, entry_price: float, entry_time: datetime):
        self.direction = signal['direction']
        self.setup_type = signal['setup_type']
        self.entry_price = entry_price
        self.entry_time = entry_time
        self.stop_loss = signal['stop_loss']
        self.take_profit = signal['take_profit']
        self.quality = signal['quality']

        self.exit_price = None
        self.exit_time = None
        self.exit_reason = ""
        self.pnl = 0.0
        self.r_multiple = 0.0

    @property
    def risk(self) -> float:
        return abs(self.entry_price - self.stop_loss)

    def close(self, exit_price: float, exit_time: datetime, reason: str):
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.exit_reason = reason

        if self.direction == "BUY":
            self.pnl = exit_price - self.entry_price
        else:
            self.pnl = self.entry_price - exit_price

        if self.risk > 0:
            self.r_multiple = self.pnl / self.risk
        else:
            self.r_multiple = 0.0

    @property
    def is_winner(self) -> bool:
        return self.pnl > 0


def run_backtest(df: pd.DataFrame, spread: float = 0.50, slippage: float = 0.30):
    """Run backtest on historical data"""

    print(f"\nRunning backtest on {len(df)} candles...")
    print(f"Spread: ${spread}, Slippage: ${slippage}")

    # Add indicators
    df = add_all_indicators(df)

    strategy = StrategyEngine()
    trades = []
    signals_generated = 0
    current_trade = None

    # Walk through each candle
    for i in range(60, len(df)):
        historical = df.iloc[:i+1].copy()
        current_candle = df.iloc[i]
        current_time = current_candle['timestamp']

        # Check for exit if we have an open trade
        if current_trade is not None:
            if current_trade.direction == "BUY":
                if current_candle['low'] <= current_trade.stop_loss:
                    current_trade.close(current_trade.stop_loss, current_time, "Stop Loss")
                    trades.append(current_trade)
                    current_trade = None
                elif current_candle['high'] >= current_trade.take_profit:
                    current_trade.close(current_trade.take_profit, current_time, "Take Profit")
                    trades.append(current_trade)
                    current_trade = None
            else:  # SELL
                if current_candle['high'] >= current_trade.stop_loss:
                    current_trade.close(current_trade.stop_loss, current_time, "Stop Loss")
                    trades.append(current_trade)
                    current_trade = None
                elif current_candle['low'] <= current_trade.take_profit:
                    current_trade.close(current_trade.take_profit, current_time, "Take Profit")
                    trades.append(current_trade)
                    current_trade = None

        # Check for new signal if no open trade
        if current_trade is None:
            signal = strategy.evaluate(historical)

            if signal is not None:
                signals_generated += 1

                if i + 1 < len(df):
                    next_candle = df.iloc[i + 1]
                    total_cost = spread + slippage

                    if signal['direction'] == "BUY":
                        entry_price = next_candle['open'] + total_cost
                    else:
                        entry_price = next_candle['open'] - total_cost

                    entry_time = next_candle['timestamp']
                    current_trade = Trade(signal, entry_price, entry_time)

    # Close any remaining trade
    if current_trade is not None:
        last_candle = df.iloc[-1]
        current_trade.close(last_candle['close'], last_candle['timestamp'], "End of data")
        trades.append(current_trade)

    return trades, signals_generated


def print_results(trades: list, signals_generated: int):
    """Print backtest results"""

    print("\n" + "=" * 70)
    print("                    6-MONTH BACKTEST RESULTS")
    print("                    Mean Reversion Strategy")
    print("=" * 70)

    print(f"\nSignals Generated: {signals_generated}")
    print(f"Trades Executed:   {len(trades)}")

    if not trades:
        print("\nNo trades to analyze")
        return {}

    winners = [t for t in trades if t.is_winner]
    losers = [t for t in trades if not t.is_winner]

    win_rate = len(winners) / len(trades) if trades else 0

    r_values = [t.r_multiple for t in trades]
    avg_r = np.mean(r_values)
    total_r = sum(r_values)

    gross_profit = sum(t.pnl for t in winners) if winners else 0
    gross_loss = abs(sum(t.pnl for t in losers)) if losers else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Calculate max drawdown
    equity = [0]
    for t in trades:
        equity.append(equity[-1] + t.r_multiple)

    peak = 0
    max_dd = 0
    for e in equity:
        peak = max(peak, e)
        dd = peak - e
        max_dd = max(max_dd, dd)

    print(f"\n{'─' * 40}")
    print("PERFORMANCE METRICS")
    print(f"{'─' * 40}")
    print(f"  Win Rate:        {win_rate:.1%}")
    print(f"  Winners:         {len(winners)}")
    print(f"  Losers:          {len(losers)}")
    print(f"  Average R:       {avg_r:+.2f}R")
    print(f"  Total R:         {total_r:+.2f}R")
    print(f"  Profit Factor:   {profit_factor:.2f}")

    print(f"\n{'─' * 40}")
    print("RISK METRICS")
    print(f"{'─' * 40}")
    print(f"  Max Drawdown:    {max_dd:.2f}R")

    print(f"\n{'─' * 40}")
    print("PROFIT PROJECTIONS (1% risk per trade)")
    print(f"{'─' * 40}")

    for bank in [1000, 5000, 10000]:
        risk_per_trade = bank * 0.01
        profit = total_r * risk_per_trade
        print(f"  ${bank:,} bank:     ${profit:+,.2f} ({profit/bank*100:+.1f}%)")

    # Monthly breakdown
    print(f"\n{'─' * 40}")
    print("MONTHLY BREAKDOWN")
    print(f"{'─' * 40}")

    monthly = {}
    for t in trades:
        month_key = t.entry_time.strftime('%Y-%m')
        if month_key not in monthly:
            monthly[month_key] = {'trades': 0, 'r': 0, 'wins': 0}
        monthly[month_key]['trades'] += 1
        monthly[month_key]['r'] += t.r_multiple
        if t.is_winner:
            monthly[month_key]['wins'] += 1

    for month, stats in sorted(monthly.items()):
        wr = stats['wins'] / stats['trades'] if stats['trades'] > 0 else 0
        print(f"  {month}: {stats['trades']:3d} trades, {wr:5.1%} WR, {stats['r']:+6.1f}R")

    print("\n" + "=" * 70)
    print("DISCLAIMER: Past performance does not guarantee future results.")
    print("These are educational backtests on historical data only.")
    print("=" * 70)

    return {
        'total_trades': len(trades),
        'win_rate': win_rate,
        'total_r': total_r,
        'avg_r': avg_r,
        'max_drawdown': max_dd,
        'profit_factor': profit_factor
    }


def main():
    # Fetch 6 months of data
    df = fetch_yahoo_historical(months=6)

    if df is None or len(df) < 100:
        print("Failed to fetch sufficient data")
        return

    # Save data for reference
    os.makedirs('sample_data', exist_ok=True)
    df.to_csv('sample_data/gold_6months_1h.csv', index=False)
    print(f"Saved data to sample_data/gold_6months_1h.csv")

    # Run backtest
    trades, signals = run_backtest(df)

    # Print results
    results = print_results(trades, signals)

    # Export trades
    if trades:
        os.makedirs('logs', exist_ok=True)
        trade_data = []
        for t in trades:
            trade_data.append({
                'entry_time': t.entry_time,
                'exit_time': t.exit_time,
                'direction': t.direction,
                'entry_price': round(t.entry_price, 2),
                'stop_loss': t.stop_loss,
                'take_profit': t.take_profit,
                'exit_price': round(t.exit_price, 2) if t.exit_price else '',
                'exit_reason': t.exit_reason,
                'pnl': round(t.pnl, 2),
                'r_multiple': round(t.r_multiple, 2),
                'winner': 'Y' if t.is_winner else 'N'
            })

        trade_df = pd.DataFrame(trade_data)
        trade_df.to_csv('logs/backtest_6months_trades.csv', index=False)
        print(f"\nTrade log saved to logs/backtest_6months_trades.csv")


if __name__ == "__main__":
    main()
