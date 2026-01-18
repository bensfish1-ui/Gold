"""
Backtesting Module for Gold Trading Bot

Runs strategy over historical 5-minute candle data and produces:
- Number of signals
- Win rate
- Average R (risk-adjusted return)
- Maximum drawdown (approximate)
- Sample trades log CSV

Usage:
    python backtest.py [csv_file] [--start YYYY-MM-DD] [--end YYYY-MM-DD]

CSV format required:
    timestamp,open,high,low,close
    2024-01-01 00:00:00,2050.00,2051.50,2049.00,2050.50
    ...
"""

import argparse
import csv
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

import config
from indicators import add_all_indicators
from strategy import StrategyEngine


class Trade:
    """Represents a single backtested trade"""

    def __init__(self, signal: Dict[str, Any], entry_price: float, entry_time: datetime):
        self.direction = signal['direction']
        self.setup_type = signal['setup_type']
        self.signal_entry = signal['entry']
        self.entry_price = entry_price  # Actual fill price (with spread/slippage)
        self.entry_time = entry_time
        self.stop_loss = signal['stop_loss']
        self.take_profit = signal['take_profit']
        self.quality = signal['quality']

        self.exit_price: Optional[float] = None
        self.exit_time: Optional[datetime] = None
        self.exit_reason: str = ""
        self.pnl: float = 0.0
        self.r_multiple: float = 0.0

    @property
    def risk(self) -> float:
        """Calculate risk in price terms"""
        return abs(self.entry_price - self.stop_loss)

    def close(self, exit_price: float, exit_time: datetime, reason: str):
        """Close the trade and calculate P&L"""
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.exit_reason = reason

        if self.direction == "BUY":
            self.pnl = exit_price - self.entry_price
        else:
            self.pnl = self.entry_price - exit_price

        # Calculate R-multiple
        if self.risk > 0:
            self.r_multiple = self.pnl / self.risk
        else:
            self.r_multiple = 0.0

    @property
    def is_winner(self) -> bool:
        return self.pnl > 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for CSV export"""
        return {
            'entry_time': self.entry_time.isoformat() if self.entry_time else '',
            'exit_time': self.exit_time.isoformat() if self.exit_time else '',
            'direction': self.direction,
            'setup_type': self.setup_type,
            'quality': self.quality,
            'signal_entry': self.signal_entry,
            'entry_price': round(self.entry_price, 2),
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'exit_price': round(self.exit_price, 2) if self.exit_price else '',
            'exit_reason': self.exit_reason,
            'pnl': round(self.pnl, 2),
            'r_multiple': round(self.r_multiple, 2),
            'winner': 'Y' if self.is_winner else 'N'
        }


class Backtester:
    """Main backtesting engine"""

    def __init__(self, spread: float = None, slippage: float = None,
                 fill_type: str = None):
        self.spread = spread if spread is not None else config.BACKTEST_SPREAD
        self.slippage = slippage if slippage is not None else config.BACKTEST_SLIPPAGE
        self.fill_type = fill_type if fill_type is not None else config.BACKTEST_FILL_TYPE

        self.strategy = StrategyEngine()
        self.trades: List[Trade] = []
        self.signals_generated = 0
        self.current_trade: Optional[Trade] = None

    def load_data(self, filepath: str) -> Optional[pd.DataFrame]:
        """
        Load historical data from CSV file

        Expected columns: timestamp, open, high, low, close
        """
        if not os.path.exists(filepath):
            print(f"Error: File not found: {filepath}")
            return None

        try:
            df = pd.read_csv(filepath)

            # Validate required columns
            required_cols = ['timestamp', 'open', 'high', 'low', 'close']
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                # Try alternative column names
                col_map = {
                    'date': 'timestamp', 'time': 'timestamp', 'datetime': 'timestamp',
                    'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close'
                }
                df = df.rename(columns=col_map)

                missing = [c for c in required_cols if c not in df.columns]
                if missing:
                    print(f"Error: Missing columns: {missing}")
                    return None

            # Parse timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            # Sort by timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)

            # Ensure numeric columns
            for col in ['open', 'high', 'low', 'close']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Remove NaN rows
            df = df.dropna()

            print(f"Loaded {len(df)} candles from {filepath}")
            print(f"Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")

            return df

        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def run(self, df: pd.DataFrame, start_date: str = None,
            end_date: str = None) -> Dict[str, Any]:
        """
        Run backtest over the data

        Args:
            df: DataFrame with OHLC data
            start_date: optional start date filter (YYYY-MM-DD)
            end_date: optional end date filter (YYYY-MM-DD)

        Returns:
            dict with backtest results
        """
        # Filter date range
        if start_date:
            start_dt = pd.to_datetime(start_date)
            df = df[df['timestamp'] >= start_dt]

        if end_date:
            end_dt = pd.to_datetime(end_date)
            df = df[df['timestamp'] <= end_dt]

        if len(df) < 100:
            print("Error: Need at least 100 candles for backtesting")
            return {}

        print(f"\nRunning backtest on {len(df)} candles...")
        print(f"Spread: ${self.spread}, Slippage: ${self.slippage}")
        print(f"Fill type: {self.fill_type}")

        # Add indicators
        df = add_all_indicators(df)

        # Reset state
        self.trades = []
        self.signals_generated = 0
        self.current_trade = None
        self.strategy = StrategyEngine()  # Reset strategy state

        # Track equity for drawdown
        equity = [0.0]
        peak_equity = 0.0
        max_drawdown = 0.0

        # Walk through each candle (start from candle 60 to ensure indicators are valid)
        for i in range(60, len(df)):
            # Get historical data up to current candle (inclusive)
            historical = df.iloc[:i+1].copy()
            current_candle = df.iloc[i]
            current_time = current_candle['timestamp']

            # If we have an open trade, check for exit
            if self.current_trade is not None:
                exit_triggered = self._check_exit(current_candle, current_time)
                if exit_triggered:
                    # Update equity
                    equity.append(equity[-1] + self.current_trade.pnl)

                    # Track drawdown
                    peak_equity = max(peak_equity, equity[-1])
                    drawdown = peak_equity - equity[-1]
                    max_drawdown = max(max_drawdown, drawdown)

                    self.current_trade = None

            # If no open trade, check for new signal
            if self.current_trade is None:
                signal = self.strategy.evaluate(historical)

                if signal is not None:
                    self.signals_generated += 1

                    # Get next candle for fill (if available)
                    if i + 1 < len(df):
                        next_candle = df.iloc[i + 1]

                        # Calculate entry price with spread/slippage
                        entry_price = self._calculate_entry_price(
                            signal, current_candle, next_candle
                        )

                        if entry_price is not None:
                            # Open trade
                            entry_time = next_candle['timestamp']
                            self.current_trade = Trade(signal, entry_price, entry_time)

        # Close any remaining open trade at last price
        if self.current_trade is not None:
            last_candle = df.iloc[-1]
            self.current_trade.close(
                last_candle['close'],
                last_candle['timestamp'],
                "End of data"
            )
            self.trades.append(self.current_trade)
            equity.append(equity[-1] + self.current_trade.pnl)

        # Calculate statistics
        results = self._calculate_statistics(equity, max_drawdown)

        return results

    def _calculate_entry_price(self, signal: Dict[str, Any],
                                current_candle: pd.Series,
                                next_candle: pd.Series) -> Optional[float]:
        """
        Calculate entry price based on fill type with spread/slippage

        Args:
            signal: signal dict
            current_candle: candle that generated the signal
            next_candle: next candle for fill

        Returns:
            fill price or None if fill would be invalid
        """
        if self.fill_type == "signal_close":
            base_price = current_candle['close']
        else:  # next_open
            base_price = next_candle['open']

        # Apply spread and slippage
        total_cost = self.spread + self.slippage

        if signal['direction'] == "BUY":
            entry_price = base_price + total_cost
        else:
            entry_price = base_price - total_cost

        # Validate entry makes sense vs stop loss
        if signal['direction'] == "BUY":
            if entry_price >= signal['stop_loss']:
                return entry_price
        else:
            if entry_price <= signal['stop_loss']:
                return entry_price

        return entry_price  # Allow it anyway, might be slippage issue

    def _check_exit(self, candle: pd.Series, candle_time: datetime) -> bool:
        """
        Check if current candle triggers exit

        Args:
            candle: current OHLC candle
            candle_time: timestamp of candle

        Returns:
            True if trade was closed
        """
        trade = self.current_trade

        if trade.direction == "BUY":
            # Check stop loss (low touches or crosses SL)
            if candle['low'] <= trade.stop_loss:
                trade.close(trade.stop_loss, candle_time, "Stop Loss")
                self.trades.append(trade)
                return True

            # Check take profit (high touches or crosses TP)
            if candle['high'] >= trade.take_profit:
                trade.close(trade.take_profit, candle_time, "Take Profit")
                self.trades.append(trade)
                return True

        else:  # SELL
            # Check stop loss (high touches or crosses SL)
            if candle['high'] >= trade.stop_loss:
                trade.close(trade.stop_loss, candle_time, "Stop Loss")
                self.trades.append(trade)
                return True

            # Check take profit (low touches or crosses TP)
            if candle['low'] <= trade.take_profit:
                trade.close(trade.take_profit, candle_time, "Take Profit")
                self.trades.append(trade)
                return True

        return False

    def _calculate_statistics(self, equity: List[float],
                               max_drawdown: float) -> Dict[str, Any]:
        """Calculate backtest statistics"""
        total_trades = len(self.trades)

        if total_trades == 0:
            return {
                'signals_generated': self.signals_generated,
                'trades_taken': 0,
                'win_rate': 0,
                'avg_r': 0,
                'total_r': 0,
                'max_drawdown': 0,
                'profit_factor': 0
            }

        winners = [t for t in self.trades if t.is_winner]
        losers = [t for t in self.trades if not t.is_winner]

        win_count = len(winners)
        win_rate = win_count / total_trades if total_trades > 0 else 0

        # R-multiples
        r_values = [t.r_multiple for t in self.trades]
        avg_r = np.mean(r_values) if r_values else 0
        total_r = sum(r_values)

        # Profit factor
        gross_profit = sum(t.pnl for t in winners) if winners else 0
        gross_loss = abs(sum(t.pnl for t in losers)) if losers else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # By setup type
        setup_stats = {}
        for setup_type in set(t.setup_type for t in self.trades):
            setup_trades = [t for t in self.trades if t.setup_type == setup_type]
            setup_winners = [t for t in setup_trades if t.is_winner]
            setup_stats[setup_type] = {
                'count': len(setup_trades),
                'win_rate': len(setup_winners) / len(setup_trades) if setup_trades else 0,
                'avg_r': np.mean([t.r_multiple for t in setup_trades]) if setup_trades else 0
            }

        return {
            'signals_generated': self.signals_generated,
            'trades_taken': total_trades,
            'winners': win_count,
            'losers': len(losers),
            'win_rate': win_rate,
            'avg_r': avg_r,
            'total_r': total_r,
            'max_drawdown': max_drawdown,
            'max_drawdown_r': max_drawdown / (self.trades[0].risk if self.trades else 1),
            'profit_factor': profit_factor,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'final_equity': equity[-1] if equity else 0,
            'setup_breakdown': setup_stats
        }

    def export_trades(self, filepath: str):
        """Export trades to CSV file"""
        if not self.trades:
            print("No trades to export")
            return

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.trades[0].to_dict().keys())
            writer.writeheader()
            for trade in self.trades:
                writer.writerow(trade.to_dict())

        print(f"Exported {len(self.trades)} trades to {filepath}")

    def print_results(self, results: Dict[str, Any]):
        """Print formatted backtest results"""
        print("\n" + "=" * 60)
        print("BACKTEST RESULTS")
        print("=" * 60)

        print(f"\nSignals Generated: {results['signals_generated']}")
        print(f"Trades Taken:      {results['trades_taken']}")

        if results['trades_taken'] > 0:
            print(f"\nPerformance:")
            print(f"  Win Rate:        {results['win_rate']:.1%}")
            print(f"  Winners:         {results['winners']}")
            print(f"  Losers:          {results['losers']}")
            print(f"  Average R:       {results['avg_r']:.2f}R")
            print(f"  Total R:         {results['total_r']:.2f}R")
            print(f"  Profit Factor:   {results['profit_factor']:.2f}")

            print(f"\nRisk:")
            print(f"  Max Drawdown:    ${results['max_drawdown']:.2f}")
            print(f"  Max DD (R):      {results.get('max_drawdown_r', 0):.2f}R")

            print(f"\nBy Setup Type:")
            for setup, stats in results.get('setup_breakdown', {}).items():
                print(f"  {setup}:")
                print(f"    Trades: {stats['count']}, "
                      f"Win Rate: {stats['win_rate']:.1%}, "
                      f"Avg R: {stats['avg_r']:.2f}")

        print("\n" + "=" * 60)


def generate_sample_data(filepath: str = "sample_data/xauusd_m5.csv",
                         candles: int = 5000):
    """
    Generate synthetic sample data for testing

    Args:
        filepath: output file path
        candles: number of candles to generate
    """
    print(f"Generating {candles} synthetic candles...")

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    np.random.seed(42)

    base_price = 2000.0
    volatility = 0.0003  # ~0.03% per candle

    timestamps = pd.date_range(
        start='2024-01-01 00:00:00',
        periods=candles,
        freq='5min'
    )

    prices = [base_price]
    for i in range(1, candles):
        # Random walk with mean reversion
        change = np.random.randn() * volatility * prices[-1]
        trend = (base_price - prices[-1]) * 0.001  # Mean reversion
        prices.append(prices[-1] + change + trend)

    # Generate OHLC from prices
    data = []
    for i, (ts, price) in enumerate(zip(timestamps, prices)):
        range_mult = np.random.uniform(0.5, 1.5)
        range_size = abs(np.random.randn()) * 2 * range_mult

        open_price = price + np.random.uniform(-1, 1)
        close_price = price
        high = max(open_price, close_price) + np.random.uniform(0, range_size)
        low = min(open_price, close_price) - np.random.uniform(0, range_size)

        data.append({
            'timestamp': ts.strftime('%Y-%m-%d %H:%M:%S'),
            'open': round(open_price, 2),
            'high': round(high, 2),
            'low': round(low, 2),
            'close': round(close_price, 2)
        })

    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)
    print(f"Saved to {filepath}")

    return filepath


def main():
    parser = argparse.ArgumentParser(
        description='Backtest Gold Trading Strategy',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python backtest.py sample_data/xauusd_m5.csv
  python backtest.py data.csv --start 2024-01-01 --end 2024-06-30
  python backtest.py --generate  # Generate sample data first
        """
    )

    parser.add_argument('csv_file', nargs='?', default=config.BACKTEST_DATA_FILE,
                        help='Path to CSV file with historical data')
    parser.add_argument('--start', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', help='End date (YYYY-MM-DD)')
    parser.add_argument('--spread', type=float, help=f'Spread in USD (default: {config.BACKTEST_SPREAD})')
    parser.add_argument('--slippage', type=float, help=f'Slippage in USD (default: {config.BACKTEST_SLIPPAGE})')
    parser.add_argument('--output', default='logs/backtest_trades.csv',
                        help='Output file for trade log')
    parser.add_argument('--generate', action='store_true',
                        help='Generate sample data before running backtest')

    args = parser.parse_args()

    # Generate sample data if requested or file doesn't exist
    if args.generate or not os.path.exists(args.csv_file):
        if not os.path.exists(args.csv_file):
            print(f"Data file not found: {args.csv_file}")
            print("Generating sample data...")

        generate_sample_data(args.csv_file)

    # Create backtester
    backtester = Backtester(
        spread=args.spread,
        slippage=args.slippage
    )

    # Load data
    df = backtester.load_data(args.csv_file)
    if df is None:
        return

    # Run backtest
    results = backtester.run(df, start_date=args.start, end_date=args.end)

    if not results:
        print("Backtest failed")
        return

    # Print results
    backtester.print_results(results)

    # Export trades
    backtester.export_trades(args.output)

    print(f"\nTrade log saved to: {args.output}")
    print("\n*** DISCLAIMER: These are educational backtests only. ***")
    print("*** Past performance does not guarantee future results. ***")


if __name__ == "__main__":
    main()
