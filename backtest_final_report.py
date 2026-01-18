"""
Final 6-Month Backtest Report
Shows detailed P&L with £1,000 starting capital
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from data import get_historical_candles_yahoo
from indicators import add_all_indicators
from strategy import StrategyEngine
import config


def calculate_position_size(account_balance: float, risk_pct: float, stop_distance: float, 
                           price: float) -> float:
    """
    Calculate position size based on risk management
    
    Args:
        account_balance: Current account size in GBP
        risk_pct: Risk percentage per trade (e.g., 0.01 for 1%)
        stop_distance: Distance to stop loss in USD
        price: Current gold price in USD
    
    Returns:
        Position size in ounces
    """
    # Convert to same currency (assume GBP/USD = 1.27)
    gbp_usd_rate = 1.27
    risk_amount_usd = account_balance * gbp_usd_rate * risk_pct
    
    # Position size = Risk Amount / Stop Distance
    position_size = risk_amount_usd / stop_distance if stop_distance > 0 else 0
    
    return position_size


def run_detailed_backtest():
    """
    Run complete 6-month backtest with detailed trade log
    """
    print("=" * 90)
    print("                    6-MONTH BACKTEST FINAL REPORT")
    print("                    Starting Capital: £1,000")
    print("                    Risk Per Trade: 1%")
    print("=" * 90)
    
    # Fetch data
    print("\nFetching 6 months of historical data...")
    df = get_historical_candles_yahoo(num_candles=3000)
    
    if df is None or len(df) < 100:
        print("Error: Could not fetch sufficient data")
        return
    
    df = add_all_indicators(df)
    
    # Filter to exactly 6 months
    end_date = df['timestamp'].max()
    start_date = end_date - timedelta(days=180)
    df = df[df['timestamp'] >= start_date].reset_index(drop=True)
    
    print(f"Data Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Total Candles: {len(df)}")
    
    # Initialize
    strategy = StrategyEngine()
    
    # Account tracking
    starting_balance = 1000.0  # £1,000
    account_balance = starting_balance
    risk_per_trade = 0.01  # 1%
    
    trades = []
    equity_curve = [starting_balance]
    
    print("\nRunning backtest...")
    
    # Simulate trading
    for i in range(100, len(df)):
        current_slice = df.iloc[:i+1].copy()
        
        # Get signal
        signal = strategy.evaluate(current_slice)
        
        if signal and signal['setup_quality'] >= config.MIN_SIGNAL_QUALITY:
            entry_price = signal['entry_price']
            stop_loss = signal['stop_loss']
            take_profit = signal['take_profit']
            direction = signal['direction']
            
            # Calculate position size
            stop_distance = abs(entry_price - stop_loss)
            position_size = calculate_position_size(account_balance, risk_per_trade, 
                                                   stop_distance, entry_price)
            
            # Simulate trade outcome
            # Look ahead to see if SL or TP hit first
            max_candles_ahead = min(50, len(df) - i - 1)
            outcome = None
            exit_price = None
            exit_candle = None
            
            for j in range(1, max_candles_ahead + 1):
                future_candle = df.iloc[i + j]
                
                if direction == 'BUY':
                    # Check if stop loss hit
                    if future_candle['low'] <= stop_loss:
                        outcome = 'LOSS'
                        exit_price = stop_loss
                        exit_candle = j
                        break
                    # Check if take profit hit
                    elif future_candle['high'] >= take_profit:
                        outcome = 'WIN'
                        exit_price = take_profit
                        exit_candle = j
                        break
                else:  # SELL
                    # Check if stop loss hit
                    if future_candle['high'] >= stop_loss:
                        outcome = 'LOSS'
                        exit_price = stop_loss
                        exit_candle = j
                        break
                    # Check if take profit hit
                    elif future_candle['low'] <= take_profit:
                        outcome = 'WIN'
                        exit_price = take_profit
                        exit_candle = j
                        break
            
            # If no outcome, assume stopped out at end
            if outcome is None:
                outcome = 'LOSS'
                exit_price = stop_loss
                exit_candle = max_candles_ahead
            
            # Calculate P&L
            if direction == 'BUY':
                price_change = exit_price - entry_price
            else:
                price_change = entry_price - exit_price
            
            # Apply spread/slippage
            spread_cost = 0.5  # $0.50 per ounce
            pnl_usd = (price_change * position_size) - (spread_cost * position_size)
            
            # Convert to GBP
            pnl_gbp = pnl_usd / 1.27
            
            # Update account
            account_balance += pnl_gbp
            equity_curve.append(account_balance)
            
            # Calculate R-multiple
            risk_amount_gbp = starting_balance * risk_per_trade
            r_multiple = pnl_gbp / risk_amount_gbp if risk_amount_gbp > 0 else 0
            
            # Record trade
            trades.append({
                'timestamp': df.iloc[i]['timestamp'],
                'strategy': signal['setup_type'],
                'direction': direction,
                'entry': entry_price,
                'exit': exit_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'outcome': outcome,
                'position_size': position_size,
                'pnl_usd': pnl_usd,
                'pnl_gbp': pnl_gbp,
                'r_multiple': r_multiple,
                'account_balance': account_balance,
                'candles_held': exit_candle
            })
    
    # Create trades DataFrame
    if not trades:
        print("\n⚠️  No trades generated in this period")
        return
    
    trades_df = pd.DataFrame(trades)
    
    # Calculate statistics
    total_trades = len(trades_df)
    winning_trades = len(trades_df[trades_df['outcome'] == 'WIN'])
    losing_trades = len(trades_df[trades_df['outcome'] == 'LOSS'])
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    
    total_pnl_gbp = account_balance - starting_balance
    total_pnl_pct = (total_pnl_gbp / starting_balance) * 100
    
    avg_win = trades_df[trades_df['outcome'] == 'WIN']['pnl_gbp'].mean() if winning_trades > 0 else 0
    avg_loss = trades_df[trades_df['outcome'] == 'LOSS']['pnl_gbp'].mean() if losing_trades > 0 else 0
    
    total_wins = trades_df[trades_df['outcome'] == 'WIN']['pnl_gbp'].sum()
    total_losses = abs(trades_df[trades_df['outcome'] == 'LOSS']['pnl_gbp'].sum())
    profit_factor = total_wins / total_losses if total_losses > 0 else 0
    
    avg_r = trades_df['r_multiple'].mean()
    total_r = trades_df['r_multiple'].sum()
    
    # Max drawdown
    equity_series = pd.Series(equity_curve)
    running_max = equity_series.expanding().max()
    drawdown = (equity_series - running_max) / running_max * 100
    max_drawdown = drawdown.min()
    
    # Print summary
    print("\n" + "=" * 90)
    print("                         PERFORMANCE SUMMARY")
    print("=" * 90)
    
    print(f"\n💰 ACCOUNT PERFORMANCE")
    print(f"   Starting Balance:     £{starting_balance:,.2f}")
    print(f"   Ending Balance:       £{account_balance:,.2f}")
    print(f"   Total P&L:            £{total_pnl_gbp:+,.2f} ({total_pnl_pct:+.2f}%)")
    print(f"   Max Drawdown:         {max_drawdown:.2f}%")
    
    print(f"\n📊 TRADE STATISTICS")
    print(f"   Total Trades:         {total_trades}")
    print(f"   Winning Trades:       {winning_trades} ({win_rate:.1f}%)")
    print(f"   Losing Trades:        {losing_trades} ({100-win_rate:.1f}%)")
    print(f"   Profit Factor:        {profit_factor:.2f}")
    
    print(f"\n💵 MONETARY METRICS")
    print(f"   Average Win:          £{avg_win:+,.2f}")
    print(f"   Average Loss:         £{avg_loss:+,.2f}")
    print(f"   Total Wins:           £{total_wins:+,.2f}")
    print(f"   Total Losses:         £{total_losses:+,.2f}")
    
    print(f"\n📈 RISK METRICS")
    print(f"   Risk Per Trade:       {risk_per_trade * 100:.1f}%")
    print(f"   Average R-Multiple:   {avg_r:+.2f}R")
    print(f"   Total R:              {total_r:+.2f}R")
    
    # Strategy breakdown
    print(f"\n🎯 STRATEGY BREAKDOWN")
    strategy_stats = trades_df.groupby('strategy').agg({
        'outcome': 'count',
        'pnl_gbp': 'sum',
        'r_multiple': 'sum'
    }).rename(columns={'outcome': 'trades'})
    
    for strategy_name, row in strategy_stats.iterrows():
        strategy_trades = trades_df[trades_df['strategy'] == strategy_name]
        strategy_wins = len(strategy_trades[strategy_trades['outcome'] == 'WIN'])
        strategy_wr = (strategy_wins / row['trades'] * 100) if row['trades'] > 0 else 0
        
        print(f"   {strategy_name:25} {int(row['trades']):3} trades | "
              f"WR: {strategy_wr:5.1f}% | P&L: £{row['pnl_gbp']:+7.2f} | "
              f"R: {row['r_multiple']:+6.2f}")
    
    # Monthly breakdown
    print(f"\n📅 MONTHLY BREAKDOWN")
    trades_df['month'] = pd.to_datetime(trades_df['timestamp']).dt.to_period('M')
    monthly = trades_df.groupby('month').agg({
        'pnl_gbp': 'sum',
        'outcome': 'count'
    }).rename(columns={'outcome': 'trades'})
    
    for month, row in monthly.iterrows():
        month_trades = trades_df[trades_df['month'] == month]
        month_wins = len(month_trades[month_trades['outcome'] == 'WIN'])
        month_wr = (month_wins / row['trades'] * 100) if row['trades'] > 0 else 0
        
        print(f"   {str(month):8} | {int(row['trades']):3} trades | "
              f"WR: {month_wr:5.1f}% | P&L: £{row['pnl_gbp']:+8.2f}")
    
    # Save trades to CSV
    trades_df.to_csv('backtest_trades_6month.csv', index=False)
    print(f"\n✓ Detailed trade log saved to: backtest_trades_6month.csv")
    
    # Best and worst trades
    print(f"\n🏆 BEST TRADE")
    best_trade = trades_df.loc[trades_df['pnl_gbp'].idxmax()]
    print(f"   {best_trade['timestamp']} | {best_trade['strategy']} {best_trade['direction']}")
    print(f"   Entry: ${best_trade['entry']:.2f} → Exit: ${best_trade['exit']:.2f}")
    print(f"   P&L: £{best_trade['pnl_gbp']:+.2f} ({best_trade['r_multiple']:+.2f}R)")
    
    print(f"\n💔 WORST TRADE")
    worst_trade = trades_df.loc[trades_df['pnl_gbp'].idxmin()]
    print(f"   {worst_trade['timestamp']} | {worst_trade['strategy']} {worst_trade['direction']}")
    print(f"   Entry: ${worst_trade['entry']:.2f} → Exit: ${worst_trade['exit']:.2f}")
    print(f"   P&L: £{worst_trade['pnl_gbp']:+.2f} ({worst_trade['r_multiple']:+.2f}R)")
    
    print("\n" + "=" * 90)
    print("                         CONCLUSION")
    print("=" * 90)
    
    if total_pnl_gbp > 0:
        roi_annual = (total_pnl_pct / 6) * 12
        print(f"\n✅ PROFITABLE SYSTEM")
        print(f"   6-Month Return: {total_pnl_pct:+.2f}%")
        print(f"   Annualized ROI: ~{roi_annual:+.2f}%")
        print(f"   With £{starting_balance:,.0f} → £{account_balance:,.2f}")
        print(f"   With £5,000 → £{5000 + (total_pnl_gbp * 5):,.2f}")
        print(f"   With £10,000 → £{10000 + (total_pnl_gbp * 10):,.2f}")
    else:
        print(f"\n⚠️  LOSS OVER 6 MONTHS")
        print(f"   Total Loss: £{total_pnl_gbp:,.2f} ({total_pnl_pct:.2f}%)")
        print(f"   Strategy needs further optimization")
    
    print("\n" + "=" * 90)


if __name__ == "__main__":
    run_detailed_backtest()
