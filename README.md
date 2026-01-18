# Gold (XAUUSD) Trading Signal Bot

Automated trading signal bot that monitors Gold prices on 5-minute candles and sends Telegram alerts for high-probability **educational** trade setups.

> **DISCLAIMER**: This bot generates **educational signals only**. It is not financial advice. Trading involves substantial risk of loss. Past performance does not guarantee future results. Always do your own research and never risk more than you can afford to lose.

## Features

- **5-Minute Candle Analysis** - Scans at each candle close (no repainting)
- **Three Trading Strategies**:
  - **Trend Pullback** (Primary) - EMA-based trend following with rejection entries
  - **Breakout + Retest** (Secondary) - Range breakout with retest confirmation
  - **Mean Reversion** (Optional) - Bollinger Band fades in ranging markets
- **Quality Filtering** - Only alerts on high-probability setups (configurable threshold)
- **Risk Management** - ATR-based stops, structure-based targets, R:R calculations
- **Guardrails**:
  - Session filters (London/NY)
  - News blackout windows
  - Volatility sanity checks (chop/chaos detection)
  - Cooldown between signals
- **Backtesting** - Test strategy on historical data with detailed statistics
- **Telegram Alerts** - Instant notifications with complete trade plans

## Project Structure

```
gold/
├── bot.py              # Main bot loop (candle-aligned)
├── strategy.py         # StrategyEngine with 3 modules
├── indicators.py       # EMA, RSI, ATR, Bollinger Bands
├── data.py             # Price fetching from API
├── utils.py            # Time alignment, state, formatting
├── config.py           # All configuration settings
├── backtest.py         # Historical testing module
├── state.json          # Persistent bot state
├── requirements.txt    # Python dependencies
├── logs/               # Log files and trade records
│   └── .gitkeep
├── sample_data/        # Historical data for backtesting
│   └── .gitkeep
└── README.md           # This file
```

## Quick Start

### 1. Install Dependencies

```bash
cd gold
pip install -r requirements.txt
```

### 2. Configure API Keys

Edit `config.py` and add your credentials:

```python
# Metal Price API (get free key from https://metalpriceapi.com/)
METAL_PRICE_API_KEY = "your_api_key_here"

# Telegram (from @BotFather)
TELEGRAM_BOT_TOKEN = "your_bot_token_here"
TELEGRAM_CHAT_ID = "your_chat_id_here"
```

#### Getting Telegram Chat ID

1. Create a bot via @BotFather on Telegram
2. Start a chat with your bot (send `/start`)
3. Visit: `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates`
4. Look for `"chat":{"id":YOUR_CHAT_ID}`

### 3. Run Live Mode

```bash
python bot.py
```

The bot will:
- Wait for each 5-minute candle close
- Evaluate all strategies
- Apply quality filters
- Send Telegram alerts for valid setups

### 4. Run Backtest

```bash
# Generate sample data and run backtest
python backtest.py --generate

# Run on your own data
python backtest.py your_data.csv

# With date filters
python backtest.py data.csv --start 2024-01-01 --end 2024-06-30
```

## Trading Strategies

### A. Trend Pullback (Primary)

**BUY Setup:**
- EMA20 > EMA50 (bullish trend)
- Price pulls back to EMA20 zone (within 1 ATR)
- RSI 40-60 and rising
- Bullish rejection candle or strong close
- Entry: Market at close or limit at EMA20
- Stop: Below recent swing low or 1.5 ATR
- Target: 2R (configurable)

**SELL Setup:** Mirror conditions for bearish trend.

### B. Breakout + Retest (Secondary)

**BUY Setup:**
- Price closes above 20-candle range high by 0.5 ATR
- Within 1-3 candles, price retests breakout level
- Retest holds (no close back inside range)
- Entry: At retest confirmation
- Stop: Below breakout level
- Target: 2R or measured move (range height)

### C. Mean Reversion (Disabled by default)

Only activates when market is ranging (flat EMA + low ATR):
- Price touches outer Bollinger Band
- Snaps back inside on close
- RSI at extreme (oversold/overbought)
- Entry: At close
- Stop: Outside band by 1 ATR
- Target: Middle band or 1.5R

Enable in `config.py`:
```python
STRATEGY_MEAN_REVERSION_ENABLED = True
```

## Configuration Reference

### Strategy Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `STRATEGY_TREND_PULLBACK_ENABLED` | True | Enable pullback signals |
| `STRATEGY_BREAKOUT_ENABLED` | True | Enable breakout signals |
| `STRATEGY_MEAN_REVERSION_ENABLED` | False | Enable mean reversion |
| `PULLBACK_ATR_DISTANCE` | 1.0 | Max distance to EMA in ATR multiples |
| `BREAKOUT_RANGE_CANDLES` | 20 | Candles for range definition |
| `BREAKOUT_ATR_THRESHOLD` | 0.5 | Min breakout beyond range |

### Risk Management

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ATR_MULTIPLIER_SL` | 1.5 | Stop loss ATR multiplier |
| `RISK_REWARD_RATIO` | 2.0 | Take profit ratio |
| `MIN_STOP_DISTANCE` | 3.0 | Minimum stop in USD |
| `MAX_STOP_DISTANCE` | 25.0 | Maximum stop in USD |

### Quality Filters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MIN_SIGNAL_QUALITY` | 0.6 | Min quality score (0-1) |
| `COOLDOWN_MINUTES` | 30 | Cooldown per direction |
| `ATR_VOLATILITY_MIN_PERCENTILE` | 25 | Skip if ATR too low |
| `ATR_VOLATILITY_MAX_PERCENTILE` | 95 | Skip if ATR too high |

### Session Filters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SESSION_FILTER_ENABLED` | True | Enable session filtering |
| `LONDON_SESSION_ENABLED` | True | Trade London session |
| `NY_SESSION_ENABLED` | True | Trade NY session |

### News Blackout

Add blackout windows in `config.py`:
```python
NEWS_BLACKOUT_WINDOWS = [
    ("12:30", "14:00", "NFP Release"),
    ("18:00", "20:00", "FOMC Statement"),
]
```

## Alert Format

```
🟢 GOLD BUY SETUP (XAUUSD)
━━━━━━━━━━━━━━━━━━━━
Strategy: Trend Pullback
Timeframe: M5
Quality: 80%

📍 ENTRY PLAN
  Type: Market at close
  Price: $2050.50

🛡️ RISK MANAGEMENT
  Stop Loss: $2045.00
  Take Profit: $2061.50
  Risk/Reward: 2.00R

❌ INVALIDATION
  Closes below $2045.00 OR EMA20 crosses below EMA50

📊 RATIONALE
  • Uptrend: EMA20 ($2049.20) > EMA50 ($2042.80)
  • Pullback to EMA20 support zone
  • RSI at 48.5 - momentum building
  • Bullish rejection candle formed

🕐 2024-01-15 14:35 UTC

⚠️ EDUCATIONAL SETUP ONLY
Not financial advice. Do your own research.
```

## Backtest Output

```
============================================================
BACKTEST RESULTS
============================================================

Signals Generated: 245
Trades Taken:      198

Performance:
  Win Rate:        54.5%
  Winners:         108
  Losers:          90
  Average R:       0.42R
  Total R:         83.16R
  Profit Factor:   1.45

Risk:
  Max Drawdown:    $125.50
  Max DD (R):      8.25R

By Setup Type:
  Trend Pullback:
    Trades: 142, Win Rate: 56.3%, Avg R: 0.48
  Breakout Retest:
    Trades: 56, Win Rate: 50.0%, Avg R: 0.31
============================================================
```

## Testing Components

```bash
# Test data fetching
python data.py

# Test indicators
python indicators.py

# Test strategy
python strategy.py

# Test utilities
python utils.py
```

## Logs

The bot creates several log files in `logs/`:

- `gold_bot.log` - Main bot activity log
- `gold_bot_evaluations.csv` - Every evaluation cycle
- `trade_signals.csv` - All generated signals

## Troubleshooting

### "API Connection failed"
- Check your Metal Price API key in `config.py`
- Verify you have API credits remaining
- Check internet connection

### "Telegram message failed"
- Verify bot token is correct
- Ensure you've sent `/start` to your bot
- Check chat ID is correct

### "No signals detected"
- This is normal - setups are specific and filtered
- Check logs for filter reasons
- Consider lowering `MIN_SIGNAL_QUALITY` for testing

### Bot drifts from candle close times
- The bot uses clock-aligned timing, not fixed intervals
- If timing seems off, check system clock sync

## Advanced Usage

### Custom Data for Backtesting

Prepare a CSV file with columns:
```
timestamp,open,high,low,close
2024-01-01 00:00:00,2050.00,2051.50,2049.00,2050.50
2024-01-01 00:05:00,2050.50,2052.00,2050.00,2051.20
...
```

### Adjusting for Different Market Conditions

For higher volatility periods:
```python
ATR_MULTIPLIER_SL = 2.0
MIN_SIGNAL_QUALITY = 0.7
```

For quieter markets:
```python
PULLBACK_ATR_DISTANCE = 0.5
STRATEGY_MEAN_REVERSION_ENABLED = True
```

## License

This project is for educational purposes only.

## Disclaimer

**IMPORTANT**: This software is provided "as is" without warranty of any kind. The signals generated are for educational and research purposes only. They do not constitute financial advice, investment recommendations, or trading signals for actual execution.

Trading Gold (XAUUSD) and other financial instruments involves substantial risk of loss. You should not trade with money you cannot afford to lose. Past performance, whether actual or indicated by historical tests, does not guarantee future results.

The authors and contributors of this software accept no responsibility for any financial losses incurred through the use of this software.

Always:
- Do your own research
- Use proper risk management
- Never risk more than you can afford to lose
- Consider consulting a licensed financial advisor
