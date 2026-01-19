"""
Utility functions for Gold Trading Bot
Includes time alignment, formatting, state management, and session filters
"""

import json
import os
import csv
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple
import config


class BotState:
    """
    Manages bot state persistence for cooldowns, signal tracking, etc.
    State is persisted to JSON file to survive restarts.
    """

    def __init__(self, state_file: str = None):
        self.state_file = state_file or config.STATE_FILE
        self.state = self._load_state()

    def _load_state(self) -> Dict[str, Any]:
        """Load state from JSON file"""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass

        return self._get_default_state()

    def _get_default_state(self) -> Dict[str, Any]:
        """Return default state structure"""
        return {
            'last_signal_buy': None,
            'last_signal_sell': None,
            'total_signals': 0,
            'signals_today': 0,
            'last_date': None,
            'last_evaluation_time': None,
            'consecutive_no_signals': 0,
            'last_heartbeat': None,
            'last_daily_summary': None,
            'total_r': 0.0,
            'starting_bank': 1000.0,
            'trades_history': []
        }

    def _save_state(self):
        """Save state to JSON file"""
        try:
            with open(self.state_file, 'w') as f:
                json.dump(self.state, f, indent=2, default=str)
        except IOError as e:
            print(f"Warning: Could not save state: {e}")

    def check_cooldown(self, direction: str) -> Tuple[bool, Optional[str]]:
        """
        Check if cooldown period has passed for given direction

        Args:
            direction: 'BUY' or 'SELL'

        Returns:
            (can_signal: bool, reason: str or None)
        """
        key = f'last_signal_{direction.lower()}'
        last_signal_time = self.state.get(key)

        if last_signal_time is None:
            return True, None

        try:
            last_time = datetime.fromisoformat(last_signal_time)
            cooldown_delta = timedelta(minutes=config.COOLDOWN_MINUTES)
            time_elapsed = datetime.now() - last_time
            time_remaining = cooldown_delta - time_elapsed

            if time_elapsed >= cooldown_delta:
                return True, None
            else:
                mins_remaining = int(time_remaining.total_seconds() / 60)
                return False, f"Cooldown active for {direction}: {mins_remaining}min remaining"

        except (ValueError, TypeError):
            return True, None

    def update_last_signal(self, direction: str, signal: Dict[str, Any] = None):
        """
        Update last signal timestamp for given direction

        Args:
            direction: 'BUY' or 'SELL'
            signal: optional signal dict to store
        """
        key = f'last_signal_{direction.lower()}'
        self.state[key] = datetime.now().isoformat()
        self.state['total_signals'] = self.state.get('total_signals', 0) + 1

        # Reset daily counter if new day
        today = datetime.now().strftime('%Y-%m-%d')
        if self.state.get('last_date') != today:
            self.state['signals_today'] = 1
            self.state['last_date'] = today
        else:
            self.state['signals_today'] = self.state.get('signals_today', 0) + 1

        # Reset consecutive no-signal counter
        self.state['consecutive_no_signals'] = 0

        self._save_state()

    def record_evaluation(self, had_signal: bool):
        """Record an evaluation cycle"""
        self.state['last_evaluation_time'] = datetime.now().isoformat()

        if not had_signal:
            self.state['consecutive_no_signals'] = \
                self.state.get('consecutive_no_signals', 0) + 1
        else:
            self.state['consecutive_no_signals'] = 0

        self._save_state()

    def get_stats(self) -> Dict[str, Any]:
        """Get current bot statistics"""
        return {
            'total_signals': self.state.get('total_signals', 0),
            'signals_today': self.state.get('signals_today', 0),
            'last_buy': self.state.get('last_signal_buy'),
            'last_sell': self.state.get('last_signal_sell'),
            'consecutive_no_signals': self.state.get('consecutive_no_signals', 0),
            'total_r': self.state.get('total_r', 0.0),
            'starting_bank': self.state.get('starting_bank', 1000.0)
        }

    def reset(self):
        """Reset state to defaults"""
        self.state = self._get_default_state()
        self._save_state()

    def record_trade_result(self, r_result: float, signal: Dict[str, Any] = None):
        """
        Record a completed trade result

        Args:
            r_result: R-multiple result (e.g., 2.0 for 2R win, -1.0 for loss)
            signal: optional signal dict for history
        """
        self.state['total_r'] = self.state.get('total_r', 0.0) + r_result

        # Add to history
        history = self.state.get('trades_history', [])
        history.append({
            'timestamp': datetime.now().isoformat(),
            'r_result': r_result,
            'direction': signal.get('direction') if signal else None
        })
        # Keep last 100 trades
        self.state['trades_history'] = history[-100:]
        self._save_state()

    def set_starting_bank(self, amount: float):
        """Set the starting bank amount"""
        self.state['starting_bank'] = amount
        self._save_state()

    def get_bank_status(self) -> Dict[str, Any]:
        """Get current bank status and growth"""
        starting = self.state.get('starting_bank', 1000.0)
        total_r = self.state.get('total_r', 0.0)
        risk_per_trade = starting * 0.01  # 1% risk
        profit = total_r * risk_per_trade
        current_bank = starting + profit
        growth_pct = (profit / starting) * 100 if starting > 0 else 0

        return {
            'starting_bank': starting,
            'current_bank': current_bank,
            'total_r': total_r,
            'profit': profit,
            'growth_pct': growth_pct,
            'total_signals': self.state.get('total_signals', 0)
        }

    def should_send_heartbeat(self, interval_hours: int = 2) -> bool:
        """Check if it's time to send a heartbeat message"""
        last_hb = self.state.get('last_heartbeat')
        if last_hb is None:
            return True

        try:
            last_time = datetime.fromisoformat(last_hb)
            hours_elapsed = (datetime.now() - last_time).total_seconds() / 3600
            return hours_elapsed >= interval_hours
        except (ValueError, TypeError):
            return True

    def record_heartbeat(self):
        """Record that a heartbeat was sent"""
        self.state['last_heartbeat'] = datetime.now().isoformat()
        self._save_state()

    def should_send_daily_summary(self) -> bool:
        """Check if it's time to send daily summary (9am UTC)"""
        now = datetime.utcnow()

        # Only at 9am hour
        if now.hour != 9:
            return False

        # Check if already sent today
        last_summary = self.state.get('last_daily_summary')
        if last_summary:
            try:
                last_date = datetime.fromisoformat(last_summary).date()
                if last_date == now.date():
                    return False  # Already sent today
            except (ValueError, TypeError):
                pass

        return True

    def record_daily_summary(self):
        """Record that daily summary was sent"""
        self.state['last_daily_summary'] = datetime.now().isoformat()
        self._save_state()


def get_next_candle_close_time(interval_seconds: int = 300) -> datetime:
    """
    Calculate the next 5-minute candle close time (aligned to clock)

    Args:
        interval_seconds: candle interval in seconds (default 300 = 5 min)

    Returns:
        datetime object for next candle close
    """
    now = datetime.now()

    # Calculate seconds since midnight
    seconds_since_midnight = now.hour * 3600 + now.minute * 60 + now.second

    # Calculate seconds to next interval boundary
    current_interval = seconds_since_midnight // interval_seconds
    next_interval_seconds = (current_interval + 1) * interval_seconds

    # Calculate target time
    target_hour = next_interval_seconds // 3600
    remaining = next_interval_seconds % 3600
    target_minute = remaining // 60
    target_second = remaining % 60

    # Handle day overflow
    if target_hour >= 24:
        next_close = now.replace(hour=0, minute=0, second=0, microsecond=0)
        next_close += timedelta(days=1)
    else:
        next_close = now.replace(
            hour=target_hour,
            minute=target_minute,
            second=target_second,
            microsecond=0
        )

    return next_close


def get_seconds_to_candle_close(interval_seconds: int = 300) -> float:
    """
    Get seconds until next candle close

    Args:
        interval_seconds: candle interval in seconds

    Returns:
        seconds until next close
    """
    next_close = get_next_candle_close_time(interval_seconds)
    now = datetime.now()
    return (next_close - now).total_seconds()


def wait_for_candle_close(interval_seconds: int = 300, buffer_seconds: int = 2) -> float:
    """
    Sleep until next candle close time plus buffer

    Args:
        interval_seconds: candle interval in seconds
        buffer_seconds: extra seconds to wait after close for data availability

    Returns:
        seconds slept
    """
    import time

    seconds_to_wait = get_seconds_to_candle_close(interval_seconds)

    if seconds_to_wait > 0:
        total_wait = seconds_to_wait + buffer_seconds
        time.sleep(total_wait)
        return total_wait

    return 0


def is_trading_session_active() -> Tuple[bool, str]:
    """
    Check if current time is within enabled trading sessions

    Returns:
        (is_active: bool, session_name: str or reason)
    """
    # If session filtering is disabled, always active
    if not config.SESSION_FILTER_ENABLED:
        return True, "Session filtering disabled"

    # If no sessions are enabled, allow trading 24/7
    if not config.LONDON_SESSION_ENABLED and not config.NY_SESSION_ENABLED:
        return True, "No sessions configured"

    now = datetime.utcnow()
    current_time = now.strftime('%H:%M')

    sessions_active = []

    # Check London session
    if config.LONDON_SESSION_ENABLED:
        if config.LONDON_SESSION_START <= current_time <= config.LONDON_SESSION_END:
            sessions_active.append("London")

    # Check NY session
    if config.NY_SESSION_ENABLED:
        if config.NY_SESSION_START <= current_time <= config.NY_SESSION_END:
            sessions_active.append("New York")

    if sessions_active:
        return True, " & ".join(sessions_active)
    else:
        return False, "Outside trading sessions"


def is_news_blackout() -> Tuple[bool, Optional[str]]:
    """
    Check if current time is within a news blackout window

    Returns:
        (is_blackout: bool, description: Optional[str])
    """
    if not config.NEWS_BLACKOUT_WINDOWS:
        return False, None

    now = datetime.utcnow()
    current_time = now.strftime('%H:%M')

    for start_time, end_time, description in config.NEWS_BLACKOUT_WINDOWS:
        if start_time <= current_time <= end_time:
            return True, description

    return False, None


def format_price(price: float, decimals: int = 2) -> str:
    """Format price with dollar sign and specified decimals"""
    return f"${price:.{decimals}f}"


def format_risk_reward(rr: float) -> str:
    """Format risk/reward ratio"""
    return f"{rr:.2f}R"


def format_quality(quality: float) -> str:
    """Format quality score as percentage"""
    return f"{quality:.0%}"


def format_telegram_message(signal: Dict[str, Any]) -> str:
    """
    Format trading signal for Telegram message

    Args:
        signal: signal dictionary from StrategyEngine

    Returns:
        formatted HTML message
    """
    emoji = "🟢" if signal['direction'] == 'BUY' else "🔴"

    # Header
    message = f"{emoji} <b>GOLD {signal['direction']} SETUP (XAUUSD)</b>\n"
    message += f"━━━━━━━━━━━━━━━━━━━━\n"
    message += f"<b>Strategy:</b> {signal['setup_type']}\n"
    message += f"<b>Timeframe:</b> {signal.get('timeframe', 'M5')}\n"
    message += f"<b>Quality:</b> {format_quality(signal['quality'])}\n\n"

    # Entry details
    message += f"<b>📍 ENTRY PLAN</b>\n"
    message += f"  Type: {signal.get('entry_type', 'Market')}\n"
    message += f"  Price: {format_price(signal['entry'])}\n\n"

    # Risk management
    message += f"<b>🛡️ RISK MANAGEMENT</b>\n"
    message += f"  Stop Loss: {format_price(signal['stop_loss'])}\n"
    message += f"  Take Profit: {format_price(signal['take_profit'])}\n"
    message += f"  Risk/Reward: {format_risk_reward(signal['risk_reward'])}\n\n"

    # Invalidation
    message += f"<b>❌ INVALIDATION</b>\n"
    message += f"  {signal['invalidation']}\n\n"

    # Rationale
    message += f"<b>📊 RATIONALE</b>\n"
    for reason in signal['rationale']:
        message += f"  • {reason}\n"

    # Footer
    timestamp = signal.get('timestamp', datetime.utcnow().isoformat())
    if isinstance(timestamp, str):
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            formatted_time = dt.strftime('%Y-%m-%d %H:%M UTC')
        except (ValueError, TypeError):
            formatted_time = timestamp
    else:
        formatted_time = timestamp.strftime('%Y-%m-%d %H:%M UTC')

    message += f"\n<i>🕐 {formatted_time}</i>"
    message += f"\n\n⚠️ <i>EDUCATIONAL SETUP ONLY</i>"
    message += f"\n<i>Not financial advice. Do your own research.</i>"

    return message


def log_signal_to_csv(signal: Dict[str, Any], filepath: str = None):
    """
    Log signal to CSV file for record keeping

    Args:
        signal: signal dictionary
        filepath: path to CSV file (default from config)
    """
    if filepath is None:
        filepath = config.TRADE_LOG_FILE

    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Check if file exists to determine if we need headers
    file_exists = os.path.exists(filepath)

    try:
        with open(filepath, 'a', newline='') as f:
            writer = csv.writer(f)

            # Write header if new file
            if not file_exists:
                writer.writerow([
                    'timestamp', 'direction', 'setup_type', 'entry',
                    'stop_loss', 'take_profit', 'risk_reward', 'quality',
                    'entry_type', 'invalidation', 'rationale'
                ])

            # Write signal data
            rationale_str = ' | '.join(signal.get('rationale', []))

            writer.writerow([
                signal.get('timestamp', datetime.now().isoformat()),
                signal['direction'],
                signal['setup_type'],
                signal['entry'],
                signal['stop_loss'],
                signal['take_profit'],
                signal['risk_reward'],
                signal['quality'],
                signal.get('entry_type', 'Market'),
                signal['invalidation'],
                rationale_str
            ])

    except IOError as e:
        print(f"Warning: Could not log signal to CSV: {e}")


def log_evaluation(result: str, details: Dict[str, Any] = None,
                   filepath: str = None):
    """
    Log each evaluation cycle to a separate log file

    Args:
        result: evaluation result (e.g., "SIGNAL", "NO_SIGNAL", "FILTERED")
        details: optional details dict
        filepath: log file path
    """
    if filepath is None:
        filepath = config.LOG_FILE.replace('.log', '_evaluations.csv')

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    file_exists = os.path.exists(filepath)

    try:
        with open(filepath, 'a', newline='') as f:
            writer = csv.writer(f)

            if not file_exists:
                writer.writerow([
                    'timestamp', 'result', 'session', 'atr_percentile',
                    'trend', 'close_price', 'details'
                ])

            writer.writerow([
                datetime.now().isoformat(),
                result,
                details.get('session', '') if details else '',
                details.get('atr_percentile', '') if details else '',
                details.get('trend', '') if details else '',
                details.get('close_price', '') if details else '',
                json.dumps(details) if details else ''
            ])

    except IOError:
        pass  # Silent fail for evaluation logs


def calculate_risk_reward(entry: float, stop_loss: float,
                          take_profit: float) -> float:
    """
    Calculate risk/reward ratio

    Args:
        entry: entry price
        stop_loss: stop loss price
        take_profit: take profit price

    Returns:
        risk/reward ratio
    """
    risk = abs(entry - stop_loss)
    reward = abs(take_profit - entry)

    if risk == 0:
        return 0

    return reward / risk


def ensure_directories():
    """Create required directories if they don't exist"""
    directories = [
        'logs',
        'sample_data'
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def create_state_template():
    """Create a template state.json file if it doesn't exist"""
    if not os.path.exists(config.STATE_FILE):
        state = BotState()
        state._save_state()
        print(f"Created {config.STATE_FILE}")


if __name__ == "__main__":
    print("Testing utility functions...\n")

    # Ensure directories exist
    ensure_directories()
    print("Directories created/verified")

    # Test state management
    state = BotState()
    print(f"\nCurrent stats: {state.get_stats()}")

    # Test cooldown
    can_buy, reason = state.check_cooldown('BUY')
    print(f"Can send BUY signal: {can_buy}")
    if reason:
        print(f"  Reason: {reason}")

    # Test session check
    in_session, session_info = is_trading_session_active()
    print(f"\nTrading session active: {in_session}")
    print(f"  Info: {session_info}")

    # Test news blackout
    is_blackout, blackout_reason = is_news_blackout()
    print(f"\nNews blackout active: {is_blackout}")
    if blackout_reason:
        print(f"  Reason: {blackout_reason}")

    # Test candle timing
    next_close = get_next_candle_close_time()
    seconds_to_close = get_seconds_to_candle_close()
    print(f"\nNext 5-min candle close: {next_close.strftime('%H:%M:%S')}")
    print(f"Seconds until close: {seconds_to_close:.1f}")

    # Test R:R calculation
    rr = calculate_risk_reward(2050, 2045, 2060)
    print(f"\nExample R:R (Entry $2050, SL $2045, TP $2060): {format_risk_reward(rr)}")

    # Test message formatting
    test_signal = {
        'direction': 'BUY',
        'setup_type': 'Trend Pullback',
        'entry': 2050.50,
        'entry_type': 'Market at close',
        'stop_loss': 2045.00,
        'take_profit': 2061.50,
        'risk_reward': 2.0,
        'quality': 0.80,
        'invalidation': 'Closes below $2045.00',
        'rationale': [
            'Uptrend confirmed: EMA20 > EMA50',
            'Pullback to EMA20 support',
            'RSI momentum building'
        ],
        'timeframe': 'M5',
        'timestamp': datetime.utcnow().isoformat()
    }

    print("\n" + "=" * 50)
    print("Sample Telegram Message:")
    print("=" * 50)
    print(format_telegram_message(test_signal))

    print("\nUtilities tested successfully!")
