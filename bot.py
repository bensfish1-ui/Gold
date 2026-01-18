"""
Gold Trading Bot - Main Module
Runs continuous scanning aligned to 5-minute candle closes
Evaluates strategies and sends Telegram alerts for high-quality setups

Features:
- Candle-aligned scanning (triggers at candle close, not arbitrary intervals)
- Session filtering (London/NY)
- News blackout windows
- Volatility filters (chop/chaos detection)
- Cooldown management
- Quality threshold filtering
- Comprehensive logging
"""

import time
import requests
from datetime import datetime
import logging
import os
import config
from data import get_current_gold_price, get_historical_candles
from indicators import add_all_indicators, calculate_atr_percentile, get_trend_bias
from strategy import StrategyEngine, check_volatility_filter
from utils import (
    BotState, format_telegram_message, log_signal_to_csv, log_evaluation,
    is_trading_session_active, is_news_blackout, get_seconds_to_candle_close,
    get_next_candle_close_time, ensure_directories
)


# Ensure log directory exists
ensure_directories()

# Setup logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class GoldTradingBot:
    """
    Main bot class that orchestrates scanning, strategy evaluation, and alerting
    """

    def __init__(self):
        self.strategy = StrategyEngine()
        self.state = BotState()
        self.candles_df = None
        self.last_refresh = None
        self.scan_count = 0
        self.startup_time = datetime.now()

    def send_telegram_message(self, message: str) -> bool:
        """
        Send message to Telegram

        Args:
            message: HTML-formatted message

        Returns:
            bool - success status
        """
        try:
            url = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage"

            payload = {
                'chat_id': config.TELEGRAM_CHAT_ID,
                'text': message,
                'parse_mode': 'HTML',
                'disable_web_page_preview': True
            }

            response = requests.post(url, json=payload, timeout=15)
            response.raise_for_status()

            return True

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False

    def send_signal_alert(self, signal: dict) -> bool:
        """
        Send trading signal alert to Telegram

        Args:
            signal: signal dict from StrategyEngine

        Returns:
            bool - success status
        """
        message = format_telegram_message(signal)
        success = self.send_telegram_message(message)

        if success:
            logger.info(f"Telegram alert sent: {signal['direction']} {signal['setup_type']}")
        else:
            logger.error("Failed to send Telegram alert")

        return success

    def refresh_candles(self) -> bool:
        """
        Fetch fresh historical candles and compute indicators

        Returns:
            bool - success status
        """
        logger.info("Refreshing candle data...")

        try:
            self.candles_df = get_historical_candles(config.LOOKBACK_CANDLES)

            if self.candles_df is not None and not self.candles_df.empty:
                self.candles_df = add_all_indicators(self.candles_df)
                self.last_refresh = datetime.now()
                logger.info(f"Loaded {len(self.candles_df)} candles with indicators")
                return True
            else:
                logger.error("Failed to fetch candles - empty data")
                return False

        except Exception as e:
            logger.error(f"Error refreshing candles: {e}")
            return False

    def evaluate_market(self) -> dict:
        """
        Main evaluation cycle - checks all filters and evaluates strategy

        Returns:
            dict with evaluation result and details
        """
        result = {
            'has_signal': False,
            'signal': None,
            'filtered_reason': None,
            'details': {}
        }

        try:
            # Get current price
            current_price = get_current_gold_price()
            if current_price is None:
                result['filtered_reason'] = "Failed to fetch current price"
                return result

            result['details']['close_price'] = current_price
            logger.info(f"Gold Price: ${current_price}")

            # Ensure we have candle data
            if self.candles_df is None or self.candles_df.empty:
                if not self.refresh_candles():
                    result['filtered_reason'] = "No candle data available"
                    return result

            # Update latest candle with current price
            self.candles_df.iloc[-1, self.candles_df.columns.get_loc('close')] = current_price
            self.candles_df.iloc[-1, self.candles_df.columns.get_loc('high')] = max(
                self.candles_df.iloc[-1]['high'], current_price
            )
            self.candles_df.iloc[-1, self.candles_df.columns.get_loc('low')] = min(
                self.candles_df.iloc[-1]['low'], current_price
            )

            # Recalculate indicators
            self.candles_df = add_all_indicators(self.candles_df)

            # Get market context for logging
            trend = get_trend_bias(self.candles_df)
            atr_percentile = calculate_atr_percentile(
                self.candles_df, config.ATR_LOOKBACK_VOLATILITY
            )
            result['details']['trend'] = trend
            result['details']['atr_percentile'] = round(atr_percentile, 1)

            # === FILTER 1: Session Check ===
            in_session, session_info = is_trading_session_active()
            result['details']['session'] = session_info

            if not in_session:
                result['filtered_reason'] = f"Outside trading session ({session_info})"
                logger.info(f"FILTER: {result['filtered_reason']}")
                return result

            # === FILTER 2: News Blackout ===
            is_blackout, blackout_reason = is_news_blackout()
            if is_blackout:
                result['filtered_reason'] = f"News blackout: {blackout_reason}"
                logger.info(f"FILTER: {result['filtered_reason']}")
                return result

            # === FILTER 3: Volatility Sanity ===
            vol_ok, vol_reason = check_volatility_filter(self.candles_df)
            if not vol_ok:
                result['filtered_reason'] = vol_reason
                logger.info(f"FILTER: {result['filtered_reason']}")
                return result

            # === EVALUATE STRATEGY ===
            signal = self.strategy.evaluate(self.candles_df)

            if signal is None:
                result['filtered_reason'] = "No valid setup detected"
                logger.info("No signal - conditions not met")
                return result

            # === FILTER 4: Quality Threshold ===
            if signal['quality'] < config.MIN_SIGNAL_QUALITY:
                result['filtered_reason'] = (
                    f"Quality too low ({signal['quality']:.0%} < "
                    f"{config.MIN_SIGNAL_QUALITY:.0%})"
                )
                logger.info(f"FILTER: {result['filtered_reason']}")
                return result

            # === FILTER 5: Cooldown Check ===
            can_signal, cooldown_reason = self.state.check_cooldown(signal['direction'])
            if not can_signal:
                result['filtered_reason'] = cooldown_reason
                logger.info(f"FILTER: {result['filtered_reason']}")
                return result

            # === SIGNAL PASSED ALL FILTERS ===
            result['has_signal'] = True
            result['signal'] = signal
            logger.info(
                f"SIGNAL: {signal['direction']} {signal['setup_type']} "
                f"(Quality: {signal['quality']:.0%})"
            )

            return result

        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            result['filtered_reason'] = f"Evaluation error: {str(e)}"
            return result

    def run_evaluation_cycle(self):
        """
        Execute one complete evaluation cycle
        """
        self.scan_count += 1
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        logger.info(f"\n{'='*60}")
        logger.info(f"Scan #{self.scan_count} - {timestamp}")
        logger.info(f"{'='*60}")

        # Run evaluation
        result = self.evaluate_market()

        # Log evaluation
        log_result = "SIGNAL" if result['has_signal'] else "FILTERED"
        if result['filtered_reason']:
            log_result = f"FILTERED: {result['filtered_reason']}"

        log_evaluation(log_result, result['details'])

        # Process signal if found
        if result['has_signal'] and result['signal']:
            signal = result['signal']

            # Send Telegram alert
            if self.send_signal_alert(signal):
                # Update state
                self.state.update_last_signal(signal['direction'], signal)

                # Log to CSV
                log_signal_to_csv(signal)

                # Console output
                self._print_signal(signal)

            self.state.record_evaluation(had_signal=True)
        else:
            self.state.record_evaluation(had_signal=False)

        # Periodic candle refresh (every 20 scans ≈ 100 minutes)
        if self.scan_count % 20 == 0:
            self.refresh_candles()

    def _print_signal(self, signal: dict):
        """Print signal details to console"""
        emoji = "🟢" if signal['direction'] == 'BUY' else "🔴"

        print("\n" + "=" * 60)
        print(f"{emoji} {signal['direction']} SIGNAL DETECTED!")
        print("=" * 60)
        print(f"Strategy: {signal['setup_type']}")
        print(f"Quality:  {signal['quality']:.0%}")
        print(f"Entry:    ${signal['entry']}")
        print(f"Stop:     ${signal['stop_loss']}")
        print(f"Target:   ${signal['take_profit']}")
        print(f"R:R:      {signal['risk_reward']:.2f}")
        print("\nRationale:")
        for reason in signal['rationale']:
            print(f"  - {reason}")
        print(f"\nInvalidation: {signal['invalidation']}")
        print("=" * 60 + "\n")

    def run(self):
        """
        Main bot loop - runs continuously aligned to candle closes
        """
        logger.info("=" * 60)
        logger.info("Gold Trading Bot Started")
        logger.info("=" * 60)
        logger.info(f"Timeframe: {config.TIMEFRAME} (5-minute candles)")
        logger.info(f"Quality threshold: {config.MIN_SIGNAL_QUALITY:.0%}")
        logger.info(f"Cooldown: {config.COOLDOWN_MINUTES} minutes per direction")
        logger.info("=" * 60)

        # Send startup notification
        startup_msg = (
            "🤖 <b>Gold Trading Bot Activated</b>\n\n"
            f"Monitoring XAUUSD on {config.TIMEFRAME} timeframe\n"
            f"Quality threshold: {config.MIN_SIGNAL_QUALITY:.0%}\n\n"
            "<i>Scanning for high-probability setups...</i>"
        )

        if self.send_telegram_message(startup_msg):
            logger.info("Startup notification sent to Telegram")
        else:
            logger.warning("Could not send startup notification")

        # Initial candle data load
        logger.info("\nLoading initial candle data...")
        if not self.refresh_candles():
            logger.error("Failed to load initial data. Check API configuration.")
            return

        logger.info("\nBot is ready! Waiting for next candle close...")

        # Print current stats
        stats = self.state.get_stats()
        logger.info(f"Session stats: {stats['total_signals']} total signals, "
                   f"{stats['signals_today']} today")

        # Main loop - aligned to candle closes
        while True:
            try:
                # Calculate wait time until next candle close
                seconds_to_wait = get_seconds_to_candle_close(config.SCAN_INTERVAL)
                next_close = get_next_candle_close_time(config.SCAN_INTERVAL)

                logger.info(
                    f"Next evaluation at {next_close.strftime('%H:%M:%S')} "
                    f"({seconds_to_wait:.0f}s)"
                )

                # Wait for candle close (plus small buffer for data availability)
                if seconds_to_wait > 0:
                    time.sleep(seconds_to_wait + 2)

                # Run evaluation cycle
                self.run_evaluation_cycle()

            except KeyboardInterrupt:
                logger.info("\n\nBot stopped by user")
                break

            except Exception as e:
                logger.error(f"Unexpected error in main loop: {e}")
                logger.info("Retrying in 30 seconds...")
                time.sleep(30)

        # Shutdown
        shutdown_msg = "🛑 <b>Gold Trading Bot Stopped</b>"
        self.send_telegram_message(shutdown_msg)
        logger.info("Bot shutdown complete")


def main():
    """
    Entry point - validates configuration and starts bot
    """
    # Check API keys
    if config.METAL_PRICE_API_KEY == "YOUR_METAL_PRICE_API_KEY_HERE":
        print("\n" + "=" * 60)
        print("ERROR: API keys not configured!")
        print("=" * 60)
        print("\nPlease edit config.py and add your API keys:")
        print("  1. METAL_PRICE_API_KEY from https://metalpriceapi.com/")
        print("  2. TELEGRAM_BOT_TOKEN from @BotFather")
        print("  3. TELEGRAM_CHAT_ID (your chat ID)")
        print("\nSee README.md for detailed setup instructions.")
        print("=" * 60 + "\n")
        return

    if config.TELEGRAM_BOT_TOKEN == "YOUR_TELEGRAM_BOT_TOKEN_HERE":
        print("\nWARNING: Telegram not configured. Alerts will not be sent.")
        print("Configure TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in config.py\n")

    # Ensure directories exist
    ensure_directories()

    # Start bot
    bot = GoldTradingBot()
    bot.run()


if __name__ == "__main__":
    main()
def run_bot():
    main()
