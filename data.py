"""
Data fetching module for Gold Trading Bot
Uses Yahoo Finance for real 5-minute OHLC candles (FREE)
Falls back to Metal Price API for spot price if needed
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import config


def get_historical_candles_yahoo(num_candles: int = 100) -> pd.DataFrame:
    """
    Fetch real 5-minute OHLC candles from Yahoo Finance

    Args:
        num_candles: Number of candles to fetch

    Returns:
        DataFrame with timestamp, open, high, low, close
    """
    try:
        symbol = 'GC=F'  # Gold Futures
        url = f'https://query1.finance.yahoo.com/v8/finance/chart/{symbol}'

        # Calculate range needed (5 days max for 5m candles)
        params = {
            'interval': '5m',
            'range': '5d'
        }

        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, params=params, headers=headers, timeout=30)
        response.raise_for_status()

        data = response.json()

        if 'chart' not in data or not data['chart']['result']:
            print("Yahoo Finance: No data returned")
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

        # Return requested number of candles
        if len(df) > num_candles:
            df = df.iloc[-num_candles:]

        df = df.reset_index(drop=True)

        return df

    except Exception as e:
        print(f"Yahoo Finance error: {e}")
        return None


def get_current_price_yahoo() -> float:
    """
    Get current Gold price from Yahoo Finance

    Returns:
        Current price or None
    """
    try:
        symbol = 'GC=F'
        url = f'https://query1.finance.yahoo.com/v8/finance/chart/{symbol}'

        params = {
            'interval': '1m',
            'range': '1d'
        }

        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, params=params, headers=headers, timeout=15)
        response.raise_for_status()

        data = response.json()

        if 'chart' in data and data['chart']['result']:
            result = data['chart']['result'][0]
            meta = result.get('meta', {})
            price = meta.get('regularMarketPrice')
            if price:
                return float(price)

            # Fallback to last close
            quotes = result['indicators']['quote'][0]
            closes = [c for c in quotes['close'] if c is not None]
            if closes:
                return float(closes[-1])

        return None

    except Exception as e:
        print(f"Yahoo Finance price error: {e}")
        return None


def get_current_gold_price() -> float:
    """
    Get current Gold price - tries Yahoo first, falls back to Metal Price API

    Returns:
        Current price or None
    """
    # Try Yahoo Finance first
    price = get_current_price_yahoo()
    if price:
        return price

    # Fallback to Metal Price API
    try:
        url = f"{config.METAL_PRICE_API_BASE_URL}/latest"
        params = {
            'api_key': config.METAL_PRICE_API_KEY,
            'base': 'XAU',
            'currencies': 'USD'
        }

        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()

        if data.get('success') and 'rates' in data:
            return float(data['rates']['USD'])

    except Exception as e:
        print(f"Metal Price API error: {e}")

    return None


def get_historical_candles(num_candles: int = 100) -> pd.DataFrame:
    """
    Get historical 5-minute candles
    Uses Yahoo Finance for real OHLC data

    Args:
        num_candles: Number of candles to fetch

    Returns:
        DataFrame with OHLC data
    """
    return get_historical_candles_yahoo(num_candles)


if __name__ == "__main__":
    print("Testing Data Module...")
    print()

    # Test Yahoo Finance
    print("=" * 50)
    print("Yahoo Finance (Primary Data Source)")
    print("=" * 50)

    price = get_current_price_yahoo()
    if price:
        print(f"Current Gold Price: ${price:.2f}")
    else:
        print("Failed to get current price")

    print()
    print("Fetching 5-minute candles...")
    df = get_historical_candles_yahoo(50)

    if df is not None and not df.empty:
        print(f"Got {len(df)} candles")
        print()
        print("Latest 5 candles:")
        print(df.tail().to_string(index=False))
        print()
        print("Data includes REAL Open/High/Low/Close values!")
    else:
        print("Failed to fetch candles")

    print()
    print("=" * 50)
    print("Metal Price API (Backup)")
    print("=" * 50)

    try:
        url = f"{config.METAL_PRICE_API_BASE_URL}/latest"
        params = {
            'api_key': config.METAL_PRICE_API_KEY,
            'base': 'XAU',
            'currencies': 'USD'
        }
        response = requests.get(url, params=params, timeout=15)
        data = response.json()

        if data.get('success'):
            print(f"Spot Price: ${data['rates']['USD']:.2f}")
            print("(Only provides spot price - no OHLC)")
        else:
            print(f"API Error: {data}")
    except Exception as e:
        print(f"Error: {e}")
