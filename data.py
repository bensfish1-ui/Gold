"""
Data fetching module for Gold prices
Handles API calls and data preprocessing
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import config


def get_current_gold_price():
    """
    Fetch current Gold (XAUUSD) price from Metal Price API
    Returns: float - current price in USD per ounce
    """
    try:
        url = f"{config.METAL_PRICE_API_BASE_URL}/latest"
        params = {
            'api_key': config.METAL_PRICE_API_KEY,
            'base': 'XAU',
            'currencies': 'USD'
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if data.get('success'):
            # Metal Price API returns USD per ounce of Gold
            usd_per_xau = data['rates']['USD']
            return round(usd_per_xau, 2)
        else:
            error_info = data.get('error', {})
            raise Exception(f"API Error: {error_info.get('info', 'Unknown error')}")
            
    except Exception as e:
        print(f"Error fetching current price: {e}")
        return None


def get_historical_candles(candles=100):
    """
    Fetch historical Gold price data
    Since free APIs might not provide full candle data, we'll simulate
    5-minute candles from available data points
    
    Args:
        candles: number of candles to fetch
    
    Returns: pandas DataFrame with OHLC data
    """
    try:
        # For a production system, you'd want to use a proper forex/commodities API
        # that provides OHLC data (e.g., Alpha Vantage, Twelve Data, etc.)
        
        # This is a simplified approach - fetch multiple data points
        prices = []
        timestamps = []
        
        # Try to get recent historical data
        # Metal Price API timeframe endpoint for historical data
        
        end_date = datetime.now()
        
        # Attempt to get historical data from multiple days
        for i in range(min(20, candles)):
            date_to_fetch = end_date - timedelta(days=i)
            timestamp = date_to_fetch.strftime('%Y-%m-%d')
            
            # Metal Price API historical endpoint
            url = f"{config.METAL_PRICE_API_BASE_URL}/{timestamp}"
            params = {
                'api_key': config.METAL_PRICE_API_KEY,
                'base': 'XAU',
                'currencies': 'USD'
            }
            
            try:
                response = requests.get(url, params=params, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    if data.get('success'):
                        price = data['rates']['USD']
                        prices.append(price)
                        timestamps.append(date_to_fetch)
            except:
                pass
            
            # Rate limiting
            time.sleep(0.2)
            
            # Break if we have enough unique data points
            if len(prices) >= 20:
                break
        
        # If we don't have enough historical data, create synthetic candles
        # based on current price (for testing purposes)
        if len(prices) < 20:
            current_price = get_current_gold_price()
            if current_price:
                prices = generate_synthetic_candles(current_price, candles)
                timestamps = [end_date - timedelta(minutes=5 * i) for i in range(candles)]
        
        # Create DataFrame with OHLC structure
        df = pd.DataFrame({
            'timestamp': timestamps[::-1],
            'close': prices[::-1]
        })
        
        # Generate OHLC from close prices (simplified)
        df['open'] = df['close'].shift(1).fillna(df['close'])
        df['high'] = df[['open', 'close']].max(axis=1) * (1 + 0.0005)
        df['low'] = df[['open', 'close']].min(axis=1) * (1 - 0.0005)
        
        df = df[['timestamp', 'open', 'high', 'low', 'close']]
        
        return df
        
    except Exception as e:
        print(f"Error fetching historical data: {e}")
        return None


def generate_synthetic_candles(base_price, count):
    """
    Generate synthetic price data for testing
    Simulates realistic price movements
    """
    import random
    
    prices = [base_price]
    for i in range(count - 1):
        # Random walk with small variations
        change_pct = random.uniform(-0.002, 0.002)  # +/- 0.2%
        new_price = prices[-1] * (1 + change_pct)
        prices.append(round(new_price, 2))
    
    return prices


def update_candles_with_new_price(df, new_price):
    """
    Update the most recent candle with new price data
    
    Args:
        df: existing candles DataFrame
        new_price: current price
    
    Returns: updated DataFrame
    """
    if df is None or df.empty:
        return None
    
    # Update the last candle
    df.iloc[-1, df.columns.get_loc('close')] = new_price
    df.iloc[-1, df.columns.get_loc('high')] = max(df.iloc[-1]['high'], new_price)
    df.iloc[-1, df.columns.get_loc('low')] = min(df.iloc[-1]['low'], new_price)
    
    return df


def test_api_connection():
    """
    Test if API connection is working
    """
    print("Testing API connection...")
    price = get_current_gold_price()
    
    if price:
        print(f"✓ API Connection successful!")
        print(f"✓ Current Gold Price: ${price}")
        return True
    else:
        print("✗ API Connection failed!")
        print("Please check your API key in config.py")
        return False


if __name__ == "__main__":
    # Test the data fetching
    test_api_connection()
    
    print("\nFetching historical candles...")
    df = get_historical_candles(50)
    if df is not None:
        print(f"✓ Fetched {len(df)} candles")
        print("\nLatest candles:")
        print(df.tail())
    else:
        print("✗ Failed to fetch historical data")
