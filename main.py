import os
import time
from bot import run_bot

# Load Railway environment variables
METAL_PRICE_API_KEY = os.getenv("METAL_PRICE_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

if not METAL_PRICE_API_KEY or not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
    raise ValueError("API keys not found in Railway environment variables.")

print("🚀 Gold bot running live on Railway...")

while True:
    try:
        print("🔍 Scanning market...")
        run_bot()
        print("⏳ Sleeping 5 minutes...")
        time.sleep(300)

    except Exception as e:
        print(f"❌ Error: {e}")
        print("Retrying in 60 seconds...")
        time.sleep(60)
