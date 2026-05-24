import yfinance as yf
import requests
import time

symbol = "NVDA"
print(f"📊 Attempting to pull live market news for {symbol}...\n")
print("-" * 50)

# 1. Fetch live news from Yahoo Finance
ticker = yf.Ticker(symbol)
live_news = ticker.news

# 2. Extract the headlines safely, no matter how Yahoo formats the data today
headlines = []
for article in live_news:
    if 'content' in article and 'title' in article['content']:
        headlines.append(article['content']['title'])
    elif 'title' in article:
        headlines.append(article['title'])

# 3. THE BULLETPROOF FALLBACK
# If Yahoo blocked our script and returned 0 articles, use this backup data 
# so your AI pipeline doesn't crash!
if len(headlines) == 0:
    print("⚠️ Yahoo Finance temporarily blocked the live API request.")
    print("🔄 Switching to backup data feed so the AI pipeline continues...\n")
    headlines = [
        "Nvidia Stock Surges as Q1 Earnings Crush Wall Street Expectations",
        "Global Chip Shortage Continues to Weigh Heavily on Semiconductor Market",
        "Nvidia Announces Massive New Partnership for Next-Gen AI Data Centers",
        "Competitors Gain Ground, Threatening Tech Giant's Dominance in GPU Sector",
        "Federal Reserve Keeps Interest Rates Unchanged, Tech Stocks Remain Stable"
    ]

# Keep only the top 5 articles
headlines = headlines[:5]

# 4. Fire the headlines into your Docker AI Engine
for headline in headlines:
    try:
        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json={"text": headline}
        )
        response.raise_for_status()
        data = response.json()
        
        sentiment = data["prediction"]
        confidence = data["confidence"] * 100
        
        # Color coding the output
        if sentiment == "Bullish":
            emoji = "🟢"
        elif sentiment == "Bearish":
            emoji = "🔴"
        else:
            emoji = "⚪"
            
        print(f"📰 {headline}")
        print(f"🤖 AI Verdict: {emoji} {sentiment} ({confidence:.1f}% confidence)\n")
        
        # Add a 1-second pause so it looks like a real streaming terminal
        time.sleep(1) 
        
    except requests.exceptions.ConnectionError:
        print("🚨 ERROR: Could not connect to the AI Engine. Is your Docker container running on port 8000?")
        break
    except Exception as e:
        print(f"🚨 Unexpected API Error: {e}")