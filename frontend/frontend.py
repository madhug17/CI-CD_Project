import streamlit as st
import requests
import yfinance as yf
import time

st.set_page_config(page_title="FinBERT AI", page_icon="📈", layout="centered")

st.title("📈 Live Financial News Analyzer")
st.write("Powered by FinBERT, PyTorch, Docker, and Yahoo Finance")
st.markdown("---")

# User selects which stock to track
symbol = st.text_input("Enter a Stock Ticker (e.g., NVDA, AAPL, TSLA):", value="NVDA").upper()

if st.button(f"Fetch Live Pulse for {symbol}", type="primary"):
    with st.spinner(f"Scraping live news for {symbol}..."):
        try:
            # 1. Scrape the internet
            ticker = yf.Ticker(symbol)
            live_news = ticker.news
            
            headlines = []
            for article in live_news:
                if 'content' in article and 'title' in article['content']:
                    headlines.append(article['content']['title'])
                elif 'title' in article:
                    headlines.append(article['title'])
                    
            if not headlines:
                st.warning("Yahoo Finance blocked the request or returned no news. Try again in a minute.")
            else:
                st.success(f"Successfully pulled {min(len(headlines), 10)} live articles!")
                st.markdown("---")
                
                # 2. Score each headline through the Docker API
                for headline in headlines[:10]:
                    st.write(f"📰 **{headline}**")
                    
                    response = requests.post(
                        "https://ci-cd-project-01.onrender.com/predict",
                        json={"text": headline}
                    )
                    response.raise_for_status()
                    data = response.json()
                    
                    sentiment = data['prediction']
                    confidence = data['confidence'] * 100
                    
                    if sentiment == "Bullish":
                        st.success(f"🤖 AI Verdict: 🟢 {sentiment} ({confidence:.1f}% confidence)")
                    elif sentiment == "Bearish":
                        st.error(f"🤖 AI Verdict: 🔴 {sentiment} ({confidence:.1f}% confidence)")
                    else:
                        st.info(f"🤖 AI Verdict: ⚪ {sentiment} ({confidence:.1f}% confidence)")
                    
                    st.write("") # Add a little space between articles
                    time.sleep(0.5) # Slight pause so it doesn't overload your Docker container
                    
        except requests.exceptions.ConnectionError:
            st.error("🚨 API Connection Failed! Make sure your Docker container is running on port 8000.")
        except Exception as e:
            st.error(f"An error occurred: {e}")