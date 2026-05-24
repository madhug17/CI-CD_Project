import finnhub
import pandas as pd
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("FINNHUB_API_KEY")

if not api_key:
    raise EnvironmentError(
        "FINNHUB_API_KEY not found. "
        "Add it to your .env file: FINNHUB_API_KEY=your_key_here"
    )

finnhub_client = finnhub.Client(api_key=api_key)


def fetch_and_save_daily_news():

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Fetching daily market news from Finnhub...")

    try:
        news_data = finnhub_client.general_news("general", min_id=0)

        if not news_data:
            print("No news returned from Finnhub API.")
            return None

        df = pd.DataFrame(news_data)

        # Keep only the columns we care about
        columns_to_keep = ["datetime", "headline", "summary", "source", "url"]
        df = df[columns_to_keep]

        # Convert unix timestamp → readable datetime
        df["datetime"] = pd.to_datetime(df["datetime"], unit="s")

        # --------------------------------------------------
        # BUG FIX: create directory if it doesn't exist yet
        # Without this, df.to_csv() crashes silently
        # --------------------------------------------------
        output_dir = "data/inference_logs"
        os.makedirs(output_dir, exist_ok=True)

        today_str = datetime.now().strftime("%Y-%m-%d")
        file_path = f"{output_dir}/news_batch_{today_str}.csv"

        df.to_csv(file_path, index=False, encoding="utf-8")

        print(f"Saved {len(df)} articles → {file_path}")
        return file_path

    except Exception as e:
        print(f"Error fetching news: {e}")
        return None


if __name__ == "__main__":
    fetch_and_save_daily_news()