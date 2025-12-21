from flask import Flask, render_template_string, request, jsonify
import pandas as pd
import numpy as np
import yfinance as yf
import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.linear_model import LogisticRegression
from datetime import datetime, timedelta
import time

app = Flask(__name__)

# Cache dictionary: {symbol: (timestamp, sentiment)}
sentiment_cache = {}

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Live Stock Dashboard</title>
    <style>
        body { font-family: Arial; background: #f4f6f8; padding: 30px; }
        .card { background: white; padding: 20px; border-radius: 10px; width: 520px; margin: auto; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
        h1 { text-align: center; }
        .buy { color: green; font-weight: bold; }
        .sell { color: red; font-weight: bold; }
        select { padding: 6px; width: 100%; margin-bottom: 10px; }
    </style>
    <script>
        function fetchData() {
            const symbol = document.getElementById("symbol").value;
            fetch("/update?symbol=" + symbol)
            .then(response => response.json())
            .then(data => {
                if(data.error){ alert(data.error); return; }
                document.getElementById("date").innerText = data.date;
                document.getElementById("price").innerText = data.price;
                document.getElementById("sentiment").innerText = data.sentiment;
                document.getElementById("confidence").innerText = data.confidence + "%";
                document.getElementById("signal").innerText = data.signal;
                document.getElementById("signal").className = data.css;
            });
        }
        setInterval(fetchData, 60000); // every 60 seconds
        window.onload = fetchData;
    </script>
</head>
<body>
<div class="card">
    <h1>📈 Live Stock Dashboard</h1>
    <form>
        <select id="symbol" onchange="fetchData()">
            <option value="AAPL">Apple (AAPL)</option>
            <option value="TSLA">Tesla (TSLA)</option>
            <option value="MSFT">Microsoft (MSFT)</option>
        </select>
    </form>
    <p><b>Date:</b> <span id="date"></span></p>
    <p><b>Close Price:</b> <span id="price"></span></p>
    <p><b>Sentiment (VADER):</b> <span id="sentiment"></span></p>
    <p><b>Confidence:</b> <span id="confidence"></span></p>
    <p><b>Prediction:</b> <span id="signal"></span></p>
</div>
</body>
</html>
"""

# Function to get live sentiment
def get_sentiment(symbol):
    now = time.time()
    if symbol in sentiment_cache:
        ts, cached_sentiment = sentiment_cache[symbol]
        if now - ts < 120:  # cache for 2 minutes
            return cached_sentiment

    news_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US"
    feed = feedparser.parse(news_url)
    analyzer = SentimentIntensityAnalyzer()

    if not feed.entries:
        avg_sentiment = 0.0
    else:
        sentiments = [analyzer.polarity_scores(e.title)['compound'] for e in feed.entries[:20]]
        avg_sentiment = sum(sentiments)/len(sentiments)

    sentiment_cache[symbol] = (now, avg_sentiment)
    return avg_sentiment

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/update")
def update():
    symbol = request.args.get("symbol", "AAPL")

    # 1. Fetch stock data
    end_date = datetime.today()
    start_date = end_date - timedelta(days=30)
    stock = yf.download(symbol, start=start_date, end=end_date, auto_adjust=False, progress=False, threads=False)
    if stock.empty:
        return jsonify({"error":"No stock data available"})

    stock.reset_index(inplace=True)
    stock.columns = stock.columns.get_level_values(0)  # flatten multi-index if exists

    latest_price = round(stock.iloc[-1]['Close'], 2)
    date_str = pd.to_datetime(stock['Date'].iloc[-1]).strftime('%Y-%m-%d')

    # 2. Get sentiment
    sentiment = round(get_sentiment(symbol),3)

    # 3. Model prediction
    stock['target'] = (stock['Close'] > stock['Open']).astype(int)
    X = stock[['Open','Volume']]
    y = stock['target']

    if len(X) < 5:
        return jsonify({"error":"Not enough data to train model"})

    model = LogisticRegression(max_iter=200)
    model.fit(X[:-1], y[:-1])

    latest = X.iloc[[-1]]
    prob = model.predict_proba(latest)[0]
    prediction = np.argmax(prob)
    confidence = round(max(prob)*100,2)
    signal = "BUY" if prediction==1 else "SELL"
    css = "buy" if signal=="BUY" else "sell"

    # Debug output (optional)
    print(f"{symbol}: price={latest_price}, sentiment={sentiment}, confidence={confidence}, signal={signal}")

    return jsonify({
        "date": date_str,
        "price": latest_price,
        "sentiment": sentiment,
        "confidence": confidence,
        "signal": signal,
        "css": css
    })

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=False, use_reloader=False)
