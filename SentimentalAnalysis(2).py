from flask import Flask, redirect, request, jsonify, render_template_string
from kiteconnect import KiteConnect
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from datetime import datetime, timedelta
import webbrowser, threading, time
import plotly.graph_objs as go
import json
from functools import lru_cache

# =====================
# KITE CONFIG
# =====================
API_KEY = "0x1niqzaa6tvxuid"
API_SECRET = "974md4plu221gr0xislzf3h4f9v15t1e"

kite = KiteConnect(api_key=API_KEY)
access_token = None

app = Flask(__name__)

# =====================
# HTML DASHBOARD
# =====================
HTML = """
<!DOCTYPE html>
<html>
<head>
  <title>Intraday Multi-Stock Dashboard</title>
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  <style>
    body { font-family: Arial; background:#f4f6f8; padding:20px; }
    table { width:100%; border-collapse:collapse; background:#fff; }
    th, td { padding:10px; border:1px solid #ccc; text-align:center; }
    th { background:#333; color:#fff; }
    .buy { color:green; font-weight:bold; }
    .sell { color:red; font-weight:bold; }
    .up { background:#d4edda; }
    .down { background:#f8d7da; }
    .box { margin-top:20px; background:#fff; padding:15px; border-radius:10px; }
  </style>
  <script>
    let countdown=60;
    function updateCountdown(){
        document.getElementById('timer').innerText=countdown+'s';
        countdown--;
        if(countdown<0){ countdown=60; fetchTable(); }
    }
    setInterval(updateCountdown,1000);

    function fetchTable(){
      fetch('/table').then(r=>r.json()).then(rows=>{
        const tb=document.getElementById('rows'); tb.innerHTML='';
        rows.forEach(r=>{
          const priceClass = r.price_change>0 ? 'up' : r.price_change<0 ? 'down' : '';
          const tr=document.createElement('tr');
          tr.innerHTML=`<td onclick=loadChart('${r.symbol}')>${r.symbol}</td>
                        <td class='${priceClass}'>${r.price.toFixed(2)}</td>
                        <td class='${r.css}'>${r.signal}</td>
                        <td>${r.confidence}%</td>
                        <td>${r.strength}</td>
                        <td>${r.rsi}</td>
                        <td>${r.trend}</td>`;
          tb.appendChild(tr);
        });
      });
    }

    function loadChart(symbol){
      fetch('/chart?symbol='+symbol).then(r=>r.json()).then(fig=>{
        Plotly.newPlot('chart', fig.data, fig.layout);
        const ul=document.getElementById('why'); ul.innerHTML='';
        fig.why.forEach(x=>{
          const li=document.createElement('li');
          li.innerText=x;
          ul.appendChild(li);
        });
      });
    }

    window.onload=fetchTable;
  </script>
</head>
<body>
<h2>📊 Intraday Multi-Stock Dashboard <span id='timer'>60s</span></h2>
<table>
<thead>
<tr>
  <th>Stock</th><th>Price</th><th>Signal</th><th>Confidence</th>
  <th>Strength</th><th>RSI</th><th>Trend</th>
</tr>
</thead>
<tbody id='rows'></tbody>
</table>

<div class='box'>
  <h3>📈 Intraday Chart</h3>
  <div id='chart'></div>
</div>

<div class='box'>
  <h3>Why BUY / SELL?</h3>
  <ul id='why'></ul>
</div>
</body>
</html>
"""

# =====================
# CACHING
# =====================
CACHE = {}
CACHE_TTL = 60  # seconds

@lru_cache(maxsize=1)
def get_instruments():
    return {
        i['tradingsymbol']: i['instrument_token']
        for i in kite.instruments('NSE')
    }

# =====================
# INDICATORS
# =====================
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def explain_and_strength(row, confidence, signal):
    reasons = []
    strength = int(confidence)

    if signal == 'BUY':
        if row['rsi'] < 30:
            reasons.append('RSI indicates oversold')
            strength += 10
        if row['ma5'] > row['ma10']:
            reasons.append('Uptrend MA5 > MA10')
            strength += 10
    else:
        if row['rsi'] > 70:
            reasons.append('RSI indicates overbought')
            strength += 10
        if row['ma5'] < row['ma10']:
            reasons.append('Downtrend MA5 < MA10')
            strength += 10

    strength = min(strength, 100)
    if not reasons:
        reasons.append('Mixed intraday signals')

    return reasons, strength

# =====================
# CORE COMPUTE
# =====================
def compute(symbol):
    now = time.time()

    if symbol in CACHE and now - CACHE[symbol]['ts'] < CACHE_TTL:
        return CACHE[symbol]['data']

    token = get_instruments().get(symbol)
    if not token:
        return None

    to_dt = datetime.now()
    from_dt = to_dt - timedelta(days=10)

    df = pd.DataFrame(
        kite.historical_data(token, from_dt, to_dt, interval='5minute')
    )

    if df.empty or len(df) < 40:
        return None

    df['ma5'] = df['close'].rolling(5).mean()
    df['ma10'] = df['close'].rolling(10).mean()
    df['rsi'] = compute_rsi(df['close'])
    df['return'] = df['close'].pct_change()
    df['vol_change'] = df['volume'].pct_change()
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    df.dropna(inplace=True)

    X = df[['return', 'ma5', 'ma10', 'rsi', 'vol_change']]
    y = df['target']

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(max_iter=300))
    ])
    model.fit(X[:-1], y[:-1])

    prob = model.predict_proba(X.iloc[[-1]])[0]
    pred = prob.argmax()

    confidence = round(prob.max() * 100, 2)
    signal = 'BUY' if pred == 1 else 'SELL'

    row = df.iloc[-1]
    why, strength = explain_and_strength(row, confidence, signal)

    result = {
        'symbol': symbol,
        'price': float(row['close']),
        'price_change': float(row['close'] - row['open']),
        'signal': signal,
        'css': 'buy' if signal == 'BUY' else 'sell',
        'confidence': confidence,
        'strength': strength,
        'rsi': round(row['rsi'], 2),
        'trend': 'UP' if row['ma5'] > row['ma10'] else 'DOWN',
        'why': why,
        'df': {
            'date': df['date'].astype(str).tolist(),
            'close': df['close'].tolist(),
            'ma5': df['ma5'].tolist(),
            'ma10': df['ma10'].tolist()
        }
    }

    CACHE[symbol] = {'data': result, 'ts': now}
    return result

# =====================
# ROUTES
# =====================
@app.route('/')
def start():
    return redirect(kite.login_url())

@app.route('/callback')
def callback():
    global access_token
    rt = request.args.get('request_token')
    data = kite.generate_session(rt, api_secret=API_SECRET)
    access_token = data['access_token']
    kite.set_access_token(access_token)
    return redirect('/dashboard')

@app.route('/dashboard')
def dashboard():
    return render_template_string(HTML)

@app.route('/table')
def table():
    symbols = ['RELIANCE', 'TCS', 'INFY']
    return jsonify([compute(s) for s in symbols if compute(s)])

@app.route('/chart')
def chart():
    symbol = request.args.get('symbol')
    r = compute(symbol)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=r['df']['date'], y=r['df']['close'], name='Price'))
    fig.add_trace(go.Scatter(x=r['df']['date'], y=r['df']['ma5'], name='MA5'))
    fig.add_trace(go.Scatter(x=r['df']['date'], y=r['df']['ma10'], name='MA10'))
    fig.update_layout(height=400)

    fig_json = json.loads(fig.to_json())
    return jsonify({'data': fig_json['data'], 'layout': fig_json['layout'], 'why': r['why']})

# =====================
# RUN
# =====================
def open_browser():
    webbrowser.open('http://127.0.0.1:8000')

if __name__ == '__main__':
    threading.Timer(1, open_browser).start()
    app.run(host='127.0.0.1', port=8000, debug=False, use_reloader=False)
