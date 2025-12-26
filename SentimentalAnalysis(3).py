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
# OPTIONS CONFIG
# =====================
INDEX = "NIFTY"          # NIFTY / BANKNIFTY
STRIKE_STEP = 50
MIN_CONFIDENCE = 60

# =====================
# HTML DASHBOARD
# =====================
HTML = """
<!DOCTYPE html>
<html>
<head>
<title>Live Options Dashboard</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
body { font-family: Arial; background:#f4f6f8; padding:20px; }
table { width:100%; border-collapse:collapse; background:#fff; }
th, td { padding:10px; border:1px solid #ccc; text-align:center; }
th { background:#333; color:#fff; }
.buy { color:green; font-weight:bold; }
.sell { color:red; font-weight:bold; }
.box { margin-top:20px; background:#fff; padding:15px; border-radius:10px; }
</style>
<script>
function fetchTable(){
 fetch('/table').then(r=>r.json()).then(rows=>{
  const tb=document.getElementById('rows'); tb.innerHTML='';
  rows.forEach(r=>{
   const tr=document.createElement('tr');
   tr.innerHTML=`
    <td onclick=loadChart('${r.symbol}')>${r.symbol}</td>
    <td>${r.price.toFixed(2)}</td>
    <td class='${r.signal.includes("BUY")?"buy":"sell"}'>${r.signal}</td>
    <td>${r.option || '-'}</td>
    <td>${r.option_ltp ? r.option_ltp.toFixed(2) : '-'}</td>
    <td>${r.confidence}%</td>`;
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
setInterval(fetchTable,5000);
window.onload=fetchTable;
</script>
</head>
<body>

<h2>📊 Live Options Trading Dashboard</h2>

<table>
<thead>
<tr>
<th>Stock</th>
<th>Price</th>
<th>Signal</th>
<th>Option</th>
<th>Option LTP</th>
<th>Confidence</th>
</tr>
</thead>
<tbody id="rows"></tbody>
</table>

<div class="box">
<h3>📈 Chart</h3>
<div id="chart"></div>
</div>

<div class="box">
<h3>Why Trade?</h3>
<ul id="why"></ul>
</div>

</body>
</html>
"""

# =====================
# HELPERS
# =====================
@lru_cache(maxsize=1)
def get_instruments():
    instruments = kite.instruments()
    return {f"{i['exchange']}:{i['tradingsymbol']}": i['instrument_token']
            for i in instruments}

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def get_atm_strike(price):
    return round(price / STRIKE_STEP) * STRIKE_STEP

def build_option_symbol(index, expiry, strike, opt_type):
    return f"{index}{expiry}{strike}{opt_type}"

def get_option_ltp(symbol):
    try:
        data = kite.ltp([f"NFO:{symbol}"])
        return data[f"NFO:{symbol}"]["last_price"]
    except:
        return None

# =====================
# CORE LOGIC
# =====================
def compute(symbol):
    eq_token = get_instruments().get(f"NSE:{symbol}")
    if not eq_token:
        return None

    to_dt = datetime.now()
    from_dt = to_dt - timedelta(days=7)

    df = pd.DataFrame(
        kite.historical_data(eq_token, from_dt, to_dt, interval='5minute')
    )

    if df.empty or len(df) < 40:
        return None

    df['ma5'] = df['close'].rolling(5).mean()
    df['ma10'] = df['close'].rolling(10).mean()
    df['rsi'] = compute_rsi(df['close'])
    df['ret'] = df['close'].pct_change()
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    df.dropna(inplace=True)

    X = df[['ret', 'ma5', 'ma10', 'rsi']]
    y = df['target']

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(max_iter=300))
    ])
    model.fit(X[:-1], y[:-1])

    prob = model.predict_proba(X.iloc[[-1]])[0]
    confidence = round(prob.max() * 100, 2)
    pred = prob.argmax()
    row = df.iloc[-1]

    option_symbol = None
    option_ltp = None
    signal = "NO TRADE"

    if confidence >= MIN_CONFIDENCE:
        strike = get_atm_strike(row['close'])
        expiry = datetime.now().strftime('%y%b').upper()
        if pred == 1:
            signal = "BUY CE"
            option_symbol = build_option_symbol(INDEX, expiry, strike, "CE")
        else:
            signal = "BUY PE"
            option_symbol = build_option_symbol(INDEX, expiry, strike, "PE")

        option_ltp = get_option_ltp(option_symbol)

    why = []
    if row['rsi'] < 30: why.append("RSI Oversold")
    if row['rsi'] > 70: why.append("RSI Overbought")
    if row['ma5'] > row['ma10']: why.append("Uptrend")
    else: why.append("Downtrend")

    return {
        'symbol': symbol,
        'price': float(row['close']),
        'signal': signal,
        'option': option_symbol,
        'option_ltp': option_ltp,
        'confidence': confidence,
        'why': why,
        'df': {
            'date': df['date'].astype(str).tolist(),
            'close': df['close'].tolist(),
            'ma5': df['ma5'].tolist(),
            'ma10': df['ma10'].tolist()
        }
    }

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
    webbrowser.open("http://127.0.0.1:8000")

if __name__ == '__main__':
    threading.Timer(1, open_browser).start()
    app.run(port=8000, debug=False, use_reloader=False)
