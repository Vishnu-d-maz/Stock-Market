# =====================================================
# SINGLE-FILE KITE INTRADAY DASHBOARD – 5-MIN PREDICTIONS
# Fully integrated with auto-login, access token, and explainable predictions
# =====================================================

from flask import Flask, redirect, request, jsonify, render_template_string
from kiteconnect import KiteConnect
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from datetime import datetime, timedelta
import webbrowser, threading

# =====================
# KITE CONFIG (FILL THESE)
# =====================
API_KEY = "0x1niqzaa6tvxuid"
API_SECRET = "974md4plu221gr0xislzf3h4f9v15t1e"
REDIRECT_URL = "http://127.0.0.1:8000/callback"

kite = KiteConnect(api_key=API_KEY)
access_token = None

app = Flask(__name__)

# =====================
# DASHBOARD HTML
# =====================
HTML = """
<!DOCTYPE html>
<html>
<head>
  <title>Kite 5-min Intraday Dashboard</title>
  <style>
    body { font-family: Arial; background:#f4f6f8; padding:30px; }
    .card { background:#fff; padding:20px; border-radius:10px; width:720px; margin:auto; box-shadow:0 0 10px rgba(0,0,0,.1); }
    .buy { color:green; font-weight:bold; }
    .sell { color:red; font-weight:bold; }
    select { padding:6px; width:100%; margin-bottom:10px; }
  </style>
  <script>
    function fetchData(){
      const s=document.getElementById('symbol').value;
      fetch('/update?symbol='+s).then(r=>r.json()).then(d=>{
        if(d.error){ alert(d.error); return; }
        date.innerText=d.date; price.innerText=d.price;
        rsi.innerText=d.rsi; ma5.innerText=d.ma5; ma10.innerText=d.ma10;
        confidence.innerText=d.confidence+'%';
        signal.innerText=d.signal; signal.className=d.css;
        const ul=document.getElementById('why'); ul.innerHTML='';
        d.why.forEach(x=>{const li=document.createElement('li'); li.innerText=x; ul.appendChild(li);});
      });
    }
    setInterval(fetchData,60000); window.onload=fetchData;
  </script>
</head>
<body>
  <div class="card">
    <h2>📈 Kite 5-min Intraday Dashboard</h2>
    <select id="symbol" onchange="fetchData()">
      <option value="RELIANCE">RELIANCE</option>
      <option value="TCS">TCS</option>
      <option value="INFY">INFY</option>
    </select>
    <p><b>Date:</b> <span id="date"></span></p>
    <p><b>Last Price:</b> <span id="price"></span></p>
    <p><b>RSI:</b> <span id="rsi"></span></p>
    <p><b>MA5:</b> <span id="ma5"></span></p>
    <p><b>MA10:</b> <span id="ma10"></span></p>
    <p><b>Confidence:</b> <span id="confidence"></span></p>
    <p><b>Prediction:</b> <span id="signal"></span></p>
    <h3>Why this decision?</h3>
    <ul id="why"></ul>
  </div>
</body>
</html>
"""

# =====================
# EXPLAIN LOGIC
# =====================
def explain_decision(row, confidence, signal):
    reasons=[]
    if signal=='BUY':
        if row['rsi']<30: reasons.append('RSI indicates oversold conditions')
        if row['ma5']>row['ma10']: reasons.append('Uptrend detected (MA5 > MA10)')
        if confidence>60: reasons.append('Model confidence is high')
    else:
        if row['rsi']>70: reasons.append('RSI indicates overbought conditions')
        if row['ma5']<row['ma10']: reasons.append('Downtrend detected (MA5 < MA10)')
    if not reasons: reasons.append('Decision based on mixed signals')
    return reasons

# =====================
# ROUTES
# =====================
@app.route('/')
def start():
    return redirect(kite.login_url())

@app.route('/callback')
def callback():
    global access_token
    rt=request.args.get('request_token')
    if not rt: return 'ERROR: request_token missing'
    data=kite.generate_session(rt, api_secret=API_SECRET)
    access_token=data['access_token']
    kite.set_access_token(access_token)
    return redirect('/dashboard')

@app.route('/dashboard')
def dashboard():
    if not access_token: return redirect('/')
    return render_template_string(HTML)

@app.route('/update')
def update():
    if not access_token: return jsonify({'error':'Not authenticated'})

    symbol=request.args.get('symbol','RELIANCE')
    inst=[i for i in kite.instruments('NSE') if i['tradingsymbol']==symbol]
    if not inst: return jsonify({'error':'Symbol not found'})
    token=inst[0]['instrument_token']

    # Fetch last 10 days intraday 5-min bars
    to_dt=datetime.now()
    from_dt=to_dt - timedelta(days=10)
    data=kite.historical_data(token, from_dt, to_dt, interval='5minute')
    df=pd.DataFrame(data)
    if df.empty or len(df)<20: return jsonify({'error':'Not enough data'})

    # Indicators
    df['return']=df['close'].pct_change()
    df['ma5']=df['close'].rolling(5).mean()
    df['ma10']=df['close'].rolling(10).mean()
    d=df['close'].diff(); g=d.clip(lower=0).rolling(14).mean(); l=(-d.clip(upper=0)).rolling(14).mean()
    rs=g/l; df['rsi']=100-(100/(1+rs))
    df['vol_change']=df['volume'].pct_change()
    df['target']=(df['close'].shift(-1)>df['close']).astype(int)

    df.dropna(inplace=True)
    if df.empty: return jsonify({'error':'Insufficient valid data'})

    # Model
    X=df[['return','ma5','ma10','rsi','vol_change']]; y=df['target']
    model=Pipeline([('scaler',StandardScaler()),('lr',LogisticRegression(max_iter=500))])
    model.fit(X[:-1],y[:-1])

    latest=X.iloc[[-1]]
    prob=model.predict_proba(latest)[0]; pred=np.argmax(prob)
    confidence=round(max(prob)*100,2)
    signal='BUY' if pred==1 else 'SELL'; css='buy' if signal=='BUY' else 'sell'

    row=df.iloc[-1]
    why=explain_decision(row, confidence, signal)

    return jsonify({
        'date': row['date'].strftime('%Y-%m-%d %H:%M'),
        'price': round(row['close'],2),
        'rsi': round(row['rsi'],2),
        'ma5': round(row['ma5'],2),
        'ma10': round(row['ma10'],2),
        'confidence': confidence,
        'signal': signal,
        'css': css,
        'why': why
    })

# =====================
# RUN
# =====================
def open_browser(): webbrowser.open('http://127.0.0.1:8000')

if __name__=='__main__':
    threading.Timer(1, open_browser).start()
    app.run(host='127.0.0.1', port=8000, debug=False, use_reloader=False)
