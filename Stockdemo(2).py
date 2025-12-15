# ======================================================
# MEAN REVERSION OPTIONS TRADING
# LIVE TP / SL + CONTINUOUS LIVE PnL (PAPER)
# ======================================================

print("🔥 FILE STARTED", flush=True)

from kiteconnect import KiteConnect
import pandas as pd
import datetime as dt
import time

# ======================================================
# KITE API SETUP
# ======================================================
API_KEY = "0x1niqzaa6tvxuid"
ACCESS_TOKEN = "AKCw2vhD0Sgl5ETLWKJt6VUl3GVjT6pS"

kite = KiteConnect(api_key=API_KEY)
kite.set_access_token(ACCESS_TOKEN)

print("✅ Kite connected", flush=True)

# ======================================================
# CONFIGURATION
# ======================================================
SYMBOL = "BANKNIFTY"        # Change if needed
INSTRUMENT_TOKEN = 260105
LOT_SIZE = 15

INTERVAL = "minute"
TARGET_POINTS = 2
STOPLOSS_POINTS = 2

# ======================================================
# LOAD INSTRUMENTS
# ======================================================
print("⏳ Loading instruments...", flush=True)
INSTRUMENTS = pd.DataFrame(kite.instruments("NFO"))
print("✅ Instruments loaded", flush=True)

# ======================================================
# TRADE STATE
# ======================================================
TRADE_ACTIVE = False
ENTRY_PRICE = 0.0
CURRENT_OPTION = None
TARGET_PRICE = 0.0
STOPLOSS_PRICE = 0.0

# ======================================================
# FUNCTIONS
# ======================================================

def get_1min_data(token):
    to_date = dt.datetime.now()
    from_date = to_date - dt.timedelta(days=2)

    data = kite.historical_data(
        token,
        from_date,
        to_date,
        interval=INTERVAL,
        oi=True
    )
    return pd.DataFrame(data)


def add_bollinger(df, period=3):
    df["SMA"] = df["close"].rolling(period).mean()
    df["STD"] = df["close"].rolling(period).std()
    df["UPPER"] = df["SMA"] + 1 * df["STD"]
    df["LOWER"] = df["SMA"] - 1 * df["STD"]
    return df


def get_signal(df):
    last = df.iloc[-1]
    if last["close"] <= last["LOWER"]:
        return "BUY"
    elif last["close"] >= last["UPPER"]:
        return "SELL"
    return "HOLD"


def get_nearest_expiry(symbol):
    df = INSTRUMENTS[INSTRUMENTS["name"] == symbol]
    return sorted(df["expiry"].unique())[0]


def select_option(symbol, signal, spot_price):
    expiry = get_nearest_expiry(symbol)

    df = INSTRUMENTS[
        (INSTRUMENTS["name"] == symbol) &
        (INSTRUMENTS["expiry"] == expiry)
    ].copy()

    df["distance"] = abs(df["strike"] - spot_price)
    df = df[df["distance"] <= 200]

    if df.empty:
        return None

    if signal == "BUY":
        opt = df[df["instrument_type"] == "CE"]
    else:
        opt = df[df["instrument_type"] == "PE"]

    return opt.sort_values("distance").iloc[0]["tradingsymbol"]


def enter_trade(option):
    global TRADE_ACTIVE, ENTRY_PRICE, CURRENT_OPTION
    global TARGET_PRICE, STOPLOSS_PRICE

    ENTRY_PRICE = kite.ltp(f"NFO:{option}")[f"NFO:{option}"]["last_price"]
    TARGET_PRICE = ENTRY_PRICE + TARGET_POINTS
    STOPLOSS_PRICE = ENTRY_PRICE - STOPLOSS_POINTS
    CURRENT_OPTION = option
    TRADE_ACTIVE = True

    print(
        f"🟢 ENTER TRADE → {option} @ {ENTRY_PRICE}",
        flush=True
    )


def monitor_trade_live():
    global TRADE_ACTIVE

    while TRADE_ACTIVE:
        ltp = kite.ltp(f"NFO:{CURRENT_OPTION}")[f"NFO:{CURRENT_OPTION}"]["last_price"]
        pnl = (ltp - ENTRY_PRICE) * LOT_SIZE

        print(
            f"| 🎯 TP: {TARGET_PRICE} | 🛑 SL: {STOPLOSS_PRICE} | "
            f"📈 LTP: {ltp} | Entry: {ENTRY_PRICE} | Live PnL: ₹{round(pnl,2)}",
            flush=True
        )

        if ltp >= TARGET_PRICE:
            print("🎯 TARGET HIT → EXIT TRADE\n", flush=True)
            TRADE_ACTIVE = False
            break

        if ltp <= STOPLOSS_PRICE:
            print("🛑 STOPLOSS HIT → EXIT TRADE\n", flush=True)
            TRADE_ACTIVE = False
            break

        time.sleep(2)

# ======================================================
# MAIN LOOP
# ======================================================
def main():
    print("🚀 Strategy Started (LIVE TP / SL MODE)", flush=True)

    while True:
        df = get_1min_data(INSTRUMENT_TOKEN)

        if df is None or len(df) < 3:
            print("⏳ Waiting for sufficient candles...", flush=True)
            time.sleep(10)
            continue

        df = add_bollinger(df)
        signal = get_signal(df)
        spot_price = df.iloc[-1]["close"]

        print(f"📊 Signal: {signal} | Spot: {spot_price}", flush=True)

        if signal in ["BUY", "SELL"] and not TRADE_ACTIVE:
            option = select_option(SYMBOL, signal, spot_price)
            if option:
                enter_trade(option)
                monitor_trade_live()

        print("⏳ Waiting for next candle...\n", flush=True)
        time.sleep(60)


if __name__ == "__main__":
    main()
