# -*- coding: utf-8 -*-
"""
Paper-trader MTF EMA200 (Kraken compatible)
- Timeframe base = 5m
- Filtre MTF : LONG si close > EMA200 sur 1h/4h/1d/1w, SHORT si close < EMA200
- Entrée : close vs EMA200 base (+ breakout optionnel)
- Sortie : SL/TP ou si le biais MTF se casse
- Résultats persistés : trades.csv, equity.csv, state.json
"""

import os, json, math, time
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
from typing import Dict
import ccxt
import datetime as dt

PARAMS = {
    "exchange": "kraken",
    "symbol": "BTC/USDT",   # XRP/USDT peut aussi marcher, BTC/USDT est sûr

    # Timeframes
    "tf_base": "5m",
    "tf_1h": "1h",
    "tf_4h": "4h",
    "tf_1d": "1d",
    "tf_1m": "1w",  # Kraken ne supporte pas '1M', on remplace par '1w'

    # Historique initial
    "since": "2024-01-01T00:00:00Z",
    "ccxt_limit": 500,
    "rate_sleep_sec": 1.2,

    # Indicateurs
    "ema_period": 200,

    # Déclencheurs
    "use_breakout": True,
    "breakout_lookback": 10,

    # Gestion du capital
    "capital_start": 200.0,
    "position_mode": "percent",
    "risk_pct": 0.20,
    "fixed_trade_size": 7.0,
    "fee_bps": 8.0,
    "sl_pct": 0.003,
    "tp_pct": 0.008,
    "allow_short": True,
    "one_position_at_a_time": True,
    "slippage_pct": 0.000,
    "cooldown_bars": 1,
    "intrabar_priority": "worst",

    # Levier
    "use_leverage": True,
    "leverage": 2.0,

    # Fichiers
    "out_dir": "out",
    "state_path": "state.json",
}

# ========= Fonctions utilitaires =========

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def highest(series: pd.Series, lookback: int) -> pd.Series:
    return series.rolling(lookback, min_periods=1).max()

def lowest(series: pd.Series, lookback: int) -> pd.Series:
    return series.rolling(lookback, min_periods=1).min()

@dataclass
class Position:
    side: str
    entry_time: str
    entry_price: float
    qty: float
    bar_index: int

def load_state(p: Dict) -> Dict:
    if not os.path.exists(p["state_path"]):
        return {"capital": p["capital_start"], "open_position": None, "last_bar_ts": None}
    with open(p["state_path"], "r", encoding="utf-8") as f:
        return json.load(f)

def save_state(p: Dict, state: Dict):
    with open(p["state_path"], "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

# ========= Gestion des timeframes =========

def tf_to_minutes(tf: str) -> int:
    m = {"1m":1,"3m":3,"5m":5,"15m":15,"30m":30,
         "1h":60,"2h":120,"4h":240,"6h":360,"8h":480,"12h":720,
         "1d":1440,"1w":10080}
    return m.get(tf, 1440)  # fallback jour

def bars_needed(tf: str, p: dict, is_base: bool) -> int:
    warm = 50
    if is_base:
        return p["ema_period"] + p.get("breakout_lookback", 0) + warm
    return p["ema_period"] + warm

def since_for(tf: str, p: dict, is_base: bool) -> int:
    minutes = tf_to_minutes(tf) * bars_needed(tf, p, is_base)
    start = pd.Timestamp.utcnow() - pd.Timedelta(minutes=minutes)
    return int(start.timestamp() * 1000)

def _ccxt_fetch(ex, symbol, tf, since_ms, limit, sleep_s, max_retries=6):
    rows, fetch_since = [], since_ms
    while True:
        for attempt in range(max_retries):
            try:
                ohlcv = ex.fetch_ohlcv(symbol, tf, since=fetch_since, limit=limit)
                break
            except (ccxt.DDoSProtection, ccxt.RateLimitExceeded, ccxt.NetworkError) as e:
                wait = min(2 ** attempt, 20)
                print(f"[RATE] {e.__class__.__name__} on {tf}, retry in {wait}s...")
                time.sleep(wait)
        else:
            raise
        if not ohlcv:
            break
        rows.extend(ohlcv)
        last_ts = ohlcv[-1][0]
        nxt = last_ts + 1
        if fetch_since and nxt <= fetch_since:
            break
        fetch_since = nxt
        time.sleep(sleep_s)
        if len(rows) > 10000:
            break
    cols = ["timestamp", "open", "high", "low", "close", "volume"]
    df = pd.DataFrame(rows, columns=["ts", *cols[1:]])
    if df.empty:
        return pd.DataFrame(columns=cols).set_index(pd.Index([], name="timestamp"))
    df["timestamp"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.drop(columns=["ts"]).drop_duplicates("timestamp").sort_values("timestamp").set_index("timestamp")
    return df

def load_all_timeframes(p: dict):
    ex = getattr(ccxt, p["exchange"])({"enableRateLimit": True, "timeout": 20000})
    tf_m = p.get("tf_1m", "1w")
    if tf_m.lower() == "1m":  # kraken n’a pas de monthly natif
        tf_m = "1w"

    s_base = since_for(p["tf_base"], p, True)
    s_1h   = since_for(p["tf_1h"],   p, False)
    s_4h   = since_for(p["tf_4h"],   p, False)
    s_1d   = since_for(p["tf_1d"],   p, False)
    s_1m   = since_for(tf_m,         p, False)

    base = _ccxt_fetch(ex, p["symbol"], p["tf_base"], s_base, p["ccxt_limit"], p["rate_sleep_sec"])
    time.sleep(1.5)
    tf1h = _ccxt_fetch(ex, p["symbol"], p["tf_1h"], s_1h, p["ccxt_limit"], p["rate_sleep_sec"])
    time.sleep(1.5)
    tf4h = _ccxt_fetch(ex, p["symbol"], p["tf_4h"], s_4h, p["ccxt_limit"], p["rate_sleep_sec"])
    time.sleep(1.5)
    tf1d = _ccxt_fetch(ex, p["symbol"], p["tf_1d"], s_1d, p["ccxt_limit"], p["rate_sleep_sec"])
    time.sleep(1.5)
    tf1m = _ccxt_fetch(ex, p["symbol"], tf_m,       s_1m, p["ccxt_limit"], p["rate_sleep_sec"])

    return base, tf1h, tf4h, tf1d, tf1m

# ========= Signaux =========

def align_htf_on_base(df_base, df_htf, ema_col):
    keep = df_htf[["close", ema_col]].rename(columns={"close": "htf_close"})
    return keep.reindex(df_base.index, method="ffill")

def compute_signals_mtf(df_base, df_1h, df_4h, df_1d, df_1m, p):
    df_base["ema200_base"] = ema(df_base["close"], p["ema_period"])
    for df in (df_1h, df_4h, df_1d, df_1m):
        df["ema200"] = ema(df["close"], p["ema_period"])
    a1h, a4h, a1d, a1m = [align_htf_on_base(df_base, df, "ema200") for df in (df_1h, df_4h, df_1d, df_1m)]
    mtf_long = (a1h["htf_close"] > a1h["ema200"]) & (a4h["htf_close"] > a4h["ema200"]) & (a1d["htf_close"] > a1d["ema200"]) & (a1m["htf_close"] > a1m["ema200"])
    mtf_short = (a1h["htf_close"] < a1h["ema200"]) & (a4h["htf_close"] < a4h["ema200"]) & (a1d["htf_close"] < a1d["ema200"]) & (a1m["htf_close"] < a1m["ema200"])
    cond_long, cond_short = df_base["close"] > df_base["ema200_base"], df_base["close"] < df_base["ema200_base"]
    if p["use_breakout"]:
        hh = highest(df_base["high"], p["breakout_lookback"]).shift(1)
        ll = lowest(df_base["low"], p["breakout_lookback"]).shift(1)
        cond_long &= df_base["close"] > hh
        cond_short &= df_base["close"] < ll
    sig = pd.Series(0, index=df_base.index)
    sig[(mtf_long & cond_long)] = 1
    sig[(mtf_short & cond_short)] = -1
    return sig, mtf_long, mtf_short

def append_csv(path, header, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame([row])
    if os.path.exists(path):
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, header=header, index=False)

# ========= Run principal =========

def run_once(p: Dict):
    state = load_state(p)
    base, tf1h, tf4h, tf1d, tf1m = load_all_timeframes(p)
    sig, mtf_long, mtf_short = compute_signals_mtf(base, tf1h, tf4h, tf1d, tf1m, p)
    if len(base) < 2:
        return
    ts = base.index[-2]
    if state.get("last_bar_ts") == ts.isoformat():
        return
    c, o, h, l = base["close"].iloc[-2], base["open"].iloc[-2], base["high"].iloc[-2], base["low"].iloc[-2]
    pos = state.get("open_position")
    fee = p["fee_bps"]/10000.0
    if pos:
        side, ep, qty = pos["side"], pos["entry_price"], pos["qty"]
        closed, reason = False, None
        if side == "long":
            if l <= ep*(1-p["sl_pct"]): closed, reason, px = True, "sl", ep*(1-p["sl_pct"])
            elif h >= ep*(1+p["tp_pct"]): closed, reason, px = True, "tp", ep*(1+p["tp_pct"])
        else:
            if h >= ep*(1+p["sl_pct"]): closed, reason, px = True, "sl", ep*(1+p["sl_pct"])
            elif l <= ep*(1-p["tp_pct"]): closed, reason, px = True, "tp", ep*(1-p["tp_pct"])
        if closed:
            pnl = (px-c*fee-ep) * qty if side=="long" else (ep-c*fee-px) * qty
            state["capital"] += pnl
            trade = {"entry_time": pos["entry_time"], "exit_time": ts.isoformat(), "side": side, "entry_price": ep, "exit_price": px, "qty": qty, "pnl": pnl, "reason": reason}
            append_csv(os.path.join(p["out_dir"],"trades.csv"), trade.keys(), trade)
            state["open_position"] = None
    if not state.get("open_position"):
        if sig.iloc[-2] == 1 and mtf_long.iloc[-2]:
            notion = state["capital"]*p["risk_pct"]
            qty = (notion*p["leverage"])/c
            state["open_position"] = asdict(Position("long", ts.isoformat(), c, qty, len(base)-2))
        elif sig.iloc[-2] == -1 and mtf_short.iloc[-2]:
            notion = state["capital"]*p["risk_pct"]
            qty = (notion*p["leverage"])/c
            state["open_position"] = asdict(Position("short", ts.isoformat(), c, qty, len(base)-2))
    append_csv(os.path.join(p["out_dir"],"equity.csv"), ["timestamp","equity"], {"timestamp": ts.isoformat(), "equity": state["capital"]})
    state["last_bar_ts"] = ts.isoformat()
    save_state(p, state)

if __name__ == "__main__":
    run_once(PARAMS)
    print("OK")
