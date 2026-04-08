# ============================================================================
# HedGEX — Cascade Backtest Engine  v3
# Powered by NYZTrade Analytics Pvt. Ltd.
#
# v3 changes vs v2:
#   • Results tab: full tabulation with Buy Price, Sell Price, P&L per lot,
#     sub-tabs for Summary / CALL trades / PUT trades / Monthly / Winners-Losers
#   • Intraday vs CNC backtesting modes (separate runs, separate DB tables)
#   • Data Management tab: granular reset — trades only, signals only,
#     raw chain only, or full wipe per symbol or all symbols
# ============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm
from datetime import datetime, timedelta, date
import pytz, requests, time, sqlite3, json, os, warnings
from typing import List, Dict, Optional, Tuple
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="HedGEX Cascade Backtest",
                   page_icon="📊", layout="wide",
                   initial_sidebar_state="expanded")

# ── Constants ─────────────────────────────────────────────────────────────────
IST       = pytz.timezone("Asia/Kolkata")
DHAN_BASE = "https://api.dhan.co/v2"
RISK_FREE = 0.07
DB_PATH   = "hedgex_backtest.db"
CKPT_PATH = "hedgex_checkpoint.json"

DHAN_CLIENT_ID    = "1100480354"
DHAN_ACCESS_TOKEN = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9.eyJpc3MiOiJkaGFuIiwicGFydG5lcklkIjoiIiwiZXhwIjoxNzc1NzIwNzAwLCJhcHBfaWQiOiJhYjYxZmJmOSIsImlhdCI6MTc3NTYzNDMwMCwidG9rZW5Db25zdW1lclR5cGUiOiJBUFAiLCJ3ZWJob29rVXJsIjoiIiwiZGhhbkNsaWVudElkIjoiMTEwMDQ4MDM1NCJ9.FQxQjX8a3pc4SmMCjqd5yk2S-juo140hlWNGg_0_MqpHIm6mgki5v0315FFvAIfWxlnuNY5R31RjTTfSPxD0-w"
DHAN_INDEX_SECURITY_IDS = {
    "NIFTY": 13, "BANKNIFTY": 25, "FINNIFTY": 27, "MIDCPNIFTY": 442, "SENSEX": 51,
}
BSE_FNO_SYMBOLS = {"SENSEX"}

INDEX_CONFIG = {
    "NIFTY":      {"contract_size": 25,  "strike_interval": 50},
    "BANKNIFTY":  {"contract_size": 15,  "strike_interval": 100},
    "FINNIFTY":   {"contract_size": 40,  "strike_interval": 50},
    "MIDCPNIFTY": {"contract_size": 75,  "strike_interval": 25},
    "SENSEX":     {"contract_size": 10,  "strike_interval": 200},
}

INSTRUMENT_PARAMS = {
    "NIFTY":      {"pts_per_unit": 0.010, "strike_cap": 150},
    "BANKNIFTY":  {"pts_per_unit": 0.033, "strike_cap": 300},
    "FINNIFTY":   {"pts_per_unit": 0.050, "strike_cap": 150},
    "MIDCPNIFTY": {"pts_per_unit": 0.050, "strike_cap":  75},
    "SENSEX":     {"pts_per_unit": 0.025, "strike_cap": 500},
}

# CNC holding days (simulated multi-day hold)
CNC_HOLD_DAYS = 5

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Space+Grotesk:wght@400;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Space Grotesk',sans-serif;}
.bt-header{background:linear-gradient(135deg,#0f0c29,#302b63,#24243e);
  border:1px solid rgba(168,85,247,0.3);border-radius:16px;padding:28px 36px;margin-bottom:20px;}
.bt-title{font-size:2rem;font-weight:800;letter-spacing:-0.02em;
  background:linear-gradient(135deg,#00f5c4,#00d4ff,#a78bfa);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;}
.bt-sub{font-family:"JetBrains Mono",monospace;color:rgba(255,255,255,0.45);font-size:0.82rem;margin-top:4px;}
.metric-card{background:rgba(255,255,255,0.04);border:1px solid rgba(168,85,247,0.2);
  border-radius:12px;padding:16px 20px;text-align:center;}
.metric-val{font-size:1.6rem;font-weight:800;}
.metric-lbl{font-size:0.72rem;color:rgba(255,255,255,0.45);font-family:"JetBrains Mono",monospace;margin-top:4px;}
.g{color:#10b981;} .r{color:#ef4444;} .n{color:#94a3b8;}
.info-box{background:rgba(6,182,212,0.08);border-left:3px solid #06b6d4;
  border-radius:6px;padding:10px 14px;font-family:"JetBrains Mono",monospace;font-size:0.80rem;line-height:1.8;}
.warn-box{background:rgba(245,158,11,0.08);border-left:3px solid #f59e0b;
  border-radius:6px;padding:10px 14px;font-family:"JetBrains Mono",monospace;font-size:0.80rem;line-height:1.8;}
.danger-box{background:rgba(239,68,68,0.08);border-left:3px solid #ef4444;
  border-radius:6px;padding:10px 14px;font-family:"JetBrains Mono",monospace;font-size:0.80rem;line-height:1.8;}
.mode-intraday{background:linear-gradient(135deg,rgba(0,212,255,0.15),rgba(0,212,255,0.05));
  border:1px solid rgba(0,212,255,0.4);border-radius:10px;padding:10px 16px;margin-bottom:8px;
  font-family:"JetBrains Mono",monospace;font-size:0.78rem;}
.mode-cnc{background:linear-gradient(135deg,rgba(167,139,250,0.15),rgba(167,139,250,0.05));
  border:1px solid rgba(167,139,250,0.4);border-radius:10px;padding:10px 16px;margin-bottom:8px;
  font-family:"JetBrains Mono",monospace;font-size:0.78rem;}
</style>
""", unsafe_allow_html=True)

# ── Black-Scholes ─────────────────────────────────────────────────────────────
class BS:
    @staticmethod
    def d1(S, K, T, r, sigma):
        if T <= 0 or sigma <= 0: return 0.0
        return (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    @staticmethod
    def d2(S, K, T, r, sigma):
        return BS.d1(S,K,T,r,sigma) - sigma*np.sqrt(T)
    @staticmethod
    def gamma(S, K, T, r, sigma):
        if T<=0 or sigma<=0 or S<=0 or K<=0: return 0.0
        try:    return norm.pdf(BS.d1(S,K,T,r,sigma))/(S*sigma*np.sqrt(T))
        except: return 0.0
    @staticmethod
    def vanna(S, K, T, r, sigma):
        if T<=0 or sigma<=0 or S<=0 or K<=0: return 0.0
        try:
            d1=BS.d1(S,K,T,r,sigma); d2=BS.d2(S,K,T,r,sigma)
            return -norm.pdf(d1)*d2/sigma
        except: return 0.0

# ── Headers ───────────────────────────────────────────────────────────────────
def get_headers() -> Dict:
    return {"access-token": DHAN_ACCESS_TOKEN,
            "client-id":    DHAN_CLIENT_ID,
            "Content-Type": "application/json"}

# ── Database ──────────────────────────────────────────────────────────────────
def init_db():
    con = sqlite3.connect(DB_PATH, timeout=20); cur = con.cursor()
    cur.execute("PRAGMA journal_mode=WAL")   # allow concurrent reads during writes
    cur.execute("PRAGMA synchronous=NORMAL")

    cur.execute("""CREATE TABLE IF NOT EXISTS raw_chain(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT, trade_date TEXT, timestamp TEXT,
        strike_type TEXT, strike REAL, spot_price REAL,
        call_oi REAL, put_oi REAL, call_vol REAL, put_vol REAL,
        call_iv REAL, put_iv REAL,
        call_gex REAL, put_gex REAL, net_gex REAL,
        call_vanna REAL, put_vanna REAL, net_vanna REAL,
        call_oi_chg REAL, put_oi_chg REAL,
        interval TEXT, expiry_code INTEGER, expiry_flag TEXT,
        UNIQUE(symbol,trade_date,timestamp,strike_type,expiry_code,expiry_flag))""")

    cur.execute("""CREATE TABLE IF NOT EXISTS cascade_signals(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT, trade_date TEXT, timestamp TEXT, spot_price REAL,
        bear_fuel_pts REAL, bear_absorb_pts REAL,
        bull_fuel_pts REAL, bull_absorb_pts REAL,
        bear_quality REAL, bull_quality REAL,
        iv_regime TEXT, avg_iv REAL, iv_skew REAL,
        net_gex_total REAL, signal TEXT, signal_strength REAL,
        cascade_target REAL, cascade_stop REAL,
        UNIQUE(symbol,trade_date,timestamp))""")

    # bt_trades now has bt_mode column (INTRADAY / CNC)
    cur.execute("""CREATE TABLE IF NOT EXISTS bt_trades(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT, trade_date TEXT, entry_time TEXT, exit_time TEXT,
        direction TEXT, entry_price REAL, exit_price REAL,
        pts_captured REAL, pnl_per_lot REAL,
        cascade_target REAL, cascade_stop REAL,
        exit_reason TEXT, bear_quality REAL, bull_quality REAL,
        iv_regime TEXT, signal_strength REAL,
        is_expiry_day INTEGER, expiry_flag TEXT,
        bt_mode TEXT DEFAULT 'INTRADAY')""")

    # Migration: add new columns if DB was created by v2 (errors = column already exists, safe to ignore)
    try: cur.execute("ALTER TABLE bt_trades ADD COLUMN bt_mode TEXT DEFAULT 'INTRADAY'")
    except: pass
    try: cur.execute("ALTER TABLE bt_trades ADD COLUMN pnl_per_lot REAL DEFAULT 0.0")
    except: pass
    # Back-fill pnl_per_lot for rows migrated from v2 (NULL or corrupted text value)
    # Use typeof() instead of GLOB to avoid Python sqlite3 driver misparse
    cur.execute("UPDATE bt_trades SET pnl_per_lot=0.0 WHERE pnl_per_lot IS NULL")
    cur.execute("UPDATE bt_trades SET pnl_per_lot=0.0 WHERE typeof(pnl_per_lot)='text'")
    cur.execute("UPDATE bt_trades SET bt_mode='INTRADAY' WHERE bt_mode IS NULL")

    cur.execute("""CREATE TABLE IF NOT EXISTS fetch_log(
        symbol TEXT, trade_date TEXT, expiry_code INTEGER, expiry_flag TEXT,
        status TEXT, rows_fetched INTEGER, fetched_at TEXT,
        PRIMARY KEY(symbol,trade_date,expiry_code,expiry_flag))""")

    con.commit(); con.close()

def get_fetch_log(symbol, expiry_code, expiry_flag):
    con = sqlite3.connect(DB_PATH, timeout=20); cur = con.cursor()
    cur.execute("SELECT trade_date FROM fetch_log WHERE symbol=? AND expiry_code=? AND expiry_flag=? AND status='ok'",
                (symbol, expiry_code, expiry_flag))
    done = {r[0] for r in cur.fetchall()}; con.close(); return done

def log_fetch(symbol, trade_date, expiry_code, expiry_flag, status, rows):
    con = sqlite3.connect(DB_PATH, timeout=20)
    con.execute("INSERT OR REPLACE INTO fetch_log VALUES(?,?,?,?,?,?,?)",
                (symbol,trade_date,expiry_code,expiry_flag,status,rows,
                 datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")))
    con.commit(); con.close()

def save_raw_chain(rows):
    if not rows: return
    con = sqlite3.connect(DB_PATH, timeout=20)
    con.executemany("""INSERT OR IGNORE INTO raw_chain
        (symbol,trade_date,timestamp,strike_type,strike,spot_price,
         call_oi,put_oi,call_vol,put_vol,call_iv,put_iv,
         call_gex,put_gex,net_gex,call_vanna,put_vanna,net_vanna,
         call_oi_chg,put_oi_chg,interval,expiry_code,expiry_flag)
        VALUES(:symbol,:trade_date,:timestamp,:strike_type,:strike,:spot_price,
         :call_oi,:put_oi,:call_vol,:put_vol,:call_iv,:put_iv,
         :call_gex,:put_gex,:net_gex,:call_vanna,:put_vanna,:net_vanna,
         :call_oi_chg,:put_oi_chg,:interval,:expiry_code,:expiry_flag)""", rows)
    con.commit(); con.close()

def load_raw_chain(symbol, date_str, expiry_code, expiry_flag):
    con = sqlite3.connect(DB_PATH, timeout=20)
    df = pd.read_sql_query(
        "SELECT * FROM raw_chain WHERE symbol=? AND trade_date=? AND expiry_code=? AND expiry_flag=? ORDER BY timestamp,strike",
        con, params=(symbol,date_str,expiry_code,expiry_flag)); con.close()
    if not df.empty: df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

def save_signals(rows):
    if not rows: return
    con = sqlite3.connect(DB_PATH, timeout=20)
    con.executemany("""INSERT OR REPLACE INTO cascade_signals
        (symbol,trade_date,timestamp,spot_price,bear_fuel_pts,bear_absorb_pts,
         bull_fuel_pts,bull_absorb_pts,bear_quality,bull_quality,
         iv_regime,avg_iv,iv_skew,net_gex_total,signal,signal_strength,
         cascade_target,cascade_stop)
        VALUES(:symbol,:trade_date,:timestamp,:spot_price,:bear_fuel_pts,:bear_absorb_pts,
         :bull_fuel_pts,:bull_absorb_pts,:bear_quality,:bull_quality,
         :iv_regime,:avg_iv,:iv_skew,:net_gex_total,:signal,:signal_strength,
         :cascade_target,:cascade_stop)""", rows)
    con.commit(); con.close()

def load_signals(symbol, date_str):
    con = sqlite3.connect(DB_PATH, timeout=20)
    df = pd.read_sql_query(
        "SELECT * FROM cascade_signals WHERE symbol=? AND trade_date=? ORDER BY timestamp",
        con, params=(symbol,date_str)); con.close()
    if not df.empty: df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

def save_trades(rows):
    if not rows: return
    con = sqlite3.connect(DB_PATH, timeout=20)
    con.executemany("""INSERT INTO bt_trades
        (symbol,trade_date,entry_time,exit_time,direction,entry_price,exit_price,
         pts_captured,pnl_per_lot,cascade_target,cascade_stop,exit_reason,
         bear_quality,bull_quality,iv_regime,signal_strength,is_expiry_day,expiry_flag,bt_mode)
        VALUES(:symbol,:trade_date,:entry_time,:exit_time,:direction,:entry_price,:exit_price,
         :pts_captured,:pnl_per_lot,:cascade_target,:cascade_stop,:exit_reason,
         :bear_quality,:bull_quality,:iv_regime,:signal_strength,:is_expiry_day,:expiry_flag,:bt_mode)""", rows)
    con.commit(); con.close()

def load_trades(symbol=None, bt_mode=None):
    con = sqlite3.connect(DB_PATH, timeout=20)
    conditions = []
    params = []
    if symbol:
        conditions.append("symbol=?"); params.append(symbol)
    if bt_mode:
        conditions.append("bt_mode=?"); params.append(bt_mode)
    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    q = f"SELECT * FROM bt_trades {where} ORDER BY trade_date,entry_time"
    df = pd.read_sql_query(q, con, params=params); con.close()
    return df

def clear_trades(symbol=None, bt_mode=None):
    con = sqlite3.connect(DB_PATH, timeout=20)
    conditions = []
    params = []
    if symbol:
        conditions.append("symbol=?"); params.append(symbol)
    if bt_mode:
        conditions.append("bt_mode=?"); params.append(bt_mode)
    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    con.execute(f"DELETE FROM bt_trades {where}", params)
    con.commit(); con.close()

def clear_signals(symbol=None):
    con = sqlite3.connect(DB_PATH, timeout=20)
    if symbol:
        con.execute("DELETE FROM cascade_signals WHERE symbol=?", (symbol,))
    else:
        con.execute("DELETE FROM cascade_signals")
    con.commit(); con.close()

def clear_raw_chain(symbol=None):
    con = sqlite3.connect(DB_PATH, timeout=20)
    if symbol:
        con.execute("DELETE FROM raw_chain WHERE symbol=?", (symbol,))
        con.execute("DELETE FROM fetch_log WHERE symbol=?", (symbol,))
    else:
        con.execute("DELETE FROM raw_chain")
        con.execute("DELETE FROM fetch_log")
    con.commit(); con.close()

def db_stats():
    con = sqlite3.connect(DB_PATH, timeout=20); cur = con.cursor()
    cur.execute("SELECT COUNT(*) FROM raw_chain");         rr = cur.fetchone()[0]
    cur.execute("SELECT COUNT(DISTINCT trade_date),COUNT(DISTINCT symbol) FROM raw_chain")
    days, syms = cur.fetchone()
    cur.execute("SELECT COUNT(*) FROM bt_trades WHERE bt_mode='INTRADAY'"); ti = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM bt_trades WHERE bt_mode='CNC'");      tc = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM cascade_signals");   sg = cur.fetchone()[0]
    con.close()
    return {"raw_rows":rr,"days":days,"symbols":syms,
            "trades_intraday":ti,"trades_cnc":tc,"signals":sg}

# ── Checkpoint ────────────────────────────────────────────────────────────────
def save_checkpoint(symbol, trade_date, expiry_code, expiry_flag, completed, rows):
    try:
        with open(CKPT_PATH,"w") as f:
            json.dump({"symbol":symbol,"trade_date":trade_date,
                       "expiry_code":expiry_code,"expiry_flag":expiry_flag,
                       "completed_strikes":completed,"partial_rows":rows,
                       "saved_at":datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")},f)
    except: pass

def load_checkpoint(symbol, trade_date, expiry_code, expiry_flag):
    if not os.path.exists(CKPT_PATH): return [],[]
    try:
        with open(CKPT_PATH) as f: ckpt=json.load(f)
        if (ckpt.get("symbol")==symbol and ckpt.get("trade_date")==trade_date
                and ckpt.get("expiry_code")==expiry_code and ckpt.get("expiry_flag")==expiry_flag):
            return ckpt.get("completed_strikes",[]), ckpt.get("partial_rows",[])
    except: pass
    return [],[]

def clear_checkpoint():
    try:
        if os.path.exists(CKPT_PATH): os.remove(CKPT_PATH)
    except: pass

def checkpoint_status():
    if not os.path.exists(CKPT_PATH): return None
    try:
        with open(CKPT_PATH) as f: return json.load(f)
    except: return None

# ── Dhan API ──────────────────────────────────────────────────────────────────
def fetch_rolling_option(symbol, from_date, to_date, strike_type,
                         option_type, interval, expiry_code, expiry_flag, silent=True):
    sec_id   = DHAN_INDEX_SECURITY_IDS.get(symbol)
    if not sec_id:
        if not silent: st.error(f"Unknown symbol: {symbol}")
        return None
    exchange = "BSE_FNO" if symbol in BSE_FNO_SYMBOLS else "NSE_FNO"
    payload  = {
        "exchangeSegment": exchange, "interval": interval,
        "securityId": sec_id, "instrument": "OPTIDX",
        "expiryFlag": expiry_flag, "expiryCode": expiry_code,
        "strike": strike_type, "drvOptionType": option_type,
        "requiredData": ["open","high","low","close","volume","oi","iv","strike","spot"],
        "fromDate": from_date, "toDate": to_date,
    }
    try:
        resp = requests.post(f"{DHAN_BASE}/charts/rollingoption",
                             headers=get_headers(), json=payload, timeout=30)
        if resp.status_code == 200:
            return resp.json().get("data", {}) or None
        if not silent:
            st.error(f"HTTP {resp.status_code}: {resp.text[:400]}\n\nPayload: {json.dumps(payload,indent=2)}")
        return None
    except Exception as e:
        if not silent: st.error(f"Exception: {e}")
        return None

def fetch_one_day(symbol, trade_date, strikes, interval,
                  expiry_code, expiry_flag, progress_bar=None, status_text=None):
    cfg           = INDEX_CONFIG.get(symbol, {})
    contract_size = cfg.get("contract_size", 25)
    scaling       = 1e9
    tte           = 7/365 if expiry_flag == "WEEK" else 30/365
    target_dt     = datetime.strptime(trade_date, "%Y-%m-%d")
    from_date = (target_dt - timedelta(days=2)).strftime("%Y-%m-%d")
    to_date   = (target_dt + timedelta(days=2)).strftime("%Y-%m-%d")

    completed, all_rows = load_checkpoint(symbol, trade_date, expiry_code, expiry_flag)
    remaining = [s for s in strikes if s not in completed]
    total = len(strikes) * 2
    done  = len(completed) * 2

    for stype in remaining:
        if status_text:
            status_text.text(f"Fetching {symbol} {stype} | {trade_date} ({len(completed)+1}/{len(strikes)})")

        call_data = fetch_rolling_option(symbol, from_date, to_date,
                                         stype, "CALL", interval, expiry_code, expiry_flag)
        done += 1
        if progress_bar: progress_bar.progress(min(done/total, 1.0))
        time.sleep(0.3)

        put_data = fetch_rolling_option(symbol, from_date, to_date,
                                         stype, "PUT", interval, expiry_code, expiry_flag)
        done += 1
        if progress_bar: progress_bar.progress(min(done/total, 1.0))
        time.sleep(0.3)

        if not call_data or not put_data:
            completed.append(stype)
            save_checkpoint(symbol,trade_date,expiry_code,expiry_flag,completed,all_rows)
            continue

        ce = call_data.get("ce", {})
        pe = put_data.get("pe",  {})
        if not ce:
            completed.append(stype)
            save_checkpoint(symbol,trade_date,expiry_code,expiry_flag,completed,all_rows)
            continue

        ts_list = ce.get("timestamp", [])
        for i, ts in enumerate(ts_list):
            try:
                dt_ist = datetime.fromtimestamp(ts, tz=pytz.UTC).astimezone(IST)
                if dt_ist.date() != target_dt.date(): continue

                def _g(src, key, default):
                    arr = src.get(key, [])
                    return arr[i] if i < len(arr) else default

                spot   = _g(ce,"spot",0)   or 0
                strike = _g(ce,"strike",0) or 0
                if spot == 0 or strike == 0: continue

                c_oi  = _g(ce,"oi",0)     or 0
                p_oi  = _g(pe,"oi",0)     or 0
                c_vol = _g(ce,"volume",0) or 0
                p_vol = _g(pe,"volume",0) or 0
                c_iv  = _g(ce,"iv",15)    or 15
                p_iv  = _g(pe,"iv",15)    or 15

                civ = max(c_iv/100 if c_iv>1 else float(c_iv), 0.01)
                piv = max(p_iv/100 if p_iv>1 else float(p_iv), 0.01)

                cg = BS.gamma(spot, strike, tte, RISK_FREE, civ)
                pg = BS.gamma(spot, strike, tte, RISK_FREE, piv)
                cv = BS.vanna(spot, strike, tte, RISK_FREE, civ)
                pv = BS.vanna(spot, strike, tte, RISK_FREE, piv)

                all_rows.append({
                    "symbol": symbol, "trade_date": trade_date,
                    "timestamp": dt_ist.strftime("%Y-%m-%d %H:%M:%S"),
                    "strike_type": stype, "strike": float(strike), "spot_price": float(spot),
                    "call_oi": float(c_oi), "put_oi": float(p_oi),
                    "call_vol": float(c_vol), "put_vol": float(p_vol),
                    "call_iv": float(c_iv), "put_iv": float(p_iv),
                    "call_gex": float((c_oi*cg*spot**2*contract_size)/scaling),
                    "put_gex":  float(-(p_oi*pg*spot**2*contract_size)/scaling),
                    "net_gex":  float((c_oi*cg - p_oi*pg)*spot**2*contract_size/scaling),
                    "call_vanna": float(c_oi*cv*spot*contract_size/scaling),
                    "put_vanna":  float(p_oi*pv*spot*contract_size/scaling),
                    "net_vanna":  float((c_oi*cv + p_oi*pv)*spot*contract_size/scaling),
                    "call_oi_chg": 0.0, "put_oi_chg": 0.0,
                    "interval": interval, "expiry_code": expiry_code, "expiry_flag": expiry_flag,
                })
            except: continue

        completed.append(stype)
        save_checkpoint(symbol,trade_date,expiry_code,expiry_flag,completed,all_rows)

    if all_rows:
        df = pd.DataFrame(all_rows).sort_values(["strike","timestamp"])
        for sv in df["strike"].unique():
            m = df["strike"]==sv
            df.loc[m,"call_oi_chg"] = df.loc[m,"call_oi"].diff().fillna(0)
            df.loc[m,"put_oi_chg"]  = df.loc[m,"put_oi"].diff().fillna(0)
        all_rows = df.to_dict("records")

    save_raw_chain(all_rows)
    clear_checkpoint()
    return len(all_rows)

# ── Trading dates ─────────────────────────────────────────────────────────────
def get_trading_dates(start, end):
    dates, cur = [], start
    while cur <= end:
        if cur.weekday() < 5: dates.append(cur.strftime("%Y-%m-%d"))
        cur += timedelta(days=1)
    return dates

# ── IV regime ─────────────────────────────────────────────────────────────────
def compute_iv_regime_series(df):
    rows = []
    for ts, grp in df.groupby("timestamp", sort=True):
        avg_iv  = (grp["call_iv"].mean() + grp["put_iv"].mean()) / 2
        iv_skew = grp["put_iv"].mean() - grp["call_iv"].mean()
        rows.append({"timestamp": ts, "avg_iv": avg_iv, "iv_skew": iv_skew})
    iv_df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    if len(iv_df) < 3:
        iv_df["iv_regime"] = "FLAT"; return iv_df
    iv_df["iv_change"] = iv_df["avg_iv"].diff().fillna(0)
    thr = iv_df["avg_iv"].std() * 0.15
    iv_df["iv_regime"] = iv_df["iv_change"].apply(
        lambda x: "EXPANDING" if x>thr else ("COMPRESSING" if x<-thr else "FLAT"))
    return iv_df

# ── Cascade engine (Tab 2 signals — kept intact) ──────────────────────────────
VANNA_ADJ = {
    "SUPPORT_FLOOR":      {"COMPRESSING":0.60,"FLAT":0.35,"EXPANDING":-0.20},
    "TRAP_DOOR":          {"EXPANDING":-0.30,"FLAT":0.10,"COMPRESSING":0.15},
    "VACUUM_ZONE":        {"EXPANDING":0.50,"FLAT":0.20,"COMPRESSING":0.10},
    "RESISTANCE_CEILING": {"EXPANDING":-0.20,"FLAT":0.00,"COMPRESSING":0.15},
}

def compute_cascade_for_snapshot(df_ts, spot, symbol, iv_regime):
    params = INSTRUMENT_PARAMS.get(symbol, INSTRUMENT_PARAMS["NIFTY"])
    ppu    = params["pts_per_unit"]
    cap    = params["strike_cap"]
    vz_map = {}
    strikes_s = sorted(df_ts["strike"].unique())
    vs = [(s, df_ts[df_ts["strike"]==s]["net_vanna"].iloc[0])
          for s in strikes_s if len(df_ts[df_ts["strike"]==s])>0]
    for j in range(1, len(vs)):
        sp, vp = vs[j-1]; sc, vc = vs[j]
        if vp*vc < 0:
            role = None
            if sp < spot and vp < 0 and vc > 0: role = "SUPPORT_FLOOR"
            elif sp < spot and vp > 0 and vc < 0: role = "TRAP_DOOR"
            elif sc > spot and vp < 0 and vc > 0: role = "VACUUM_ZONE"
            elif sc > spot and vp > 0 and vc < 0: role = "RESISTANCE_CEILING"
            if role: vz_map[sc] = role

    bf=ba=uf=ua=0.0; bt=ut=spot
    for _, row in df_ts.iterrows():
        s=row["strike"]; gex=row["net_gex"]
        rp = min(abs(gex)*ppu, cap)
        role = vz_map.get(s)
        adj  = VANNA_ADJ.get(role,{}).get(iv_regime,0.0) if role else 0.0
        ap   = rp*(1+adj)
        if s < spot:
            if gex<0: bf+=ap; bt=min(bt,s)
            else:     ba+=ap
        else:
            if gex<0: uf+=ap; ut=max(ut,s)
            else:     ua+=ap

    bq = bf/max(ba,1.0); uq = uf/max(ua,1.0)
    nbr = max(0, bf-ba*0.5); nur = max(0, uf-ua*0.5)
    if bq >= uq:
        sig="BEAR"; strength=bq; tgt=spot-nbr; stp=spot+(ba*ppu*0.5)
    else:
        sig="BULL"; strength=uq; tgt=spot+nur; stp=spot-(ua*ppu*0.5)
    if max(bq,uq) < 0.5: sig="NONE"
    return {"bear_fuel_pts":round(bf,2),"bear_absorb_pts":round(ba,2),
            "bull_fuel_pts":round(uf,2),"bull_absorb_pts":round(ua,2),
            "bear_quality":round(bq,4),"bull_quality":round(uq,4),
            "net_gex_total":round(df_ts["net_gex"].sum(),4),
            "signal":sig,"signal_strength":round(strength,4),
            "cascade_target":round(tgt,2),"cascade_stop":round(stp,2)}

def compute_signals_for_day(df_day, symbol, trade_date):
    iv_df = compute_iv_regime_series(df_day)
    rows  = []
    for ts in sorted(df_day["timestamp"].unique()):
        df_ts  = df_day[df_day["timestamp"]==ts].copy()
        spot   = df_ts["spot_price"].mean()
        iv_row = iv_df[iv_df["timestamp"]==ts]
        if iv_row.empty: iv_regime,avg_iv,iv_skew = "FLAT",15.0,0.0
        else:
            iv_regime = str(iv_row.iloc[0]["iv_regime"])
            avg_iv    = float(iv_row.iloc[0]["avg_iv"])
            iv_skew   = float(iv_row.iloc[0]["iv_skew"])
        cas = compute_cascade_for_snapshot(df_ts, spot, symbol, iv_regime)
        rows.append({"symbol":symbol,"trade_date":trade_date,
                     "timestamp":ts.strftime("%Y-%m-%d %H:%M:%S") if hasattr(ts,"strftime") else str(ts),
                     "spot_price":round(spot,2),"iv_regime":iv_regime,
                     "avg_iv":round(avg_iv,2),"iv_skew":round(iv_skew,4),**cas})
    return rows

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CUSTOM STRATEGY — helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def get_strike_cascade_pts(df_ts, strike_types, symbol):
    ppu    = INSTRUMENT_PARAMS.get(symbol, INSTRUMENT_PARAMS["NIFTY"])["pts_per_unit"]
    result = {}
    for st in strike_types:
        rows = df_ts[df_ts["strike_type"] == st]
        result[st] = 0.0 if rows.empty else abs(rows["net_gex"].iloc[0]) * ppu
    return result

def compute_strike_signals_for_day(df_day, symbol, trade_date, iv_df,
                                   cascade_entry_threshold, max_target_pts):
    bars = []
    for ts in sorted(df_day["timestamp"].unique()):
        df_ts = df_day[df_day["timestamp"] == ts]
        spot  = df_ts["spot_price"].mean()

        iv_row    = iv_df[iv_df["timestamp"] == ts]
        iv_regime = str(iv_row.iloc[0]["iv_regime"]) if not iv_row.empty else "FLAT"

        c_pts        = get_strike_cascade_pts(df_ts, ["ATM+1","ATM+2","ATM+3"], symbol)
        atm1_pts     = c_pts.get("ATM+1", 0.0)
        call_cum     = sum(c_pts.values())
        call_tgt_pts = min(call_cum, max_target_pts)
        call_signal  = (iv_regime == "EXPANDING") and (atm1_pts > cascade_entry_threshold)

        p_pts        = get_strike_cascade_pts(df_ts, ["ATM-1","ATM-2","ATM-3"], symbol)
        atm_neg1_pts = p_pts.get("ATM-1", 0.0)
        put_cum      = sum(p_pts.values())
        put_tgt_pts  = min(put_cum, max_target_pts)
        put_signal   = (iv_regime == "EXPANDING") and (atm_neg1_pts > cascade_entry_threshold)

        bars.append({
            "timestamp":    ts,
            "spot":         round(spot, 2),
            "iv_regime":    iv_regime,
            "call_signal":  call_signal,
            "call_target":  round(spot + call_tgt_pts, 2),
            "put_signal":   put_signal,
            "put_target":   round(spot - put_tgt_pts, 2),
            "atm1_pts":     round(atm1_pts, 2),
            "atm_neg1_pts": round(atm_neg1_pts, 2),
            "call_cum_pts": round(call_cum, 2),
            "put_cum_pts":  round(put_cum, 2),
        })
    return bars

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BACKTEST ENGINE — INTRADAY
# Trades open and close within the same day (EOD forced exit)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_intraday_day(df_day, symbol, trade_date, expiry_flag,
                     cascade_entry_threshold, max_target_pts, fixed_sl_pts,
                     direction_filter, max_trades_per_day, lots, iv_df):
    if df_day is None or df_day.empty: return []
    contract_size = INDEX_CONFIG.get(symbol,{}).get("contract_size", 25)

    bar_signals = compute_strike_signals_for_day(
        df_day, symbol, trade_date, iv_df, cascade_entry_threshold, max_target_pts)
    if not bar_signals: return []

    try:
        dt        = datetime.strptime(trade_date, "%Y-%m-%d")
        is_expiry = (dt.weekday() == 3 if expiry_flag == "WEEK" else False)
    except:
        is_expiry = False

    trades      = []
    trade_count = 0
    call_open   = put_open = False
    call_entry_ts = call_entry_px = call_tgt = call_stp = call_cum = None
    put_entry_ts  = put_entry_px  = put_tgt  = put_stp  = put_cum  = None

    for bar in bar_signals:
        ts     = bar["timestamp"]
        spot   = bar["spot"]
        iv_reg = bar["iv_regime"]

        if call_open:
            pts = spot - call_entry_px
            if spot >= call_tgt:
                trades.append(_mk_v3(symbol,trade_date,call_entry_ts,ts,"CALL",
                    call_entry_px,spot,pts,call_tgt,call_stp,"TARGET_HIT",
                    iv_reg,call_cum,is_expiry,expiry_flag,lots,contract_size,"INTRADAY"))
                call_open=False; trade_count+=1
            elif spot <= call_stp:
                trades.append(_mk_v3(symbol,trade_date,call_entry_ts,ts,"CALL",
                    call_entry_px,spot,pts,call_tgt,call_stp,"STOP_HIT",
                    iv_reg,call_cum,is_expiry,expiry_flag,lots,contract_size,"INTRADAY"))
                call_open=False; trade_count+=1

        if put_open:
            pts = put_entry_px - spot
            if spot <= put_tgt:
                trades.append(_mk_v3(symbol,trade_date,put_entry_ts,ts,"PUT",
                    put_entry_px,spot,pts,put_tgt,put_stp,"TARGET_HIT",
                    iv_reg,put_cum,is_expiry,expiry_flag,lots,contract_size,"INTRADAY"))
                put_open=False; trade_count+=1
            elif spot >= put_stp:
                trades.append(_mk_v3(symbol,trade_date,put_entry_ts,ts,"PUT",
                    put_entry_px,spot,pts,put_tgt,put_stp,"STOP_HIT",
                    iv_reg,put_cum,is_expiry,expiry_flag,lots,contract_size,"INTRADAY"))
                put_open=False; trade_count+=1

        if trade_count >= max_trades_per_day: continue

        if (not call_open and direction_filter in ("BOTH","CALL only")
                and bar["call_signal"] and bar["atm1_pts"] > cascade_entry_threshold):
            call_open=True; call_entry_ts=ts; call_entry_px=spot
            call_tgt=bar["call_target"]; call_stp=round(spot-fixed_sl_pts,2)
            call_cum=bar["call_cum_pts"]

        if (not put_open and direction_filter in ("BOTH","PUT only")
                and bar["put_signal"] and bar["atm_neg1_pts"] > cascade_entry_threshold):
            put_open=True; put_entry_ts=ts; put_entry_px=spot
            put_tgt=bar["put_target"]; put_stp=round(spot+fixed_sl_pts,2)
            put_cum=bar["put_cum_pts"]

    if bar_signals:
        last=bar_signals[-1]; lts=last["timestamp"]; lspot=last["spot"]; iv_reg=last["iv_regime"]
        if call_open:
            pts=lspot-call_entry_px
            trades.append(_mk_v3(symbol,trade_date,call_entry_ts,lts,"CALL",
                call_entry_px,lspot,pts,call_tgt,call_stp,"EOD_EXIT",
                iv_reg,call_cum,is_expiry,expiry_flag,lots,contract_size,"INTRADAY"))
        if put_open:
            pts=put_entry_px-lspot
            trades.append(_mk_v3(symbol,trade_date,put_entry_ts,lts,"PUT",
                put_entry_px,lspot,pts,put_tgt,put_stp,"EOD_EXIT",
                iv_reg,put_cum,is_expiry,expiry_flag,lots,contract_size,"INTRADAY"))
    return trades

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BACKTEST ENGINE — CNC (positional / multi-day hold)
# Entry on signal bar; exit on first of:
#   1. Target hit on any subsequent day's EOD price
#   2. Stop hit on any subsequent day's EOD price
#   3. Max hold days reached (forced exit at EOD)
# All_dates list is needed to simulate multi-day progression.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_cnc_backtest(all_dates, symbol, expiry_flag, expiry_code,
                     cascade_entry_threshold, max_target_pts, fixed_sl_pts,
                     direction_filter, max_hold_days, lots,
                     status_placeholder=None):
    """
    CNC positional backtest — entry intrabar, exit on subsequent EOD prices.
    Uses the last bar of each day as the EOD/settlement price.
    """
    contract_size = INDEX_CONFIG.get(symbol,{}).get("contract_size", 25)
    trades        = []
    open_positions= []   # list of dicts: {direction, entry_date, entry_price, tgt, stp, cum, iv_reg, is_expiry}

    for date_idx, trade_date in enumerate(all_dates):
        if status_placeholder:
            status_placeholder.text(f"CNC: {trade_date} ({date_idx+1}/{len(all_dates)})")

        df_day = load_raw_chain(symbol, trade_date, expiry_code, expiry_flag)
        if df_day.empty: continue
        iv_df  = compute_iv_regime_series(df_day)

        # EOD price = last bar's spot
        bars     = compute_strike_signals_for_day(
            df_day, symbol, trade_date, iv_df, cascade_entry_threshold, max_target_pts)
        if not bars: continue
        eod_spot = bars[-1]["spot"]
        eod_iv   = bars[-1]["iv_regime"]

        try:
            dt        = datetime.strptime(trade_date, "%Y-%m-%d")
            is_expiry = (dt.weekday() == 3 if expiry_flag == "WEEK" else False)
        except:
            is_expiry = False

        # ── Step 1: manage existing open positions against today's EOD price ──
        still_open = []
        for pos in open_positions:
            held_days = date_idx - pos["entry_day_idx"]
            pts_call  = eod_spot - pos["entry_price"] if pos["direction"]=="CALL" else pos["entry_price"] - eod_spot

            exit_reason = None
            if pos["direction"] == "CALL":
                pts = eod_spot - pos["entry_price"]
                if eod_spot >= pos["tgt"]:      exit_reason = "TARGET_HIT"
                elif eod_spot <= pos["stp"]:     exit_reason = "STOP_HIT"
                elif held_days >= max_hold_days: exit_reason = "MAX_HOLD_EXIT"
            else:
                pts = pos["entry_price"] - eod_spot
                if eod_spot <= pos["tgt"]:      exit_reason = "TARGET_HIT"
                elif eod_spot >= pos["stp"]:     exit_reason = "STOP_HIT"
                elif held_days >= max_hold_days: exit_reason = "MAX_HOLD_EXIT"

            if exit_reason:
                if pos["direction"] == "CALL": pts = eod_spot - pos["entry_price"]
                else:                          pts = pos["entry_price"] - eod_spot
                trades.append(_mk_v3(
                    symbol, pos["entry_date"], pos["entry_ts"], trade_date + " EOD",
                    pos["direction"], pos["entry_price"], eod_spot, pts,
                    pos["tgt"], pos["stp"], exit_reason,
                    eod_iv, pos["cum"], pos["is_expiry"], expiry_flag,
                    lots, contract_size, "CNC"))
            else:
                still_open.append(pos)
        open_positions = still_open

        # ── Step 2: look for new entries on today's intraday bars ─────────────
        for bar in bars:
            if len(open_positions) >= 2: break   # max 2 concurrent CNC positions

            ts     = bar["timestamp"]
            spot   = bar["spot"]
            iv_reg = bar["iv_regime"]

            if (direction_filter in ("BOTH","CALL only") and bar["call_signal"]
                    and bar["atm1_pts"] > cascade_entry_threshold
                    and not any(p["direction"]=="CALL" for p in open_positions)):
                open_positions.append({
                    "direction": "CALL", "entry_date": trade_date,
                    "entry_ts": ts, "entry_price": spot,
                    "tgt": bar["call_target"],
                    "stp": round(spot - fixed_sl_pts, 2),
                    "cum": bar["call_cum_pts"], "iv_reg": iv_reg,
                    "is_expiry": is_expiry, "entry_day_idx": date_idx,
                })

            if (direction_filter in ("BOTH","PUT only") and bar["put_signal"]
                    and bar["atm_neg1_pts"] > cascade_entry_threshold
                    and not any(p["direction"]=="PUT" for p in open_positions)):
                open_positions.append({
                    "direction": "PUT", "entry_date": trade_date,
                    "entry_ts": ts, "entry_price": spot,
                    "tgt": bar["put_target"],
                    "stp": round(spot + fixed_sl_pts, 2),
                    "cum": bar["put_cum_pts"], "iv_reg": iv_reg,
                    "is_expiry": is_expiry, "entry_day_idx": date_idx,
                })

    # Force-close any positions still open at end of data
    for pos in open_positions:
        # Use last known eod_spot
        if pos["direction"] == "CALL": pts = eod_spot - pos["entry_price"]
        else:                          pts = pos["entry_price"] - eod_spot
        trades.append(_mk_v3(
            symbol, pos["entry_date"], pos["entry_ts"], all_dates[-1] + " EOD",
            pos["direction"], pos["entry_price"], eod_spot, pts,
            pos["tgt"], pos["stp"], "DATA_END_EXIT",
            eod_iv, pos["cum"], pos["is_expiry"], expiry_flag,
            lots, contract_size, "CNC"))

    return trades

# ── Trade record builder ──────────────────────────────────────────────────────
def _mk_v3(symbol, trade_date, entry_ts, exit_ts, direction,
           entry_px, exit_px, pts, tgt, stp,
           reason, iv_reg, cascade_cum_pts, is_expiry, expiry_flag,
           lots, contract_size, bt_mode):
    pnl_per_lot = round(pts * contract_size, 2)
    return {
        "symbol":          symbol,
        "trade_date":      trade_date,
        "entry_time":      str(entry_ts),
        "exit_time":       str(exit_ts),
        "direction":       direction,
        "entry_price":     round(entry_px, 2),
        "exit_price":      round(exit_px, 2),
        "pts_captured":    round(pts, 2),
        "pnl_per_lot":     pnl_per_lot,         # pts × contract_size (₹ per lot)
        "cascade_target":  round(tgt, 2),
        "cascade_stop":    round(stp, 2),
        "exit_reason":     reason,
        "bear_quality":    round(cascade_cum_pts, 2),
        "bull_quality":    0.0,
        "iv_regime":       iv_reg,
        "signal_strength": round(cascade_cum_pts, 2),
        "is_expiry_day":   int(is_expiry),
        "expiry_flag":     expiry_flag,
        "bt_mode":         bt_mode,
    }

# ── Legacy wrapper kept for backward compat ───────────────────────────────────
def run_backtest_for_day(signals, symbol, trade_date, expiry_flag,
                         min_quality, require_iv,
                         cascade_entry_threshold=50.0, max_target_pts=200.0,
                         fixed_sl_pts=50.0, direction_filter="BOTH",
                         max_trades_per_day=2, df_day=None, iv_df=None,
                         lots=1):
    if df_day is None or df_day.empty: return []
    contract_size = INDEX_CONFIG.get(symbol,{}).get("contract_size",25)
    if iv_df is None: iv_df = compute_iv_regime_series(df_day)
    return run_intraday_day(df_day, symbol, trade_date, expiry_flag,
                            cascade_entry_threshold, max_target_pts, fixed_sl_pts,
                            direction_filter, max_trades_per_day, lots, iv_df)

# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(trades, contract_size=25, lots=1):
    if trades.empty: return {}

    # Force numeric — SQLite migrations can leave TEXT in numeric columns
    trades = trades.copy()
    for col in ["pts_captured","pnl_per_lot","cascade_target","entry_price","is_expiry_day"]:
        if col in trades.columns:
            trades[col] = pd.to_numeric(trades[col], errors="coerce").fillna(0.0)

    pts  = trades["pts_captured"]
    # pnl_per_lot: use column if present and non-zero, otherwise derive
    if "pnl_per_lot" in trades.columns and trades["pnl_per_lot"].abs().sum() > 0:
        pnl = trades["pnl_per_lot"]
    else:
        pnl = pts * contract_size * lots

    wins   = pts[pts > 0];  losses = pts[pts <= 0]
    pnl_w  = pnl[pts > 0];  pnl_l  = pnl[pts <= 0]
    total  = len(trades);   hit = len(wins)/total*100 if total else 0
    pf     = (wins.sum()/abs(losses.sum())) if losses.sum() != 0 else float("inf")
    cum    = pts.cumsum();  dd = (cum.cummax() - cum).max()
    sh     = (pts.mean()/pts.std()*np.sqrt(250)) if pts.std() > 0 else 0
    neg    = pts[pts < 0].std()
    so     = (pts.mean()/neg*np.sqrt(250)) if (neg and neg > 0) else 0
    t2     = trades.copy()
    t2["tm"] = (t2["cascade_target"] - t2["entry_price"]).abs()
    t2["am"] = t2["pts_captured"].abs()
    ca     = (t2["am"] >= t2["tm"]*0.7).mean()*100 if len(t2) else 0

    exp_mask     = trades["is_expiry_day"].astype(int) == 1
    non_exp_mask = trades["is_expiry_day"].astype(int) == 0

    return {
        "total_trades":    total,
        "hit_rate":        round(hit, 1),
        "total_pts":       round(float(pts.sum()), 1),
        "total_pnl":       round(float(pnl.sum()), 1),
        "avg_pts_win":     round(float(wins.mean()), 1)   if len(wins)   else 0.0,
        "avg_pts_loss":    round(float(losses.mean()), 1) if len(losses) else 0.0,
        "avg_pnl_win":     round(float(pnl_w.mean()), 1) if len(pnl_w)  else 0.0,
        "avg_pnl_loss":    round(float(pnl_l.mean()), 1) if len(pnl_l)  else 0.0,
        "profit_factor":   round(float(pf), 2),
        "max_drawdown":    round(float(dd), 1),
        "sharpe":          round(float(sh), 2),
        "sortino":         round(float(so), 2),
        "cascade_accuracy":round(float(ca), 1),
        "expiry_trades":   int(exp_mask.sum()),
        "expiry_pts":      round(float(pts[exp_mask].sum()), 1),
        "non_expiry_pts":  round(float(pts[non_exp_mask].sum()), 1),
    }

# ── Chart helpers ─────────────────────────────────────────────────────────────
def equity_curve_chart(trades):
    t=trades.copy().reset_index(drop=True); t["cum"]=t["pts_captured"].cumsum()
    fig=make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[0.65,0.35],
                      subplot_titles=["Equity Curve (Cumulative Pts)","Per-Trade P&L (pts)"])
    fig.add_trace(go.Scatter(x=t.index,y=t["cum"],mode="lines",
        line=dict(color="#00f5c4",width=2.5),fill="tozeroy",
        fillcolor="rgba(0,245,196,0.08)"),row=1,col=1)
    fig.add_trace(go.Bar(x=t.index,y=t["pts_captured"],
        marker_color=t["pts_captured"].apply(lambda x:"#10b981" if x>0 else "#ef4444")),row=2,col=1)
    fig.update_layout(template="plotly_dark",height=500,paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,10,20,0.95)",showlegend=False,margin=dict(l=0,r=0,t=30,b=0))
    return fig

def pnl_lot_chart(trades):
    t = trades.copy().reset_index(drop=True)
    if "pnl_per_lot" not in t.columns:
        t["pnl_per_lot"] = 0.0
    t["pnl_per_lot"] = pd.to_numeric(t["pnl_per_lot"], errors="coerce").fillna(0.0)
    t["cum_pnl"] = t["pnl_per_lot"].cumsum()
    fig=make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[0.65,0.35],
                      subplot_titles=["Cumulative P&L per Lot (₹)","Per-Trade P&L per Lot (₹)"])
    fig.add_trace(go.Scatter(x=t.index,y=t["cum_pnl"],mode="lines",
        line=dict(color="#a78bfa",width=2.5),fill="tozeroy",
        fillcolor="rgba(167,139,250,0.08)"),row=1,col=1)
    fig.add_trace(go.Bar(x=t.index,y=t["pnl_per_lot"],
        marker_color=t["pnl_per_lot"].apply(lambda x:"#10b981" if x>0 else "#ef4444")),row=2,col=1)
    fig.update_layout(template="plotly_dark",height=500,paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,10,20,0.95)",showlegend=False,margin=dict(l=0,r=0,t=30,b=0))
    return fig

def direction_breakdown_chart(trades):
    grp=trades.groupby("direction")["pts_captured"].agg(["sum","count","mean"]).reset_index()
    cm={"CALL":"#00d4ff","PUT":"#a78bfa"}
    fig=go.Figure(go.Bar(x=grp["direction"],y=grp["sum"],
        marker_color=[cm.get(d,"#8b5cf6") for d in grp["direction"]],
        text=[f"{row['count']} trades | avg {row['mean']:.1f}pts" for _,row in grp.iterrows()],
        textposition="outside"))
    fig.update_layout(template="plotly_dark",height=300,paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,10,20,0.95)",yaxis_title="Total Pts",
        title_text="CALL vs PUT",margin=dict(l=0,r=0,t=30,b=0))
    return fig

def exit_reason_chart(trades):
    grp=trades.groupby("exit_reason")["pts_captured"].agg(["sum","count"]).reset_index()
    cm={"TARGET_HIT":"#10b981","STOP_HIT":"#ef4444","EOD_EXIT":"#f59e0b",
        "MAX_HOLD_EXIT":"#f97316","DATA_END_EXIT":"#6b7280","TRAIL_STOP":"#fb923c"}
    fig=go.Figure(go.Bar(x=grp["exit_reason"],y=grp["sum"],
        marker_color=[cm.get(r,"#8b5cf6") for r in grp["exit_reason"]],
        text=grp["count"].apply(lambda x: f"{x}"),textposition="outside"))
    fig.update_layout(template="plotly_dark",height=300,paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,10,20,0.95)",yaxis_title="Total Pts",margin=dict(l=0,r=0,t=10,b=0))
    return fig

def monthly_pnl_chart(trades):
    trades=trades.copy()
    trades["month"]=pd.to_datetime(trades["trade_date"]).dt.to_period("M").astype(str)
    mo=trades.groupby("month")["pts_captured"].agg(["sum","count"]).reset_index()
    fig=go.Figure(go.Bar(x=mo["month"],y=mo["sum"],
        marker_color=mo["sum"].apply(lambda x:"#10b981" if x>0 else "#ef4444"),
        text=mo["count"].apply(lambda x: f"{x} trades"),textposition="outside"))
    fig.update_layout(template="plotly_dark",height=320,paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,10,20,0.95)",margin=dict(l=0,r=0,t=10,b=0))
    return fig

def iv_regime_breakdown_chart(trades):
    grp=trades.groupby("iv_regime")["pts_captured"].agg(["sum","count","mean"]).reset_index()
    colors={"EXPANDING":"#ef4444","COMPRESSING":"#10b981","FLAT":"#94a3b8"}
    fig=go.Figure(go.Bar(x=grp["iv_regime"],y=grp["sum"],
        marker_color=[colors.get(r,"#8b5cf6") for r in grp["iv_regime"]],
        text=[f"{row['count']} trades" for _,row in grp.iterrows()],textposition="outside"))
    fig.update_layout(template="plotly_dark",height=300,paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,10,20,0.95)",yaxis_title="Total Pts",margin=dict(l=0,r=0,t=10,b=0))
    return fig

# ── Trade table builder ───────────────────────────────────────────────────────
def build_trade_table(trades, contract_size):
    """Formatted trade table with Buy Price, Sell Price, P&L per lot."""
    if trades.empty: return pd.DataFrame()
    t = trades.copy()

    # Force numeric to survive SQLite text migration
    for col in ["entry_price","exit_price","pts_captured","pnl_per_lot",
                "cascade_target","cascade_stop","is_expiry_day"]:
        if col in t.columns:
            t[col] = pd.to_numeric(t[col], errors="coerce").fillna(0.0)

    # Buy / Sell labelling based on direction
    t["Buy Price"]  = t.apply(lambda r: r["entry_price"] if r["direction"]=="CALL"
                               else r["exit_price"], axis=1)
    t["Sell Price"] = t.apply(lambda r: r["exit_price"] if r["direction"]=="CALL"
                               else r["entry_price"], axis=1)

    # P&L per lot in ₹
    if "pnl_per_lot" not in t.columns or t["pnl_per_lot"].abs().sum() == 0:
        t["pnl_per_lot"] = t["pts_captured"] * contract_size

    t["P&L/Lot (₹)"] = t["pnl_per_lot"].apply(lambda x: f"{'▲' if x>=0 else '▼'} ₹{x:,.0f}")
    t["Pts"]         = t["pts_captured"].apply(lambda x: f"{x:+.1f}")

    # Entry/Exit times cleaned
    def _ftime(ts_str):
        try:    return pd.to_datetime(ts_str).strftime("%d-%b %H:%M")
        except: return str(ts_str)[:16]

    t["Entry Time"] = t["entry_time"].apply(_ftime)
    t["Exit Time"]  = t["exit_time"].apply(_ftime)

    disp = t[["trade_date","Entry Time","Exit Time","direction",
               "Buy Price","Sell Price","Pts","P&L/Lot (₹)",
               "cascade_target","cascade_stop","exit_reason","iv_regime","is_expiry_day"]].copy()
    disp.columns = ["Date","Entry","Exit","Dir",
                     "Buy (₹)","Sell (₹)","Pts","P&L/Lot",
                     "Target","Stop","Exit","IV","Expiry"]
    disp["Expiry"] = disp["Expiry"].map({0:"",1:"✓"})
    return disp

def style_trade_table(df):
    def rc(row):
        clr = "rgba(16,185,129,0.12)" if "▲" in str(row.get("P&L/Lot","")) \
              else "rgba(239,68,68,0.10)"
        return [f"background-color:{clr}"]*len(row)
    return df.style.apply(rc, axis=1)

# ── Metric card renderer ──────────────────────────────────────────────────────
def render_kpi_row(kpis, cols):
    for col,(label,val,cls) in zip(cols,kpis):
        col.markdown(
            f'<div class="metric-card"><div class="metric-val {cls}">{val}</div>'
            f'<div class="metric-lbl">{label}</div></div>',
            unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# RESULTS SECTION — full tabulated view
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def render_results_section(trades, symbol, mode_label, contract_size, lots):
    if trades.empty:
        st.info(f"No {mode_label} trades yet — run backtest in Tab 3 first.")
        return

    m = compute_metrics(trades, contract_size, lots)

    # ── KPI rows ─────────────────────────────────────────────────────────
    st.markdown(f"#### 📊 {mode_label} — Performance Summary")
    cols = st.columns(5)
    render_kpi_row([
        ("Total Trades",  str(m["total_trades"]),           "n"),
        ("Hit Rate",      f"{m['hit_rate']}%",              "g" if m["hit_rate"]>=50 else "r"),
        ("Total Pts",     f"{m['total_pts']:+.1f}",         "g" if m["total_pts"]>0 else "r"),
        ("Total P&L/Lot", f"₹{m['total_pnl']:,.0f}",       "g" if m["total_pnl"]>0 else "r"),
        ("Profit Factor", f"{m['profit_factor']:.2f}x",    "g" if m["profit_factor"]>1 else "r"),
    ], cols)

    cols2 = st.columns(5)
    render_kpi_row([
        ("Avg Win Pts",   f"{m['avg_pts_win']:+.1f}",       "g"),
        ("Avg Loss Pts",  f"{m['avg_pts_loss']:+.1f}",      "r"),
        ("Avg Win/Lot",   f"₹{m['avg_pnl_win']:,.0f}",     "g"),
        ("Avg Loss/Lot",  f"₹{m['avg_pnl_loss']:,.0f}",    "r"),
        ("Max Drawdown",  f"{m['max_drawdown']:.1f} pts",   "r"),
    ], cols2)

    cols3 = st.columns(5)
    render_kpi_row([
        ("Sharpe",        f"{m['sharpe']:.2f}",             "g" if m["sharpe"]>1 else "n"),
        ("Sortino",       f"{m['sortino']:.2f}",            "g" if m["sortino"]>1 else "n"),
        ("Cascade Acc.",  f"{m['cascade_accuracy']:.1f}%",  "g" if m["cascade_accuracy"]>60 else "n"),
        ("Expiry Pts",    f"{m['expiry_pts']:+.1f}",        "g" if m["expiry_pts"]>0 else "r"),
        ("Non-Exp Pts",   f"{m['non_expiry_pts']:+.1f}",   "g" if m["non_expiry_pts"]>0 else "r"),
    ], cols3)

    # ── Sub-tabs ─────────────────────────────────────────────────────────
    st.markdown("---")
    rt1, rt2, rt3, rt4, rt5 = st.tabs([
        "📈 Charts", "📋 All Trades", "📗 CALL Trades", "📕 PUT Trades", "📅 Monthly"])

    with rt1:
        c1, c2 = st.columns(2)
        with c1: st.plotly_chart(equity_curve_chart(trades), use_container_width=True)
        with c2: st.plotly_chart(pnl_lot_chart(trades), use_container_width=True)
        c3, c4 = st.columns(2)
        with c3: st.plotly_chart(direction_breakdown_chart(trades), use_container_width=True)
        with c4: st.plotly_chart(exit_reason_chart(trades), use_container_width=True)
        c5, c6 = st.columns(2)
        with c5: st.plotly_chart(iv_regime_breakdown_chart(trades), use_container_width=True)
        with c6: st.plotly_chart(monthly_pnl_chart(trades), use_container_width=True)

    with rt2:
        st.markdown(f"**All {mode_label} Trades** — {len(trades)} total")
        tbl = build_trade_table(trades, contract_size)
        st.dataframe(style_trade_table(tbl), use_container_width=True, height=600, hide_index=True)
        st.download_button(
            f"📥 Download {mode_label} CSV",
            data=trades.to_csv(index=False),
            file_name=f"hedgex_{symbol}_{mode_label.lower()}.csv",
            mime="text/csv", use_container_width=True)

    with rt3:
        calls = trades[trades["direction"]=="CALL"]
        st.markdown(f"**CALL Trades** — {len(calls)} total")
        if not calls.empty:
            tbl = build_trade_table(calls, contract_size)
            st.dataframe(style_trade_table(tbl), use_container_width=True, height=500, hide_index=True)
            cm = compute_metrics(calls, contract_size, lots)
            st.markdown(
                f'<div class="info-box">CALL Hit Rate: <b>{cm["hit_rate"]}%</b> &nbsp;|&nbsp; '
                f'Total Pts: <b>{cm["total_pts"]:+.1f}</b> &nbsp;|&nbsp; '
                f'P&L/Lot: <b>₹{cm["total_pnl"]:,.0f}</b></div>',
                unsafe_allow_html=True)
        else:
            st.info("No CALL trades in this run.")

    with rt4:
        puts = trades[trades["direction"]=="PUT"]
        st.markdown(f"**PUT Trades** — {len(puts)} total")
        if not puts.empty:
            tbl = build_trade_table(puts, contract_size)
            st.dataframe(style_trade_table(tbl), use_container_width=True, height=500, hide_index=True)
            pm = compute_metrics(puts, contract_size, lots)
            st.markdown(
                f'<div class="info-box">PUT Hit Rate: <b>{pm["hit_rate"]}%</b> &nbsp;|&nbsp; '
                f'Total Pts: <b>{pm["total_pts"]:+.1f}</b> &nbsp;|&nbsp; '
                f'P&L/Lot: <b>₹{pm["total_pnl"]:,.0f}</b></div>',
                unsafe_allow_html=True)
        else:
            st.info("No PUT trades in this run.")

    with rt5:
        trades2 = trades.copy()
        trades2["pnl_per_lot"] = pd.to_numeric(trades2.get("pnl_per_lot", 0),
                                                errors="coerce").fillna(0.0)
        trades2["pts_captured"] = pd.to_numeric(trades2["pts_captured"],
                                                errors="coerce").fillna(0.0)
        trades2["Month"] = pd.to_datetime(trades2["trade_date"]).dt.to_period("M").astype(str)
        mo_grp = trades2.groupby("Month").agg(
            Trades=("pts_captured","count"),
            Pts=("pts_captured","sum"),
            PnL_per_Lot=("pnl_per_lot","sum"),
            Wins=("pts_captured", lambda x: (x>0).sum()),
            Losses=("pts_captured", lambda x: (x<=0).sum()),
        ).reset_index()
        mo_grp["Hit%"]   = (mo_grp["Wins"]/mo_grp["Trades"]*100).round(1)
        mo_grp["Pts"]    = mo_grp["Pts"].round(1)
        mo_grp["P&L (₹)"]= mo_grp["PnL_per_Lot"].apply(lambda x: f"₹{x:,.0f}")
        mo_grp["Result"] = mo_grp["Pts"].apply(lambda x: "🟢" if x>0 else "🔴")
        disp_mo = mo_grp[["Month","Result","Trades","Wins","Losses","Hit%","Pts","P&L (₹)"]].copy()
        st.dataframe(disp_mo, use_container_width=True, hide_index=True)
        st.plotly_chart(monthly_pnl_chart(trades), use_container_width=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    init_db()

    st.markdown("""<div class="bt-header">
        <div class="bt-title">HedGEX — Cascade Backtest Engine</div>
        <div class="bt-sub">Strike-Level Custom Strategy &nbsp;·&nbsp;
        Intraday &amp; CNC Modes &nbsp;·&nbsp; Powered by Dhan Rolling Option API v2</div>
    </div>""", unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        st.markdown(
            '<div class="info-box">Client: <b>1100480354</b><br>Token: <b>Hardcoded</b></div>',
            unsafe_allow_html=True)
        st.markdown("---")
        symbol      = st.selectbox("Index", list(INDEX_CONFIG.keys()), index=0)
        expiry_flag = st.selectbox("Expiry Type", ["WEEK","MONTH"], index=0)
        expiry_code = st.selectbox("Expiry Code", [1,2,3], index=0,
                                   format_func=lambda x:{1:"Current",2:"Next",3:"Far"}[x])
        interval    = st.selectbox("Bar Interval", ["5","15","60"], index=0,
                                   format_func=lambda x: f"{x} min")
        st.markdown("---")
        st.markdown("### 📅 Date Range")
        today   = date.today()
        d_end   = st.date_input("End Date",   value=today-timedelta(days=1))
        d_start = st.date_input("Start Date", value=today-timedelta(days=365))
        st.markdown("---")
        st.markdown("### ⚡ Strikes")
        n_strikes   = st.slider("ATM ± N", 3, 10, 5)
        all_strikes = (["ATM"]
                       +[f"ATM+{i}" for i in range(1,n_strikes+1)]
                       +[f"ATM-{i}" for i in range(1,n_strikes+1)])
        st.caption(f"{len(all_strikes)} strikes selected")
        st.markdown("---")

        # ── Strategy parameters ───────────────────────────────────────────
        st.markdown("### 🎯 Strike-Level Strategy")
        st.markdown(
            '<div class="info-box">'
            '📈 <b>CALL</b>: IV Expanding + ATM+1 cascade &gt; trigger<br>'
            '📉 <b>PUT</b>: IV Expanding + ATM-1 cascade &gt; trigger<br>'
            'Target = min(ATM±1+2+3, max) &nbsp;|&nbsp; SL = fixed pts'
            '</div>', unsafe_allow_html=True)

        cascade_entry_threshold = st.slider("ATM±1 Cascade Trigger (pts)", 10, 200, 50, 5)
        max_target_pts          = st.slider("Max Target (pts)", 50, 500, 200, 25)
        fixed_sl_pts            = st.slider("Stop Loss (pts)", 10, 200, 50, 5)
        direction_filter        = st.selectbox("Direction", ["BOTH","CALL only","PUT only"])
        lots                    = st.number_input("Lots", min_value=1, max_value=100, value=1, step=1)
        max_trades_per_day      = st.selectbox("Max Trades/Day", [1,2,3,5,99], index=1,
                                               format_func=lambda x: str(x) if x<99 else "Unlimited")

        # CNC extra
        st.markdown("**CNC Settings**")
        max_hold_days = st.slider("Max Hold Days (CNC)", 1, 30, 5, 1,
                                  help="Force-close CNC position after this many trading days.")

        # Legacy aliases
        min_quality = cascade_entry_threshold / 100.0
        require_iv  = True

        st.markdown("---")
        st.markdown("### 🗄️ Database")
        stats = db_stats()
        st.markdown(
            '<div class="info-box">'
            f'Raw rows: <b>{stats["raw_rows"]}</b><br>'
            f'Days: <b>{stats["days"]}</b><br>'
            f'Signals: <b>{stats["signals"]}</b><br>'
            f'Intraday trades: <b>{stats["trades_intraday"]}</b><br>'
            f'CNC trades: <b>{stats["trades_cnc"]}</b>'
            '</div>', unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    (tab_fetch, tab_signals, tab_intraday, tab_cnc,
     tab_res_intraday, tab_res_cnc, tab_data_mgmt) = st.tabs([
        "1️⃣ Fetch Data",
        "2️⃣ Compute Signals",
        "3️⃣ Run Intraday BT",
        "4️⃣ Run CNC BT",
        "5️⃣ Intraday Results",
        "6️⃣ CNC Results",
        "🗄️ Data Management",
    ])

    # ── Tab 1: Fetch ──────────────────────────────────────────────────────────
    with tab_fetch:
        st.markdown("### 📡 Fetch Historical Options Chain")
        st.markdown(
            '<div class="info-box">Calls <code>POST /v2/charts/rollingoption</code> for each strike. '
            'Data stored in SQLite. Already-fetched days skipped. '
            'Checkpoints after every strike — safe to resume if interrupted.</div>',
            unsafe_allow_html=True)

        ckpt = checkpoint_status()
        if ckpt:
            st.markdown(
                '<div class="warn-box">⚡ Checkpoint — '
                + str(ckpt.get("trade_date","?")) + " | "
                + str(len(ckpt.get("completed_strikes",[]))) + " strikes done | "
                + str(len(ckpt.get("partial_rows",[]))) + " rows saved.</div>",
                unsafe_allow_html=True)
            if st.button("🗑️ Discard Checkpoint"):
                clear_checkpoint(); st.rerun()

        trading_dates = get_trading_dates(d_start, d_end)
        done_dates    = get_fetch_log(symbol, expiry_code, expiry_flag)
        pending       = [d for d in trading_dates if d not in done_dates]
        c1,c2,c3 = st.columns(3)
        c1.metric("Total days", len(trading_dates))
        c2.metric("Fetched",    len(done_dates))
        c3.metric("Pending",    len(pending))
        if pending:
            est = len(pending)*len(all_strikes)*2*0.35/60
            st.markdown(f'<div class="warn-box">Estimated time: ~{est:.1f} min</div>',
                        unsafe_allow_html=True)

        fetch_btn = st.button(f"🚀 Fetch {len(pending)} Days", type="primary",
                              use_container_width=True, disabled=(len(pending)==0))
        if fetch_btn:
            overall=st.progress(0); day_bar=st.progress(0)
            status=st.empty(); day_status=st.empty(); log_box=st.empty()
            log_lines=[]
            for idx, trade_date in enumerate(pending):
                day_status.text(f"Day {idx+1}/{len(pending)}: {trade_date}")
                ckpt_now = checkpoint_status()
                if ckpt_now and ckpt_now.get("trade_date") != trade_date:
                    clear_checkpoint()
                try:
                    n = fetch_one_day(symbol, trade_date, all_strikes, interval,
                                      expiry_code, expiry_flag, day_bar, status)
                    log_fetch(symbol, trade_date, expiry_code, expiry_flag, "ok", n)
                    log_lines.append(f"✅ {trade_date} — {n:,} rows")
                    if n == 0:
                        log_lines.append("   ⚠️ 0 rows — check expiry_code/date")
                except Exception as e:
                    log_lines.append(f"⚠️ {trade_date} — {e}")
                    log_box.text("\n".join(log_lines[-15:]))
                    st.warning(f"Interrupted at {trade_date}. Restart to resume.")
                    break
                overall.progress((idx+1)/len(pending))
                log_box.text("\n".join(log_lines[-15:]))
            overall.empty(); day_bar.empty(); status.empty(); day_status.empty()

        with st.expander("🔬 API Response Inspector"):
            dc1,dc2,dc3 = st.columns(3)
            dbg_date   = dc1.text_input("Test Date", value=str(date.today()-timedelta(days=5)))
            dbg_strike = dc2.selectbox("Strike", ["ATM","ATM+1","ATM-1"], key="dbg_s")
            dbg_otype  = dc3.selectbox("Type", ["CALL","PUT"], key="dbg_o")
            if st.button("🔍 Inspect", key="dbg_btn"):
                dbg_dt   = datetime.strptime(dbg_date,"%Y-%m-%d").date()
                dbg_from = (dbg_dt-timedelta(days=2)).strftime("%Y-%m-%d")
                dbg_to   = (dbg_dt+timedelta(days=2)).strftime("%Y-%m-%d")
                st.caption(f"Range: {dbg_from} → {dbg_to}")
                raw = fetch_rolling_option(symbol,dbg_from,dbg_to,
                                           dbg_strike,dbg_otype,interval,expiry_code,expiry_flag,
                                           silent=False)
                if raw:
                    ce = raw.get("ce",{}); ts_list = ce.get("timestamp",[])
                    match = [t for t in ts_list
                             if datetime.fromtimestamp(t,tz=pytz.UTC).astimezone(IST).date()==dbg_dt]
                    st.success(f"✅ {len(ts_list)} timestamps | {len(match)} match {dbg_date}")
                    if ts_list:
                        show=ts_list[:8]
                        st.dataframe(pd.DataFrame({
                            "epoch":show,
                            "UTC→IST":[datetime.fromtimestamp(t,tz=pytz.UTC).astimezone(IST).strftime("%Y-%m-%d %H:%M") for t in show],
                            "spot":ce.get("spot",[])[:8],"strike":ce.get("strike",[])[:8],
                            "oi":ce.get("oi",[])[:8],"iv":ce.get("iv",[])[:8],
                        }), hide_index=True, use_container_width=True)
                else:
                    st.error("Empty response.")

        with st.expander("🔎 Verify Strike Labels in DB"):
            if st.button("Check strike_type values", key="stcheck"):
                con = sqlite3.connect(DB_PATH, timeout=20)
                df_st = pd.read_sql_query(
                    "SELECT DISTINCT strike_type FROM raw_chain WHERE symbol=? LIMIT 30",
                    con, params=(symbol,)); con.close()
                if df_st.empty:
                    st.warning("No data in DB yet.")
                else:
                    st.dataframe(df_st, hide_index=True)
                    expected = {"ATM","ATM+1","ATM+2","ATM+3","ATM-1","ATM-2","ATM-3"}
                    missing  = expected - set(df_st["strike_type"].tolist())
                    if missing: st.warning(f"⚠️ Missing: {missing}")
                    else:       st.success("✅ All required strike labels present.")

        if done_dates:
            st.markdown("#### ✅ Fetched Days")
            con=sqlite3.connect(DB_PATH, timeout=20)
            st.dataframe(pd.read_sql_query(
                "SELECT trade_date,rows_fetched,fetched_at FROM fetch_log "
                "WHERE symbol=? AND expiry_code=? AND expiry_flag=? AND status='ok' "
                "ORDER BY trade_date DESC",
                con,params=(symbol,expiry_code,expiry_flag)),
                use_container_width=True,height=300,hide_index=True)
            con.close()

    # ── Tab 2: Signals ────────────────────────────────────────────────────────
    with tab_signals:
        st.markdown("### ⚡ Compute Cascade Signals")
        st.markdown(
            '<div class="info-box">Optional for the custom strategy — '
            'the intraday/CNC backtests read raw_chain directly. '
            'Compute signals here for the legacy cascade dashboard.</div>',
            unsafe_allow_html=True)
        done_dates_s = get_fetch_log(symbol, expiry_code, expiry_flag)
        con=sqlite3.connect(DB_PATH, timeout=20)
        sig_d=pd.read_sql_query("SELECT DISTINCT trade_date FROM cascade_signals WHERE symbol=?",
                                con,params=(symbol,)); con.close()
        sig_dates   = set(sig_d["trade_date"].tolist()) if not sig_d.empty else set()
        pending_sig = [d for d in sorted(done_dates_s) if d not in sig_dates]
        c1,c2,c3 = st.columns(3)
        c1.metric("Fetched days", len(done_dates_s))
        c2.metric("Signals ready",len(sig_dates))
        c3.metric("Pending",      len(pending_sig))

        if st.button(f"⚡ Compute Signals for {len(pending_sig)} Days", type="primary",
                     use_container_width=True, disabled=(len(pending_sig)==0)):
            prog=st.progress(0); status=st.empty()
            for idx,trade_date in enumerate(pending_sig):
                status.text(f"Computing {trade_date} ({idx+1}/{len(pending_sig)})")
                df_d=load_raw_chain(symbol,trade_date,expiry_code,expiry_flag)
                if not df_d.empty:
                    save_signals(compute_signals_for_day(df_d,symbol,trade_date))
                prog.progress((idx+1)/len(pending_sig))
            prog.empty(); status.empty()
            st.success("✅ Signals computed"); st.rerun()

        if sig_dates:
            latest=max(sig_dates); sig_df=load_signals(symbol,latest)
            st.markdown(f"#### Preview — {latest}")
            if not sig_df.empty:
                st.dataframe(sig_df[["timestamp","spot_price","bear_quality","bull_quality",
                                      "signal","signal_strength","iv_regime","avg_iv",
                                      "bear_fuel_pts","bear_absorb_pts"]],
                             use_container_width=True,height=300,hide_index=True)

    # ── Tab 3: Run Intraday Backtest ──────────────────────────────────────────
    with tab_intraday:
        st.markdown("### 3️⃣ Run Intraday Backtest")
        st.markdown(
            '<div class="mode-intraday">'
            '⚡ <b>INTRADAY MODE</b> — entry and exit within the same trading day.<br>'
            'EOD forced exit if target/stop not hit. Suitable for options day-trading simulation.'
            '</div>', unsafe_allow_html=True)

        done_dates_i = get_fetch_log(symbol, expiry_code, expiry_flag)
        all_bt_dates = sorted(done_dates_i)

        st.markdown(
            '<div class="info-box">'
            f'Trigger: <b>{cascade_entry_threshold} pts</b> &nbsp;|&nbsp; '
            f'Max Target: <b>{max_target_pts} pts</b> &nbsp;|&nbsp; '
            f'SL: <b>{fixed_sl_pts} pts</b> &nbsp;|&nbsp; '
            f'Dir: <b>{direction_filter}</b> &nbsp;|&nbsp; '
            f'Lots: <b>{lots}</b> &nbsp;|&nbsp; '
            f'Days: <b>{len(all_bt_dates)}</b>'
            '</div>', unsafe_allow_html=True)

        contract_size = INDEX_CONFIG.get(symbol,{}).get("contract_size",25)
        st.caption(f"Contract size: {contract_size} units/lot  ·  "
                   f"P&L per 1pt move per lot = ₹{contract_size}")

        col_run, col_clr = st.columns([3,1])
        run_intra = col_run.button(f"▶ Run Intraday BT on {len(all_bt_dates)} Days",
                                   type="primary", use_container_width=True,
                                   disabled=(len(all_bt_dates)==0))
        if col_clr.button("🗑️ Clear Intraday", use_container_width=True):
            clear_trades(symbol, "INTRADAY"); st.success("Intraday trades cleared"); st.rerun()

        if run_intra:
            clear_trades(symbol, "INTRADAY")
            prog=st.progress(0); status=st.empty(); all_trades=[]
            for idx, td in enumerate(all_bt_dates):
                status.text(f"Simulating {td} ({idx+1}/{len(all_bt_dates)})")
                df_day = load_raw_chain(symbol, td, expiry_code, expiry_flag)
                if not df_day.empty:
                    iv_df = compute_iv_regime_series(df_day)
                    all_trades.extend(run_intraday_day(
                        df_day, symbol, td, expiry_flag,
                        cascade_entry_threshold, max_target_pts, fixed_sl_pts,
                        direction_filter, max_trades_per_day, lots, iv_df))
                prog.progress((idx+1)/len(all_bt_dates))
            save_trades(all_trades)
            prog.empty(); status.empty()
            st.success(f"✅ Intraday done — {len(all_trades)} trades"); st.rerun()

    # ── Tab 4: Run CNC Backtest ───────────────────────────────────────────────
    with tab_cnc:
        st.markdown("### 4️⃣ Run CNC / Positional Backtest")
        st.markdown(
            '<div class="mode-cnc">'
            '📦 <b>CNC MODE</b> — positional / multi-day holding.<br>'
            f'Entry on signal bar; exit at EOD when target/stop hit or after max <b>{max_hold_days}</b> days.<br>'
            'Max 2 concurrent positions (1 CALL + 1 PUT). Suitable for swing options simulation.'
            '</div>', unsafe_allow_html=True)

        done_dates_c = get_fetch_log(symbol, expiry_code, expiry_flag)
        all_cnc_dates = sorted(done_dates_c)

        st.markdown(
            '<div class="info-box">'
            f'Trigger: <b>{cascade_entry_threshold} pts</b> &nbsp;|&nbsp; '
            f'Max Target: <b>{max_target_pts} pts</b> &nbsp;|&nbsp; '
            f'SL: <b>{fixed_sl_pts} pts</b> &nbsp;|&nbsp; '
            f'Dir: <b>{direction_filter}</b> &nbsp;|&nbsp; '
            f'Lots: <b>{lots}</b> &nbsp;|&nbsp; '
            f'Max Hold: <b>{max_hold_days} days</b> &nbsp;|&nbsp; '
            f'Days: <b>{len(all_cnc_dates)}</b>'
            '</div>', unsafe_allow_html=True)

        contract_size = INDEX_CONFIG.get(symbol,{}).get("contract_size",25)
        st.caption(f"Contract size: {contract_size} units/lot  ·  "
                   f"P&L per 1pt move per lot = ₹{contract_size}")

        col_run2, col_clr2 = st.columns([3,1])
        run_cnc = col_run2.button(f"▶ Run CNC BT on {len(all_cnc_dates)} Days",
                                   type="primary", use_container_width=True,
                                   disabled=(len(all_cnc_dates)==0))
        if col_clr2.button("🗑️ Clear CNC", use_container_width=True):
            clear_trades(symbol, "CNC"); st.success("CNC trades cleared"); st.rerun()

        if run_cnc:
            clear_trades(symbol, "CNC")
            status2 = st.empty(); prog2 = st.progress(0)
            cnc_trades = run_cnc_backtest(
                all_cnc_dates, symbol, expiry_flag, expiry_code,
                cascade_entry_threshold, max_target_pts, fixed_sl_pts,
                direction_filter, max_hold_days, lots, status2)
            save_trades(cnc_trades)
            prog2.progress(1.0); prog2.empty(); status2.empty()
            st.success(f"✅ CNC done — {len(cnc_trades)} trades"); st.rerun()

    # ── Tab 5: Intraday Results ───────────────────────────────────────────────
    with tab_res_intraday:
        st.markdown("### 5️⃣ Intraday Results")
        contract_size = INDEX_CONFIG.get(symbol,{}).get("contract_size",25)
        trades_i = load_trades(symbol, "INTRADAY")
        render_results_section(trades_i, symbol, "Intraday", contract_size, lots)

    # ── Tab 6: CNC Results ────────────────────────────────────────────────────
    with tab_res_cnc:
        st.markdown("### 6️⃣ CNC Results")
        contract_size = INDEX_CONFIG.get(symbol,{}).get("contract_size",25)
        trades_c = load_trades(symbol, "CNC")
        render_results_section(trades_c, symbol, "CNC", contract_size, lots)

    # ── Tab 7: Data Management ────────────────────────────────────────────────
    with tab_data_mgmt:
        st.markdown("### 🗄️ Data Management")
        st.markdown(
            '<div class="danger-box">⚠️ Deletions are <b>permanent and irreversible</b>. '
            'Raw chain data must be re-fetched from Dhan API after clearing.</div>',
            unsafe_allow_html=True)

        st.markdown("---")

        # ── Section 1: Clear Trades ───────────────────────────────────────
        st.markdown("#### 🗑️ Clear Backtest Trades")
        dm_c1, dm_c2, dm_c3 = st.columns(3)

        with dm_c1:
            st.markdown("**Intraday trades only**")
            count_i = len(load_trades(symbol, "INTRADAY"))
            st.caption(f"{count_i} intraday trades for {symbol}")
            if st.button(f"Clear {symbol} Intraday Trades", use_container_width=True):
                clear_trades(symbol, "INTRADAY")
                st.success(f"Cleared intraday trades for {symbol}"); st.rerun()

        with dm_c2:
            st.markdown("**CNC trades only**")
            count_c = len(load_trades(symbol, "CNC"))
            st.caption(f"{count_c} CNC trades for {symbol}")
            if st.button(f"Clear {symbol} CNC Trades", use_container_width=True):
                clear_trades(symbol, "CNC")
                st.success(f"Cleared CNC trades for {symbol}"); st.rerun()

        with dm_c3:
            st.markdown("**All trades (all symbols)**")
            count_all = stats["trades_intraday"] + stats["trades_cnc"]
            st.caption(f"{count_all} total trades in DB")
            if st.button("Clear ALL Trades (All Symbols)", use_container_width=True,
                         type="secondary"):
                clear_trades(); st.success("All trades cleared"); st.rerun()

        st.markdown("---")

        # ── Section 2: Clear Signals ──────────────────────────────────────
        st.markdown("#### 🔻 Clear Cascade Signals")
        dm_s1, dm_s2 = st.columns(2)

        with dm_s1:
            st.markdown(f"**Signals for {symbol}**")
            con=sqlite3.connect(DB_PATH, timeout=20)
            sc = pd.read_sql_query(
                "SELECT COUNT(*) as c FROM cascade_signals WHERE symbol=?",
                con,params=(symbol,)).iloc[0]["c"]; con.close()
            st.caption(f"{int(sc)} signal rows for {symbol}")
            if st.button(f"Clear {symbol} Signals", use_container_width=True):
                clear_signals(symbol)
                st.success(f"Cleared signals for {symbol}"); st.rerun()

        with dm_s2:
            st.markdown("**All signals (all symbols)**")
            st.caption(f"{stats['signals']} total signal rows")
            if st.button("Clear ALL Signals", use_container_width=True, type="secondary"):
                clear_signals()
                st.success("All signals cleared"); st.rerun()

        st.markdown("---")

        # ── Section 3: Clear Raw Chain ────────────────────────────────────
        st.markdown("#### ☢️ Clear Raw Chain Data")
        st.markdown(
            '<div class="warn-box">Clearing raw chain also clears the fetch log. '
            'You will need to re-fetch all data from Dhan API.</div>',
            unsafe_allow_html=True)
        dm_r1, dm_r2 = st.columns(2)

        with dm_r1:
            st.markdown(f"**Raw chain for {symbol}**")
            con=sqlite3.connect(DB_PATH, timeout=20)
            rc_cnt = pd.read_sql_query(
                "SELECT COUNT(*) as c FROM raw_chain WHERE symbol=?",
                con,params=(symbol,)).iloc[0]["c"]; con.close()
            st.caption(f"{int(rc_cnt)} raw rows for {symbol}")
            confirm_sym = st.text_input(
                f'Type "{symbol}" to confirm', key="confirm_sym",
                placeholder=symbol)
            if st.button(f"Clear {symbol} Raw Chain", use_container_width=True,
                         type="secondary", disabled=(confirm_sym != symbol)):
                clear_raw_chain(symbol)
                st.success(f"Cleared raw chain for {symbol}"); st.rerun()

        with dm_r2:
            st.markdown("**⚠️ FULL WIPE — All Data**")
            st.caption(f"{stats['raw_rows']} raw rows · all symbols")
            confirm_all = st.text_input(
                'Type "WIPE ALL" to confirm', key="confirm_all",
                placeholder="WIPE ALL")
            if st.button("☢️ Full Database Wipe", use_container_width=True,
                         type="secondary", disabled=(confirm_all != "WIPE ALL")):
                clear_raw_chain()     # clears raw_chain + fetch_log
                clear_signals()       # clears cascade_signals
                clear_trades()        # clears bt_trades
                clear_checkpoint()    # clears checkpoint file
                st.success("✅ Full database wipe complete. Start fresh by fetching data.")
                st.rerun()

        st.markdown("---")

        # ── DB Stats ──────────────────────────────────────────────────────
        st.markdown("#### 📊 Current Database Stats")
        fresh_stats = db_stats()
        s1,s2,s3,s4,s5 = st.columns(5)
        s1.metric("Raw Rows",       fresh_stats["raw_rows"])
        s2.metric("Trading Days",   fresh_stats["days"])
        s3.metric("Intraday Trades",fresh_stats["trades_intraday"])
        s4.metric("CNC Trades",     fresh_stats["trades_cnc"])
        s5.metric("Signal Rows",    fresh_stats["signals"])

        # DB file size
        if os.path.exists(DB_PATH):
            size_mb = os.path.getsize(DB_PATH) / 1024 / 1024
            st.caption(f"📁 {DB_PATH} — {size_mb:.2f} MB")

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown(
        '<div style="text-align:center;padding:16px;font-family:JetBrains Mono,monospace;'
        'font-size:0.68rem;color:rgba(255,255,255,0.2);">'
        'HedGEX Cascade Backtest v3 · NYZTrade Analytics Pvt. Ltd. · Research purposes only'
        '</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
