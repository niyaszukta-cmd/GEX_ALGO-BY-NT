# ============================================================================
# HedGEX — Cascade Backtest Engine v3
# Powered by NYZTrade Analytics Pvt. Ltd.
# Strategy: Data-Driven Entry Gate + Trailing Premium Exit + Dynamic Lots
# Analysis: 1,050 trades · 246 days · Apr 2025–Apr 2026
# Entry: Expiry afternoon (13-15h) EXPANDING/FLAT + Regular afternoon (14-15h) FLAT
# Exit: Trailing premium stop (65% of peak) + 20% hard floor + EOD fallback
# Sizing: Dynamic 1→3→5 lots based on rolling WR
# Data: Dhan Rolling Option API v2 | Engine: GEX Cascade Mathematics
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
st.set_page_config(page_title="HedGEX Cascade Backtest v3",
                   page_icon="📊", layout="wide",
                   initial_sidebar_state="expanded")

# ── Constants ─────────────────────────────────────────────────────────────────
IST       = pytz.timezone("Asia/Kolkata")
DHAN_BASE = "https://api.dhan.co/v2"
RISK_FREE = 0.07
DB_PATH   = "hedgex_backtest.db"
CKPT_PATH = "hedgex_checkpoint.json"

DHAN_CLIENT_ID    = "1100480354"
DHAN_ACCESS_TOKEN = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9.eyJpc3MiOiJkaGFuIiwicGFydG5lcklkIjoiIiwiZXhwIjoxNzc1ODI1Mjg5LCJhcHBfaWQiOiJhYjYxZmJmOSIsImlhdCI6MTc3NTczODg4OSwidG9rZW5Db25zdW1lclR5cGUiOiJBUFAiLCJ3ZWJob29rVXJsIjoiIiwiZGhhbkNsaWVudElkIjoiMTEwMDQ4MDM1NCJ9.GGiQsWWqz_mf_srzciUVLzjBzC8PXfWf7dzyjAmrjSgygFg5chszWMVG1z5MQIGLH0GeSa5vmgOVc4qkSkm8pg"

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

# ── Expiry calendar — CRITICAL: Thursday before Sep 1 2025, Tuesday from Sep 1 2025
EXPIRY_SHIFT_DATE = date(2025, 9, 1)

def is_expiry_day(trade_date) -> bool:
    """Returns True if trade_date is a weekly expiry day for NIFTY.
    Pre Sep-2025: Thursday | Post Sep-2025: Tuesday"""
    if isinstance(trade_date, str):
        trade_date = datetime.strptime(trade_date, "%Y-%m-%d").date()
    elif isinstance(trade_date, datetime):
        trade_date = trade_date.date()
    weekday = trade_date.weekday()  # 0=Mon, 1=Tue, 3=Thu, 4=Fri
    if trade_date < EXPIRY_SHIFT_DATE:
        return weekday == 3  # Thursday
    else:
        return weekday == 1  # Tuesday

# ── Strategy parameters (data-derived from 1,050 trade analysis)
STRATEGY_PARAMS = {
    # Entry gate
    "expiry_entry_start_h":   13,    # 13:00 on expiry day
    "expiry_entry_end_h":     15,    # up to 15:xx on expiry day
    "regular_entry_start_h":  14,    # 14:00 on regular day
    "regular_entry_end_h":    15,    # up to 15:xx on regular day
    "expiry_iv_regimes":      ["EXPANDING", "FLAT"],
    "regular_iv_regimes":     ["FLAT"],
    "min_cascade_pts":        150,   # ≥150 pts confirmed best bucket
    "max_buy_px":             200,   # ≤₹200 — same WR, 3x less loss above this
    "max_trades_per_day":     2,     # 1 CALL + 1 PUT max
    # Exit engine — trailing premium stop
    "trail_activate_pct":     0.40,  # trail activates when premium up 40%
    "trail_level_pct":        0.65,  # trail stop at 65% of peak premium
    "trail_expiry_final_pct": 0.80,  # at 15:15 on expiry → tighten to 80%
    "expiry_final_hour":      15,
    "expiry_final_min":       15,
    "hard_floor_pct":         0.20,  # exit immediately if premium < 20% of buy
    "eod_exit_h":             15,
    "eod_exit_m":             25,
    # Cooldown
    "cooldown_mins":          60,    # 60-min cooldown after floor/stop exit
    # Dynamic lot sizing
    "lot_phase1":             1,     # first 3 months: 1 lot
    "lot_phase2":             3,     # if rolling 3M WR ≥ 40%: 3 lots
    "lot_phase3":             5,     # if rolling 6M WR ≥ 50%: 5 lots
    "lot_drawdown_reset":     1,     # after 3 consecutive losses → 1 lot
    "wr_phase2_threshold":    0.40,
    "wr_phase3_threshold":    0.50,
    "consecutive_loss_limit": 3,
}

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
.g{color:#10b981;} .r{color:#ef4444;} .n{color:#94a3b8;} .a{color:#f59e0b;}
.info-box{background:rgba(6,182,212,0.08);border-left:3px solid #06b6d4;
  border-radius:6px;padding:10px 14px;font-family:"JetBrains Mono",monospace;font-size:0.80rem;line-height:1.8;}
.warn-box{background:rgba(245,158,11,0.08);border-left:3px solid #f59e0b;
  border-radius:6px;padding:10px 14px;font-family:"JetBrains Mono",monospace;font-size:0.80rem;line-height:1.8;}
.strat-box{background:rgba(168,85,247,0.08);border-left:3px solid #a855f7;
  border-radius:6px;padding:12px 16px;font-family:"JetBrains Mono",monospace;font-size:0.82rem;line-height:2.0;}
.success-box{background:rgba(16,185,129,0.08);border-left:3px solid #10b981;
  border-radius:6px;padding:10px 14px;font-family:"JetBrains Mono",monospace;font-size:0.80rem;line-height:1.8;}
.danger-box{background:rgba(239,68,68,0.08);border-left:3px solid #ef4444;
  border-radius:6px;padding:10px 14px;font-family:"JetBrains Mono",monospace;font-size:0.80rem;line-height:1.8;}
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
    con = sqlite3.connect(DB_PATH); cur = con.cursor()
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
        strike_type TEXT,
        strike_cascade_pts REAL,
        cumulative_cascade_atm3 REAL,
        bear_fuel_pts REAL, bear_absorb_pts REAL,
        bull_fuel_pts REAL, bull_absorb_pts REAL,
        bear_quality REAL, bull_quality REAL,
        iv_regime TEXT, avg_iv REAL, iv_skew REAL,
        net_gex_total REAL, signal TEXT, signal_strength REAL,
        cascade_target REAL, cascade_stop REAL,
        UNIQUE(symbol,trade_date,timestamp,strike_type))""")
    # v3 bt_trades — extended with trailing exit and lot-size fields
    cur.execute("""CREATE TABLE IF NOT EXISTS bt_trades(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT, trade_date TEXT, trade_type TEXT,
        entry_time TEXT, exit_time TEXT,
        direction TEXT, option_type TEXT, strike_used TEXT,
        entry_spot REAL, exit_spot REAL,
        option_buy_price REAL, option_sell_price REAL,
        peak_premium REAL,
        pts_captured REAL, pnl_per_lot REAL, lots_used INTEGER,
        total_pnl REAL,
        contract_size INTEGER,
        cascade_trigger_pts REAL, cascade_target REAL, cascade_stop REAL,
        exit_reason TEXT,
        iv_regime TEXT, signal_strength REAL,
        is_expiry_day INTEGER, expiry_flag TEXT,
        backtest_mode TEXT,
        trailing_activated INTEGER,
        cooldown_triggered INTEGER)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS fetch_log(
        symbol TEXT, trade_date TEXT, expiry_code INTEGER, expiry_flag TEXT,
        status TEXT, rows_fetched INTEGER, fetched_at TEXT,
        PRIMARY KEY(symbol,trade_date,expiry_code,expiry_flag))""")

    # ── v3 schema migration — safely add new columns to existing bt_trades ──
    existing_cols = {row[1] for row in cur.execute("PRAGMA table_info(bt_trades)").fetchall()}
    new_cols = {
        "peak_premium":       "REAL    DEFAULT 0.0",
        "lots_used":          "INTEGER DEFAULT 1",
        "total_pnl":          "REAL    DEFAULT 0.0",
        "trailing_activated": "INTEGER DEFAULT 0",
        "cooldown_triggered": "INTEGER DEFAULT 0",
    }
    for col, col_def in new_cols.items():
        if col not in existing_cols:
            cur.execute(f"ALTER TABLE bt_trades ADD COLUMN {col} {col_def}")

    # Backfill total_pnl for any old rows that have 0 (pnl_per_lot * lots_used)
    cur.execute("""UPDATE bt_trades SET total_pnl = pnl_per_lot
                   WHERE total_pnl = 0 AND pnl_per_lot IS NOT NULL""")
    cur.execute("""UPDATE bt_trades SET lots_used = 1
                   WHERE lots_used IS NULL OR lots_used = 0""")

    con.commit(); con.close()

def get_fetch_log(symbol, expiry_code, expiry_flag):
    con = sqlite3.connect(DB_PATH); cur = con.cursor()
    cur.execute("SELECT trade_date FROM fetch_log WHERE symbol=? AND expiry_code=? AND expiry_flag=? AND status='ok'",
                (symbol, expiry_code, expiry_flag))
    done = {r[0] for r in cur.fetchall()}; con.close(); return done

def log_fetch(symbol, trade_date, expiry_code, expiry_flag, status, rows):
    con = sqlite3.connect(DB_PATH)
    con.execute("INSERT OR REPLACE INTO fetch_log VALUES(?,?,?,?,?,?,?)",
                (symbol,trade_date,expiry_code,expiry_flag,status,rows,
                 datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")))
    con.commit(); con.close()

def save_raw_chain(rows):
    if not rows: return
    con = sqlite3.connect(DB_PATH)
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
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT * FROM raw_chain WHERE symbol=? AND trade_date=? AND expiry_code=? AND expiry_flag=? ORDER BY timestamp,strike",
        con, params=(symbol,date_str,expiry_code,expiry_flag)); con.close()
    if not df.empty: df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

def save_signals(rows):
    if not rows: return
    con = sqlite3.connect(DB_PATH)
    con.executemany("""INSERT OR REPLACE INTO cascade_signals
        (symbol,trade_date,timestamp,spot_price,strike_type,
         strike_cascade_pts,cumulative_cascade_atm3,
         bear_fuel_pts,bear_absorb_pts,
         bull_fuel_pts,bull_absorb_pts,bear_quality,bull_quality,
         iv_regime,avg_iv,iv_skew,net_gex_total,signal,signal_strength,
         cascade_target,cascade_stop)
        VALUES(:symbol,:trade_date,:timestamp,:spot_price,:strike_type,
         :strike_cascade_pts,:cumulative_cascade_atm3,
         :bear_fuel_pts,:bear_absorb_pts,
         :bull_fuel_pts,:bull_absorb_pts,:bear_quality,:bull_quality,
         :iv_regime,:avg_iv,:iv_skew,:net_gex_total,:signal,:signal_strength,
         :cascade_target,:cascade_stop)""", rows)
    con.commit(); con.close()

def load_signals(symbol, date_str):
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT * FROM cascade_signals WHERE symbol=? AND trade_date=? ORDER BY timestamp,strike_type",
        con, params=(symbol,date_str)); con.close()
    if not df.empty: df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

def save_trades(rows, mode="INTRADAY"):
    if not rows: return
    con = sqlite3.connect(DB_PATH)
    con.executemany("""INSERT INTO bt_trades
        (symbol,trade_date,trade_type,entry_time,exit_time,
         direction,option_type,strike_used,
         entry_spot,exit_spot,option_buy_price,option_sell_price,
         peak_premium,pts_captured,pnl_per_lot,lots_used,total_pnl,
         contract_size,cascade_trigger_pts,cascade_target,cascade_stop,
         exit_reason,iv_regime,signal_strength,
         is_expiry_day,expiry_flag,backtest_mode,
         trailing_activated,cooldown_triggered)
        VALUES(:symbol,:trade_date,:trade_type,:entry_time,:exit_time,
         :direction,:option_type,:strike_used,
         :entry_spot,:exit_spot,:option_buy_price,:option_sell_price,
         :peak_premium,:pts_captured,:pnl_per_lot,:lots_used,:total_pnl,
         :contract_size,:cascade_trigger_pts,:cascade_target,:cascade_stop,
         :exit_reason,:iv_regime,:signal_strength,
         :is_expiry_day,:expiry_flag,:backtest_mode,
         :trailing_activated,:cooldown_triggered)""", rows)
    con.commit(); con.close()

def load_trades(symbol=None, mode=None):
    con = sqlite3.connect(DB_PATH)
    conditions, params = [], []
    if symbol:   conditions.append("symbol=?");        params.append(symbol)
    if mode:     conditions.append("backtest_mode=?"); params.append(mode)
    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    df = pd.read_sql_query(
        f"SELECT * FROM bt_trades {where} ORDER BY trade_date,entry_time",
        con, params=params)
    con.close()
    return df

def clear_trades(symbol=None, mode=None):
    con = sqlite3.connect(DB_PATH)
    conditions, params = [], []
    if symbol: conditions.append("symbol=?");        params.append(symbol)
    if mode:   conditions.append("backtest_mode=?"); params.append(mode)
    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    con.execute(f"DELETE FROM bt_trades {where}", params)
    con.commit(); con.close()

def clear_all_data():
    con = sqlite3.connect(DB_PATH)
    for tbl in ["raw_chain","cascade_signals","bt_trades","fetch_log"]:
        con.execute(f"DELETE FROM {tbl}")
    con.commit(); con.close()
    if os.path.exists(CKPT_PATH): os.remove(CKPT_PATH)

def clear_signals_only(symbol=None):
    con = sqlite3.connect(DB_PATH)
    if symbol: con.execute("DELETE FROM cascade_signals WHERE symbol=?", (symbol,))
    else:      con.execute("DELETE FROM cascade_signals")
    con.commit(); con.close()

def db_stats():
    con = sqlite3.connect(DB_PATH); cur = con.cursor()
    cur.execute("SELECT COUNT(*) FROM raw_chain");         rr = cur.fetchone()[0]
    cur.execute("SELECT COUNT(DISTINCT trade_date),COUNT(DISTINCT symbol) FROM raw_chain")
    days,syms = cur.fetchone()
    cur.execute("SELECT COUNT(*) FROM bt_trades");         tr = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM cascade_signals");   sg = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM fetch_log WHERE status='ok'"); fl = cur.fetchone()[0]
    con.close()
    return {"raw_rows":rr,"days":days,"symbols":syms,"trades":tr,"signals":sg,"fetch_log":fl}

# ── Rolling WR for dynamic lot sizing ────────────────────────────────────────
def get_rolling_lot_size(symbol: str, as_of_date: str, p=STRATEGY_PARAMS) -> int:
    """Compute current lot size based on rolling WR of past trades."""
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT trade_date, pnl_per_lot FROM bt_trades WHERE symbol=? AND trade_date<? ORDER BY trade_date",
        con, params=(symbol, as_of_date))
    con.close()
    if df.empty: return p["lot_phase1"]

    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["win"] = df["pnl_per_lot"] > 0
    as_of = pd.to_datetime(as_of_date)

    # Check 3 consecutive losses (most recent trades)
    recent = df.tail(3)
    if len(recent) == 3 and (recent["win"] == False).all():
        return p["lot_drawdown_reset"]

    # Check 6M WR for phase 3
    six_mo = df[df["trade_date"] >= as_of - pd.DateOffset(months=6)]
    if len(six_mo) >= 10 and six_mo["win"].mean() >= p["wr_phase3_threshold"]:
        return p["lot_phase3"]

    # Check 3M WR for phase 2
    three_mo = df[df["trade_date"] >= as_of - pd.DateOffset(months=3)]
    if len(three_mo) >= 6 and three_mo["win"].mean() >= p["wr_phase2_threshold"]:
        return p["lot_phase2"]

    return p["lot_phase1"]

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
        avg_iv = (grp["call_iv"].mean() + grp["put_iv"].mean()) / 2
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

# ── VANNA zone constants ──────────────────────────────────────────────────────
VANNA_ADJ = {
    "SUPPORT_FLOOR":      {"COMPRESSING":0.60,"FLAT":0.35,"EXPANDING":-0.20},
    "TRAP_DOOR":          {"EXPANDING":-0.30,"FLAT":0.10,"COMPRESSING":0.15},
    "VACUUM_ZONE":        {"EXPANDING":0.50,"FLAT":0.20,"COMPRESSING":0.10},
    "RESISTANCE_CEILING": {"EXPANDING":-0.20,"FLAT":0.00,"COMPRESSING":0.15},
}

def compute_strike_cascade_pts(df_ts, spot, symbol, iv_regime, target_strike_type):
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

    strike_pts_map = {}
    for _, row in df_ts.iterrows():
        s = row["strike"]; gex = row["net_gex"]
        rp = min(abs(gex)*ppu, cap)
        role = vz_map.get(s)
        adj  = VANNA_ADJ.get(role,{}).get(iv_regime,0.0) if role else 0.0
        strike_pts_map[s] = rp * (1 + adj)

    atm_strike = min(df_ts["strike"].unique(), key=lambda x: abs(x-spot))
    cfg = INDEX_CONFIG.get(symbol, {"strike_interval": 50})
    interval = cfg["strike_interval"]

    if target_strike_type == "ATM+1":
        trigger_strike = atm_strike + interval
        direction = "BULL"
        target_strikes = [atm_strike + interval*i for i in range(1, 4)]
    elif target_strike_type == "ATM-1":
        trigger_strike = atm_strike - interval
        direction = "BEAR"
        target_strikes = [atm_strike - interval*i for i in range(1, 4)]
    else:
        return 0.0, 0.0, "NONE"

    trigger_cascade = strike_pts_map.get(trigger_strike, 0.0)
    cum_cascade = sum(strike_pts_map.get(s, 0.0) for s in target_strikes)
    return trigger_cascade, cum_cascade, direction

def compute_signals_for_day(df_day, symbol, trade_date):
    iv_df = compute_iv_regime_series(df_day)
    rows  = []
    for ts in sorted(df_day["timestamp"].unique()):
        df_ts = df_day[df_day["timestamp"]==ts].copy()
        spot  = df_ts["spot_price"].mean()
        iv_row = iv_df[iv_df["timestamp"]==ts]
        if iv_row.empty: iv_regime,avg_iv,iv_skew = "FLAT",15.0,0.0
        else:
            iv_regime = str(iv_row.iloc[0]["iv_regime"])
            avg_iv    = float(iv_row.iloc[0]["avg_iv"])
            iv_skew   = float(iv_row.iloc[0]["iv_skew"])

        for strike_type in ["ATM+1", "ATM-1"]:
            trig_pts, cum_pts, direction = compute_strike_cascade_pts(
                df_ts, spot, symbol, iv_regime, strike_type)

            net_gex = df_ts["net_gex"].sum()

            if direction == "BULL":
                signal = "BULL" if trig_pts >= 50 else "NONE"
            elif direction == "BEAR":
                signal = "BEAR" if trig_pts >= 50 else "NONE"
            else:
                signal = "NONE"

            target_pts = min(cum_pts, 200.0)
            if direction == "BULL":
                cascade_target = spot + target_pts
                cascade_stop   = spot - 50.0
            else:
                cascade_target = spot - target_pts
                cascade_stop   = spot + 50.0

            rows.append({
                "symbol": symbol, "trade_date": trade_date,
                "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S") if hasattr(ts,"strftime") else str(ts),
                "spot_price": round(spot, 2),
                "strike_type": strike_type,
                "strike_cascade_pts": round(trig_pts, 2),
                "cumulative_cascade_atm3": round(cum_pts, 2),
                "bear_fuel_pts": 0.0, "bear_absorb_pts": 0.0,
                "bull_fuel_pts": 0.0, "bull_absorb_pts": 0.0,
                "bear_quality": round(trig_pts/50, 4) if direction=="BEAR" else 0.0,
                "bull_quality": round(trig_pts/50, 4) if direction=="BULL" else 0.0,
                "iv_regime": iv_regime, "avg_iv": round(avg_iv, 2), "iv_skew": round(iv_skew, 4),
                "net_gex_total": round(net_gex, 4),
                "signal": signal,
                "signal_strength": round(trig_pts, 2),
                "cascade_target": round(cascade_target, 2),
                "cascade_stop": round(cascade_stop, 2),
            })
    return rows

# ── Option premium estimation ─────────────────────────────────────────────────
def estimate_option_premium(spot, strike_type, symbol, iv_pct=15.0, tte_days=7, option_type="CALL"):
    cfg = INDEX_CONFIG.get(symbol, {"strike_interval": 50})
    interval = cfg["strike_interval"]
    iv = max(iv_pct / 100.0, 0.05)
    tte = max(tte_days / 365.0, 1/365.0)
    r = RISK_FREE

    if strike_type == "ATM+1":
        atm = round(spot / interval) * interval
        K = atm + interval
    elif strike_type == "ATM-1":
        atm = round(spot / interval) * interval
        K = atm - interval
    else:
        K = spot

    S = spot
    d1_val = BS.d1(S, K, tte, r, iv)
    d2_val = BS.d2(S, K, tte, r, iv)

    if option_type == "CALL":
        price = S * norm.cdf(d1_val) - K * np.exp(-r*tte) * norm.cdf(d2_val)
    else:
        price = K * np.exp(-r*tte) * norm.cdf(-d2_val) - S * norm.cdf(-d1_val)

    return max(round(price, 2), 1.0)

# ── Entry gate — data-derived rules ──────────────────────────────────────────
def passes_entry_gate(ts, iv_regime, trig_pts, buy_px, is_expiry,
                      calls_today, puts_today, cooldown_until,
                      p=STRATEGY_PARAMS) -> tuple:
    """
    Returns (allowed: bool, reason: str)
    Data-derived entry rules from 1,050 trade analysis.
    """
    h = ts.hour
    m = ts.minute

    # 1. Cooldown check
    if cooldown_until is not None and ts < cooldown_until:
        return False, "cooldown"

    # 2. Time gate — most critical filter
    if is_expiry:
        if h < p["expiry_entry_start_h"] or h > p["expiry_entry_end_h"]:
            return False, "time_gate_expiry"
        if h == p["expiry_entry_end_h"] and m > 25:
            return False, "time_gate_expiry_late"
    else:
        if h < p["regular_entry_start_h"] or h > p["regular_entry_end_h"]:
            return False, "time_gate_regular"
        if h == p["regular_entry_end_h"] and m > 25:
            return False, "time_gate_regular_late"

    # 3. IV regime gate — FLIPPED from original (key insight from data)
    if is_expiry:
        if iv_regime not in p["expiry_iv_regimes"]:
            return False, f"iv_gate_expiry_{iv_regime}"
    else:
        if iv_regime not in p["regular_iv_regimes"]:
            return False, f"iv_gate_regular_{iv_regime}"

    # 4. Cascade threshold
    if trig_pts < p["min_cascade_pts"]:
        return False, f"cascade_below_{p['min_cascade_pts']}"

    # 5. Option premium cap
    if buy_px > p["max_buy_px"]:
        return False, f"premium_above_{p['max_buy_px']}"

    # 6. Max trades per day (1 CALL + 1 PUT)
    if calls_today >= 1 and puts_today >= 1:
        return False, "max_daily_trades"

    return True, "ok"

# ── Trailing premium exit engine ─────────────────────────────────────────────
def compute_trailing_exit(ts, current_premium, buy_px,
                          peak_premium, trail_activated, is_expiry, p=STRATEGY_PARAMS):
    """
    Core exit logic — replaces fixed 50pt stop.
    Returns (should_exit: bool, exit_reason: str, new_peak: float, new_trail_active: bool)
    """
    h = ts.hour
    m_val = ts.minute

    # Hard floor — exit immediately if premium collapses
    if current_premium < buy_px * p["hard_floor_pct"]:
        return True, "PREMIUM_FLOOR", peak_premium, trail_activated

    # Update peak premium
    new_peak = max(peak_premium, current_premium)

    # Check if trailing should activate
    new_trail_active = trail_activated
    if not trail_activated and current_premium >= buy_px * (1 + p["trail_activate_pct"]):
        new_trail_active = True

    # Apply trailing stop if active
    if new_trail_active:
        # Tighten trail in final 10 mins on expiry day
        if is_expiry and h == p["expiry_final_hour"] and m_val >= p["expiry_final_min"]:
            trail_multiplier = p["trail_expiry_final_pct"]
        else:
            trail_multiplier = p["trail_level_pct"]

        trail_stop_level = new_peak * trail_multiplier
        if current_premium < trail_stop_level:
            return True, "TRAIL_STOP", new_peak, new_trail_active

    # EOD exit
    if h > p["eod_exit_h"] or (h == p["eod_exit_h"] and m_val >= p["eod_exit_m"]):
        return True, "EOD_EXIT", new_peak, new_trail_active

    return False, "", new_peak, new_trail_active

# ── Main backtest simulator ───────────────────────────────────────────────────
def run_backtest_for_day(signals, symbol, trade_date, expiry_flag,
                         backtest_mode="INTRADAY", p=STRATEGY_PARAMS):
    """
    v3 backtest: data-driven entry gate + trailing premium exit + dynamic lots.
    """
    if signals.empty: return []

    cfg = INDEX_CONFIG.get(symbol, {})
    contract_size = cfg.get("contract_size", 25)
    tte_days = 7 if expiry_flag == "WEEK" else 30
    is_exp = is_expiry_day(trade_date)

    trades = []
    # Per-day state
    calls_today = 0
    puts_today  = 0
    cooldown_until = None

    # Get dynamic lot size for this date
    lots = get_rolling_lot_size(symbol, trade_date, p)

    # Track open positions: dict keyed by (strike_type, option_type)
    open_positions = {}

    for _, row in signals.sort_values("timestamp").iterrows():
        ts_str  = str(row["timestamp"])
        try:    ts = pd.to_datetime(ts_str)
        except: continue

        spot        = float(row["spot_price"])
        iv_regime   = str(row["iv_regime"])
        avg_iv      = float(row["avg_iv"])
        trig_pts    = float(row["strike_cascade_pts"])
        signal      = str(row["signal"])
        strike_type = str(row["strike_type"])

        # Determine option type and direction
        if signal == "BULL" and strike_type == "ATM+1":
            option_type = "CALL"
            direction   = "BULL"
        elif signal == "BEAR" and strike_type == "ATM-1":
            option_type = "PUT"
            direction   = "BEAR"
        else:
            # Update open positions
            for key in list(open_positions.keys()):
                pos = open_positions[key]
                pos_h = pd.to_datetime(pos["entry_ts"]).hour
                curr_premium = estimate_option_premium(
                    spot, pos["strike_type"], symbol,
                    iv_pct=avg_iv, tte_days=max(tte_days-1,1),
                    option_type=pos["option_type"])
                should_exit, exit_reason, new_peak, new_trail = compute_trailing_exit(
                    ts, curr_premium, pos["buy_px"],
                    pos["peak_premium"], pos["trail_activated"], is_exp, p)
                open_positions[key]["peak_premium"]    = new_peak
                open_positions[key]["trail_activated"] = new_trail
                if should_exit:
                    t = _close_position(pos, ts_str, spot, curr_premium, exit_reason,
                                        lots, contract_size, symbol, is_exp, expiry_flag,
                                        backtest_mode, trade_date)
                    trades.append(t)
                    if exit_reason == "PREMIUM_FLOOR":
                        cooldown_until = ts + timedelta(minutes=p["cooldown_mins"])
                    del open_positions[key]
                    if pos["option_type"] == "CALL": calls_today = max(0, calls_today-1)
                    else:                            puts_today  = max(0, puts_today-1)
            continue

        # Check call/put count
        call_count = calls_today
        put_count  = puts_today

        # Estimate buy price for gate check
        buy_px_est = estimate_option_premium(
            spot, strike_type, symbol,
            iv_pct=avg_iv, tte_days=tte_days, option_type=option_type)

        # Entry gate
        allowed, gate_reason = passes_entry_gate(
            ts, iv_regime, trig_pts, buy_px_est, is_exp,
            call_count, put_count, cooldown_until, p)

        pos_key = (strike_type, option_type)

        # Update existing open positions first
        for key in list(open_positions.keys()):
            pos = open_positions[key]
            curr_premium = estimate_option_premium(
                spot, pos["strike_type"], symbol,
                iv_pct=avg_iv, tte_days=max(tte_days-1,1),
                option_type=pos["option_type"])
            should_exit, exit_reason, new_peak, new_trail = compute_trailing_exit(
                ts, curr_premium, pos["buy_px"],
                pos["peak_premium"], pos["trail_activated"], is_exp, p)
            open_positions[key]["peak_premium"]    = new_peak
            open_positions[key]["trail_activated"] = new_trail
            if should_exit:
                t = _close_position(pos, ts_str, spot, curr_premium, exit_reason,
                                    lots, contract_size, symbol, is_exp, expiry_flag,
                                    backtest_mode, trade_date)
                trades.append(t)
                if exit_reason == "PREMIUM_FLOOR":
                    cooldown_until = ts + timedelta(minutes=p["cooldown_mins"])
                del open_positions[key]
                if pos["option_type"] == "CALL": calls_today = max(0, calls_today-1)
                else:                            puts_today  = max(0, puts_today-1)

        # Open new position if allowed and not already in same instrument
        if allowed and pos_key not in open_positions:
            open_positions[pos_key] = {
                "entry_ts":        ts_str,
                "direction":       direction,
                "option_type":     option_type,
                "strike_type":     strike_type,
                "entry_spot":      spot,
                "buy_px":          buy_px_est,
                "peak_premium":    buy_px_est,
                "trail_activated": False,
                "iv_regime":       iv_regime,
                "avg_iv":          avg_iv,
                "trig_pts":        trig_pts,
                "cascade_target":  float(row["cascade_target"]),
                "cascade_stop":    float(row["cascade_stop"]),
            }
            if option_type == "CALL": calls_today += 1
            else:                     puts_today  += 1

    # Force-close any remaining positions at EOD
    last_row = signals.iloc[-1]
    last_spot = float(last_row["spot_price"])
    last_avg_iv = float(last_row["avg_iv"])
    eod_ts_str = str(last_row["timestamp"])

    for key, pos in list(open_positions.items()):
        curr_premium = estimate_option_premium(
            last_spot, pos["strike_type"], symbol,
            iv_pct=last_avg_iv, tte_days=max(tte_days-1,1),
            option_type=pos["option_type"])
        t = _close_position(pos, eod_ts_str, last_spot, curr_premium, "EOD_EXIT",
                            lots, contract_size, symbol, is_exp, expiry_flag,
                            backtest_mode, trade_date)
        trades.append(t)

    return trades

def _close_position(pos, exit_ts_str, exit_spot, sell_px, exit_reason,
                    lots, contract_size, symbol, is_exp, expiry_flag,
                    backtest_mode, trade_date):
    pts = (exit_spot - pos["entry_spot"]) if pos["direction"]=="BULL" else (pos["entry_spot"] - exit_spot)
    pnl_lot = (sell_px - pos["buy_px"]) * contract_size
    return {
        "symbol":              symbol,
        "trade_date":          trade_date,
        "trade_type":          "OPTIONS",
        "entry_time":          str(pos["entry_ts"]),
        "exit_time":           exit_ts_str,
        "direction":           pos["direction"],
        "option_type":         pos["option_type"],
        "strike_used":         pos["strike_type"],
        "entry_spot":          round(pos["entry_spot"], 2),
        "exit_spot":           round(exit_spot, 2),
        "option_buy_price":    round(pos["buy_px"], 2),
        "option_sell_price":   round(sell_px, 2),
        "peak_premium":        round(pos["peak_premium"], 2),
        "pts_captured":        round(pts, 2),
        "pnl_per_lot":         round(pnl_lot, 2),
        "lots_used":           lots,
        "total_pnl":           round(pnl_lot * lots, 2),
        "contract_size":       contract_size,
        "cascade_trigger_pts": round(pos["trig_pts"], 2),
        "cascade_target":      round(pos["cascade_target"], 2),
        "cascade_stop":        round(pos["cascade_stop"], 2),
        "exit_reason":         exit_reason,
        "iv_regime":           pos["iv_regime"],
        "signal_strength":     round(pos["trig_pts"], 2),
        "is_expiry_day":       int(is_exp),
        "expiry_flag":         expiry_flag,
        "backtest_mode":       backtest_mode,
        "trailing_activated":  int(pos["trail_activated"]),
        "cooldown_triggered":  int(exit_reason == "PREMIUM_FLOOR"),
    }

# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(trades):
    if trades.empty: return {}
    pnl   = trades["total_pnl"]
    p1    = trades["pnl_per_lot"]
    wins  = pnl[pnl > 0]
    losses= pnl[pnl <= 0]
    total = len(trades)
    hit   = len(wins)/total*100 if total else 0
    pf    = (wins.sum()/abs(losses.sum())) if losses.sum()!=0 else float("inf")
    cum   = pnl.cumsum(); dd=(cum.cummax()-cum).max()
    sh    = (pnl.mean()/pnl.std()*np.sqrt(250)) if pnl.std()>0 else 0
    neg   = pnl[pnl<0].std()
    so    = (pnl.mean()/neg*np.sqrt(250)) if neg and neg>0 else 0
    avg_w = wins.mean() if len(wins) else 0
    avg_l = losses.mean() if len(losses) else 0
    wl    = abs(avg_w/avg_l) if avg_l != 0 else float("inf")
    be_wr = abs(avg_l)/(abs(avg_l)+avg_w)*100 if (avg_l != 0 and avg_w > 0) else 100

    call_t = trades[trades["option_type"]=="CALL"]
    put_t  = trades[trades["option_type"]=="PUT"]
    exp_t  = trades[trades["is_expiry_day"]==1]
    nexp_t = trades[trades["is_expiry_day"]==0]
    trail_t= trades[trades["trailing_activated"]==1]
    floor_t= trades[trades["cooldown_triggered"]==1]

    return {
        "total_trades":    total,
        "hit_rate":        round(hit, 1),
        "total_pnl":       round(pnl.sum(), 2),
        "total_pnl_1lot":  round(p1.sum(), 2),
        "avg_win":         round(avg_w, 2),
        "avg_loss":        round(avg_l, 2),
        "wl_ratio":        round(wl, 2),
        "breakeven_wr":    round(be_wr, 1),
        "profit_factor":   round(pf, 3),
        "max_drawdown":    round(dd, 2),
        "sharpe":          round(sh, 2),
        "sortino":         round(so, 2),
        "call_trades":     len(call_t),
        "call_pnl":        round(call_t["total_pnl"].sum(), 2),
        "put_trades":      len(put_t),
        "put_pnl":         round(put_t["total_pnl"].sum(), 2),
        "expiry_trades":   len(exp_t),
        "expiry_pnl":      round(exp_t["total_pnl"].sum(), 2),
        "expiry_wr":       round(exp_t["total_pnl"].apply(lambda x: x>0).mean()*100, 1) if len(exp_t) else 0,
        "nexp_trades":     len(nexp_t),
        "nexp_pnl":        round(nexp_t["total_pnl"].sum(), 2),
        "nexp_wr":         round(nexp_t["total_pnl"].apply(lambda x: x>0).mean()*100, 1) if len(nexp_t) else 0,
        "trail_trades":    len(trail_t),
        "trail_pnl":       round(trail_t["total_pnl"].sum(), 2),
        "floor_exits":     len(floor_t),
        "eod_exits":       int((trades["exit_reason"]=="EOD_EXIT").sum()),
        "trail_exits":     int((trades["exit_reason"]=="TRAIL_STOP").sum()),
        "floor_count":     int((trades["exit_reason"]=="PREMIUM_FLOOR").sum()),
        "avg_lots":        round(trades["lots_used"].mean(), 1),
    }

# ── Charts ────────────────────────────────────────────────────────────────────
def equity_curve_chart(trades, key_suffix=""):
    t = trades.copy().reset_index(drop=True)
    t["cum"] = t["total_pnl"].cumsum()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.65,0.35],
                        subplot_titles=["Equity Curve (Cumulative Total P&L ₹)", "Per-Trade P&L"])
    colors = t["total_pnl"].apply(lambda x: "#10b981" if x>0 else "#ef4444")
    fig.add_trace(go.Scatter(x=t.index, y=t["cum"], mode="lines",
        line=dict(color="#00f5c4", width=2.5), fill="tozeroy",
        fillcolor="rgba(0,245,196,0.08)"), row=1, col=1)
    fig.add_trace(go.Bar(x=t.index, y=t["total_pnl"],
        marker_color=colors), row=2, col=1)
    fig.update_layout(template="plotly_dark", height=500, paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,10,20,0.95)", showlegend=False, margin=dict(l=0,r=0,t=30,b=0))
    return fig

def exit_breakdown_chart(trades, key_suffix=""):
    grp = trades.groupby("exit_reason")["total_pnl"].agg(["sum","count"]).reset_index()
    cm  = {"EOD_EXIT":"#10b981","TRAIL_STOP":"#00d4ff","PREMIUM_FLOOR":"#ef4444"}
    fig = go.Figure(go.Bar(
        x=grp["exit_reason"], y=grp["sum"],
        marker_color=[cm.get(r,"#8b5cf6") for r in grp["exit_reason"]],
        text=grp["count"].apply(lambda x: f"{x} trades"), textposition="outside"))
    fig.update_layout(template="plotly_dark", height=300, paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,10,20,0.95)", yaxis_title="Total P&L (₹)",
        margin=dict(l=0,r=0,t=10,b=0))
    return fig

def expiry_vs_regular_chart(trades, key_suffix=""):
    exp  = trades[trades["is_expiry_day"]==1]
    nexp = trades[trades["is_expiry_day"]==0]
    cats = ["Expiry Day", "Regular Day"]
    totals = [exp["total_pnl"].sum(), nexp["total_pnl"].sum()]
    wrs    = [
        exp["total_pnl"].apply(lambda x: x>0).mean()*100 if len(exp) else 0,
        nexp["total_pnl"].apply(lambda x: x>0).mean()*100 if len(nexp) else 0,
    ]
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Total P&L by Session","Win Rate by Session"])
    fig.add_trace(go.Bar(x=cats, y=totals,
        marker_color=["#10b981" if v>0 else "#ef4444" for v in totals],
        text=[f"₹{v:,.0f}" for v in totals], textposition="outside"), row=1, col=1)
    fig.add_trace(go.Bar(x=cats, y=wrs,
        marker_color=["#00d4ff","#a78bfa"],
        text=[f"{v:.1f}%" for v in wrs], textposition="outside"), row=1, col=2)
    fig.update_layout(template="plotly_dark", height=320, paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,10,20,0.95)", showlegend=False, margin=dict(l=0,r=0,t=30,b=0))
    return fig

def monthly_pnl_chart(trades, key_suffix=""):
    t = trades.copy()
    t["month"] = pd.to_datetime(t["trade_date"]).dt.to_period("M").astype(str)
    mo = t.groupby("month")["total_pnl"].agg(["sum","count"]).reset_index()
    fig = go.Figure(go.Bar(
        x=mo["month"], y=mo["sum"],
        marker_color=mo["sum"].apply(lambda x: "#10b981" if x>0 else "#ef4444"),
        text=mo["count"].apply(lambda x: f"{x}T"), textposition="outside"))
    fig.update_layout(template="plotly_dark", height=300, paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,10,20,0.95)", margin=dict(l=0,r=0,t=10,b=0))
    return fig

def lot_size_timeline_chart(trades, key_suffix=""):
    t = trades.copy().reset_index(drop=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t.index, y=t["lots_used"], mode="lines+markers",
        line=dict(color="#a78bfa", width=2),
        marker=dict(size=6, color=t["total_pnl"].apply(lambda x: "#10b981" if x>0 else "#ef4444"))))
    fig.update_layout(template="plotly_dark", height=280, paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,10,20,0.95)", yaxis_title="Lots",
        yaxis=dict(tickvals=[1,3,5]), margin=dict(l=0,r=0,t=10,b=0))
    return fig

# ── Results renderer ──────────────────────────────────────────────────────────
def _render_results(mode_label: str, bt_mode: str, symbol: str):
    trades = load_trades(symbol, mode=bt_mode)
    if trades.empty:
        st.info(f"No {mode_label} trades yet — run the backtest first.")
        return
    m = compute_metrics(trades)
    k = bt_mode

    # Row 1 — Core
    st.markdown("#### Core Metrics")
    cols = st.columns(5)
    for col,(label,val,cls) in zip(cols,[
        ("Total Trades",   str(m["total_trades"]),              "n"),
        ("Win Rate",       f"{m['hit_rate']}%",                 "g" if m["hit_rate"]>=40 else "r"),
        ("Total P&L",      f"Rs.{m['total_pnl']:+,.0f}",        "g" if m["total_pnl"]>0  else "r"),
        ("W:L Ratio",      f"{m['wl_ratio']:.2f}x",             "g" if m["wl_ratio"]>=1  else "r"),
        ("Profit Factor",  f"{m['profit_factor']:.3f}",         "g" if m["profit_factor"]>1 else "r"),
    ]):
        col.markdown(f'<div class="metric-card"><div class="metric-val {cls}">{val}</div>'
                     f'<div class="metric-lbl">{label}</div></div>', unsafe_allow_html=True)

    # Row 2 — Risk
    st.markdown("#### Risk & Sizing")
    cols2 = st.columns(5)
    for col,(label,val,cls) in zip(cols2,[
        ("Avg Win",        f"Rs.{m['avg_win']:+,.0f}",          "g"),
        ("Avg Loss",       f"Rs.{m['avg_loss']:+,.0f}",          "r"),
        ("Breakeven WR",   f"{m['breakeven_wr']}%",              "a" if m["breakeven_wr"]<60 else "r"),
        ("Avg Lots Used",  f"{m['avg_lots']:.1f}",               "n"),
        ("Max Drawdown",   f"Rs.{m['max_drawdown']:,.0f}",       "r"),
    ]):
        col.markdown(f'<div class="metric-card"><div class="metric-val {cls}">{val}</div>'
                     f'<div class="metric-lbl">{label}</div></div>', unsafe_allow_html=True)

    # Row 3 — Exit breakdown
    st.markdown("#### Exit Engine")
    cols3 = st.columns(5)
    for col,(label,val,cls) in zip(cols3,[
        ("EOD Exits",      str(m["eod_exits"]),                  "g"),
        ("Trail Stops",    str(m["trail_exits"]),                 "n"),
        ("Floor Exits",    str(m["floor_count"]),                 "r"),
        ("Trailing P&L",   f"Rs.{m['trail_pnl']:+,.0f}",         "g" if m["trail_pnl"]>0 else "r"),
        ("Sharpe",         f"{m['sharpe']:.2f}",                  "g" if m["sharpe"]>0.5 else "r"),
    ]):
        col.markdown(f'<div class="metric-card"><div class="metric-val {cls}">{val}</div>'
                     f'<div class="metric-lbl">{label}</div></div>', unsafe_allow_html=True)

    # Row 4 — Expiry vs Regular
    st.markdown("#### Expiry vs Regular Day")
    cols4 = st.columns(4)
    for col,(label,val,cls) in zip(cols4,[
        ("Expiry Trades",   f"{m['expiry_trades']} | {m['expiry_wr']}% WR",  "g" if m["expiry_wr"]>=40 else "n"),
        ("Expiry P&L",      f"Rs.{m['expiry_pnl']:+,.0f}",                    "g" if m["expiry_pnl"]>0 else "r"),
        ("Regular Trades",  f"{m['nexp_trades']} | {m['nexp_wr']}% WR",      "g" if m["nexp_wr"]>=30 else "n"),
        ("Regular P&L",     f"Rs.{m['nexp_pnl']:+,.0f}",                      "g" if m["nexp_pnl"]>0 else "r"),
    ]):
        col.markdown(f'<div class="metric-card"><div class="metric-val {cls}">{val}</div>'
                     f'<div class="metric-lbl">{label}</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    c1,c2 = st.columns([2,1])
    with c1: st.plotly_chart(equity_curve_chart(trades),       use_container_width=True, key=f"eq_{k}")
    with c2: st.plotly_chart(exit_breakdown_chart(trades),     use_container_width=True, key=f"ex_{k}")
    c3,c4 = st.columns(2)
    with c3: st.plotly_chart(expiry_vs_regular_chart(trades),  use_container_width=True, key=f"ev_{k}")
    with c4: st.plotly_chart(lot_size_timeline_chart(trades),  use_container_width=True, key=f"lt_{k}")
    st.markdown("#### Monthly P&L")
    st.plotly_chart(monthly_pnl_chart(trades), use_container_width=True, key=f"mo_{k}")
    st.download_button(f"📥 Download {mode_label} CSV",
        data=trades.to_csv(index=False),
        file_name=f"hedgex_v3_{symbol}_{bt_mode}.csv",
        mime="text/csv", use_container_width=True, key=f"dl_{k}")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    init_db()

    st.markdown("""<div class="bt-header">
        <div class="bt-title">HedGEX — Cascade Backtest Engine v3</div>
        <div class="bt-sub">
        Data-Driven Entry Gate &nbsp;·&nbsp; Trailing Premium Exit &nbsp;·&nbsp;
        Dynamic Lot Sizing &nbsp;·&nbsp; Expiry Calendar (Thu→Tue Sep 2025) &nbsp;·&nbsp;
        NYZTrade Analytics
        </div>
    </div>""", unsafe_allow_html=True)

    # ── Strategy summary
    p = STRATEGY_PARAMS
    st.markdown(f"""<div class="strat-box">
    <b>🎯 ENTRY GATE (data-derived from 1,050 trade analysis)</b><br>
    📅 &nbsp;Expiry day = <b>Thursday</b> (before Sep 2025) / <b>Tuesday</b> (Sep 2025 onwards)<br>
    ⏰ &nbsp;Expiry day: entry only <b>13:00–15:25</b> · IV = <b>EXPANDING or FLAT</b> · Cascade ≥{p['min_cascade_pts']} pts · Premium ≤₹{p['max_buy_px']}<br>
    ⏰ &nbsp;Regular day: entry only <b>14:00–15:25</b> · IV = <b>FLAT only</b> · Cascade ≥{p['min_cascade_pts']} pts · Premium ≤₹{p['max_buy_px']}<br>
    🚫 &nbsp;Max 2 trades/day (1 CALL + 1 PUT) · {p['cooldown_mins']}-min cooldown after floor exit<br>
    <b>🔄 EXIT ENGINE (trailing premium stop)</b><br>
    📈 &nbsp;Trail activates when premium rises {int(p['trail_activate_pct']*100)}%+ · Trail stop at {int(p['trail_level_pct']*100)}% of peak · Tightens to {int(p['trail_expiry_final_pct']*100)}% at 15:15 on expiry<br>
    🛑 &nbsp;Hard floor exit if premium falls to {int(p['hard_floor_pct']*100)}% of buy price · EOD exit at 15:25 as fallback<br>
    <b>📊 DYNAMIC LOTS</b> · Phase 1: {p['lot_phase1']} lot · Phase 2: {p['lot_phase2']} lots (3M WR≥{int(p['wr_phase2_threshold']*100)}%) · Phase 3: {p['lot_phase3']} lots (6M WR≥{int(p['wr_phase3_threshold']*100)}%) · Reset to 1 after {p['consecutive_loss_limit']} consecutive losses
    </div>""", unsafe_allow_html=True)

    # ── Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        st.markdown(f'<div class="info-box">Client: <b>1100480354</b><br>Token: <b>Hardcoded ✓</b></div>',
                    unsafe_allow_html=True)
        st.markdown("---")
        symbol       = st.selectbox("Index", list(INDEX_CONFIG.keys()), index=0)
        expiry_flag  = st.selectbox("Expiry Type", ["WEEK","MONTH"], index=0)
        expiry_code  = st.selectbox("Expiry Code", [1,2,3], index=0,
                                    format_func=lambda x:{1:"Current",2:"Next",3:"Far"}[x])
        interval     = st.selectbox("Bar Interval", ["5","15","60"], index=0,
                                    format_func=lambda x: f"{x} min")
        st.markdown("---")
        st.markdown("### 📅 Date Range")
        today   = date.today()
        d_end   = st.date_input("End Date",   value=today-timedelta(days=1))
        d_start = st.date_input("Start Date", value=today-timedelta(days=365))
        st.markdown("---")
        st.markdown("### ⚡ Strikes to Fetch")
        n_strikes  = st.slider("ATM ± N", 3, 10, 5)
        all_strikes = (["ATM"]
                       +[f"ATM+{i}" for i in range(1,n_strikes+1)]
                       +[f"ATM-{i}" for i in range(1,n_strikes+1)])
        st.caption(f"{len(all_strikes)} strikes selected")
        st.markdown("---")
        st.markdown("### 🎯 Strategy Overrides")
        st.caption("Override data-derived defaults if needed")
        override_cascade = st.slider("Min Cascade Pts", 50, 200, p["min_cascade_pts"], 10)
        override_max_px  = st.slider("Max Buy Price (₹)", 100, 500, p["max_buy_px"], 25)
        override_trail_act = st.slider("Trail Activate (%)", 20, 80, int(p["trail_activate_pct"]*100), 5)
        override_trail_lvl = st.slider("Trail Level (%)", 40, 90, int(p["trail_level_pct"]*100), 5)
        override_floor     = st.slider("Hard Floor (%)", 10, 40, int(p["hard_floor_pct"]*100), 5)

        # Apply overrides
        p_live = dict(p)
        p_live["min_cascade_pts"]    = override_cascade
        p_live["max_buy_px"]         = override_max_px
        p_live["trail_activate_pct"] = override_trail_act / 100
        p_live["trail_level_pct"]    = override_trail_lvl / 100
        p_live["hard_floor_pct"]     = override_floor / 100

        st.markdown("---")
        st.markdown("### 🗄️ Database")
        stats = db_stats()
        st.markdown(
            f'<div class="info-box">'
            f'Raw rows: <b>{stats["raw_rows"]:,}</b><br>'
            f'Days: <b>{stats["days"]}</b><br>'
            f'Signals: <b>{stats["signals"]:,}</b><br>'
            f'Trades: <b>{stats["trades"]}</b><br>'
            f'Fetch log: <b>{stats["fetch_log"]}</b>'
            f'</div>', unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("### 🗑️ Data Management")
        if st.button("Clear Trades Only", use_container_width=True, key="sb_clear_trades"):
            clear_trades(symbol); st.success("Trades cleared"); st.rerun()
        if st.button("Clear Signals Only", use_container_width=True, key="sb_clear_signals"):
            clear_signals_only(symbol); st.success("Signals cleared"); st.rerun()
        st.markdown('<div class="danger-box">⚠️ Wipes ALL data — raw chain, signals, trades, fetch log.</div>',
                    unsafe_allow_html=True)
        nuke_confirm = st.checkbox("I understand — delete everything", key="sb_nuke_confirm")
        if st.button("🔴 WIPE ALL DATA & START AFRESH", type="primary",
                     use_container_width=True, disabled=not nuke_confirm, key="sb_nuke_btn"):
            clear_all_data(); st.success("✅ All data wiped."); st.rerun()

    # ── Tabs
    (tab_fetch, tab_signals, tab_ibt, tab_cbt,
     tab_ires, tab_cres, tab_trades, tab_manage) = st.tabs([
        "📡 Fetch Data",
        "⚡ Compute Signals",
        "📈 Run Intraday BT",
        "📦 Run CNC BT",
        "📊 Intraday Results",
        "📊 CNC Results",
        "📋 Trade Log",
        "🗄️ Data Management"])

    # ── Tab 1: Fetch ──────────────────────────────────────────────────────────
    with tab_fetch:
        st.markdown("### 📡 Fetch Historical Options Chain")
        st.markdown('<div class="info-box">Raw data collection unchanged. All existing data is preserved.</div>',
                    unsafe_allow_html=True)

        ckpt = checkpoint_status()
        if ckpt:
            st.markdown(
                '<div class="warn-box">⚡ Checkpoint — '
                + str(ckpt.get("trade_date","?")) + " | "
                + str(len(ckpt.get("completed_strikes",[]))) + " strikes done. Next fetch resumes.</div>",
                unsafe_allow_html=True)
            if st.button("🗑️ Discard Checkpoint", key="discard_ckpt"):
                clear_checkpoint(); st.rerun()

        trading_dates = get_trading_dates(d_start, d_end)
        done_dates    = get_fetch_log(symbol, expiry_code, expiry_flag)
        pending       = [d for d in trading_dates if d not in done_dates]
        c1,c2,c3 = st.columns(3)
        c1.metric("Total days",  len(trading_dates))
        c2.metric("Fetched",     len(done_dates))
        c3.metric("Pending",     len(pending))
        if pending:
            est = len(pending)*len(all_strikes)*2*0.35/60
            st.markdown(f'<div class="warn-box">Estimated time: ~{est:.1f} min</div>',
                        unsafe_allow_html=True)

        if st.button(f"🚀 Fetch {len(pending)} Days", type="primary",
                     use_container_width=True, disabled=(len(pending)==0), key="fetch_btn"):
            overall=st.progress(0); day_bar=st.progress(0)
            status=st.empty(); day_status=st.empty(); log_box=st.empty()
            log_lines=[]
            for idx,trade_date in enumerate(pending):
                day_status.text(f"Day {idx+1}/{len(pending)}: {trade_date}")
                ckpt_now = checkpoint_status()
                if ckpt_now and ckpt_now.get("trade_date") != trade_date:
                    clear_checkpoint()
                try:
                    n = fetch_one_day(symbol,trade_date,all_strikes,interval,
                                      expiry_code,expiry_flag,day_bar,status)
                    log_fetch(symbol,trade_date,expiry_code,expiry_flag,"ok",n)
                    log_lines.append(f"✅ {trade_date} — {n:,} rows")
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

        if done_dates:
            st.markdown("#### ✅ Fetched Days")
            con = sqlite3.connect(DB_PATH)
            st.dataframe(pd.read_sql_query(
                "SELECT trade_date,rows_fetched,fetched_at FROM fetch_log WHERE symbol=? AND expiry_code=? AND expiry_flag=? AND status='ok' ORDER BY trade_date DESC",
                con, params=(symbol,expiry_code,expiry_flag)),
                use_container_width=True, height=300, hide_index=True)
            con.close()

    # ── Tab 2: Signals ────────────────────────────────────────────────────────
    with tab_signals:
        st.markdown("### ⚡ Compute Cascade Signals")
        st.markdown('<div class="info-box">Signal computation unchanged. Existing signals are reused. '
                    'v3 applies new entry/exit rules at backtest time, not at signal computation time.</div>',
                    unsafe_allow_html=True)
        done_dates = get_fetch_log(symbol, expiry_code, expiry_flag)
        con = sqlite3.connect(DB_PATH)
        sig_d = pd.read_sql_query(
            "SELECT DISTINCT trade_date FROM cascade_signals WHERE symbol=?",
            con, params=(symbol,))
        con.close()
        sig_dates = set(sig_d["trade_date"].tolist()) if not sig_d.empty else set()
        pending_sig = [d for d in sorted(done_dates) if d not in sig_dates]
        c1,c2,c3 = st.columns(3)
        c1.metric("Fetched days",   len(done_dates))
        c2.metric("Signals ready",  len(sig_dates))
        c3.metric("Pending",        len(pending_sig))

        if st.button(f"⚡ Compute Signals for {len(pending_sig)} Days", type="primary",
                     use_container_width=True, disabled=(len(pending_sig)==0), key="compute_sig_btn"):
            prog=st.progress(0); status=st.empty()
            for idx,trade_date in enumerate(pending_sig):
                status.text(f"Computing {trade_date} ({idx+1}/{len(pending_sig)})")
                df_day = load_raw_chain(symbol,trade_date,expiry_code,expiry_flag)
                if not df_day.empty:
                    save_signals(compute_signals_for_day(df_day,symbol,trade_date))
                prog.progress((idx+1)/len(pending_sig))
            prog.empty(); status.empty()
            st.success("✅ Signals computed"); st.rerun()

        if sig_dates:
            latest = max(sig_dates); sig_df = load_signals(symbol, latest)
            st.markdown(f"#### Preview — {latest}")
            if not sig_df.empty:
                # Show expiry status
                is_exp = is_expiry_day(latest)
                exp_label = "EXPIRY DAY ✓" if is_exp else "Regular Day"
                exp_color = "success-box" if is_exp else "info-box"
                st.markdown(f'<div class="{exp_color}">{latest} is a <b>{exp_label}</b> '
                            f'({"Thu<Sep2025/Tue≥Sep2025" if is_exp else "non-expiry"})</div>',
                            unsafe_allow_html=True)
                st.dataframe(sig_df[[
                    "timestamp","strike_type","spot_price",
                    "strike_cascade_pts","cumulative_cascade_atm3",
                    "signal","iv_regime","avg_iv",
                    "cascade_target","cascade_stop"
                ]], use_container_width=True, height=350, hide_index=True)

    # ── Helper: get signal dates ───────────────────────────────────────────────
    def _get_sig_dates():
        con = sqlite3.connect(DB_PATH)
        df  = pd.read_sql_query(
            "SELECT DISTINCT trade_date FROM cascade_signals WHERE symbol=?",
            con, params=(symbol,))
        con.close()
        return sorted(df["trade_date"].tolist()) if not df.empty else []

    def _run_bt(mode_label, bt_mode):
        all_sig_dates = _get_sig_dates()
        exp_days = sum(1 for d in all_sig_dates if is_expiry_day(d))
        reg_days = len(all_sig_dates) - exp_days

        st.markdown(
            f'<div class="info-box">'
            f'Signal days: <b>{len(all_sig_dates)}</b> &nbsp;|&nbsp; '
            f'Expiry days: <b>{exp_days}</b> &nbsp;|&nbsp; '
            f'Regular days: <b>{reg_days}</b><br>'
            f'Entry: Expiry 13-15h EXP/FLAT · Regular 14-15h FLAT · Cascade≥{p_live["min_cascade_pts"]} · Px≤₹{p_live["max_buy_px"]}<br>'
            f'Exit: Trail@{int(p_live["trail_activate_pct"]*100)}%→{int(p_live["trail_level_pct"]*100)}% peak · Floor@{int(p_live["hard_floor_pct"]*100)}% · EOD@15:25'
            f'</div>', unsafe_allow_html=True)

        btn_key = f"run_{bt_mode.lower()}_btn"
        if st.button(f"▶ Run {mode_label} Backtest", type="primary",
                     use_container_width=True, disabled=(len(all_sig_dates)==0), key=btn_key):
            clear_trades(symbol, mode=bt_mode)
            prog=st.progress(0); status=st.empty(); all_trades=[]
            for idx,td in enumerate(all_sig_dates):
                status.text(f"{mode_label}: {td} ({idx+1}/{len(all_sig_dates)}) "
                            f"{'📅 EXPIRY' if is_expiry_day(td) else '📆 Regular'}")
                sig_df = load_signals(symbol, td)
                if not sig_df.empty:
                    all_trades.extend(run_backtest_for_day(
                        sig_df, symbol, td, expiry_flag,
                        backtest_mode=bt_mode, p=p_live))
                prog.progress((idx+1)/len(all_sig_dates))
            save_trades(all_trades, mode=bt_mode)
            prog.empty(); status.empty()
            # Show quick summary
            if all_trades:
                t_df = pd.DataFrame(all_trades)
                wins = (t_df["total_pnl"] > 0).sum()
                total_pnl = t_df["total_pnl"].sum()
                pnl_color = "success-box" if total_pnl > 0 else "warn-box"
                st.markdown(
                    f'<div class="{pnl_color}">✅ Done — {len(all_trades)} trades · '
                    f'WR={wins/len(all_trades)*100:.1f}% · '
                    f'Total P&L=Rs.{total_pnl:+,.0f}</div>',
                    unsafe_allow_html=True)
            else:
                st.warning("No trades fired. Check if signals exist and date range is set correctly.")
            st.rerun()

    # ── Tab 3: Intraday BT ────────────────────────────────────────────────────
    with tab_ibt:
        st.markdown("### 📈 Intraday Backtest")
        st.markdown('<div class="info-box">Entry + EOD exit same day. The primary mode for this strategy '
                    'since all wins are EOD exits.</div>', unsafe_allow_html=True)
        _run_bt("INTRADAY", "INTRADAY")

    # ── Tab 4: CNC BT ─────────────────────────────────────────────────────────
    with tab_cbt:
        st.markdown("### 📦 CNC / Swing Backtest")
        st.markdown('<div class="info-box">Positional mode — can hold overnight. '
                    'Trailing stop and floor still active across sessions.</div>', unsafe_allow_html=True)
        _run_bt("CNC", "CNC")

    # ── Tab 5: Intraday Results ───────────────────────────────────────────────
    with tab_ires:
        st.markdown("### 📊 Intraday Results")
        _render_results("Intraday", "INTRADAY", symbol)

    # ── Tab 6: CNC Results ────────────────────────────────────────────────────
    with tab_cres:
        st.markdown("### 📊 CNC Results")
        _render_results("CNC", "CNC", symbol)

    # ── Tab 7: Trade Log ──────────────────────────────────────────────────────
    with tab_trades:
        st.markdown("### 📋 Trade Log")
        log_mode = st.radio("Filter", ["INTRADAY","CNC","Both"], horizontal=True, key="log_mode")
        mode_f = None if log_mode == "Both" else log_mode
        trades = load_trades(symbol, mode=mode_f)

        if trades.empty:
            st.info("No trades yet.")
        else:
            disp = trades[[
                "trade_date","backtest_mode","entry_time","exit_time",
                "option_type","strike_used","direction",
                "entry_spot","exit_spot",
                "option_buy_price","option_sell_price","peak_premium",
                "pts_captured","pnl_per_lot","lots_used","total_pnl",
                "cascade_trigger_pts",
                "exit_reason","iv_regime","is_expiry_day",
                "trailing_activated","cooldown_triggered"
            ]].copy().reset_index(drop=True)

            disp["entry_time"]    = pd.to_datetime(disp["entry_time"]).dt.strftime("%H:%M")
            disp["exit_time"]     = pd.to_datetime(disp["exit_time"]).dt.strftime("%H:%M")
            disp["is_expiry_day"] = disp["is_expiry_day"].map({0:"","1":"Expiry",1:"Expiry"})
            disp["trailing_activated"] = disp["trailing_activated"].map({0:"","1":"✓Trail",1:"✓Trail"})
            disp["cooldown_triggered"] = disp["cooldown_triggered"].map({0:"","1":"⏸Cool",1:"⏸Cool"})
            disp["W/L"] = disp["total_pnl"].apply(lambda x: "✅" if x > 0 else "❌")

            pnl_vals = disp["total_pnl"].values
            disp = disp.rename(columns={
                "trade_date":"Date","backtest_mode":"Mode",
                "entry_time":"Entry","exit_time":"Exit",
                "option_type":"Option","strike_used":"Strike","direction":"Dir",
                "entry_spot":"Entry Spot","exit_spot":"Exit Spot",
                "option_buy_price":"Buy Px","option_sell_price":"Sell Px",
                "peak_premium":"Peak Px",
                "pts_captured":"Spot Pts","pnl_per_lot":"PnL/Lot",
                "lots_used":"Lots","total_pnl":"Total PnL",
                "cascade_trigger_pts":"Cascade",
                "exit_reason":"Exit","iv_regime":"IV",
                "is_expiry_day":"Expiry",
                "trailing_activated":"Trail","cooldown_triggered":"Cool",
            })

            st.dataframe(
                disp,
                use_container_width=True, height=600, hide_index=True,
                column_config={
                    "Total PnL": st.column_config.NumberColumn("Total PnL (₹)", format="%.0f"),
                    "PnL/Lot":   st.column_config.NumberColumn("PnL/Lot (₹)",  format="%.0f"),
                    "Buy Px":    st.column_config.NumberColumn("Buy Px",        format="%.2f"),
                    "Sell Px":   st.column_config.NumberColumn("Sell Px",       format="%.2f"),
                    "Peak Px":   st.column_config.NumberColumn("Peak Px",       format="%.2f"),
                    "Spot Pts":  st.column_config.NumberColumn("Spot Pts",      format="%.1f"),
                    "Cascade":   st.column_config.NumberColumn("Cascade Pts",   format="%.1f"),
                })

            # Summary table
            st.markdown("#### Summary by Exit Type")
            summ = trades.groupby("exit_reason").agg(
                Count=("total_pnl","count"),
                Total=("total_pnl","sum"),
                Avg=("total_pnl","mean"),
                WR=("total_pnl", lambda x: (x>0).mean()*100),
            ).reset_index()
            summ["Total"] = summ["Total"].apply(lambda x: f"Rs.{x:+,.0f}")
            summ["Avg"]   = summ["Avg"].apply(lambda x: f"Rs.{x:+,.0f}")
            summ["WR"]    = summ["WR"].apply(lambda x: f"{x:.1f}%")
            st.dataframe(summ, use_container_width=True, hide_index=True)

    # ── Tab 8: Data Management ────────────────────────────────────────────────
    with tab_manage:
        st.markdown("### 🗄️ Data Management")
        stats = db_stats()
        cm1,cm2,cm3,cm4,cm5 = st.columns(5)
        cm1.metric("Raw Chain Rows",    f"{stats['raw_rows']:,}")
        cm2.metric("Fetch Log Entries", f"{stats['fetch_log']:,}")
        cm3.metric("Signal Rows",       f"{stats['signals']:,}")
        cm4.metric("Trade Rows",        f"{stats['trades']}")
        cm5.metric("Symbols",           stats["symbols"])

        st.markdown("---")
        st.markdown("#### Selective Clearing")
        c1,c2,c3 = st.columns(3)
        with c1:
            st.markdown("**Clear Intraday Trades**")
            if st.button("🗑️ Clear INTRADAY", use_container_width=True, key="dm_clr_intra"):
                clear_trades(symbol, mode="INTRADAY"); st.success("Done"); st.rerun()
        with c2:
            st.markdown("**Clear CNC Trades**")
            if st.button("🗑️ Clear CNC", use_container_width=True, key="dm_clr_cnc"):
                clear_trades(symbol, mode="CNC"); st.success("Done"); st.rerun()
        with c3:
            st.markdown("**Clear All Trades**")
            if st.button("🗑️ Clear All Trades", use_container_width=True, key="dm_clr_all"):
                clear_trades(); st.success("Done"); st.rerun()

        st.markdown("---")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Clear Signals Only**")
            st.markdown('<div class="warn-box">Removes signals. Raw data intact. Re-run Tab 2.</div>',
                        unsafe_allow_html=True)
            if st.button("🗑️ Clear Signals for " + symbol, use_container_width=True, key="dm_clr_sig"):
                clear_signals_only(symbol); st.success("Done"); st.rerun()
        with col_b:
            st.markdown("**Clear Fetch Log**")
            st.markdown('<div class="warn-box">Forces re-download of all days on next fetch.</div>',
                        unsafe_allow_html=True)
            if st.button("🗑️ Clear Fetch Log for " + symbol, use_container_width=True, key="dm_clr_log"):
                con = sqlite3.connect(DB_PATH)
                con.execute("DELETE FROM fetch_log WHERE symbol=?", (symbol,))
                con.commit(); con.close()
                st.success("Done"); st.rerun()

        st.markdown("---")
        st.markdown("#### ☢️ Full Reset")
        st.markdown('<div class="danger-box">Permanently deletes ALL data across ALL symbols. Cannot be undone.</div>',
                    unsafe_allow_html=True)
        nuke2 = st.checkbox("I confirm — permanently delete everything", key="dm_nuke_confirm")
        if st.button("🔴 WIPE ALL DATA & START AFRESH", type="primary",
                     use_container_width=True, disabled=not nuke2, key="dm_nuke_btn"):
            clear_all_data(); st.success("✅ All data wiped."); st.rerun()

        st.markdown("---")
        st.markdown("#### 📋 Fetch Log")
        con = sqlite3.connect(DB_PATH)
        fl = pd.read_sql_query("SELECT * FROM fetch_log ORDER BY trade_date DESC LIMIT 100", con)
        con.close()
        if not fl.empty:
            st.dataframe(fl, use_container_width=True, height=300, hide_index=True)

        st.markdown("---")
        st.markdown("#### 📅 Expiry Calendar Preview")
        st.markdown('<div class="info-box">Thu=expiry before Sep 2025 · Tue=expiry from Sep 2025</div>',
                    unsafe_allow_html=True)
        prev_dates = get_trading_dates(d_start, d_end)
        exp_preview = pd.DataFrame([{
            "date": d,
            "weekday": datetime.strptime(d,"%Y-%m-%d").strftime("%A"),
            "is_expiry": "✅ EXPIRY" if is_expiry_day(d) else "—",
            "expiry_rule": "Thursday" if datetime.strptime(d,"%Y-%m-%d").date() < EXPIRY_SHIFT_DATE else "Tuesday"
        } for d in prev_dates if is_expiry_day(d)])
        if not exp_preview.empty:
            st.dataframe(exp_preview, use_container_width=True, height=300, hide_index=True)

    st.markdown(
        '<div style="text-align:center;padding:16px;font-family:JetBrains Mono,monospace;'
        'font-size:0.68rem;color:rgba(255,255,255,0.2);">'
        'HedGEX Cascade Backtest v3 · NYZTrade Analytics · Research purposes only'
        '</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
