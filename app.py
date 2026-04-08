# ============================================================================
# HedGEX — Cascade Backtest Engine
# Powered by NYZTrade Analytics Pvt. Ltd.
# Tests: Idea 2 (Fuel/Absorption Ratio) + Idea 3 (IV Regime Gate)
# Data: Dhan Rolling Option API  |  Engine: GEX Cascade Mathematics
# ============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm
from datetime import datetime, timedelta, date
import pytz
import requests
import time
import sqlite3
import json
import os
import warnings
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from pathlib import Path

try:
    import pyotp
    PYOTP_AVAILABLE = True
except ImportError:
    PYOTP_AVAILABLE = False

warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="HedGEX Cascade Backtest",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# CONSTANTS & CONFIG
# ============================================================================
IST          = pytz.timezone('Asia/Kolkata')
DHAN_BASE    = "https://api.dhan.co"
RISK_FREE    = 0.07
DB_PATH      = "hedgex_backtest.db"

_FALLBACK_TOKEN = "PASTE_YOUR_DHAN_TOKEN_HERE"

DHAN_INDEX_SECURITY_IDS = {
    "NIFTY": 13, "BANKNIFTY": 25, "FINNIFTY": 27, "MIDCPNIFTY": 442, "SENSEX": 51,
}
BSE_FNO_SYMBOLS = {"SENSEX"}

INDEX_CONFIG = {
    "NIFTY":      {"contract_size": 25,  "strike_interval": 50,  "type": "INDEX"},
    "BANKNIFTY":  {"contract_size": 15,  "strike_interval": 100, "type": "INDEX"},
    "FINNIFTY":   {"contract_size": 40,  "strike_interval": 50,  "type": "INDEX"},
    "MIDCPNIFTY": {"contract_size": 75,  "strike_interval": 25,  "type": "INDEX"},
    "SENSEX":     {"contract_size": 10,  "strike_interval": 200, "type": "INDEX"},
}

INSTRUMENT_PARAMS = {
    "NIFTY":      {"pts_per_unit": 0.010, "strike_cap": 150},
    "BANKNIFTY":  {"pts_per_unit": 0.033, "strike_cap": 300},
    "FINNIFTY":   {"pts_per_unit": 0.050, "strike_cap": 150},
    "MIDCPNIFTY": {"pts_per_unit": 0.050, "strike_cap":  75},
    "SENSEX":     {"pts_per_unit": 0.025, "strike_cap": 500},
}

# ============================================================================
# STYLING
# ============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Space+Grotesk:wght@400;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
.bt-header {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    border: 1px solid rgba(168,85,247,0.3);
    border-radius: 16px; padding: 28px 36px; margin-bottom: 20px;
}
.bt-title {
    font-size: 2rem; font-weight: 800; letter-spacing: -0.02em;
    background: linear-gradient(135deg, #00f5c4, #00d4ff, #a78bfa);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
}
.bt-sub { font-family: 'JetBrains Mono', monospace; color: rgba(255,255,255,0.45); font-size: 0.82rem; margin-top: 4px; }
.metric-card {
    background: rgba(255,255,255,0.04); border: 1px solid rgba(168,85,247,0.2);
    border-radius: 12px; padding: 16px 20px; text-align: center;
}
.metric-val { font-size: 1.6rem; font-weight: 800; font-family: 'Space Grotesk', sans-serif; }
.metric-lbl { font-size: 0.72rem; color: rgba(255,255,255,0.45); font-family: 'JetBrains Mono', monospace; margin-top: 4px; }
.signal-bull { color: #10b981; }
.signal-bear { color: #ef4444; }
.signal-neutral { color: #94a3b8; }
.info-box {
    background: rgba(6,182,212,0.08); border-left: 3px solid #06b6d4;
    border-radius: 6px; padding: 10px 14px;
    font-family: 'JetBrains Mono', monospace; font-size: 0.80rem; line-height: 1.8;
}
.warn-box {
    background: rgba(245,158,11,0.08); border-left: 3px solid #f59e0b;
    border-radius: 6px; padding: 10px 14px;
    font-family: 'JetBrains Mono', monospace; font-size: 0.80rem; line-height: 1.8;
}
.section-label {
    font-family: 'JetBrains Mono', monospace; font-size: 0.68rem;
    color: rgba(192,132,252,0.7); text-transform: uppercase;
    letter-spacing: 0.10em; margin-bottom: 8px;
}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# TOKEN MANAGER
# ============================================================================
def _get_token() -> str:
    try:
        token = st.secrets.get("DHAN_ACCESS_TOKEN", None)
        if token:
            return token
    except Exception:
        pass
    if PYOTP_AVAILABLE:
        try:
            cid    = st.secrets.get("DHAN_CLIENT_ID", None)
            pin    = st.secrets.get("DHAN_PIN", None)
            secret = st.secrets.get("DHAN_TOTP_SECRET", None)
            if all([cid, pin, secret]):
                totp = pyotp.TOTP(secret).now()
                resp = requests.post(
                    "https://auth.dhan.co/app/generateAccessToken",
                    params={"dhanClientId": cid, "pin": pin, "totp": totp},
                    timeout=10,
                )
                if resp.status_code == 200:
                    return resp.json().get("accessToken", _FALLBACK_TOKEN)
        except Exception:
            pass
    return _FALLBACK_TOKEN

def get_headers() -> Dict:
    if "dhan_token" not in st.session_state:
        st.session_state.dhan_token = _get_token()
    return {
        "access-token": st.session_state.dhan_token,
        "client-id":    "1100480354",
        "Content-Type": "application/json",
    }

# ============================================================================
# BLACK-SCHOLES CALCULATOR
# ============================================================================
class BS:
    @staticmethod
    def d1(S, K, T, r, sigma):
        if T <= 0 or sigma <= 0: return 0.0
        return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    @staticmethod
    def d2(S, K, T, r, sigma):
        return BS.d1(S, K, T, r, sigma) - sigma * np.sqrt(T)

    @staticmethod
    def gamma(S, K, T, r, sigma):
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0: return 0.0
        try:
            return norm.pdf(BS.d1(S, K, T, r, sigma)) / (S * sigma * np.sqrt(T))
        except: return 0.0

    @staticmethod
    def vanna(S, K, T, r, sigma):
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0: return 0.0
        try:
            d1 = BS.d1(S, K, T, r, sigma)
            d2 = BS.d2(S, K, T, r, sigma)
            return -norm.pdf(d1) * d2 / sigma
        except: return 0.0

    @staticmethod
    def call_delta(S, K, T, r, sigma):
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0: return 0.0
        try: return norm.cdf(BS.d1(S, K, T, r, sigma))
        except: return 0.0

    @staticmethod
    def put_delta(S, K, T, r, sigma):
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0: return 0.0
        try: return norm.cdf(BS.d1(S, K, T, r, sigma)) - 1.0
        except: return 0.0

# ============================================================================
# DATABASE — SQLite
# ============================================================================
def init_db():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    # Raw options chain snapshots
    cur.execute("""
        CREATE TABLE IF NOT EXISTS raw_chain (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol        TEXT,
            trade_date    TEXT,
            timestamp     TEXT,
            strike_type   TEXT,
            strike        REAL,
            spot_price    REAL,
            call_oi       REAL,
            put_oi        REAL,
            call_vol      REAL,
            put_vol       REAL,
            call_iv       REAL,
            put_iv        REAL,
            call_gex      REAL,
            put_gex       REAL,
            net_gex       REAL,
            call_vanna    REAL,
            put_vanna     REAL,
            net_vanna     REAL,
            call_oi_chg   REAL,
            put_oi_chg    REAL,
            interval      TEXT,
            expiry_code   INTEGER,
            expiry_flag   TEXT,
            UNIQUE(symbol, trade_date, timestamp, strike_type, expiry_code, expiry_flag)
        )
    """)

    # Computed cascade signals per timestamp
    cur.execute("""
        CREATE TABLE IF NOT EXISTS cascade_signals (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol           TEXT,
            trade_date       TEXT,
            timestamp        TEXT,
            spot_price       REAL,
            bear_fuel_pts    REAL,
            bear_absorb_pts  REAL,
            bull_fuel_pts    REAL,
            bull_absorb_pts  REAL,
            bear_quality     REAL,
            bull_quality     REAL,
            iv_regime        TEXT,
            avg_iv           REAL,
            iv_skew          REAL,
            net_gex_total    REAL,
            signal           TEXT,
            signal_strength  REAL,
            cascade_target   REAL,
            cascade_stop     REAL,
            UNIQUE(symbol, trade_date, timestamp)
        )
    """)

    # Backtest trades
    cur.execute("""
        CREATE TABLE IF NOT EXISTS bt_trades (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol           TEXT,
            trade_date       TEXT,
            entry_time       TEXT,
            exit_time        TEXT,
            direction        TEXT,
            entry_price      REAL,
            exit_price       REAL,
            pts_captured     REAL,
            cascade_target   REAL,
            cascade_stop     REAL,
            exit_reason      TEXT,
            bear_quality     REAL,
            bull_quality     REAL,
            iv_regime        TEXT,
            signal_strength  REAL,
            is_expiry_day    INTEGER,
            expiry_flag      TEXT
        )
    """)

    # Fetch log — track which dates are done
    cur.execute("""
        CREATE TABLE IF NOT EXISTS fetch_log (
            symbol       TEXT,
            trade_date   TEXT,
            expiry_code  INTEGER,
            expiry_flag  TEXT,
            status       TEXT,
            rows_fetched INTEGER,
            fetched_at   TEXT,
            PRIMARY KEY (symbol, trade_date, expiry_code, expiry_flag)
        )
    """)

    con.commit()
    con.close()

def get_fetch_log(symbol: str, expiry_code: int, expiry_flag: str) -> set:
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
        SELECT trade_date FROM fetch_log
        WHERE symbol=? AND expiry_code=? AND expiry_flag=? AND status='ok'
    """, (symbol, expiry_code, expiry_flag))
    done = {r[0] for r in cur.fetchall()}
    con.close()
    return done

def log_fetch(symbol, trade_date, expiry_code, expiry_flag, status, rows):
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        INSERT OR REPLACE INTO fetch_log
        (symbol, trade_date, expiry_code, expiry_flag, status, rows_fetched, fetched_at)
        VALUES (?,?,?,?,?,?,?)
    """, (symbol, trade_date, expiry_code, expiry_flag, status, rows,
          datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S')))
    con.commit()
    con.close()

def save_raw_chain(rows: List[Dict]):
    if not rows: return
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.executemany("""
        INSERT OR IGNORE INTO raw_chain
        (symbol,trade_date,timestamp,strike_type,strike,spot_price,
         call_oi,put_oi,call_vol,put_vol,call_iv,put_iv,
         call_gex,put_gex,net_gex,call_vanna,put_vanna,net_vanna,
         call_oi_chg,put_oi_chg,interval,expiry_code,expiry_flag)
        VALUES
        (:symbol,:trade_date,:timestamp,:strike_type,:strike,:spot_price,
         :call_oi,:put_oi,:call_vol,:put_vol,:call_iv,:put_iv,
         :call_gex,:put_gex,:net_gex,:call_vanna,:put_vanna,:net_vanna,
         :call_oi_chg,:put_oi_chg,:interval,:expiry_code,:expiry_flag)
    """, rows)
    con.commit()
    con.close()

def load_raw_chain(symbol, date_str, expiry_code, expiry_flag) -> pd.DataFrame:
    con = sqlite3.connect(DB_PATH)
    df  = pd.read_sql_query("""
        SELECT * FROM raw_chain
        WHERE symbol=? AND trade_date=? AND expiry_code=? AND expiry_flag=?
        ORDER BY timestamp, strike
    """, con, params=(symbol, date_str, expiry_code, expiry_flag))
    con.close()
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def save_signals(rows: List[Dict]):
    if not rows: return
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.executemany("""
        INSERT OR REPLACE INTO cascade_signals
        (symbol,trade_date,timestamp,spot_price,bear_fuel_pts,bear_absorb_pts,
         bull_fuel_pts,bull_absorb_pts,bear_quality,bull_quality,
         iv_regime,avg_iv,iv_skew,net_gex_total,signal,signal_strength,
         cascade_target,cascade_stop)
        VALUES
        (:symbol,:trade_date,:timestamp,:spot_price,:bear_fuel_pts,:bear_absorb_pts,
         :bull_fuel_pts,:bull_absorb_pts,:bear_quality,:bull_quality,
         :iv_regime,:avg_iv,:iv_skew,:net_gex_total,:signal,:signal_strength,
         :cascade_target,:cascade_stop)
    """, rows)
    con.commit()
    con.close()

def load_signals(symbol, date_str) -> pd.DataFrame:
    con = sqlite3.connect(DB_PATH)
    df  = pd.read_sql_query("""
        SELECT * FROM cascade_signals
        WHERE symbol=? AND trade_date=?
        ORDER BY timestamp
    """, con, params=(symbol, date_str))
    con.close()
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def save_trades(rows: List[Dict]):
    if not rows: return
    con = sqlite3.connect(DB_PATH)
    con.executemany("""
        INSERT INTO bt_trades
        (symbol,trade_date,entry_time,exit_time,direction,entry_price,exit_price,
         pts_captured,cascade_target,cascade_stop,exit_reason,
         bear_quality,bull_quality,iv_regime,signal_strength,is_expiry_day,expiry_flag)
        VALUES
        (:symbol,:trade_date,:entry_time,:exit_time,:direction,:entry_price,:exit_price,
         :pts_captured,:cascade_target,:cascade_stop,:exit_reason,
         :bear_quality,:bull_quality,:iv_regime,:signal_strength,:is_expiry_day,:expiry_flag)
    """, rows)
    con.commit()
    con.close()

def load_trades(symbol=None) -> pd.DataFrame:
    con = sqlite3.connect(DB_PATH)
    if symbol:
        df = pd.read_sql_query(
            "SELECT * FROM bt_trades WHERE symbol=? ORDER BY trade_date, entry_time",
            con, params=(symbol,))
    else:
        df = pd.read_sql_query(
            "SELECT * FROM bt_trades ORDER BY trade_date, entry_time", con)
    con.close()
    return df

def clear_trades(symbol=None):
    con = sqlite3.connect(DB_PATH)
    if symbol:
        con.execute("DELETE FROM bt_trades WHERE symbol=?", (symbol,))
    else:
        con.execute("DELETE FROM bt_trades")
    con.commit()
    con.close()

def db_stats() -> Dict:
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("SELECT COUNT(*) FROM raw_chain")
    raw_rows = cur.fetchone()[0]
    cur.execute("SELECT COUNT(DISTINCT trade_date), COUNT(DISTINCT symbol) FROM raw_chain")
    r = cur.fetchone()
    days, syms = r
    cur.execute("SELECT COUNT(*) FROM bt_trades")
    trades = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM cascade_signals")
    sigs = cur.fetchone()[0]
    con.close()
    return {"raw_rows": raw_rows, "days": days, "symbols": syms,
            "trades": trades, "signals": sigs}

# ============================================================================
# DHAN API — DATA FETCHER
# ============================================================================
def fetch_rolling_option(symbol: str, from_date: str, to_date: str,
                         strike_type: str, option_type: str,
                         interval: str, expiry_code: int, expiry_flag: str) -> Optional[Dict]:
    sec_id      = DHAN_INDEX_SECURITY_IDS.get(symbol)
    if sec_id is None: return None
    exchange    = "BSE_FNO" if symbol in BSE_FNO_SYMBOLS else "NSE_FNO"
    payload     = {
        "exchangeSegment": exchange,
        "interval":        interval,
        "securityId":      sec_id,
        "instrument":      "OPTIDX",
        "expiryFlag":      expiry_flag,
        "expiryCode":      expiry_code,
        "strike":          strike_type,
        "drvOptionType":   option_type,
        "requiredData":    ["open","high","low","close","volume","oi","iv","strike","spot"],
        "fromDate":        from_date,
        "toDate":          to_date,
    }
    try:
        resp = requests.post(
            f"{DHAN_BASE}/charts/rollingoption",
            headers=get_headers(), json=payload, timeout=30,
        )
        if resp.status_code == 200:
            return resp.json().get("data", {})
    except Exception as e:
        st.warning(f"API error {symbol} {strike_type} {option_type}: {e}")
    return None

# ============================================================================
# CHECKPOINT — mid-session resume at strike level
# ============================================================================
CKPT_PATH = "hedgex_checkpoint.json"

def save_checkpoint(symbol: str, trade_date: str, expiry_code: int,
                    expiry_flag: str, completed_strikes: List[str],
                    partial_rows: List[Dict]) -> None:
    """Persist progress so a crashed session can resume."""
    ckpt = {
        "symbol":            symbol,
        "trade_date":        trade_date,
        "expiry_code":       expiry_code,
        "expiry_flag":       expiry_flag,
        "completed_strikes": completed_strikes,
        "partial_rows":      partial_rows,
        "saved_at":          datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(CKPT_PATH, "w") as f:
        json.dump(ckpt, f)

def load_checkpoint(symbol: str, trade_date: str,
                    expiry_code: int, expiry_flag: str) -> Tuple[List[str], List[Dict]]:
    """
    Returns (completed_strikes, partial_rows) if a matching checkpoint exists,
    else ([], []).
    """
    if not os.path.exists(CKPT_PATH):
        return [], []
    try:
        with open(CKPT_PATH) as f:
            ckpt = json.load(f)
        if (ckpt["symbol"]      == symbol
                and ckpt["trade_date"]   == trade_date
                and ckpt["expiry_code"]  == expiry_code
                and ckpt["expiry_flag"]  == expiry_flag):
            return ckpt["completed_strikes"], ckpt["partial_rows"]
    except Exception:
        pass
    return [], []

def clear_checkpoint() -> None:
    if os.path.exists(CKPT_PATH):
        os.remove(CKPT_PATH)

def checkpoint_status() -> Optional[Dict]:
    if not os.path.exists(CKPT_PATH):
        return None
    try:
        with open(CKPT_PATH) as f:
            return json.load(f)
    except Exception:
        return None


def fetch_one_day(symbol: str, trade_date: str, strikes: List[str],
                  interval: str, expiry_code: int, expiry_flag: str,
                  progress_bar=None, status_text=None) -> int:
    """
    Fetch one trading day — CALL + PUT per strike, process rows, save to DB.
    Uses local dict (no function attributes) — thread-safe.
    Checkpoints after every strike so crashes can resume mid-day.
    """
    cfg           = INDEX_CONFIG.get(symbol, {})
    contract_size = cfg.get("contract_size", 25)
    scaling       = 1e9
    tte           = 7/365 if expiry_flag == "WEEK" else 30/365
    target_dt     = datetime.strptime(trade_date, "%Y-%m-%d").date()
    from_date     = (target_dt - timedelta(days=1)).strftime("%Y-%m-%d")
    to_date       = (target_dt + timedelta(days=1)).strftime("%Y-%m-%d")

    # ── Resume from checkpoint if available ──────────────────────────────────
    completed_strikes, all_rows = load_checkpoint(
        symbol, trade_date, expiry_code, expiry_flag)
    remaining_strikes = [s for s in strikes if s not in completed_strikes]

    total = len(strikes) * 2
    done  = len(completed_strikes) * 2   # already done

    # ── Per-strike: fetch CALL + PUT, pair immediately, process rows ─────────
    for stype in remaining_strikes:
        # Use plain local dict — no function attribute, fully thread-safe
        strike_data: Dict[str, Dict] = {}

        for otype in ["CALL", "PUT"]:
            if status_text:
                status_text.text(
                    f"Fetching {symbol} {stype} {otype} | {trade_date} "
                    f"({len(completed_strikes)+1}/{len(strikes)})"
                )
            data = fetch_rolling_option(
                symbol, from_date, to_date,
                stype, otype, interval, expiry_code, expiry_flag,
            )
            done += 1
            if progress_bar:
                progress_bar.progress(min(done / total, 1.0))
            time.sleep(0.3)

            if data:
                strike_data[otype] = data

        # ── Pair CALL + PUT for this strike ───────────────────────────────────
        ce = strike_data.get("CALL", {})
        pe = strike_data.get("PUT",  {})

        if ce:
            ts_list = ce.get("timestamp", [])
            n_ts    = len(ts_list)
            for i, ts in enumerate(ts_list):
                try:
                    dt_ist = datetime.fromtimestamp(ts, tz=pytz.UTC).astimezone(IST)
                    if dt_ist.date() != target_dt:
                        continue

                    def _safe(src, key, default, idx):
                        arr = src.get(key, [])
                        return arr[idx] if idx < len(arr) else default

                    spot   = _safe(ce, "spot",   0,  i) or 0
                    strike = _safe(ce, "strike", 0,  i) or 0
                    if spot == 0 or strike == 0:
                        continue

                    c_oi  = _safe(ce, "oi",     0,  i) or 0
                    p_oi  = _safe(pe, "oi",     0,  i) or 0  if pe else 0
                    c_vol = _safe(ce, "volume", 0,  i) or 0
                    p_vol = _safe(pe, "volume", 0,  i) or 0  if pe else 0
                    c_iv  = _safe(ce, "iv",    15,  i) or 15
                    p_iv  = _safe(pe, "iv",    15,  i) or 15 if pe else 15

                    civ = c_iv / 100 if c_iv > 1 else float(c_iv)
                    piv = p_iv / 100 if p_iv > 1 else float(p_iv)
                    civ = max(civ, 0.01)
                    piv = max(piv, 0.01)

                    cg = BS.gamma(spot, strike, tte, RISK_FREE, civ)
                    pg = BS.gamma(spot, strike, tte, RISK_FREE, piv)
                    cv = BS.vanna(spot, strike, tte, RISK_FREE, civ)
                    pv = BS.vanna(spot, strike, tte, RISK_FREE, piv)

                    all_rows.append({
                        "symbol":      symbol,
                        "trade_date":  trade_date,
                        "timestamp":   dt_ist.strftime("%Y-%m-%d %H:%M:%S"),
                        "strike_type": stype,
                        "strike":      float(strike),
                        "spot_price":  float(spot),
                        "call_oi":     float(c_oi),
                        "put_oi":      float(p_oi),
                        "call_vol":    float(c_vol),
                        "put_vol":     float(p_vol),
                        "call_iv":     float(c_iv),
                        "put_iv":      float(p_iv),
                        "call_gex":    float((c_oi * cg * spot**2 * contract_size) / scaling),
                        "put_gex":     float(-(p_oi * pg * spot**2 * contract_size) / scaling),
                        "net_gex":     float((c_oi * cg - p_oi * pg) * spot**2 * contract_size / scaling),
                        "call_vanna":  float(c_oi * cv * spot * contract_size / scaling),
                        "put_vanna":   float(p_oi * pv * spot * contract_size / scaling),
                        "net_vanna":   float((c_oi * cv + p_oi * pv) * spot * contract_size / scaling),
                        "call_oi_chg": 0.0,
                        "put_oi_chg":  0.0,
                        "interval":    interval,
                        "expiry_code": expiry_code,
                        "expiry_flag": expiry_flag,
                    })
                except Exception:
                    continue

        # ── Mark this strike done, checkpoint ─────────────────────────────────
        completed_strikes.append(stype)
        save_checkpoint(symbol, trade_date, expiry_code, expiry_flag,
                        completed_strikes, all_rows)

    # ── Compute OI change per strike across timestamps ────────────────────────
    if all_rows:
        df = pd.DataFrame(all_rows).sort_values(["strike", "timestamp"])
        for st_val in df["strike"].unique():
            mask = df["strike"] == st_val
            df.loc[mask, "call_oi_chg"] = df.loc[mask, "call_oi"].diff().fillna(0)
            df.loc[mask, "put_oi_chg"]  = df.loc[mask, "put_oi"].diff().fillna(0)
        all_rows = df.to_dict("records")

    save_raw_chain(all_rows)
    clear_checkpoint()        # day complete — remove checkpoint
    return len(all_rows)

# ============================================================================
# TRADING DATES HELPER
# ============================================================================
def get_trading_dates(start: date, end: date) -> List[str]:
    """Returns weekdays (Mon–Fri) between start and end as YYYY-MM-DD strings.
    Excludes obvious non-trading days — NSE holiday list not embedded,
    so weekends only filtered here. User can skip bad dates manually."""
    dates = []
    cur   = start
    while cur <= end:
        if cur.weekday() < 5:  # Mon–Fri
            dates.append(cur.strftime("%Y-%m-%d"))
        cur += timedelta(days=1)
    return dates

# ============================================================================
# CASCADE SIGNAL ENGINE
# ============================================================================
def compute_iv_regime(df_ts: pd.DataFrame) -> Tuple[str, float, float]:
    """
    Compute IV regime from a snapshot DataFrame (all strikes at one timestamp).
    Returns: regime ('EXPANDING'/'COMPRESSING'/'FLAT'), avg_iv, iv_skew
    """
    if df_ts.empty:
        return "FLAT", 15.0, 0.0
    call_iv = df_ts["call_iv"].mean()
    put_iv  = df_ts["put_iv"].mean()
    avg_iv  = (call_iv + put_iv) / 2
    iv_skew = put_iv - call_iv
    # Regime determined by comparing current vs previous timestamps
    # For single-timestamp, return FLAT — caller handles trend
    return "FLAT", avg_iv, iv_skew

def compute_iv_regime_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute IV regime per timestamp across the full day.
    Returns DataFrame: timestamp, avg_iv, iv_skew, iv_regime
    """
    rows = []
    for ts, grp in df.groupby("timestamp", sort=True):
        call_iv = grp["call_iv"].mean()
        put_iv  = grp["put_iv"].mean()
        avg_iv  = (call_iv + put_iv) / 2
        iv_skew = put_iv - call_iv
        rows.append({"timestamp": ts, "avg_iv": avg_iv, "iv_skew": iv_skew})
    iv_df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    if len(iv_df) < 3:
        iv_df["iv_regime"] = "FLAT"
        return iv_df
    iv_df["iv_ma"]     = iv_df["avg_iv"].rolling(3, min_periods=1).mean()
    iv_df["iv_change"] = iv_df["avg_iv"].diff().fillna(0)
    threshold          = iv_df["avg_iv"].std() * 0.15
    iv_df["iv_regime"] = iv_df["iv_change"].apply(
        lambda x: "EXPANDING" if x > threshold else ("COMPRESSING" if x < -threshold else "FLAT")
    )
    return iv_df

def compute_cascade_for_snapshot(df_ts: pd.DataFrame, spot: float,
                                 symbol: str, iv_regime: str) -> Dict:
    """
    Compute cascade signal for one timestamp snapshot.
    Returns dict with fuel/absorption pts and signal.
    """
    params         = INSTRUMENT_PARAMS.get(symbol, INSTRUMENT_PARAMS["NIFTY"])
    pts_per_unit   = params["pts_per_unit"]
    strike_cap_pts = params["strike_cap"]

    # VANNA absorption coefficients by role × regime
    VANNA_ADJ = {
        "SUPPORT_FLOOR":       {"COMPRESSING": 0.60, "FLAT": 0.35, "EXPANDING": -0.20},
        "TRAP_DOOR":           {"EXPANDING":  -0.30, "FLAT": 0.10, "COMPRESSING": 0.15},
        "VACUUM_ZONE":         {"EXPANDING":   0.50, "FLAT": 0.20, "COMPRESSING": 0.10},
        "RESISTANCE_CEILING":  {"EXPANDING":  -0.20, "FLAT": 0.00, "COMPRESSING": 0.15},
    }

    # VANNA zones from snapshot
    vanna_zone_map = {}
    strikes_sorted = sorted(df_ts["strike"].unique())
    if len(strikes_sorted) >= 2:
        vanna_series = []
        for s in strikes_sorted:
            row = df_ts[df_ts["strike"] == s].iloc[0] if len(df_ts[df_ts["strike"] == s]) > 0 else None
            if row is not None:
                vanna_series.append((s, row["net_vanna"]))
        for j in range(1, len(vanna_series)):
            s_prev, v_prev = vanna_series[j-1]
            s_curr, v_curr = vanna_series[j]
            if v_prev * v_curr < 0:  # sign flip
                role = None
                if s_prev < spot and v_prev < 0 and v_curr > 0:
                    role = "SUPPORT_FLOOR"
                elif s_prev < spot and v_prev > 0 and v_curr < 0:
                    role = "TRAP_DOOR"
                elif s_curr > spot and v_prev < 0 and v_curr > 0:
                    role = "VACUUM_ZONE"
                elif s_curr > spot and v_prev > 0 and v_curr < 0:
                    role = "RESISTANCE_CEILING"
                if role:
                    vanna_zone_map[s_curr] = role

    bear_fuel    = 0.0
    bear_absorb  = 0.0
    bull_fuel    = 0.0
    bull_absorb  = 0.0
    bear_target  = spot
    bull_target  = spot

    for _, row in df_ts.iterrows():
        s       = row["strike"]
        gex     = row["net_gex"]
        raw_pts = min(abs(gex) * pts_per_unit, strike_cap_pts)

        # VANNA adjustment
        role     = vanna_zone_map.get(s, None)
        adj_pct  = 0.0
        if role and role in VANNA_ADJ:
            adj_pct = VANNA_ADJ[role].get(iv_regime, 0.0)
        adj_pts  = raw_pts * (1 + adj_pct)

        if s < spot:  # Bear cascade territory
            if gex < 0:
                bear_fuel   += adj_pts
                bear_target  = min(bear_target, s)
            else:
                bear_absorb += adj_pts
        else:          # Bull cascade territory
            if gex < 0:
                bull_fuel   += adj_pts
                bull_target  = max(bull_target, s)
            else:
                bull_absorb += adj_pts

    bear_quality = bear_fuel / max(bear_absorb, 1.0)
    bull_quality = bull_fuel / max(bull_absorb, 1.0)
    net_realised_bear = max(0, bear_fuel - bear_absorb * 0.5)
    net_realised_bull = max(0, bull_fuel - bull_absorb * 0.5)

    # Signal logic
    if bear_quality >= bull_quality:
        signal   = "BEAR"
        strength = bear_quality
        target   = spot - net_realised_bear
        stop     = spot + (bear_absorb * pts_per_unit * 0.5)
    else:
        signal   = "BULL"
        strength = bull_quality
        target   = spot + net_realised_bull
        stop     = spot - (bull_absorb * pts_per_unit * 0.5)

    if max(bear_quality, bull_quality) < 0.5:
        signal = "NONE"

    return {
        "bear_fuel_pts":   round(bear_fuel,   2),
        "bear_absorb_pts": round(bear_absorb, 2),
        "bull_fuel_pts":   round(bull_fuel,   2),
        "bull_absorb_pts": round(bull_absorb, 2),
        "bear_quality":    round(bear_quality, 4),
        "bull_quality":    round(bull_quality, 4),
        "net_gex_total":   round(df_ts["net_gex"].sum(), 4),
        "signal":          signal,
        "signal_strength": round(strength, 4),
        "cascade_target":  round(target, 2),
        "cascade_stop":    round(stop, 2),
    }

def compute_signals_for_day(df_day: pd.DataFrame, symbol: str,
                             trade_date: str) -> List[Dict]:
    """Run cascade engine across all timestamps of one trading day."""
    iv_df  = compute_iv_regime_series(df_day)
    rows   = []
    timestamps = sorted(df_day["timestamp"].unique())
    for ts in timestamps:
        df_ts   = df_day[df_day["timestamp"] == ts].copy()
        spot    = df_ts["spot_price"].mean()
        iv_row  = iv_df[iv_df["timestamp"] == ts]
        if iv_row.empty:
            iv_regime = "FLAT"; avg_iv = 15.0; iv_skew = 0.0
        else:
            iv_regime = str(iv_row.iloc[0]["iv_regime"])
            avg_iv    = float(iv_row.iloc[0]["avg_iv"])
            iv_skew   = float(iv_row.iloc[0]["iv_skew"])
        cascade = compute_cascade_for_snapshot(df_ts, spot, symbol, iv_regime)
        rows.append({
            "symbol":       symbol,
            "trade_date":   trade_date,
            "timestamp":    ts.strftime("%Y-%m-%d %H:%M:%S") if hasattr(ts, "strftime") else str(ts),
            "spot_price":   round(spot, 2),
            "iv_regime":    iv_regime,
            "avg_iv":       round(avg_iv, 2),
            "iv_skew":      round(iv_skew, 4),
            **cascade,
        })
    return rows

# ============================================================================
# BACKTEST SIMULATOR
# ============================================================================
def run_backtest_for_day(signals: pd.DataFrame, symbol: str, trade_date: str,
                         expiry_flag: str, min_quality: float,
                         require_iv_expanding: bool) -> List[Dict]:
    """
    Idea 2 + 3: Trade when fuel/absorption quality > min_quality,
    optionally filtered by IV EXPANDING regime.
    Entry = bar open (approximated as spot_price at signal bar).
    Exit = when target / stop hit, or EOD.
    One trade at a time (no pyramiding).
    """
    if signals.empty: return []
    trades     = []
    in_trade   = False
    entry_ts   = None; entry_px  = None; direction = None
    tgt        = None; stop      = None
    sig_str    = None; bq = None;  buq = None; iv_reg = None

    expiry_day_weekday = 3 if expiry_flag == "WEEK" else -1  # Thu = expiry for weekly
    try:
        dt = datetime.strptime(trade_date, "%Y-%m-%d")
        is_expiry = (dt.weekday() == expiry_day_weekday)
    except Exception:
        is_expiry = False

    for _, row in signals.iterrows():
        ts      = row["timestamp"]
        spot    = row["spot_price"]
        signal  = row["signal"]
        quality = row["signal_strength"]
        regime  = row["iv_regime"]

        if in_trade:
            # Check exit conditions
            if direction == "BEAR":
                if spot <= tgt:
                    exit_reason = "TARGET_HIT"
                    pts = entry_px - spot
                    trades.append(_make_trade(symbol, trade_date, entry_ts, ts,
                        direction, entry_px, spot, pts, tgt, stop, exit_reason,
                        bq, buq, iv_reg, sig_str, is_expiry, expiry_flag))
                    in_trade = False; continue
                elif spot >= stop:
                    exit_reason = "STOP_HIT"
                    pts = entry_px - spot
                    trades.append(_make_trade(symbol, trade_date, entry_ts, ts,
                        direction, entry_px, spot, pts, tgt, stop, exit_reason,
                        bq, buq, iv_reg, sig_str, is_expiry, expiry_flag))
                    in_trade = False; continue
            else:  # BULL
                if spot >= tgt:
                    exit_reason = "TARGET_HIT"
                    pts = spot - entry_px
                    trades.append(_make_trade(symbol, trade_date, entry_ts, ts,
                        direction, entry_px, spot, pts, tgt, stop, exit_reason,
                        bq, buq, iv_reg, sig_str, is_expiry, expiry_flag))
                    in_trade = False; continue
                elif spot <= stop:
                    exit_reason = "STOP_HIT"
                    pts = spot - entry_px
                    trades.append(_make_trade(symbol, trade_date, entry_ts, ts,
                        direction, entry_px, spot, pts, tgt, stop, exit_reason,
                        bq, buq, iv_reg, sig_str, is_expiry, expiry_flag))
                    in_trade = False; continue

        if not in_trade and signal != "NONE":
            iv_ok = (not require_iv_expanding) or (regime == "EXPANDING")
            if quality >= min_quality and iv_ok:
                in_trade  = True
                direction = signal
                entry_ts  = ts
                entry_px  = spot
                tgt       = row["cascade_target"]
                stop      = row["cascade_stop"]
                bq        = row["bear_quality"]
                buq       = row["bull_quality"]
                iv_reg    = regime
                sig_str   = quality

    # EOD exit — close any open trade at last price
    if in_trade and len(signals) > 0:
        last = signals.iloc[-1]
        last_px = last["spot_price"]
        pts = (entry_px - last_px) if direction == "BEAR" else (last_px - entry_px)
        trades.append(_make_trade(symbol, trade_date, entry_ts, last["timestamp"],
            direction, entry_px, last_px, pts, tgt, stop, "EOD_EXIT",
            bq, buq, iv_reg, sig_str, is_expiry, expiry_flag))

    return trades

def _make_trade(symbol, trade_date, entry_ts, exit_ts, direction,
                entry_px, exit_px, pts, tgt, stop, exit_reason,
                bq, buq, iv_reg, sig_str, is_expiry, expiry_flag) -> Dict:
    return {
        "symbol":        symbol,
        "trade_date":    trade_date,
        "entry_time":    str(entry_ts),
        "exit_time":     str(exit_ts),
        "direction":     direction,
        "entry_price":   round(entry_px, 2),
        "exit_price":    round(exit_px, 2),
        "pts_captured":  round(pts, 2),
        "cascade_target":round(tgt, 2),
        "cascade_stop":  round(stop, 2),
        "exit_reason":   exit_reason,
        "bear_quality":  round(bq, 4),
        "bull_quality":  round(buq, 4),
        "iv_regime":     iv_reg,
        "signal_strength": round(sig_str, 4),
        "is_expiry_day": int(is_expiry),
        "expiry_flag":   expiry_flag,
    }

# ============================================================================
# PERFORMANCE ANALYTICS
# ============================================================================
def compute_metrics(trades: pd.DataFrame) -> Dict:
    if trades.empty:
        return {}
    pts        = trades["pts_captured"]
    wins       = pts[pts > 0]
    losses     = pts[pts <= 0]
    total      = len(trades)
    hit_rate   = len(wins) / total * 100 if total > 0 else 0
    avg_win    = wins.mean()  if len(wins)   > 0 else 0
    avg_loss   = losses.mean() if len(losses) > 0 else 0
    profit_fac = (wins.sum() / abs(losses.sum())) if losses.sum() != 0 else float("inf")
    cumpts     = pts.cumsum()
    drawdown   = (cumpts.cummax() - cumpts)
    max_dd     = drawdown.max()
    # Sharpe (assuming ~52 weeks, 5 trades/week)
    sharpe     = (pts.mean() / pts.std() * np.sqrt(250)) if pts.std() > 0 else 0
    sortino_neg = pts[pts < 0].std()
    sortino    = (pts.mean() / sortino_neg * np.sqrt(250)) if sortino_neg and sortino_neg > 0 else 0

    # Cascade accuracy: did actual move >= cascade target move?
    trades2 = trades.copy()
    trades2["target_move"] = abs(trades2["cascade_target"] - trades2["entry_price"])
    trades2["actual_move"] = abs(trades2["pts_captured"])
    trades2["cascade_hit"] = trades2["actual_move"] >= trades2["target_move"] * 0.7
    cascade_acc = trades2["cascade_hit"].mean() * 100

    return {
        "total_trades":    total,
        "hit_rate":        round(hit_rate, 1),
        "total_pts":       round(pts.sum(), 1),
        "avg_win":         round(avg_win, 1),
        "avg_loss":        round(avg_loss, 1),
        "profit_factor":   round(profit_fac, 2),
        "max_drawdown":    round(max_dd, 1),
        "sharpe":          round(sharpe, 2),
        "sortino":         round(sortino, 2),
        "cascade_accuracy":round(cascade_acc, 1),
        "expiry_trades":   int(trades["is_expiry_day"].sum()),
        "expiry_pts":      round(trades[trades["is_expiry_day"]==1]["pts_captured"].sum(), 1),
        "non_expiry_pts":  round(trades[trades["is_expiry_day"]==0]["pts_captured"].sum(), 1),
    }

# ============================================================================
# CHARTS
# ============================================================================
def equity_curve_chart(trades: pd.DataFrame) -> go.Figure:
    if trades.empty: return go.Figure()
    trades = trades.copy().reset_index(drop=True)
    trades["cumpts"] = trades["pts_captured"].cumsum()
    trades["color"]  = trades["pts_captured"].apply(lambda x: "#10b981" if x > 0 else "#ef4444")

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.65, 0.35],
                        subplot_titles=["Equity Curve (Cumulative Pts)", "Per-Trade P&L"])
    fig.add_trace(go.Scatter(
        x=trades.index, y=trades["cumpts"],
        mode="lines", name="Equity",
        line=dict(color="#00f5c4", width=2.5),
        fill="tozeroy", fillcolor="rgba(0,245,196,0.08)",
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=trades.index, y=trades["pts_captured"],
        marker_color=trades["color"], name="Trade P&L",
    ), row=2, col=1)
    fig.update_layout(
        template="plotly_dark", height=500,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,10,20,0.95)",
        showlegend=False,
        margin=dict(l=0, r=0, t=30, b=0),
    )
    return fig

def quality_vs_pts_chart(trades: pd.DataFrame) -> go.Figure:
    if trades.empty: return go.Figure()
    trades = trades.copy()
    colors = trades["pts_captured"].apply(lambda x: "#10b981" if x > 0 else "#ef4444")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=trades["signal_strength"], y=trades["pts_captured"],
        mode="markers",
        marker=dict(color=colors, size=8, opacity=0.75,
                    line=dict(color="rgba(255,255,255,0.2)", width=1)),
        text=trades["trade_date"] + " " + trades["direction"],
        hovertemplate="<b>%{text}</b><br>Quality: %{x:.2f}<br>P&L: %{y:.1f} pts<extra></extra>",
    ))
    fig.update_layout(
        template="plotly_dark", height=380,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,10,20,0.95)",
        xaxis_title="Signal Quality (Fuel/Absorption Ratio)",
        yaxis_title="Pts Captured",
        margin=dict(l=0, r=0, t=10, b=0),
    )
    return fig

def iv_regime_breakdown_chart(trades: pd.DataFrame) -> go.Figure:
    if trades.empty: return go.Figure()
    grp = trades.groupby("iv_regime")["pts_captured"].agg(["sum","count","mean"]).reset_index()
    grp.columns = ["iv_regime", "total_pts", "trades", "avg_pts"]
    colors = {"EXPANDING": "#ef4444", "COMPRESSING": "#10b981", "FLAT": "#94a3b8"}
    fig = go.Figure(go.Bar(
        x=grp["iv_regime"], y=grp["total_pts"],
        marker_color=[colors.get(r, "#8b5cf6") for r in grp["iv_regime"]],
        text=grp.apply(lambda r: f"{r['trades']} trades<br>avg {r['avg_pts']:.1f}pts", axis=1),
        textposition="outside",
    ))
    fig.update_layout(
        template="plotly_dark", height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,10,20,0.95)",
        yaxis_title="Total Pts", xaxis_title="IV Regime",
        margin=dict(l=0, r=0, t=10, b=0),
    )
    return fig

def cascade_accuracy_chart(trades: pd.DataFrame) -> go.Figure:
    if trades.empty: return go.Figure()
    t = trades.copy()
    t["target_move"] = abs(t["cascade_target"] - t["entry_price"])
    t["actual_move"] = abs(t["pts_captured"])
    t["accuracy_pct"] = (t["actual_move"] / t["target_move"].replace(0, np.nan) * 100).fillna(0).clip(0, 200)
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=t["accuracy_pct"], nbinsx=20,
        marker_color="#a78bfa", opacity=0.8,
        name="Cascade Accuracy %",
    ))
    fig.add_vline(x=70, line_dash="dash", line_color="#f59e0b",
                  annotation_text="70% threshold", annotation_position="top right")
    fig.update_layout(
        template="plotly_dark", height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,10,20,0.95)",
        xaxis_title="Actual Move as % of Cascade Target",
        yaxis_title="Frequency",
        margin=dict(l=0, r=0, t=10, b=0),
    )
    return fig

def exit_reason_chart(trades: pd.DataFrame) -> go.Figure:
    if trades.empty: return go.Figure()
    grp = trades.groupby("exit_reason")["pts_captured"].agg(["sum","count"]).reset_index()
    colors_map = {"TARGET_HIT": "#10b981", "STOP_HIT": "#ef4444", "EOD_EXIT": "#f59e0b"}
    fig = go.Figure(go.Bar(
        x=grp["exit_reason"], y=grp["sum"],
        marker_color=[colors_map.get(r, "#8b5cf6") for r in grp["exit_reason"]],
        text=grp["count"].apply(lambda x: f"{x} trades"),
        textposition="outside",
    ))
    fig.update_layout(
        template="plotly_dark", height=300,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,10,20,0.95)",
        yaxis_title="Total Pts", xaxis_title="Exit Reason",
        margin=dict(l=0, r=0, t=10, b=0),
    )
    return fig

# ============================================================================
# MAIN APP
# ============================================================================
def main():
    init_db()

    # Header
    st.markdown("""
    <div class="bt-header">
        <div class="bt-title">HedGEX — Cascade Backtest Engine</div>
        <div class="bt-sub">Idea 2: Fuel/Absorption Ratio Filter &nbsp;·&nbsp;
        Idea 3: IV Regime Gate &nbsp;·&nbsp; Powered by Dhan Rolling Option API</div>
    </div>
    """, unsafe_allow_html=True)

    # ── SIDEBAR ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Backtest Configuration")
        symbol = st.selectbox("Index", list(INDEX_CONFIG.keys()), index=0)
        expiry_flag = st.selectbox("Expiry Type", ["WEEK", "MONTH"], index=0)
        expiry_code = st.selectbox("Expiry Code",
                                   [1, 2, 3], index=0,
                                   format_func=lambda x: {1:"Current",2:"Next",3:"Far"}[x])
        interval = st.selectbox("Bar Interval", ["5", "15", "60"], index=0,
                                format_func=lambda x: f"{x} min")
        st.markdown("---")
        st.markdown("### 📅 Date Range")
        today  = date.today()
        d_end  = st.date_input("End Date",   value=today - timedelta(days=1))
        d_start= st.date_input("Start Date", value=today - timedelta(days=365))
        st.markdown("---")
        st.markdown("### ⚡ Strikes to Fetch")
        n_strikes = st.slider("ATM ± N strikes", 3, 10, 5)
        all_strikes = (["ATM"]
                       + [f"ATM+{i}" for i in range(1, n_strikes+1)]
                       + [f"ATM-{i}" for i in range(1, n_strikes+1)])
        st.caption(f"{len(all_strikes)} strikes selected")
        st.markdown("---")
        st.markdown("### 🎯 Signal Parameters")
        min_quality = st.slider(
            "Min Quality Score (Fuel/Absorption)", 1.0, 5.0, 2.0, 0.25,
            help="Higher = fewer but stronger signals")
        require_iv  = st.checkbox("Require IV EXPANDING", value=True,
            help="Idea 3: Only trade when IV is expanding (amplifies cascade)")
        st.markdown("---")
        st.markdown("### 🗄️ Database")
        stats = db_stats()
        st.markdown(f"""
        <div class="info-box">
        Raw rows: <b>{stats['raw_rows']:,}</b><br>
        Days fetched: <b>{stats['days']}</b><br>
        Signals: <b>{stats['signals']:,}</b><br>
        Trades: <b>{stats['trades']}</b>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🗑️ Clear All Trades", use_container_width=True):
            clear_trades(symbol)
            st.success("Trades cleared — re-run backtest")

    # ── TABS ─────────────────────────────────────────────────────────────────
    tab_fetch, tab_signals, tab_bt, tab_results, tab_trades = st.tabs([
        "1️⃣ Fetch Data",
        "2️⃣ Compute Signals",
        "3️⃣ Run Backtest",
        "4️⃣ Results",
        "5️⃣ Trade Log",
    ])

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 1: FETCH DATA
    # ─────────────────────────────────────────────────────────────────────────
    with tab_fetch:
        st.markdown("### 📡 Fetch Historical Options Chain")
        st.markdown("""
        <div class="info-box">
        <b>How it works:</b><br>
        For each trading day in your date range, the engine calls<br>
        <code>POST /charts/rollingoption</code> for each strike (CALL + PUT).<br>
        Data is stored in SQLite — each day is fetched only once.<br>
        Already-fetched days are skipped automatically.<br>
        <b>Checkpoint:</b> Progress is saved after every strike — if the session
        crashes or times out, the next fetch run resumes from where it stopped.
        </div>
        """, unsafe_allow_html=True)

        # ── Checkpoint banner ─────────────────────────────────────────────────
        ckpt = checkpoint_status()
        if ckpt:
            ckpt_done = len(ckpt.get("completed_strikes", []))
            ckpt_rows = len(ckpt.get("partial_rows", []))
            st.markdown(f"""
            <div class="warn-box">
            <b>⚡ Checkpoint detected — incomplete session found!</b><br>
            Symbol: <b>{ckpt.get("symbol")}</b> &nbsp;|&nbsp;
            Date: <b>{ckpt.get("trade_date")}</b> &nbsp;|&nbsp;
            Strikes done: <b>{ckpt_done}</b> &nbsp;|&nbsp;
            Rows saved: <b>{ckpt_rows:,}</b><br>
            Saved at: {ckpt.get("saved_at","?")}
            <br><br>The next fetch run will <b>automatically resume</b> from the saved strike.
            </div>
            """, unsafe_allow_html=True)
            if st.button("🗑️ Discard Checkpoint (start fresh for that day)",
                         key="discard_ckpt"):
                clear_checkpoint()
                st.success("Checkpoint cleared.")
                st.rerun()

        trading_dates = get_trading_dates(d_start, d_end)
        done_dates    = get_fetch_log(symbol, expiry_code, expiry_flag)
        pending       = [d for d in trading_dates if d not in done_dates]

        col1, col2, col3 = st.columns(3)
        col1.metric("Total trading days", len(trading_dates))
        col2.metric("Already fetched",    len(done_dates))
        col3.metric("Pending",            len(pending))

        if pending:
            st.markdown(f"""
            <div class="warn-box">
            Estimated fetch time: ~{len(pending) * len(all_strikes) * 2 * 0.35 / 60:.1f} minutes
            ({len(pending)} days × {len(all_strikes)} strikes × 2 options × 0.35s delay)
            </div>
            """, unsafe_allow_html=True)

        fetch_btn = st.button(
            f"🚀 Fetch {len(pending)} Pending Days",
            type="primary", use_container_width=True,
            disabled=(len(pending) == 0),
        )

        if fetch_btn:
            overall_bar  = st.progress(0)
            day_bar      = st.progress(0)
            status_txt   = st.empty()
            day_status   = st.empty()
            log_container= st.empty()
            log_lines    = []

            for idx, trade_date in enumerate(pending):
                day_status.text(f"Day {idx+1}/{len(pending)}: {trade_date}")
                # Check if checkpoint exists for this specific day
                ckpt_now = checkpoint_status()
                if ckpt_now and ckpt_now.get("trade_date") != trade_date:
                    # Stale checkpoint from a different date — clear it
                    clear_checkpoint()
                try:
                    n = fetch_one_day(
                        symbol, trade_date, all_strikes,
                        interval, expiry_code, expiry_flag,
                        progress_bar=day_bar,
                        status_text=status_txt,
                    )
                    log_fetch(symbol, trade_date, expiry_code, expiry_flag, "ok", n)
                    log_lines.append(f"✅ {trade_date} — {n:,} rows")
                except Exception as e:
                    # Save what we have — checkpoint already written inside fetch_one_day
                    # Mark as partial so it shows in pending next run
                    log_lines.append(f"⚠️ {trade_date} — interrupted: {e} (checkpoint saved)")
                    log_container.text("\n".join(log_lines[-15:]))
                    st.warning(f"Session interrupted at {trade_date}. "
                               f"Checkpoint saved — restart fetch to resume.")
                    break   # stop loop, don't mark as ok

                overall_bar.progress((idx + 1) / len(pending))
                log_container.text("\n".join(log_lines[-15:]))

            overall_bar.empty(); day_bar.empty()
            status_txt.empty(); day_status.empty()
            st.success(f"✅ Fetch complete — {len(pending)} days processed.")
            st.rerun()

        # Show fetched days table
        if done_dates:
            st.markdown("#### ✅ Already Fetched Days")
            con = sqlite3.connect(DB_PATH)
            log_df = pd.read_sql_query("""
                SELECT trade_date, rows_fetched, fetched_at
                FROM fetch_log
                WHERE symbol=? AND expiry_code=? AND expiry_flag=? AND status='ok'
                ORDER BY trade_date DESC
            """, con, params=(symbol, expiry_code, expiry_flag))
            con.close()
            st.dataframe(log_df, use_container_width=True, height=300, hide_index=True)

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 2: COMPUTE SIGNALS
    # ─────────────────────────────────────────────────────────────────────────
    with tab_signals:
        st.markdown("### ⚡ Compute Cascade Signals")
        st.markdown("""
        <div class="info-box">
        For each fetched day, compute GEX cascade signal at every 5-min bar:<br>
        <b>Bear/Bull Fuel Pts</b> · <b>Absorption Pts</b> · <b>Quality Score</b> · <b>IV Regime</b><br>
        Results stored in <code>cascade_signals</code> table for instant backtest replay.
        </div>
        """, unsafe_allow_html=True)

        done_dates   = get_fetch_log(symbol, expiry_code, expiry_flag)
        con          = sqlite3.connect(DB_PATH)
        sig_dates_df = pd.read_sql_query(
            "SELECT DISTINCT trade_date FROM cascade_signals WHERE symbol=?",
            con, params=(symbol,))
        con.close()
        sig_dates = set(sig_dates_df["trade_date"].tolist()) if not sig_dates_df.empty else set()
        pending_sig = [d for d in sorted(done_dates) if d not in sig_dates]

        c1, c2, c3 = st.columns(3)
        c1.metric("Fetched days",  len(done_dates))
        c2.metric("Signals ready", len(sig_dates))
        c3.metric("Pending",       len(pending_sig))

        sig_btn = st.button(
            f"⚡ Compute Signals for {len(pending_sig)} Days",
            type="primary", use_container_width=True,
            disabled=(len(pending_sig) == 0),
        )

        if sig_btn:
            prog   = st.progress(0)
            status = st.empty()
            for idx, trade_date in enumerate(pending_sig):
                status.text(f"Computing signals: {trade_date} ({idx+1}/{len(pending_sig)})")
                df_day = load_raw_chain(symbol, trade_date, expiry_code, expiry_flag)
                if df_day.empty:
                    continue
                sig_rows = compute_signals_for_day(df_day, symbol, trade_date)
                save_signals(sig_rows)
                prog.progress((idx+1) / len(pending_sig))
            prog.empty(); status.empty()
            st.success(f"✅ Signals computed for {len(pending_sig)} days.")
            st.rerun()

        # Preview latest signals
        if sig_dates:
            latest_date = max(sig_dates)
            st.markdown(f"#### 📊 Signal Preview — {latest_date}")
            sig_df = load_signals(symbol, latest_date)
            if not sig_df.empty:
                cols = ["timestamp","spot_price","bear_quality","bull_quality",
                        "signal","signal_strength","iv_regime","avg_iv",
                        "bear_fuel_pts","bear_absorb_pts"]
                st.dataframe(sig_df[cols], use_container_width=True,
                             height=350, hide_index=True)

                # Quick chart
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                    subplot_titles=["Signal Quality", "IV Regime"],
                                    row_heights=[0.6, 0.4])
                fig.add_trace(go.Scatter(x=sig_df["timestamp"], y=sig_df["bear_quality"],
                    name="Bear Quality", line=dict(color="#ef4444")), row=1, col=1)
                fig.add_trace(go.Scatter(x=sig_df["timestamp"], y=sig_df["bull_quality"],
                    name="Bull Quality", line=dict(color="#10b981")), row=1, col=1)
                fig.add_hline(y=min_quality, line_dash="dash",
                              line_color="#f59e0b", row=1, col=1)
                fig.add_trace(go.Scatter(x=sig_df["timestamp"], y=sig_df["avg_iv"],
                    name="Avg IV", line=dict(color="#a78bfa")), row=2, col=1)
                fig.update_layout(
                    template="plotly_dark", height=420,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(10,10,20,0.95)",
                    margin=dict(l=0, r=0, t=30, b=0),
                )
                st.plotly_chart(fig, use_container_width=True)

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 3: RUN BACKTEST
    # ─────────────────────────────────────────────────────────────────────────
    with tab_bt:
        st.markdown("### 🔄 Run Backtest Simulation")

        con = sqlite3.connect(DB_PATH)
        sig_dates_df = pd.read_sql_query(
            "SELECT DISTINCT trade_date FROM cascade_signals WHERE symbol=?",
            con, params=(symbol,))
        con.close()
        all_sig_dates = sorted(sig_dates_df["trade_date"].tolist()) if not sig_dates_df.empty else []

        st.markdown(f"""
        <div class="info-box">
        <b>Strategy:</b><br>
        &nbsp;• Signal fires when quality &ge; <b>{min_quality}</b> (fuel/absorption ratio)<br>
        &nbsp;• IV gate: <b>{"EXPANDING only" if require_iv else "Any regime"}</b><br>
        &nbsp;• Entry at signal bar spot price<br>
        &nbsp;• Target = cascade net realised pts estimate<br>
        &nbsp;• Stop = next absorption wall<br>
        &nbsp;• One trade at a time, EOD forced exit<br>
        <b>Days available:</b> {len(all_sig_dates)}
        </div>
        """, unsafe_allow_html=True)

        bt_btn = st.button(
            f"▶ Run Backtest on {len(all_sig_dates)} Days",
            type="primary", use_container_width=True,
            disabled=(len(all_sig_dates) == 0),
        )

        if bt_btn:
            clear_trades(symbol)
            prog   = st.progress(0)
            status = st.empty()
            all_trades = []
            for idx, trade_date in enumerate(all_sig_dates):
                status.text(f"Simulating {trade_date} ({idx+1}/{len(all_sig_dates)})")
                sig_df = load_signals(symbol, trade_date)
                if sig_df.empty: continue
                day_trades = run_backtest_for_day(
                    sig_df, symbol, trade_date, expiry_flag,
                    min_quality, require_iv)
                all_trades.extend(day_trades)
                prog.progress((idx+1) / len(all_sig_dates))
            save_trades(all_trades)
            prog.empty(); status.empty()
            st.success(f"✅ Backtest complete — {len(all_trades)} trades generated.")
            st.rerun()

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 4: RESULTS
    # ─────────────────────────────────────────────────────────────────────────
    with tab_results:
        st.markdown("### 📊 Backtest Results")
        trades = load_trades(symbol)
        if trades.empty:
            st.info("No trades yet — run the backtest in Tab 3 first.")
        else:
            m = compute_metrics(trades)

            # KPI cards
            cols = st.columns(5)
            kpis = [
                ("Total Trades",      str(m["total_trades"]),          "signal-neutral"),
                ("Hit Rate",          f"{m['hit_rate']}%",             "signal-bull" if m["hit_rate"]>=50 else "signal-bear"),
                ("Total Pts",         f"{m['total_pts']:+.1f}",        "signal-bull" if m["total_pts"]>0 else "signal-bear"),
                ("Profit Factor",     f"{m['profit_factor']:.2f}x",    "signal-bull" if m["profit_factor"]>1 else "signal-bear"),
                ("Max Drawdown",      f"{m['max_drawdown']:.1f} pts",  "signal-bear"),
            ]
            for col, (label, val, cls) in zip(cols, kpis):
                col.markdown(f"""
                <div class="metric-card">
                    <div class="metric-val {cls}">{val}</div>
                    <div class="metric-lbl">{label}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("")
            cols2 = st.columns(5)
            kpis2 = [
                ("Sharpe Ratio",       f"{m['sharpe']:.2f}",           "signal-bull" if m["sharpe"]>1 else "signal-neutral"),
                ("Sortino Ratio",      f"{m['sortino']:.2f}",          "signal-bull" if m["sortino"]>1 else "signal-neutral"),
                ("Cascade Accuracy",   f"{m['cascade_accuracy']:.1f}%","signal-bull" if m["cascade_accuracy"]>60 else "signal-neutral"),
                ("Expiry Day Pts",     f"{m['expiry_pts']:+.1f}",       "signal-bull" if m["expiry_pts"]>0 else "signal-bear"),
                ("Non-Expiry Pts",     f"{m['non_expiry_pts']:+.1f}",   "signal-bull" if m["non_expiry_pts"]>0 else "signal-bear"),
            ]
            for col, (label, val, cls) in zip(cols2, kpis2):
                col.markdown(f"""
                <div class="metric-card">
                    <div class="metric-val {cls}">{val}</div>
                    <div class="metric-lbl">{label}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("---")

            # Charts row 1
            c1, c2 = st.columns([2, 1])
            with c1:
                st.markdown('<div class="section-label">Equity Curve</div>', unsafe_allow_html=True)
                st.plotly_chart(equity_curve_chart(trades), use_container_width=True)
            with c2:
                st.markdown('<div class="section-label">IV Regime Breakdown</div>', unsafe_allow_html=True)
                st.plotly_chart(iv_regime_breakdown_chart(trades), use_container_width=True)

            # Charts row 2
            c3, c4 = st.columns(2)
            with c3:
                st.markdown('<div class="section-label">Quality Score vs P&L</div>', unsafe_allow_html=True)
                st.plotly_chart(quality_vs_pts_chart(trades), use_container_width=True)
            with c4:
                st.markdown('<div class="section-label">Cascade Target Accuracy</div>', unsafe_allow_html=True)
                st.plotly_chart(cascade_accuracy_chart(trades), use_container_width=True)

            # Exit reason
            st.markdown('<div class="section-label">Exit Reason Breakdown</div>', unsafe_allow_html=True)
            st.plotly_chart(exit_reason_chart(trades), use_container_width=True)

            # Monthly breakdown
            st.markdown("#### 📅 Monthly P&L Breakdown")
            trades["month"] = pd.to_datetime(trades["trade_date"]).dt.to_period("M").astype(str)
            monthly = trades.groupby("month")["pts_captured"].agg(["sum","count","mean"]).reset_index()
            monthly.columns = ["Month", "Total Pts", "Trades", "Avg Pts"]
            monthly["Total Pts"] = monthly["Total Pts"].round(1)
            monthly["Avg Pts"]   = monthly["Avg Pts"].round(1)
            fig_m = go.Figure(go.Bar(
                x=monthly["Month"], y=monthly["Total Pts"],
                marker_color=monthly["Total Pts"].apply(lambda x: "#10b981" if x>0 else "#ef4444"),
                text=monthly["Trades"].apply(lambda x: f"{x} trades"),
                textposition="outside",
            ))
            fig_m.update_layout(
                template="plotly_dark", height=320,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(10,10,20,0.95)",
                yaxis_title="Pts", margin=dict(l=0,r=0,t=10,b=0),
            )
            st.plotly_chart(fig_m, use_container_width=True)

            # Download
            st.download_button(
                "📥 Download Results CSV",
                data=trades.to_csv(index=False),
                file_name=f"hedgex_backtest_{symbol}.csv",
                mime="text/csv",
                use_container_width=True,
            )

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 5: TRADE LOG
    # ─────────────────────────────────────────────────────────────────────────
    with tab_trades:
        st.markdown("### 📋 Full Trade Log")
        trades = load_trades(symbol)
        if trades.empty:
            st.info("No trades yet.")
        else:
            disp_cols = ["trade_date","entry_time","exit_time","direction",
                         "entry_price","exit_price","pts_captured",
                         "cascade_target","exit_reason","signal_strength",
                         "iv_regime","is_expiry_day"]
            disp = trades[disp_cols].copy()
            disp["entry_time"] = pd.to_datetime(disp["entry_time"]).dt.strftime("%H:%M")
            disp["exit_time"]  = pd.to_datetime(disp["exit_time"]).dt.strftime("%H:%M")
            disp["is_expiry_day"] = disp["is_expiry_day"].map({0:"", 1:"🗓️ Expiry"})

            def row_color(row):
                if row["pts_captured"] > 0:
                    return ["background-color:rgba(16,185,129,0.12)"] * len(row)
                return ["background-color:rgba(239,68,68,0.10)"] * len(row)

            st.dataframe(
                disp.style.apply(row_color, axis=1),
                use_container_width=True, height=600, hide_index=True,
            )

    # Footer
    st.markdown("""
    <div style="text-align:center;padding:20px;font-family:'JetBrains Mono',monospace;
    font-size:0.70rem;color:rgba(255,255,255,0.25);margin-top:20px;">
    HedGEX Cascade Backtest &nbsp;·&nbsp; Powered by NYZTrade Analytics Pvt. Ltd.
    &nbsp;·&nbsp; For research purposes only — not financial advice.
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
