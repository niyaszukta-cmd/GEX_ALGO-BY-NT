# ============================================================================
# HedGEX — Cascade Backtest Engine  v2
# Powered by NYZTrade Analytics Pvt. Ltd.
#
# Strategy: Strike-Level Custom Entry/Exit
#   CALL : IV Expanding + ATM+1 cascade pts > threshold → BUY ATM+1 Call
#          Target = min(cumulative cascade ATM+1→ATM+3, max_target) from spot
#          Stop   = fixed pts below spot
#   PUT  : IV Expanding + ATM-1 cascade pts > threshold → BUY ATM-1 Put
#          Target = min(cumulative cascade ATM-1→ATM-3, max_target) from spot
#          Stop   = fixed pts above spot
#
# Data : Dhan Rolling Option API v2
# Engine: GEX Cascade Mathematics
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
DHAN_ACCESS_TOKEN = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9.eyJpc3MiOiJkaGFuIiwicGFydG5lcklkIjoiIiwiZXhwIjoxNzc1NjMyMDMyLCJhcHBfaWQiOiJhYjYxZmJmOSIsImlhdCI6MTc3NTU0NTYzMiwidG9rZW5Db25zdW1lclR5cGUiOiJBUFAiLCJ3ZWJob29rVXJsIjoiIiwiZGhhbkNsaWVudElkIjoiMTEwMDQ4MDM1NCJ9.6mLI3OTxCjgy4oEvqtjKG1BUgP3OLWuenA011y-FXfXQZbpnmZ_aeBjWMu5c-DKzhPUfLQZPVtUr2eQ7_-lmDQ"

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
        bear_fuel_pts REAL, bear_absorb_pts REAL,
        bull_fuel_pts REAL, bull_absorb_pts REAL,
        bear_quality REAL, bull_quality REAL,
        iv_regime TEXT, avg_iv REAL, iv_skew REAL,
        net_gex_total REAL, signal TEXT, signal_strength REAL,
        cascade_target REAL, cascade_stop REAL,
        UNIQUE(symbol,trade_date,timestamp))""")
    cur.execute("""CREATE TABLE IF NOT EXISTS bt_trades(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT, trade_date TEXT, entry_time TEXT, exit_time TEXT,
        direction TEXT, entry_price REAL, exit_price REAL,
        pts_captured REAL, cascade_target REAL, cascade_stop REAL,
        exit_reason TEXT, bear_quality REAL, bull_quality REAL,
        iv_regime TEXT, signal_strength REAL,
        is_expiry_day INTEGER, expiry_flag TEXT)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS fetch_log(
        symbol TEXT, trade_date TEXT, expiry_code INTEGER, expiry_flag TEXT,
        status TEXT, rows_fetched INTEGER, fetched_at TEXT,
        PRIMARY KEY(symbol,trade_date,expiry_code,expiry_flag))""")
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
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT * FROM cascade_signals WHERE symbol=? AND trade_date=? ORDER BY timestamp",
        con, params=(symbol,date_str)); con.close()
    if not df.empty: df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

def save_trades(rows):
    if not rows: return
    con = sqlite3.connect(DB_PATH)
    con.executemany("""INSERT INTO bt_trades
        (symbol,trade_date,entry_time,exit_time,direction,entry_price,exit_price,
         pts_captured,cascade_target,cascade_stop,exit_reason,
         bear_quality,bull_quality,iv_regime,signal_strength,is_expiry_day,expiry_flag)
        VALUES(:symbol,:trade_date,:entry_time,:exit_time,:direction,:entry_price,:exit_price,
         :pts_captured,:cascade_target,:cascade_stop,:exit_reason,
         :bear_quality,:bull_quality,:iv_regime,:signal_strength,:is_expiry_day,:expiry_flag)""", rows)
    con.commit(); con.close()

def load_trades(symbol=None):
    con = sqlite3.connect(DB_PATH)
    q = "SELECT * FROM bt_trades WHERE symbol=? ORDER BY trade_date,entry_time" if symbol \
        else "SELECT * FROM bt_trades ORDER BY trade_date,entry_time"
    df = pd.read_sql_query(q, con, params=(symbol,) if symbol else ()); con.close()
    return df

def clear_trades(symbol=None):
    con = sqlite3.connect(DB_PATH)
    con.execute("DELETE FROM bt_trades WHERE symbol=?" if symbol else "DELETE FROM bt_trades",
                (symbol,) if symbol else ())
    con.commit(); con.close()

def db_stats():
    con = sqlite3.connect(DB_PATH); cur = con.cursor()
    cur.execute("SELECT COUNT(*) FROM raw_chain");         rr = cur.fetchone()[0]
    cur.execute("SELECT COUNT(DISTINCT trade_date),COUNT(DISTINCT symbol) FROM raw_chain")
    days,syms = cur.fetchone()
    cur.execute("SELECT COUNT(*) FROM bt_trades");         tr = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM cascade_signals");   sg = cur.fetchone()[0]
    con.close()
    return {"raw_rows":rr,"days":days,"symbols":syms,"trades":tr,"signals":sg}

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

# ── Cascade engine (kept intact for Tab 2 signals) ────────────────────────────
VANNA_ADJ = {
    "SUPPORT_FLOOR":      {"COMPRESSING":0.60,"FLAT":0.35,"EXPANDING":-0.20},
    "TRAP_DOOR":          {"EXPANDING":-0.30,"FLAT":0.10,"COMPRESSING":0.15},
    "VACUUM_ZONE":        {"EXPANDING":0.50,"FLAT":0.20,"COMPRESSING":0.10},
    "RESISTANCE_CEILING": {"EXPANDING":-0.20,"FLAT":0.00,"COMPRESSING":0.15},
}

def compute_cascade_for_snapshot(df_ts, spot, symbol, iv_regime):
    params       = INSTRUMENT_PARAMS.get(symbol, INSTRUMENT_PARAMS["NIFTY"])
    ppu          = params["pts_per_unit"]
    cap          = params["strike_cap"]
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
        df_ts = df_day[df_day["timestamp"]==ts].copy()
        spot  = df_ts["spot_price"].mean()
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
# CUSTOM STRATEGY — Strike-level cascade energy helper
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def get_strike_cascade_pts(df_ts, strike_types, symbol):
    """
    Returns { strike_type: cascade_pts } for a single-bar snapshot.
    cascade_pts = abs(net_gex) * pts_per_unit
    """
    ppu    = INSTRUMENT_PARAMS.get(symbol, INSTRUMENT_PARAMS["NIFTY"])["pts_per_unit"]
    result = {}
    for st in strike_types:
        rows = df_ts[df_ts["strike_type"] == st]
        if rows.empty:
            result[st] = 0.0
        else:
            net_gex = rows["net_gex"].iloc[0]
            result[st] = abs(net_gex) * ppu
    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CUSTOM STRATEGY — Per-bar signal builder (reads raw_chain directly)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def compute_strike_signals_for_day(df_day, symbol, trade_date, iv_df,
                                   cascade_entry_threshold, max_target_pts):
    """
    Walks every bar in df_day and returns a list of bar-dicts with:
        timestamp, spot, iv_regime,
        call_signal, call_target, call_stop,
        put_signal,  put_target,  put_stop,
        atm1_pts, atm_neg1_pts, call_cum_pts, put_cum_pts

    CALL entry condition:
        iv_regime == EXPANDING  AND  ATM+1 cascade pts > cascade_entry_threshold
        target = spot + min(ATM+1 + ATM+2 + ATM+3 cascade pts, max_target_pts)

    PUT entry condition:
        iv_regime == EXPANDING  AND  ATM-1 cascade pts > cascade_entry_threshold
        target = spot - min(ATM-1 + ATM-2 + ATM-3 cascade pts, max_target_pts)
    """
    bars = []
    for ts in sorted(df_day["timestamp"].unique()):
        df_ts = df_day[df_day["timestamp"] == ts]
        spot  = df_ts["spot_price"].mean()

        # IV regime for this bar
        iv_row    = iv_df[iv_df["timestamp"] == ts]
        iv_regime = str(iv_row.iloc[0]["iv_regime"]) if not iv_row.empty else "FLAT"

        # ── CALL side: ATM+1, ATM+2, ATM+3 ──────────────────────────────
        call_strikes = ["ATM+1", "ATM+2", "ATM+3"]
        c_pts        = get_strike_cascade_pts(df_ts, call_strikes, symbol)
        atm1_pts     = c_pts.get("ATM+1", 0.0)
        call_cum     = sum(c_pts.values())
        call_tgt_pts = min(call_cum, max_target_pts)
        call_signal  = (iv_regime == "EXPANDING") and (atm1_pts > cascade_entry_threshold)

        # ── PUT side: ATM-1, ATM-2, ATM-3 ───────────────────────────────
        put_strikes  = ["ATM-1", "ATM-2", "ATM-3"]
        p_pts        = get_strike_cascade_pts(df_ts, put_strikes, symbol)
        atm_neg1_pts = p_pts.get("ATM-1", 0.0)
        put_cum      = sum(p_pts.values())
        put_tgt_pts  = min(put_cum, max_target_pts)
        put_signal   = (iv_regime == "EXPANDING") and (atm_neg1_pts > cascade_entry_threshold)

        bars.append({
            "timestamp":    ts,
            "spot":         round(spot, 2),
            "iv_regime":    iv_regime,
            # CALL
            "call_signal":  call_signal,
            "call_target":  round(spot + call_tgt_pts, 2),
            # PUT
            "put_signal":   put_signal,
            "put_target":   round(spot - put_tgt_pts, 2),
            # diagnostics
            "atm1_pts":     round(atm1_pts, 2),
            "atm_neg1_pts": round(atm_neg1_pts, 2),
            "call_cum_pts": round(call_cum, 2),
            "put_cum_pts":  round(put_cum, 2),
        })
    return bars


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CUSTOM STRATEGY — Backtest engine
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_backtest_for_day(signals,           # kept for Tab 3 signature compat (not used)
                         symbol,
                         trade_date,
                         expiry_flag,
                         min_quality,       # legacy param (not used in strike strategy)
                         require_iv,        # legacy param (not used in strike strategy)
                         # ── custom strategy params ──────────────────────
                         cascade_entry_threshold=50.0,
                         max_target_pts=200.0,
                         fixed_sl_pts=50.0,
                         direction_filter="BOTH",
                         max_trades_per_day=2,
                         df_day=None,
                         iv_df=None):
    """
    Strike-Level Cascade Entry/Exit Backtest

    CALL trade  (BUY ATM+1 Call):
        Entry  : iv_regime==EXPANDING  AND  ATM+1 cascade pts > cascade_entry_threshold
        Target : spot + min(ATM+1+ATM+2+ATM+3 cascade pts,  max_target_pts)
        Stop   : spot - fixed_sl_pts

    PUT trade  (BUY ATM-1 Put):
        Entry  : iv_regime==EXPANDING  AND  ATM-1 cascade pts > cascade_entry_threshold
        Target : spot - min(ATM-1+ATM-2+ATM-3 cascade pts, max_target_pts)
        Stop   : spot + fixed_sl_pts

    CALL and PUT trades are tracked independently — both can be open simultaneously.
    max_trades_per_day counts CALL + PUT combined.
    """
    if df_day is None or df_day.empty:
        return []

    if iv_df is None:
        iv_df = compute_iv_regime_series(df_day)

    bar_signals = compute_strike_signals_for_day(
        df_day, symbol, trade_date, iv_df,
        cascade_entry_threshold, max_target_pts)

    if not bar_signals:
        return []

    try:
        dt        = datetime.strptime(trade_date, "%Y-%m-%d")
        is_expiry = (dt.weekday() == 3 if expiry_flag == "WEEK" else False)
    except:
        is_expiry = False

    trades      = []
    trade_count = 0   # combined CALL + PUT count for the day

    # ── Per-direction state ───────────────────────────────────────────────
    call_open     = False
    call_entry_ts = call_entry_px = call_tgt = call_stp = call_cum = None

    put_open      = False
    put_entry_ts  = put_entry_px = put_tgt = put_stp = put_cum = None

    for bar in bar_signals:
        ts    = bar["timestamp"]
        spot  = bar["spot"]
        iv_reg = bar["iv_regime"]

        # ── Exit: open CALL trade ─────────────────────────────────────────
        if call_open:
            pts = spot - call_entry_px

            if spot >= call_tgt:
                trades.append(_mk_strike(
                    symbol, trade_date, call_entry_ts, ts, "CALL",
                    call_entry_px, spot, pts, call_tgt, call_stp,
                    "TARGET_HIT", iv_reg, call_cum, is_expiry, expiry_flag))
                call_open = False
                trade_count += 1

            elif spot <= call_stp:
                trades.append(_mk_strike(
                    symbol, trade_date, call_entry_ts, ts, "CALL",
                    call_entry_px, spot, pts, call_tgt, call_stp,
                    "STOP_HIT", iv_reg, call_cum, is_expiry, expiry_flag))
                call_open = False
                trade_count += 1

        # ── Exit: open PUT trade ──────────────────────────────────────────
        if put_open:
            pts = put_entry_px - spot

            if spot <= put_tgt:
                trades.append(_mk_strike(
                    symbol, trade_date, put_entry_ts, ts, "PUT",
                    put_entry_px, spot, pts, put_tgt, put_stp,
                    "TARGET_HIT", iv_reg, put_cum, is_expiry, expiry_flag))
                put_open = False
                trade_count += 1

            elif spot >= put_stp:
                trades.append(_mk_strike(
                    symbol, trade_date, put_entry_ts, ts, "PUT",
                    put_entry_px, spot, pts, put_tgt, put_stp,
                    "STOP_HIT", iv_reg, put_cum, is_expiry, expiry_flag))
                put_open = False
                trade_count += 1

        # Gate: max trades per day reached
        if trade_count >= max_trades_per_day:
            continue

        # ── Entry: CALL ───────────────────────────────────────────────────
        if (not call_open
                and direction_filter in ("BOTH", "CALL only")
                and bar["call_signal"]
                and bar["atm1_pts"] > cascade_entry_threshold):

            call_open     = True
            call_entry_ts = ts
            call_entry_px = spot
            call_tgt      = bar["call_target"]   # already capped at max_target_pts
            call_stp      = round(spot - fixed_sl_pts, 2)
            call_cum      = bar["call_cum_pts"]

        # ── Entry: PUT ────────────────────────────────────────────────────
        if (not put_open
                and direction_filter in ("BOTH", "PUT only")
                and bar["put_signal"]
                and bar["atm_neg1_pts"] > cascade_entry_threshold):

            put_open     = True
            put_entry_ts = ts
            put_entry_px = spot
            put_tgt      = bar["put_target"]     # already capped at max_target_pts
            put_stp      = round(spot + fixed_sl_pts, 2)
            put_cum      = bar["put_cum_pts"]

    # ── EOD exit ──────────────────────────────────────────────────────────
    if bar_signals:
        last   = bar_signals[-1]
        lts    = last["timestamp"]
        lspot  = last["spot"]
        iv_reg = last["iv_regime"]

        if call_open:
            pts = lspot - call_entry_px
            trades.append(_mk_strike(
                symbol, trade_date, call_entry_ts, lts, "CALL",
                call_entry_px, lspot, pts, call_tgt, call_stp,
                "EOD_EXIT", iv_reg, call_cum, is_expiry, expiry_flag))

        if put_open:
            pts = put_entry_px - lspot
            trades.append(_mk_strike(
                symbol, trade_date, put_entry_ts, lts, "PUT",
                put_entry_px, lspot, pts, put_tgt, put_stp,
                "EOD_EXIT", iv_reg, put_cum, is_expiry, expiry_flag))

    return trades


def _mk_strike(symbol, trade_date, entry_ts, exit_ts, direction,
               entry_px, exit_px, pts, tgt, stp,
               reason, iv_reg, cascade_cum_pts, is_expiry, expiry_flag):
    """
    Trade record builder.
    bear_quality  → stores cumulative cascade pts (ATM±1+2+3) for display.
    signal_strength → same value for chart compatibility.
    """
    return {
        "symbol":          symbol,
        "trade_date":      trade_date,
        "entry_time":      str(entry_ts),
        "exit_time":       str(exit_ts),
        "direction":       direction,        # "CALL" or "PUT"
        "entry_price":     round(entry_px, 2),
        "exit_price":      round(exit_px, 2),
        "pts_captured":    round(pts, 2),
        "cascade_target":  round(tgt, 2),
        "cascade_stop":    round(stp, 2),
        "exit_reason":     reason,
        "bear_quality":    round(cascade_cum_pts, 2),   # cascade cum pts
        "bull_quality":    0.0,
        "iv_regime":       iv_reg,
        "signal_strength": round(cascade_cum_pts, 2),
        "is_expiry_day":   int(is_expiry),
        "expiry_flag":     expiry_flag,
    }

# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(trades):
    if trades.empty: return {}
    pts=trades["pts_captured"]; wins=pts[pts>0]; losses=pts[pts<=0]
    total=len(trades); hit=len(wins)/total*100 if total else 0
    pf=(wins.sum()/abs(losses.sum())) if losses.sum()!=0 else float("inf")
    cum=pts.cumsum(); dd=(cum.cummax()-cum).max()
    sh=(pts.mean()/pts.std()*np.sqrt(250)) if pts.std()>0 else 0
    neg=pts[pts<0].std()
    so=(pts.mean()/neg*np.sqrt(250)) if neg and neg>0 else 0
    t2=trades.copy()
    t2["tm"]=abs(t2["cascade_target"]-t2["entry_price"])
    t2["am"]=abs(t2["pts_captured"])
    ca=(t2["am"]>=t2["tm"]*0.7).mean()*100
    return {"total_trades":total,"hit_rate":round(hit,1),"total_pts":round(pts.sum(),1),
            "avg_win":round(wins.mean(),1) if len(wins) else 0,
            "avg_loss":round(losses.mean(),1) if len(losses) else 0,
            "profit_factor":round(pf,2),"max_drawdown":round(dd,1),
            "sharpe":round(sh,2),"sortino":round(so,2),"cascade_accuracy":round(ca,1),
            "expiry_trades":int(trades["is_expiry_day"].sum()),
            "expiry_pts":round(trades[trades["is_expiry_day"]==1]["pts_captured"].sum(),1),
            "non_expiry_pts":round(trades[trades["is_expiry_day"]==0]["pts_captured"].sum(),1)}

# ── Charts ────────────────────────────────────────────────────────────────────
def equity_curve_chart(trades):
    t=trades.copy().reset_index(drop=True); t["cum"]=t["pts_captured"].cumsum()
    fig=make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[0.65,0.35],
                      subplot_titles=["Equity Curve (Cumulative Pts)","Per-Trade P&L"])
    fig.add_trace(go.Scatter(x=t.index,y=t["cum"],mode="lines",
        line=dict(color="#00f5c4",width=2.5),fill="tozeroy",fillcolor="rgba(0,245,196,0.08)"),row=1,col=1)
    fig.add_trace(go.Bar(x=t.index,y=t["pts_captured"],
        marker_color=t["pts_captured"].apply(lambda x:"#10b981" if x>0 else "#ef4444")),row=2,col=1)
    fig.update_layout(template="plotly_dark",height=500,paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,10,20,0.95)",showlegend=False,margin=dict(l=0,r=0,t=30,b=0))
    return fig

def quality_vs_pts_chart(trades):
    colors=trades["pts_captured"].apply(lambda x:"#10b981" if x>0 else "#ef4444")
    fig=go.Figure(go.Scatter(x=trades["signal_strength"],y=trades["pts_captured"],
        mode="markers",marker=dict(color=colors,size=8,opacity=0.75),
        text=trades["trade_date"]+" "+trades["direction"],
        hovertemplate="<b>%{text}</b><br>Cascade Pts:%{x:.1f}<br>P&L:%{y:.1f}pts<extra></extra>"))
    fig.update_layout(template="plotly_dark",height=380,paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,10,20,0.95)",xaxis_title="Cumulative Cascade Pts",
        yaxis_title="Pts Captured",margin=dict(l=0,r=0,t=10,b=0))
    return fig

def iv_regime_breakdown_chart(trades):
    grp=trades.groupby("iv_regime")["pts_captured"].agg(["sum","count","mean"]).reset_index()
    colors={"EXPANDING":"#ef4444","COMPRESSING":"#10b981","FLAT":"#94a3b8"}
    fig=go.Figure(go.Bar(x=grp["iv_regime"],y=grp["sum"],
        marker_color=[colors.get(r,"#8b5cf6") for r in grp["iv_regime"]],
        text=[f"{row['count']} trades" for _,row in grp.iterrows()],
        textposition="outside"))
    fig.update_layout(template="plotly_dark",height=320,paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,10,20,0.95)",yaxis_title="Total Pts",margin=dict(l=0,r=0,t=10,b=0))
    return fig

def exit_reason_chart(trades):
    grp=trades.groupby("exit_reason")["pts_captured"].agg(["sum","count"]).reset_index()
    cm={"TARGET_HIT":"#10b981","STOP_HIT":"#ef4444",
        "EOD_EXIT":"#f59e0b","TRAIL_STOP":"#f97316"}
    fig=go.Figure(go.Bar(x=grp["exit_reason"],y=grp["sum"],
        marker_color=[cm.get(r,"#8b5cf6") for r in grp["exit_reason"]],
        text=grp["count"].apply(lambda x: f"{x} trades"),textposition="outside"))
    fig.update_layout(template="plotly_dark",height=300,paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,10,20,0.95)",yaxis_title="Total Pts",margin=dict(l=0,r=0,t=10,b=0))
    return fig

def direction_breakdown_chart(trades):
    """New: CALL vs PUT performance breakdown."""
    grp=trades.groupby("direction")["pts_captured"].agg(["sum","count","mean"]).reset_index()
    cm={"CALL":"#00d4ff","PUT":"#a78bfa"}
    fig=go.Figure(go.Bar(x=grp["direction"],y=grp["sum"],
        marker_color=[cm.get(d,"#8b5cf6") for d in grp["direction"]],
        text=[f"{row['count']} trades | avg {row['mean']:.1f}pts" for _,row in grp.iterrows()],
        textposition="outside"))
    fig.update_layout(template="plotly_dark",height=300,paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,10,20,0.95)",yaxis_title="Total Pts",
        title_text="CALL vs PUT Breakdown",margin=dict(l=0,r=0,t=30,b=0))
    return fig

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    init_db()

    st.markdown("""<div class="bt-header">
        <div class="bt-title">HedGEX — Cascade Backtest Engine</div>
        <div class="bt-sub">Strike-Level Custom Strategy &nbsp;·&nbsp;
        ATM±1 Cascade Trigger &nbsp;·&nbsp; Powered by Dhan Rolling Option API</div>
    </div>""", unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        st.markdown(
            f'<div class="info-box">Client: <b>1100480354</b><br>Token: <b>Hardcoded</b></div>',
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
        today  = date.today()
        d_end  = st.date_input("End Date",   value=today-timedelta(days=1))
        d_start= st.date_input("Start Date", value=today-timedelta(days=365))
        st.markdown("---")
        st.markdown("### ⚡ Strikes")
        n_strikes = st.slider("ATM ± N", 3, 10, 5)
        all_strikes = (["ATM"]
                       +[f"ATM+{i}" for i in range(1,n_strikes+1)]
                       +[f"ATM-{i}" for i in range(1,n_strikes+1)])
        st.caption(f"{len(all_strikes)} strikes selected")
        st.markdown("---")

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # CUSTOM STRATEGY PARAMETERS
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        st.markdown("### 🎯 Strike-Level Strategy")
        st.markdown(
            '<div class="info-box">'
            '📈 <b>CALL</b>: IV Expanding + ATM+1 cascade &gt; trigger → BUY ATM+1 Call<br>'
            '&nbsp;&nbsp;&nbsp;&nbsp;Target = min(ATM+1+2+3 cascade, max target)<br>'
            '📉 <b>PUT &nbsp;</b>: IV Expanding + ATM-1 cascade &gt; trigger → BUY ATM-1 Put<br>'
            '&nbsp;&nbsp;&nbsp;&nbsp;Target = min(ATM-1+2+3 cascade, max target)<br>'
            '🛑 Stop Loss = fixed pts from entry spot'
            '</div>',
            unsafe_allow_html=True)

        st.markdown("**Entry**")
        cascade_entry_threshold = st.slider(
            "ATM±1 Cascade Trigger (pts)", 10, 200, 50, 5,
            help="Entry fires when abs(net_gex × pts_per_unit) at ATM+1 or ATM-1 exceeds this value.")

        st.markdown("**Target**")
        max_target_pts = st.slider(
            "Max Target (pts)", 50, 500, 200, 25,
            help="Cumulative cascade (ATM±1 through ATM±3) capped at this value.")

        st.markdown("**Stop Loss**")
        fixed_sl_pts = st.slider(
            "Stop Loss (pts)", 10, 200, 50, 5,
            help="Fixed stop loss in points from entry spot price.")

        st.markdown("**Direction**")
        direction_filter = st.selectbox(
            "Allow Trades",
            options=["BOTH", "CALL only", "PUT only"],
            index=0,
            help="Restrict to one direction or allow both simultaneously.")

        st.markdown("**Max Trades / Day**")
        max_trades_per_day = st.selectbox(
            "Max Trades Per Day",
            options=[1, 2, 3, 5, 99],
            index=1,
            format_func=lambda x: str(x) if x < 99 else "Unlimited",
            help="Combined CALL + PUT count. Stops new entries once limit is reached.")

        # Legacy aliases (used in a few display strings only)
        min_quality = cascade_entry_threshold / 100.0
        require_iv  = True

        st.markdown("---")
        st.markdown("### 🗄️ Database")
        stats = db_stats()
        st.markdown(
            '<div class="info-box">Raw rows: <b>' + str(stats["raw_rows"]) + '</b><br>'
            'Days: <b>' + str(stats["days"]) + '</b><br>Signals: <b>' + str(stats["signals"]) + '</b><br>'
            'Trades: <b>' + str(stats["trades"]) + '</b></div>',
            unsafe_allow_html=True)
        if st.button("🗑️ Clear Trades", use_container_width=True):
            clear_trades(symbol); st.success("Trades cleared"); st.rerun()

    tab_fetch, tab_signals, tab_bt, tab_results, tab_trades = st.tabs([
        "1️⃣ Fetch Data","2️⃣ Compute Signals","3️⃣ Run Backtest","4️⃣ Results","5️⃣ Trade Log"])

    # ── Tab 1: Fetch ──────────────────────────────────────────────────────────
    with tab_fetch:
        st.markdown("### 📡 Fetch Historical Options Chain")
        st.markdown('<div class="info-box">Calls <code>POST /v2/charts/rollingoption</code> for each strike. '
                    "Data stored in SQLite. Already-fetched days skipped automatically. "
                    "Checkpoints after every strike — safe to resume if interrupted.</div>",
                    unsafe_allow_html=True)

        ckpt = checkpoint_status()
        if ckpt:
            st.markdown(
                '<div class="warn-box">⚡ Checkpoint — '
                + str(ckpt.get("trade_date","?")) + " | "
                + str(len(ckpt.get("completed_strikes",[]))) + " strikes done | "
                + str(len(ckpt.get("partial_rows",[]))) + " rows saved. Next fetch resumes.</div>",
                unsafe_allow_html=True)
            if st.button("🗑️ Discard Checkpoint"):
                clear_checkpoint(); st.rerun()

        trading_dates = get_trading_dates(d_start, d_end)
        done_dates    = get_fetch_log(symbol, expiry_code, expiry_flag)
        pending       = [d for d in trading_dates if d not in done_dates]
        c1,c2,c3 = st.columns(3)
        c1.metric("Total days",     len(trading_dates))
        c2.metric("Fetched",        len(done_dates))
        c3.metric("Pending",        len(pending))
        if pending:
            est = len(pending)*len(all_strikes)*2*0.35/60
            st.markdown(f'<div class="warn-box">Estimated time: ~{est:.1f} min</div>', unsafe_allow_html=True)

        fetch_btn = st.button(f"🚀 Fetch {len(pending)} Days", type="primary",
                              use_container_width=True, disabled=(len(pending)==0))
        if fetch_btn:
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
                    if n == 0:
                        log_lines.append(f"   ⚠️ 0 rows — check expiry_code or try adjacent date")
                except Exception as e:
                    log_lines.append(f"⚠️ {trade_date} — {e} (checkpoint saved)")
                    log_box.text("\n".join(log_lines[-15:]))
                    st.warning(f"Interrupted at {trade_date}. Restart to resume.")
                    break
                overall.progress((idx+1)/len(pending))
                log_box.text("\n".join(log_lines[-15:]))
            overall.empty(); day_bar.empty(); status.empty(); day_status.empty()

        # API Inspector
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
                    match = [t for t in ts_list if datetime.fromtimestamp(t,tz=pytz.UTC).astimezone(IST).date()==dbg_dt]
                    st.success(f"✅ ce keys: {list(ce.keys())} | {len(ts_list)} timestamps | {len(match)} match {dbg_date}")
                    if ts_list:
                        show=ts_list[:8]
                        st.dataframe(pd.DataFrame({
                            "epoch":show,
                            "UTC→IST":[datetime.fromtimestamp(t,tz=pytz.UTC).astimezone(IST).strftime("%Y-%m-%d %H:%M") for t in show],
                            "spot":ce.get("spot",[])[:8],"strike":ce.get("strike",[])[:8],
                            "oi":ce.get("oi",[])[:8],"iv":ce.get("iv",[])[:8],
                        }), hide_index=True, use_container_width=True)
                else:
                    st.error("Empty response. Check token or try different expiry_code.")

        # Strike type verifier
        with st.expander("🔎 Verify Strike Type Labels in DB"):
            st.markdown(
                '<div class="info-box">Run this to confirm your DB stores strike labels '
                'as ATM+1, ATM+2 etc. The strategy reads these labels directly.</div>',
                unsafe_allow_html=True)
            if st.button("Check strike_type values", key="stcheck"):
                con = sqlite3.connect(DB_PATH)
                df_st = pd.read_sql_query(
                    "SELECT DISTINCT strike_type FROM raw_chain WHERE symbol=? LIMIT 30",
                    con, params=(symbol,))
                con.close()
                if df_st.empty:
                    st.warning("No data in DB yet for this symbol.")
                else:
                    st.dataframe(df_st, hide_index=True)
                    expected = {"ATM","ATM+1","ATM+2","ATM+3","ATM-1","ATM-2","ATM-3"}
                    found    = set(df_st["strike_type"].tolist())
                    missing  = expected - found
                    if missing:
                        st.warning(f"⚠️ Missing expected labels: {missing}")
                    else:
                        st.success("✅ All required strike labels present.")

        if done_dates:
            st.markdown("#### ✅ Fetched Days")
            con=sqlite3.connect(DB_PATH)
            st.dataframe(pd.read_sql_query(
                "SELECT trade_date,rows_fetched,fetched_at FROM fetch_log WHERE symbol=? AND expiry_code=? AND expiry_flag=? AND status='ok' ORDER BY trade_date DESC",
                con,params=(symbol,expiry_code,expiry_flag)),
                use_container_width=True,height=300,hide_index=True)
            con.close()

    # ── Tab 2: Signals ────────────────────────────────────────────────────────
    with tab_signals:
        st.markdown("### ⚡ Compute Cascade Signals")
        st.markdown(
            '<div class="info-box">Optional for the custom strategy — signals table is used '
            'by the original cascade engine. The strike-level backtest (Tab 3) reads '
            'raw_chain directly and does not require this step.</div>',
            unsafe_allow_html=True)
        done_dates = get_fetch_log(symbol, expiry_code, expiry_flag)
        con=sqlite3.connect(DB_PATH)
        sig_d=pd.read_sql_query("SELECT DISTINCT trade_date FROM cascade_signals WHERE symbol=?",
                                con,params=(symbol,))
        con.close()
        sig_dates=set(sig_d["trade_date"].tolist()) if not sig_d.empty else set()
        pending_sig=[d for d in sorted(done_dates) if d not in sig_dates]
        c1,c2,c3=st.columns(3)
        c1.metric("Fetched days",len(done_dates))
        c2.metric("Signals ready",len(sig_dates))
        c3.metric("Pending",len(pending_sig))

        if st.button(f"⚡ Compute Signals for {len(pending_sig)} Days",type="primary",
                     use_container_width=True,disabled=(len(pending_sig)==0)):
            prog=st.progress(0); status=st.empty()
            for idx,trade_date in enumerate(pending_sig):
                status.text(f"Computing {trade_date} ({idx+1}/{len(pending_sig)})")
                df_day=load_raw_chain(symbol,trade_date,expiry_code,expiry_flag)
                if not df_day.empty:
                    save_signals(compute_signals_for_day(df_day,symbol,trade_date))
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

    # ── Tab 3: Run Backtest ───────────────────────────────────────────────────
    with tab_bt:
        st.markdown("### 🔄 Run Backtest")

        done_dates = get_fetch_log(symbol, expiry_code, expiry_flag)
        all_bt_dates = sorted(done_dates)

        st.markdown(
            '<div class="info-box">'
            '📈 <b>CALL entry</b>: IV Expanding + ATM+1 cascade &gt; <b>'
            + str(cascade_entry_threshold) + ' pts</b><br>'
            '&nbsp;&nbsp;&nbsp;&nbsp;Target: min(ATM+1+2+3 cumulative, <b>'
            + str(max_target_pts) + ' pts</b>) above entry<br>'
            '📉 <b>PUT entry</b>: IV Expanding + ATM-1 cascade &gt; <b>'
            + str(cascade_entry_threshold) + ' pts</b><br>'
            '&nbsp;&nbsp;&nbsp;&nbsp;Target: min(ATM-1+2+3 cumulative, <b>'
            + str(max_target_pts) + ' pts</b>) below entry<br>'
            '🛑 Stop Loss: <b>' + str(fixed_sl_pts) + ' pts</b> &nbsp;|&nbsp;'
            'Direction: <b>' + direction_filter + '</b> &nbsp;|&nbsp;'
            'Max Trades/Day: <b>' + str(max_trades_per_day) + '</b> &nbsp;|&nbsp;'
            'Days available: <b>' + str(len(all_bt_dates)) + '</b>'
            '</div>',
            unsafe_allow_html=True)

        if st.button(f"▶ Run Backtest on {len(all_bt_dates)} Days", type="primary",
                     use_container_width=True, disabled=(len(all_bt_dates)==0)):
            clear_trades(symbol)
            prog=st.progress(0); status=st.empty(); all_trades=[]
            for idx, td in enumerate(all_bt_dates):
                status.text(f"Simulating {td} ({idx+1}/{len(all_bt_dates)})")
                df_day = load_raw_chain(symbol, td, expiry_code, expiry_flag)
                if not df_day.empty:
                    iv_df = compute_iv_regime_series(df_day)
                    all_trades.extend(run_backtest_for_day(
                        signals=None,
                        symbol=symbol,
                        trade_date=td,
                        expiry_flag=expiry_flag,
                        min_quality=min_quality,
                        require_iv=require_iv,
                        cascade_entry_threshold=cascade_entry_threshold,
                        max_target_pts=max_target_pts,
                        fixed_sl_pts=fixed_sl_pts,
                        direction_filter=direction_filter,
                        max_trades_per_day=max_trades_per_day,
                        df_day=df_day,
                        iv_df=iv_df))
                prog.progress((idx+1)/len(all_bt_dates))
            save_trades(all_trades)
            prog.empty(); status.empty()
            st.success(f"✅ Done — {len(all_trades)} trades"); st.rerun()

    # ── Tab 4: Results ────────────────────────────────────────────────────────
    with tab_results:
        st.markdown("### 📊 Results")
        trades=load_trades(symbol)
        if trades.empty:
            st.info("No trades yet — run backtest in Tab 3 first.")
        else:
            m=compute_metrics(trades)
            cols=st.columns(5)
            kpis=[
                ("Total Trades",  str(m["total_trades"]),        "n"),
                ("Hit Rate",      str(m['hit_rate'])+'%',         "g" if m["hit_rate"]>=50 else "r"),
                ("Total Pts",     f"{m['total_pts']:+.1f}",       "g" if m["total_pts"]>0  else "r"),
                ("Profit Factor", f"{m['profit_factor']:.2f}x",  "g" if m["profit_factor"]>1 else "r"),
                ("Max Drawdown",  f"{m['max_drawdown']:.1f} pts","r"),
            ]
            for col,(label,val,cls) in zip(cols,kpis):
                col.markdown(
                    '<div class="metric-card"><div class="metric-val ' + cls + '">' + str(val) + '</div>'
                    '<div class="metric-lbl">' + label + '</div></div>',
                    unsafe_allow_html=True)

            cols2=st.columns(5)
            kpis2=[
                ("Sharpe",         f"{m['sharpe']:.2f}",           "g" if m["sharpe"]>1  else "n"),
                ("Sortino",        f"{m['sortino']:.2f}",          "g" if m["sortino"]>1 else "n"),
                ("Cascade Acc.",   f"{m['cascade_accuracy']:.1f}%","g" if m["cascade_accuracy"]>60 else "n"),
                ("Expiry Pts",     f"{m['expiry_pts']:+.1f}",      "g" if m["expiry_pts"]>0 else "r"),
                ("Non-Expiry Pts", f"{m['non_expiry_pts']:+.1f}",  "g" if m["non_expiry_pts"]>0 else "r"),
            ]
            for col,(label,val,cls) in zip(cols2,kpis2):
                col.markdown(
                    '<div class="metric-card"><div class="metric-val ' + cls + '">' + str(val) + '</div>'
                    + '<div class="metric-lbl">' + label + '</div></div>',
                    unsafe_allow_html=True)

            # Charts row 1
            c1,c2=st.columns([2,1])
            with c1: st.plotly_chart(equity_curve_chart(trades),use_container_width=True)
            with c2: st.plotly_chart(direction_breakdown_chart(trades),use_container_width=True)

            # Charts row 2
            c3,c4=st.columns(2)
            with c3: st.plotly_chart(quality_vs_pts_chart(trades),use_container_width=True)
            with c4: st.plotly_chart(exit_reason_chart(trades),use_container_width=True)

            # Charts row 3
            c5,c6=st.columns(2)
            with c5: st.plotly_chart(iv_regime_breakdown_chart(trades),use_container_width=True)

            # Monthly P&L
            st.markdown("#### 📅 Monthly P&L")
            trades["month"]=pd.to_datetime(trades["trade_date"]).dt.to_period("M").astype(str)
            mo=trades.groupby("month")["pts_captured"].agg(["sum","count"]).reset_index()
            fig_m=go.Figure(go.Bar(x=mo["month"],y=mo["sum"],
                marker_color=mo["sum"].apply(lambda x:"#10b981" if x>0 else "#ef4444"),
                text=mo["count"].apply(lambda x: f"{x} trades"),textposition="outside"))
            fig_m.update_layout(template="plotly_dark",height=300,
                paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(10,10,20,0.95)",
                margin=dict(l=0,r=0,t=10,b=0))
            st.plotly_chart(fig_m,use_container_width=True)

            st.download_button("📥 Download CSV",data=trades.to_csv(index=False),
                file_name=f"hedgex_{symbol}.csv",mime="text/csv",use_container_width=True)

    # ── Tab 5: Trade Log ──────────────────────────────────────────────────────
    with tab_trades:
        st.markdown("### 📋 Trade Log")
        trades=load_trades(symbol)
        if trades.empty:
            st.info("No trades yet.")
        else:
            # Show cascade cum pts (stored in bear_quality column)
            disp=trades[["trade_date","entry_time","exit_time","direction",
                          "entry_price","exit_price","pts_captured",
                          "cascade_target","cascade_stop","exit_reason",
                          "bear_quality","iv_regime","is_expiry_day"]].copy()
            disp.rename(columns={"bear_quality":"cascade_cum_pts"}, inplace=True)
            disp["entry_time"]=pd.to_datetime(disp["entry_time"]).dt.strftime("%H:%M")
            disp["exit_time"] =pd.to_datetime(disp["exit_time"]).dt.strftime("%H:%M")
            disp["is_expiry_day"]=disp["is_expiry_day"].map({0:"",1:"Expiry"})
            def rc(row):
                if row["pts_captured"]>0: return ["background-color:rgba(16,185,129,0.12)"]*len(row)
                return ["background-color:rgba(239,68,68,0.10)"]*len(row)
            st.dataframe(disp.style.apply(rc,axis=1),
                use_container_width=True,height=600,hide_index=True)

    st.markdown(
        '<div style="text-align:center;padding:16px;font-family:JetBrains Mono,monospace;'
        'font-size:0.68rem;color:rgba(255,255,255,0.2);">'
        'HedGEX Cascade Backtest · NYZTrade Analytics · Research purposes only</div>',
        unsafe_allow_html=True)

if __name__ == "__main__":
    main()
