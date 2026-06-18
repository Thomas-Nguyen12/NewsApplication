# This will be a streamlit dashboard of vinfast predictions
# 
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = (BASE_DIR / "scripts").resolve()

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))



from preprocessing import lemmatize 
import preprocessing 

from sentiment_analyser import analyser
"""
Vinfast Stock Performance Dashboard
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os
news_api_key = st.secrets['news_api_key']
eod_key = st.secrets['fin_historical_data']

# checking the datee and time for refresh 



#
# I can now checkingall the sentiment analyser module using mod.analyser 
def sentiment_analysis(text:str) -> int: 
    text_to_analyse = analyser(text) 
    prediction = text_to_analyse.predict()[0]
    return prediction 






# day, month and year 
#
day = datetime.now().day 
month = datetime.now().month 
year = datetime.now().year 

# collecting historical stock data:wq
#
st.cache_data
def load_news_eod(): 
    # depending on the results from vinfast_data_collection.py, the eod_data will either 
    # be an updated dataset or not 
    #
    from scripts.vinfast_data_collection import eod_data, vinfast_news
    print ("Analysing the sentiment of the news...") 
    try: 
        # analysing the content column 
        # This may be a little slow
        vinfast_news['content_sentiment'] = vinfast_news['content'].apply(sentiment_analysis)
        print (vinfast_news.head()) 
    except Exception as news_e: 
        print (f"There was an exception: {news_e}") 
    finally: 
        print ("Analysis complete!") 


    return eod_data, vinfast_news  



print ("Loading the historical data...") 


eod_data, vinfast_news = load_news_eod() 

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VFS · Stock Dashboard",
    page_icon="📈",
    layout="wide",
)

# --------> Testing the display of api data 
st.header("EOD vinfast data") 
st.header("VINFAST news") 




# ── Custom styling ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Dark Vietnamese-flag-inspired palette */
    :root {
        --vf-red:    #D0222A;
        --vf-gold:   #F5C842;
        --vf-dark:   #0F1117;
        --vf-card:   #1A1D27;
        --vf-muted:  #8892A4;
    }

    .stApp { background-color: #0F1117; }

    /* Metric cards */
    .metric-card {
        background: #1A1D27;
        border: 1px solid #2A2D3A;
        border-radius: 10px;
        padding: 18px 22px;
        margin-bottom: 4px;
    }
    .metric-label {
        color: #8892A4;
        font-size: 12px;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 6px;
    }
    .metric-value {
        color: #F0F2F6;
        font-size: 28px;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    .metric-delta-pos { color: #3DD68C; font-size: 14px; }
    .metric-delta-neg { color: #D0222A; font-size: 14px; }
</style>
""", unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv("data/vinfast_data_cleaned.csv")
df.drop(['Unnamed: 0'], axis=1, inplace=True)

df["date"] = pd.to_datetime(df["date"])
df.sort_values("date", ascending=True, inplace=True)

df = pd.concat([df, eod_data], axis=0)
df['date'] = pd.to_datetime(df['date'])

# dropping the duplicates 
df = df.drop_duplicates(subset=['date']) 
print ("------------------")
print (df)
# the new df dataframe will be up to date 


# ── Header ────────────────────────────────────────────────────────────────────
col_title, col_ticker = st.columns([4, 1])
with col_title:
    st.markdown("## 🚗  VinFast Auto · **VFS**")
    st.caption("NASDAQ  ·  Historical stock data dashboard")
with col_ticker:
    latest = df.iloc[-1]
    prev   = df.iloc[-2]
    delta  = latest["close"] - prev["close"]
    pct    = delta / prev["close"] * 100
    color  = "normal" if delta >= 0 else "inverse"
    st.metric("Last Close", f"${latest['close']:.2f}",
              delta=f"{delta:+.2f}  ({pct:+.2f}%)", delta_color=color)

st.divider()

# ── Date range slider ─────────────────────────────────────────────────────────
min_date = df["date"].min().to_pydatetime()
max_date = df["date"].max().to_pydatetime()

date_range = st.slider(
    "Date range",
    min_value=min_date,
    max_value=max_date,
    value=(datetime(2025, 1, 1), max_date),
    format="MMM DD, YYYY",
)

mask = (df["date"] >= date_range[0]) & (df["date"] <= date_range[1])
dff  = df[mask].copy()

if dff.empty:
    st.warning("No data in selected range.")
    st.stop()

# ── Helper ────────────────────────────────────────────────────────────────────
def pct_change(col: str) -> float:
    start = dff.iloc[0][col]
    end   = dff.iloc[-1][col]
    return (end - start) / start * 100 if start else 0.0

# ── Key metrics ───────────────────────────────────────────────────────────────
st.subheader("Key Metrics")

m1, m2, m3, m4 = st.columns(4)

def render_metric(container, label, value, delta_pct):
    arrow = "▲" if delta_pct >= 0 else "▼"
    cls   = "metric-delta-pos" if delta_pct >= 0 else "metric-delta-neg"
    container.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">{label}</div>
      <div class="metric-value">${value:.2f}</div>
      <span class="{cls}">{arrow} {abs(delta_pct):.2f}% over period</span>
    </div>
    """, unsafe_allow_html=True)

render_metric(m1, "Open",           dff.iloc[-1]["open"],           pct_change("open"))
render_metric(m2, "Close",          dff.iloc[-1]["close"],          pct_change("close"))
render_metric(m3, "Adj. Close",     dff.iloc[-1]["adjusted_close"], pct_change("adjusted_close"))

# Volume as raw number (not price)
vol_start = dff.iloc[0]["volume"]
vol_end   = dff.iloc[-1]["volume"]
vol_delta = (vol_end - vol_start) / vol_start * 100 if vol_start else 0
arrow = "▲" if vol_delta >= 0 else "▼"
cls   = "metric-delta-pos" if vol_delta >= 0 else "metric-delta-neg"
m4.markdown(f"""
<div class="metric-card">
  <div class="metric-label">Volume (latest)</div>
  <div class="metric-value">{int(vol_end):,}</div>
  <span class="{cls}">{arrow} {abs(vol_delta):.2f}% over period</span>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Candlestick chart ─────────────────────────────────────────────────────────
st.subheader("Historical Performance")

fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=dff["date"],
    open=dff["open"],
    high=dff["high"],
    low=dff["low"],
    close=dff["close"],
    name="OHLC",
    increasing_line_color="#3DD68C",
    decreasing_line_color="#D0222A",
))

# 20-day moving average overlay
dff["MA20"] = dff["close"].rolling(20).mean()
fig.add_trace(go.Scatter(
    x=dff["date"], y=dff["MA20"],
    mode="lines",
    name="20-day MA",
    line=dict(color="#F5C842", width=1.5, dash="dot"),
))

fig.update_layout(
    paper_bgcolor="#0F1117",
    plot_bgcolor="#0F1117",
    font=dict(color="#8892A4"),
    xaxis=dict(gridcolor="#1A1D27", rangeslider_visible=False),
    yaxis=dict(gridcolor="#1A1D27", title="Price (USD)"),
    legend=dict(bgcolor="#1A1D27", bordercolor="#2A2D3A"),
    margin=dict(l=0, r=0, t=10, b=0),
    height=420,
)

st.plotly_chart(fig, use_container_width=True)

# ── Volume bar chart ──────────────────────────────────────────────────────────
st.subheader("Trading Volume")

vol_fig = px.bar(
    dff, x="date", y="volume",
    color_discrete_sequence=["#2A4D8F"],
)
vol_fig.update_layout(
    paper_bgcolor="#0F1117",
    plot_bgcolor="#0F1117",
    font=dict(color="#8892A4"),
    xaxis=dict(gridcolor="#1A1D27"),
    yaxis=dict(gridcolor="#1A1D27", title="Volume"),
    margin=dict(l=0, r=0, t=10, b=0),
    height=260,
    showlegend=False,
)
st.plotly_chart(vol_fig, use_container_width=True)

# ── Raw data table (collapsible) ──────────────────────────────────────────────
with st.expander("View raw data"):
    st.dataframe(
        dff.set_index("date").style.format({
            "open": "${:.2f}", "high": "${:.2f}", "low": "${:.2f}",
            "close": "${:.2f}", "adjusted_close": "${:.2f}",
            "volume": "{:,.0f}",
        }),
        use_container_width=True,
    )
