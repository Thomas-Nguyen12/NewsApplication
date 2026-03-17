import streamlit as st
from streamlit_shap import st_shap
from ai_detection import ai_detector
import shap

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NewsShield",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

/* ── Root & background ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0d0d0d;
    color: #e8e0d0;
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] { background: #111; }

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer { visibility: hidden; }

/* ── Global font ── */
* { font-family: 'IBM Plex Sans', sans-serif; }

/* ── Masthead ── */
.masthead {
    border-top: 4px solid #e8e0d0;
    border-bottom: 1px solid #333;
    padding: 28px 0 20px;
    margin-bottom: 0;
    display: flex;
    align-items: baseline;
    gap: 18px;
}
.masthead-title {
    font-family: 'Playfair Display', serif;
    font-weight: 900;
    font-size: 3.4rem;
    letter-spacing: -1px;
    color: #e8e0d0;
    margin: 0;
    line-height: 1;
}
.masthead-version {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: #666;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 4px;
}
.masthead-tagline {
    font-family: 'IBM Plex Sans', sans-serif;
    font-weight: 300;
    font-size: 0.9rem;
    color: #888;
    letter-spacing: 0.5px;
    margin-top: 6px;
}

/* ── Divider ── */
.rule { border: none; border-top: 1px solid #2a2a2a; margin: 24px 0; }

/* ── About blurb ── */
.about-text {
    font-size: 0.9rem;
    line-height: 1.75;
    color: #999;
    font-weight: 300;
    border-left: 2px solid #333;
    padding-left: 14px;
}
.about-text strong { color: #ccc; font-weight: 500; }

/* ── Section labels ── */
.section-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #555;
    margin-bottom: 10px;
}

/* ── Text input override ── */
[data-testid="stTextArea"] textarea,
[data-testid="stTextInput"] input {
    background-color: #161616 !important;
    border: 1px solid #2e2e2e !important;
    border-radius: 2px !important;
    color: #e8e0d0 !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 0.92rem !important;
    caret-color: #e8e0d0;
}
[data-testid="stTextArea"] textarea:focus,
[data-testid="stTextInput"] input:focus {
    border-color: #666 !important;
    box-shadow: none !important;
}

/* ── Button ── */
[data-testid="stButton"] > button {
    background: #e8e0d0 !important;
    color: #0d0d0d !important;
    border: none !important;
    border-radius: 2px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.78rem !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    padding: 10px 28px !important;
    font-weight: 500 !important;
    transition: opacity 0.15s ease !important;
}
[data-testid="stButton"] > button:hover { opacity: 0.85 !important; }

/* ── Verdict card ── */
.verdict-card {
    border: 1px solid #2a2a2a;
    background: #111;
    padding: 24px 28px;
    border-radius: 2px;
    position: relative;
    overflow: hidden;
}
.verdict-card.ai::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 4px; height: 100%;
    background: #e05252;
}
.verdict-card.human::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 4px; height: 100%;
    background: #52c97a;
}
.verdict-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #555;
    margin-bottom: 8px;
}
.verdict-result {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 700;
    line-height: 1;
    margin-bottom: 6px;
}
.verdict-result.ai-text { color: #e05252; }
.verdict-result.human-text { color: #52c97a; }
.verdict-confidence {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
    color: #666;
}
.confidence-val { color: #aaa; font-weight: 500; }

/* ── Progress bar (confidence meter) ── */
.conf-bar-bg {
    background: #1e1e1e;
    border-radius: 1px;
    height: 4px;
    margin-top: 12px;
    overflow: hidden;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 1px;
    transition: width 0.6s ease;
}

/* ── Checkbox ── */
[data-testid="stCheckbox"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.8rem !important;
    color: #888 !important;
    letter-spacing: 1px;
}

/* ── Info strip ── */
.info-strip {
    background: #111;
    border: 1px solid #2a2a2a;
    border-radius: 2px;
    padding: 14px 20px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: #555;
    letter-spacing: 0.5px;
}
.info-strip span { color: #888; }

/* ── SHAP explanation container ── */
.shap-wrapper {
    background: #111;
    border: 1px solid #2a2a2a;
    border-radius: 2px;
    padding: 20px;
    margin-top: 8px;
}
</style>
""", unsafe_allow_html=True)

# ── Masthead ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="masthead">
  <div>
    <div class="masthead-version">v1.0 · AI Content Detection</div>
    <div class="masthead-title">NewsShield</div>
    <div class="masthead-tagline">Detecting machine-generated text in journalism &amp; media</div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div class='rule'></div>", unsafe_allow_html=True)

# ── Two-column layout: about + image ─────────────────────────────────────────
col_img, col_about = st.columns([1, 2], gap="large")

with col_img:
    st.image("ny_times.jpg", use_container_width=True)

with col_about:
    st.markdown("""
    <div class="section-label">About</div>
    <div class="about-text">
        NewsShield analyses news articles and editorial content to identify text
        generated by <strong>Large Language Models</strong> such as ChatGPT, Google Gemini,
        and similar systems.<br><br>
        As AI-generated misinformation becomes harder to distinguish from genuine
        reporting, tools like this help newsrooms verify authenticity and protect
        readers from synthetic content masquerading as real journalism.
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div class='rule'></div>", unsafe_allow_html=True)

# ── Input ─────────────────────────────────────────────────────────────────────
st.markdown("<div class='section-label'>Input</div>", unsafe_allow_html=True)

text = st.text_area(
    label="Article text",
    value="",
    placeholder="Paste a news article or excerpt here…",
    height=200,
    label_visibility="collapsed",
)

run = st.button("Analyse Text")

# ── Results ───────────────────────────────────────────────────────────────────
if run and text.strip():
    st.markdown("<div class='rule'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-label'>Results</div>", unsafe_allow_html=True)

    with st.spinner("Analysing…"):
        detector = ai_detector(text)
        prediction = detector.predict()
        confidence = detector.predict_proba()

    # Determine verdict styling
    is_ai = "ai" in prediction.lower() or "generated" in prediction.lower()
    card_class = "ai" if is_ai else "human"
    result_class = "ai-text" if is_ai else "human-text"
    bar_color = "#e05252" if is_ai else "#52c97a"

    # Try to extract a numeric confidence value for the bar
    try:
        conf_num = float(str(confidence).strip().replace("%", ""))
        if conf_num <= 1.0:
            conf_num *= 100
        bar_width = f"{min(conf_num, 100):.1f}%"
        conf_display = f"{conf_num:.1f}%"
    except (ValueError, TypeError):
        bar_width = "0%"
        conf_display = str(confidence)

    st.markdown(f"""
    <div class="verdict-card {card_class}">
        <div class="verdict-label">Verdict</div>
        <div class="verdict-result {result_class}">{prediction}</div>
        <div class="verdict-confidence">
            Confidence: <span class="confidence-val">{conf_display}</span>
        </div>
        <div class="conf-bar-bg">
            <div class="conf-bar-fill"
                 style="width:{bar_width}; background:{bar_color};"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='rule'></div>", unsafe_allow_html=True)

    # ── Explanation ───────────────────────────────────────────────────────────
    st.markdown("<div class='section-label'>Explanation</div>", unsafe_allow_html=True)
    explain = st.checkbox("Show SHAP feature attribution", value=False)

    if explain:
        with st.spinner("Computing explanation…"):
            explain_plot = detector.explain()
        st.markdown("<div class='shap-wrapper'>", unsafe_allow_html=True)
        st_shap(explain_plot, height=350)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class="info-strip" style="margin-top:12px;">
            <span>ℹ</span>&nbsp; Red bars push the prediction toward <span>AI-generated</span>.
            Blue bars push toward <span>human-written</span>.
            Bar length indicates feature importance.
        </div>
        """, unsafe_allow_html=True)

elif run and not text.strip():
    st.warning("Please enter some article text before analysing.")