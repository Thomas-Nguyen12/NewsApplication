import streamlit as st
from streamlit_shap import st_shap
from ai_detection import ai_detector

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NewsShield",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500&family=DM+Mono:wght@400;500&display=swap');

/* ── Base ── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
section.main {
    background: #0f1117 !important;
    color: #e2e2e2 !important;
}
[data-testid="stHeader"]       { background: transparent !important; }
[data-testid="stSidebar"]      { background: #161820 !important; }
[data-testid="stAppViewBlockContainer"] { background: #0f1117 !important; }
#MainMenu, footer              { visibility: hidden; }
* { font-family: 'DM Sans', sans-serif; }

/* ── Top nav bar ── */
.topbar {
    background: #161820;
    border-bottom: 1px solid #2a2d3a;
    padding: 14px 40px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin: -1rem -1rem 2rem -1rem;
}
.topbar-brand {
    font-family: 'DM Serif Display', serif;
    font-size: 1.4rem;
    color: #e2e2e2;
    letter-spacing: 0.5px;
}
.topbar-tag {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: #4a4f66;
    letter-spacing: 2px;
    text-transform: uppercase;
}

/* ── Hero ── */
.hero { padding: 12px 0 28px; }
.hero h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 3rem;
    font-weight: 400;
    color: #e2e2e2;
    line-height: 1.1;
    margin: 0 0 12px;
}
.hero p {
    font-size: 1rem;
    color: #7a7f99;
    font-weight: 300;
    max-width: 560px;
    line-height: 1.75;
    margin: 0;
}

/* ── Section labels ── */
.label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #4a4f66;
    margin-bottom: 10px;
}

/* ── Divider ── */
.thin-rule {
    border: none;
    border-top: 1px solid #1e2130;
    margin: 28px 0;
}

/* ── About text ── */
.about-body {
    font-size: 0.95rem;
    line-height: 1.8;
    color: #7a7f99;
    font-weight: 300;
}
.about-body strong { color: #c8c8d8; font-weight: 500; }

/* ── Textarea ── */
[data-testid="stTextArea"] textarea {
    background: #161820 !important;
    border: 1px solid #2a2d3a !important;
    border-radius: 6px !important;
    color: #e2e2e2 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    box-shadow: none !important;
    caret-color: #e2e2e2;
}
[data-testid="stTextArea"] textarea:focus {
    border-color: #4a5080 !important;
    box-shadow: 0 0 0 2px rgba(74,80,128,0.25) !important;
}
[data-testid="stTextArea"] textarea::placeholder { color: #3a3f55 !important; }

/* ── Button ── */
[data-testid="stButton"] > button[kind="primary"] {
    background: #3d4270 !important;
    color: #e2e2e2 !important;
    border: 1px solid #4a5080 !important;
    border-radius: 6px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    padding: 12px 32px !important;
    font-weight: 500 !important;
    transition: background 0.2s ease !important;
}
[data-testid="stButton"] > button[kind="primary"]:hover {
    background: #4a5080 !important;
}

/* ── Verdict card ── */
.verdict-wrap {
    border-radius: 8px;
    padding: 28px 32px;
    margin: 4px 0 20px;
    border: 1px solid transparent;
}
.verdict-wrap.ai-card {
    background: #1f1218;
    border-color: #5c2a2a;
}
.verdict-wrap.human-card {
    background: #0e1f18;
    border-color: #1e5c3a;
}
.verdict-icon   { font-size: 2rem; margin-bottom: 10px; }
.verdict-title  {
    font-family: 'DM Serif Display', serif;
    font-size: 1.75rem;
    color: #e2e2e2;
    margin: 0 0 6px;
}
.verdict-wrap.ai-card   .verdict-title { color: #f28b82; }
.verdict-wrap.human-card .verdict-title { color: #81c995; }
.verdict-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    color: #4a4f66;
    letter-spacing: 1px;
}
.conf-bar-track {
    background: #1e2130;
    border-radius: 100px;
    height: 5px;
    margin-top: 18px;
    overflow: hidden;
}
.conf-bar-fill-ai    { height: 100%; border-radius: 100px; background: #c0392b; }
.conf-bar-fill-human { height: 100%; border-radius: 100px; background: #27ae60; }

/* ── SHAP info box ── */
.shap-info {
    background: #161820;
    border: 1px solid #2a2d3a;
    border-radius: 6px;
    padding: 14px 18px;
    font-size: 0.83rem;
    color: #7a7f99;
    line-height: 1.65;
    margin-top: 12px;
}
.shap-info strong { color: #c8c8d8; }

/* ── Checkbox ── */
[data-testid="stCheckbox"] label {
    font-size: 0.88rem !important;
    color: #7a7f99 !important;
}

/* ── Warning / info banners ── */
[data-testid="stAlert"] {
    background: #161820 !important;
    border: 1px solid #2a2d3a !important;
    color: #7a7f99 !important;
    border-radius: 6px !important;
}
</style>
""", unsafe_allow_html=True)

# ── Top nav bar ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="topbar">
    <span class="topbar-brand">🛡️ NewsShield</span>
    <span class="topbar-tag">AI Content Detection · v1.0</span>
</div>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>Is this article<br>written by AI?</h1>
    <p>Paste any news article below. NewsShield uses machine learning to detect
    whether the text was written by a human or generated by an LLM such as
    ChatGPT or Google Gemini.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("<div class='thin-rule'></div>", unsafe_allow_html=True)

# ── About + image ─────────────────────────────────────────────────────────────
col_img, col_about = st.columns([1, 2], gap="large")

with col_img:
    st.image("ny_times.jpg", use_container_width=True)

with col_about:
    st.markdown("<div class='label'>About</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='about-body'>
        NewsShield analyses editorial content to identify text generated by
        <strong>Large Language Models</strong>. As synthetic text becomes harder
        to distinguish from genuine reporting, this tool helps newsrooms verify
        authenticity and protect readers from AI-generated misinformation.
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div class='thin-rule'></div>", unsafe_allow_html=True)

# ── Input ─────────────────────────────────────────────────────────────────────
st.markdown("<div class='label'>Article Text</div>", unsafe_allow_html=True)
text = st.text_area(
    label="article",
    label_visibility="collapsed",
    placeholder="Paste a news article or excerpt here…",
    height=220,
)
run = st.button("Analyse Text", type="primary")

# ── Results ───────────────────────────────────────────────────────────────────
if run and text.strip():
    st.markdown("<div class='thin-rule'></div>", unsafe_allow_html=True)

    with st.spinner("Analysing…"):
        detector = ai_detector(text)
        prediction = detector.predict()
        confidence = detector.predict_proba()

    # Parse confidence
    try:
        conf_num = float(str(confidence).strip().replace("%", ""))
        if conf_num <= 1.0:
            conf_num *= 100
        conf_display = f"{conf_num:.1f}%"
        bar_pct = f"{min(conf_num, 100):.1f}%"
    except (ValueError, TypeError):
        conf_display = str(confidence)
        bar_pct = "0%"

    # Determine verdict
    is_ai      = "human" not in prediction.lower()
    card_class = "ai-card"          if is_ai else "human-card"
    bar_class  = "conf-bar-fill-ai" if is_ai else "conf-bar-fill-human"
    icon       = "🤖"               if is_ai else "✍️"
    title      = "AI-Generated"     if is_ai else "Human-Written"
    sub        = f"Confidence · {conf_display}"

    st.markdown("<div class='label'>Verdict</div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="verdict-wrap {card_class}">
        <div class="verdict-icon">{icon}</div>
        <div class="verdict-title">{title}</div>
        <div class="verdict-sub">{sub}</div>
        <div class="conf-bar-track">
            <div class="{bar_class}" style="width:{bar_pct};"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='thin-rule'></div>", unsafe_allow_html=True)

    # ── Explanation ───────────────────────────────────────────────────────────
    st.markdown("<div class='label'>Explanation</div>", unsafe_allow_html=True)
    
        
    explain_plot = detector.explain()  # compute inside spinner

    # ✅ Render OUTSIDE the spinner block so it isn't cleared
    st_shap(explain_plot)
    st.markdown("""
    <div class="shap-info">
        <strong>How to read this:</strong> Red bars push the prediction toward
        <em>human-written</em>. Blue bars push toward <em>AI-generated</em>.
        Bar length reflects each feature's influence on the result.
    </div>
    """, unsafe_allow_html=True)

elif run and not text.strip():
    st.warning("Please paste some article text before running the analysis.")