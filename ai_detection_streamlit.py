import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

import streamlit as st
from streamlit_shap import st_shap
from ai_detection import ai_detector
from classify_news import classifier

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
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600&family=Inter:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

/* ── Base ── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
section.main {
    background: #f5f4f0 !important;
    color: #1a1a2e !important;
}
[data-testid="stHeader"]                { background: transparent !important; }
[data-testid="stSidebar"]               { background: #eeecea !important; }
[data-testid="stAppViewBlockContainer"] { background: #f5f4f0 !important; }
#MainMenu, footer                       { visibility: hidden; }
* { font-family: 'Inter', sans-serif; }

/* ── Top nav bar ── */
.topbar {
    background: #ffffff;
    border-bottom: 1px solid #dddbd6;
    padding: 14px 40px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin: -1rem -1rem 2rem -1rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.topbar-brand {
    font-family: 'Playfair Display', serif;
    font-size: 1.4rem;
    color: #1a1a2e;
    letter-spacing: 0.3px;
}
.topbar-tag {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    color: #9a9890;
    letter-spacing: 2px;
    text-transform: uppercase;
}

/* ── Hero ── */
.hero { padding: 12px 0 28px; }
.hero h1 {
    font-family: 'Playfair Display', serif;
    font-size: 3rem;
    font-weight: 600;
    color: #1a1a2e;
    line-height: 1.1;
    margin: 0 0 12px;
}
.hero p {
    font-size: 1rem;
    color: #6b6a66;
    font-weight: 400;
    max-width: 560px;
    line-height: 1.75;
    margin: 0;
}

/* ── Section labels ── */
.label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #9a9890;
    margin-bottom: 10px;
}

/* ── Divider ── */
.thin-rule {
    border: none;
    border-top: 1px solid #dddbd6;
    margin: 28px 0;
}

/* ── About text ── */
.about-body {
    font-size: 0.95rem;
    line-height: 1.8;
    color: #6b6a66;
    font-weight: 400;
}
.about-body strong { color: #1a1a2e; font-weight: 600; }

/* ── Textarea ── */
[data-testid="stTextArea"] textarea {
    background: #ffffff !important;
    border: 1px solid #dddbd6 !important;
    border-radius: 6px !important;
    color: #1a1a2e !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
    caret-color: #1a1a2e;
}
[data-testid="stTextArea"] textarea:focus {
    border-color: #1a3a6e !important;
    box-shadow: 0 0 0 2px rgba(26,58,110,0.12) !important;
}
[data-testid="stTextArea"] textarea::placeholder { color: #b8b6b0 !important; }

/* ── Button ── */
[data-testid="stButton"] > button[kind="primary"] {
    background: #1a3a6e !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    padding: 12px 32px !important;
    font-weight: 500 !important;
    transition: background 0.2s ease, box-shadow 0.2s ease !important;
    box-shadow: 0 2px 6px rgba(26,58,110,0.18) !important;
}
[data-testid="stButton"] > button[kind="primary"]:hover {
    background: #0f2a54 !important;
    box-shadow: 0 4px 10px rgba(26,58,110,0.25) !important;
}

/* ── Verdict card ── */
.verdict-wrap {
    border-radius: 8px;
    padding: 28px 32px;
    margin: 4px 0 20px;
    border: 1px solid transparent;
}
.verdict-wrap.ai-card {
    background: #fff5f5;
    border-color: #f0c0bb;
}
.verdict-wrap.human-card {
    background: #f2faf5;
    border-color: #a8dbb8;
}
.verdict-icon   { font-size: 2rem; margin-bottom: 10px; }
.verdict-title  {
    font-family: 'Playfair Display', serif;
    font-size: 1.75rem;
    color: #1a1a2e;
    margin: 0 0 6px;
}
.verdict-wrap.ai-card    .verdict-title { color: #b91c1c; }
.verdict-wrap.human-card .verdict-title { color: #166534; }
.verdict-sub {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: #9a9890;
    letter-spacing: 1px;
}
.conf-bar-track {
    background: #e8e6e1;
    border-radius: 100px;
    height: 5px;
    margin-top: 18px;
    overflow: hidden;
}
.conf-bar-fill-ai    { height: 100%; border-radius: 100px; background: #dc2626; }
.conf-bar-fill-human { height: 100%; border-radius: 100px; background: #16a34a; }

/* ── Topic tags ── */
.topic-tag-row {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 4px;
}
.topic-tag {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    background: #eaf0fb;
    border: 1px solid #bdd0ef;
    color: #1a3a6e;
    border-radius: 4px;
    padding: 6px 14px;
}
.topic-none {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: #9a9890;
    padding: 8px 0;
}

/* ── Topic confidence bar ── */
.topic-bar-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 10px;
}
.topic-bar-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: #3a3a4a;
    width: 220px;
    flex-shrink: 0;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.topic-bar-track {
    flex: 1;
    background: #e8e6e1;
    border-radius: 100px;
    height: 5px;
    overflow: hidden;
}
.topic-bar-fill {
    height: 100%;
    border-radius: 100px;
    background: #1a3a6e;
}
.topic-bar-pct {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: #9a9890;
    width: 40px;
    text-align: right;
    flex-shrink: 0;
}

/* ── SHAP info box ── */
.shap-info {
    background: #ffffff;
    border: 1px solid #dddbd6;
    border-radius: 6px;
    padding: 14px 18px;
    font-size: 0.83rem;
    color: #6b6a66;
    line-height: 1.65;
    margin-top: 12px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.shap-info strong { color: #1a1a2e; }

/* ── Checkbox ── */
[data-testid="stCheckbox"] label {
    font-size: 0.88rem !important;
    color: #6b6a66 !important;
}

/* ── Warning / info banners ── */
[data-testid="stAlert"] {
    background: #ffffff !important;
    border: 1px solid #dddbd6 !important;
    color: #6b6a66 !important;
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

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — AI Detection
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='label'>AI Detection — Article Text</div>", unsafe_allow_html=True)
ai_text = st.text_area(
    label="ai_article",
    label_visibility="collapsed",
    placeholder="Paste a news article or excerpt here…",
    height=220,
    key="ai_input",
)
run_ai = st.button("Analyse Text", type="primary", key="run_ai")

if run_ai and ai_text.strip():
    st.markdown("<div class='thin-rule'></div>", unsafe_allow_html=True)

    with st.spinner("Analysing…"):
        detector   = ai_detector(ai_text)
        prediction = detector.predict()
        confidence = detector.predict_proba()

    # Parse confidence
    try:
        conf_num = float(str(confidence).strip().replace("%", ""))
        if conf_num <= 1.0:
            conf_num *= 100
        conf_display = f"{conf_num:.1f}%"
        bar_pct      = f"{min(conf_num, 100):.1f}%"
    except (ValueError, TypeError):
        conf_display = str(confidence)
        bar_pct      = "0%"

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

    st.markdown("""
    <div class="shap-info" style="margin-top: 0;">
        <strong>What does this confidence score mean?</strong> This percentage reflects
        how certain the model is in its verdict. A score above <strong>80%</strong>
        indicates high certainty — the text closely matches patterns typical of that
        category. Scores between <strong>50–80%</strong> suggest a probable but
        less definitive result, and anything near <strong>50%</strong> means the
        model is close to uncertain — treat those results with caution and apply
        your own judgement.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='thin-rule'></div>", unsafe_allow_html=True)

    st.markdown("<div class='label'>Explanation</div>", unsafe_allow_html=True)
    explain_plot = detector.explain()
    st_shap(explain_plot)
    st.markdown("""
    <div class="shap-info">
        <strong>How to read this:</strong> Red bars push the prediction toward
        <em>human-written</em>. Blue bars push toward <em>AI-generated</em>.
        Bar length reflects each feature's influence on the result.
    </div>
    """, unsafe_allow_html=True)

elif run_ai and not ai_text.strip():
    st.warning("Please paste some article text before running the analysis.")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Topic Classification
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='thin-rule'></div>", unsafe_allow_html=True)

st.markdown("""
<div class="hero" style="padding: 0 0 20px;">
    <h1 style="font-size:2.2rem;">What is this<br>article about?</h1>
    <p>Paste a news article below and the classifier will predict which
    topics it belongs to across up to 10 categories.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("<div class='label'>Topic Classification — Article Text</div>", unsafe_allow_html=True)
topic_text = st.text_area(
    label="topic_article",
    label_visibility="collapsed",
    placeholder="Paste a news article or excerpt here…",
    height=220,
    key="topic_input",
)
run_topic = st.button("Classify Topics", type="primary", key="run_topic")

if run_topic and topic_text.strip():
    st.markdown("<div class='thin-rule'></div>", unsafe_allow_html=True)

    with st.spinner("Classifying…"):
        clf        = classifier(text=topic_text)
        labels_df  = clf.predict()
        proba_df   = clf.predict_proba()

    topics_found = list(labels_df.columns)

    # ── Predicted topic tags ──────────────────────────────────────────────────
    st.markdown("<div class='label'>Predicted Topics</div>", unsafe_allow_html=True)

    if topics_found:
        tags_html = "".join(
            f'<span class="topic-tag">{t}</span>' for t in topics_found
        )
        st.markdown(
            f'<div class="topic-tag-row">{tags_html}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="topic-none">No topics matched the confidence threshold.</div>',
            unsafe_allow_html=True,
        )

    # ── Confidence scores ─────────────────────────────────────────────────────
    if not proba_df.empty:
        st.markdown("<div class='thin-rule'></div>", unsafe_allow_html=True)
        st.markdown("<div class='label'>Confidence Scores</div>", unsafe_allow_html=True)

        # Sort topics by confidence score descending
        sorted_topics = proba_df.iloc[0].sort_values(ascending=False).index

        bars_html = ""
        for topic in sorted_topics:
            score   = float(proba_df[topic].iloc[0])
            pct     = f"{score * 100:.1f}%"
            fill_w  = f"{min(score * 100, 100):.1f}%"

            if score >= 0.70:
                fit_label = "Strong fit"
                fit_color = "#81c995"
            elif score >= 0.40:
                fit_label = "Partial fit"
                fit_color = "#f5c842"
            else:
                fit_label = "Weak fit"
                fit_color = "#f28b82"

            bars_html += f"""
            <div class="topic-bar-row">
                <div class="topic-bar-label">{topic}</div>
                <div class="topic-bar-track">
                    <div class="topic-bar-fill" style="width:{fill_w};"></div>
                </div>
                <div class="topic-bar-pct">{pct}</div>
                <div style="font-family:'DM Mono',monospace;font-size:0.68rem;
                            letter-spacing:1px;text-transform:uppercase;
                            color:{fit_color};width:80px;text-align:right;
                            flex-shrink:0;">{fit_label}</div>
            </div>"""

        st.markdown(bars_html, unsafe_allow_html=True)

        st.markdown("""
        <div class="shap-info" style="margin-top: 16px;">
            <strong>What do these confidence scores mean?</strong> Each bar shows
            how strongly the model associates the article with that topic category.
            Scores above <strong>70%</strong> indicate the topic is a clear and
            dominant theme. Scores between <strong>40–70%</strong> suggest the
            topic is present but secondary or mixed with other themes. Scores
            below <strong>40%</strong> mean the topic is weakly represented —
            only topics that pass the detection threshold appear as predicted tags above.
        </div>
        """, unsafe_allow_html=True)

elif run_topic and not topic_text.strip():
    st.warning("Please paste some article text before classifying.")

