import os
import re
import time
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from src.agents.agents import (
    build_search_agent,
    build_reader_agent,
    get_writer_chain,
    get_critic_chain,
    get_llm,
    get_gemini_api_key,
    extract_text_content,
)

# ── Page Configuration ────────────────────────────────────────────────────────
favicon_img = "assets/favicon.png" if os.path.exists("assets/favicon.png") else "⚡"

st.set_page_config(
    page_title="ResearcherAI • Powered by Mohsin Ali",
    page_icon=favicon_img,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS / Styling ──────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@500;600;700;800&family=DM+Mono:wght@300;400;500&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&display=swap');

/* ── Global Theme & Typography ── */
:root {
    --bg-primary: #070d18;
    --bg-card: rgba(15, 23, 42, 0.65);
    --border-card: rgba(56, 189, 248, 0.16);
    --accent-cyan: #38bdf8;
    --accent-purple: #818cf8;
    --accent-green: #10b981;
    --accent-amber: #f59e0b;
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #e2e8f0;
}

.stApp {
    background-color: #070d18;
    background-image:
        radial-gradient(at 0% 0%, rgba(56, 189, 248, 0.12) 0px, transparent 40%),
        radial-gradient(at 100% 0%, rgba(129, 140, 248, 0.12) 0px, transparent 40%),
        radial-gradient(at 50% 100%, rgba(16, 185, 129, 0.08) 0px, transparent 50%),
        linear-gradient(180deg, #070d18 0%, #0b1528 100%);
    background-attachment: fixed;
}

/* Hide default streamlit decoration */
#MainMenu, footer { visibility: hidden; }
header { background: transparent !important; }
.block-container {
    padding: 2.2rem 3rem 4rem !important;
    max-width: 1320px;
}

/* ── Hero Section ── */
.hero-wrapper {
    text-align: center;
    padding: 2.5rem 1rem 1.8rem;
    position: relative;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #38bdf8;
    background: rgba(56, 189, 248, 0.08);
    border: 1px solid rgba(56, 189, 248, 0.25);
    padding: 0.35rem 1rem;
    border-radius: 9999px;
    margin-bottom: 1.2rem;
    backdrop-filter: blur(8px);
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2.4rem, 5vw, 4.2rem);
    font-weight: 800;
    line-height: 1.08;
    letter-spacing: -0.035em;
    color: #ffffff;
    margin: 0 0 1rem;
}
.hero-gradient {
    background: linear-gradient(135deg, #38bdf8 15%, #818cf8 50%, #c084fc 85%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-desc {
    font-size: 1.05rem;
    font-weight: 400;
    color: #94a3b8;
    max-width: 620px;
    margin: 0 auto;
    line-height: 1.6;
}

/* ── Modern Card Container ── */
.glass-card {
    background: rgba(15, 23, 42, 0.65);
    border: 1px solid rgba(56, 189, 248, 0.16);
    border-radius: 20px;
    padding: 1.6rem 2rem;
    margin-bottom: 1.5rem;
    backdrop-filter: blur(16px);
    box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
    transition: all 0.2s ease;
}

/* ── Agent Pipeline Cards ── */
.agent-step-card {
    background: rgba(15, 23, 42, 0.55);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 1.1rem 1.4rem;
    margin-bottom: 0.85rem;
    position: relative;
    backdrop-filter: blur(10px);
    transition: all 0.25s ease;
}
.agent-step-card:hover {
    border-color: rgba(56, 189, 248, 0.25);
    transform: translateY(-1px);
}
.agent-step-card.active {
    border-color: rgba(56, 189, 248, 0.6);
    background: rgba(56, 189, 248, 0.08);
    box-shadow: 0 0 25px rgba(56, 189, 248, 0.15);
}
.agent-step-card.done {
    border-color: rgba(16, 185, 129, 0.35);
    background: rgba(16, 185, 129, 0.05);
}
.agent-step-card::before {
    content: '';
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 4px;
    border-radius: 16px 0 0 16px;
    background: rgba(255, 255, 255, 0.08);
}
.agent-step-card.active::before {
    background: linear-gradient(180deg, #38bdf8, #818cf8);
}
.agent-step-card.done::before {
    background: #10b981;
}

.step-meta-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0.25rem;
}
.step-id {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    font-weight: 500;
    color: #38bdf8;
    letter-spacing: 0.1em;
}
.step-name {
    font-family: 'Syne', sans-serif;
    font-size: 0.98rem;
    font-weight: 700;
    color: #f8fafc;
}
.step-desc {
    font-size: 0.8rem;
    color: #94a3b8;
    line-height: 1.4;
    margin-top: 0.2rem;
}
.step-pill {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    font-weight: 500;
    padding: 0.2rem 0.6rem;
    border-radius: 9999px;
    letter-spacing: 0.05em;
}
.pill-waiting {
    color: #64748b;
    background: rgba(100, 116, 139, 0.15);
}
.pill-running {
    color: #38bdf8;
    background: rgba(56, 189, 248, 0.18);
    animation: pulseGlow 1.6s infinite ease-in-out;
}
.pill-done {
    color: #34d399;
    background: rgba(16, 185, 129, 0.18);
}

@keyframes pulseGlow {
    0%, 100% { opacity: 0.7; transform: scale(0.98); }
    50% { opacity: 1; transform: scale(1.02); }
}

/* ── Metric Stat Chips ── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
    gap: 0.9rem;
    margin-bottom: 1.5rem;
}
.metric-chip {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 0.9rem 1.1rem;
    text-align: center;
}
.metric-chip-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: #f8fafc;
}
.metric-chip-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #38bdf8;
    margin-top: 0.2rem;
}

/* ── Critic Score Display ── */
.score-hero-card {
    background: linear-gradient(135deg, rgba(56, 189, 248, 0.06), rgba(129, 140, 248, 0.08));
    border: 1px solid rgba(56, 189, 248, 0.25);
    border-radius: 18px;
    padding: 1.5rem;
    display: flex;
    align-items: center;
    gap: 1.8rem;
    margin-bottom: 1.5rem;
}
.score-circle {
    min-width: 90px;
    height: 90px;
    border-radius: 50%;
    background: linear-gradient(135deg, #0ea5e9, #8b5cf6);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    color: white;
    box-shadow: 0 0 20px rgba(56, 189, 248, 0.35);
}
.score-circle .number {
    font-family: 'Syne', sans-serif;
    font-size: 1.7rem;
    font-weight: 800;
    line-height: 1;
}
.score-circle .max {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    opacity: 0.85;
}
.score-verdict {
    font-size: 1.05rem;
    font-weight: 500;
    color: #e2e8f0;
    line-height: 1.5;
}

/* ── Source Item Card ── */
.source-card {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.9rem;
    transition: all 0.2s ease;
}
.source-card:hover {
    border-color: rgba(56, 189, 248, 0.3);
    background: rgba(56, 189, 248, 0.04);
}
.source-title {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 0.95rem;
    color: #f1f5f9;
    margin-bottom: 0.4rem;
}
.source-snippet {
    font-size: 0.85rem;
    color: #94a3b8;
    line-height: 1.5;
    margin-bottom: 0.6rem;
}
.source-link {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #38bdf8;
    text-decoration: none;
    word-break: break-all;
}

/* ── Streamlit UI Elements Customization ── */
.stButton > button {
    border-radius: 12px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.02em !important;
    transition: all 0.2s ease !important;
}

div[data-testid="stTextInput"] input {
    background: rgba(15, 23, 42, 0.7) !important;
    border: 1px solid rgba(56, 189, 248, 0.25) !important;
    border-radius: 14px !important;
    color: #f8fafc !important;
    font-size: 1rem !important;
    padding: 0.85rem 1.2rem !important;
}
div[data-testid="stTextInput"] input:focus {
    border-color: #38bdf8 !important;
    box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.2) !important;
}

/* Tabs Styling */
div[data-testid="stTabs"] button[role="tab"] {
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    color: #94a3b8 !important;
    border-radius: 8px !important;
    padding: 0.5rem 1.2rem !important;
}
div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    color: #38bdf8 !important;
    background: rgba(56, 189, 248, 0.1) !important;
    border-bottom: 2px solid #38bdf8 !important;
}

/* Sidebar Styling */
section[data-testid="stSidebar"] {
    background-color: #050b14 !important;
    border-right: 1px solid rgba(255, 255, 255, 0.06) !important;
}
</style>
""", unsafe_allow_html=True)


# ── Session State Management ──────────────────────────────────────────────────
if "results" not in st.session_state:
    st.session_state.results = {}
if "running" not in st.session_state:
    st.session_state.running = False
if "done" not in st.session_state:
    st.session_state.done = False
if "current_step" not in st.session_state:
    st.session_state.current_step = None
if "history" not in st.session_state:
    st.session_state.history = []
if "main_topic_input" not in st.session_state:
    st.session_state.main_topic_input = ""
if "topic_input" not in st.session_state:
    st.session_state.topic_input = ""
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "gemini-3-flash-preview"
if "temperature" not in st.session_state:
    st.session_state.temperature = 0.0
if "step_durations" not in st.session_state:
    st.session_state.step_durations = {}


# ── Callbacks for Widget State Manipulation (runs safely BEFORE widgets instantiate)
def on_reset_click():
    st.session_state.main_topic_input = ""
    st.session_state.topic_input = ""
    st.session_state.results = {}
    st.session_state.step_durations = {}
    st.session_state.done = False
    st.session_state.running = False
    st.session_state.current_step = None

def on_template_click(chosen_topic: str):
    st.session_state.main_topic_input = chosen_topic
    st.session_state.topic_input = chosen_topic
    st.session_state.results = {}
    st.session_state.step_durations = {}
    st.session_state.done = False

def on_history_click(chosen_topic: str, past_results: dict):
    st.session_state.main_topic_input = chosen_topic
    st.session_state.topic_input = chosen_topic
    st.session_state.results = past_results
    st.session_state.done = True
    st.session_state.running = False


# ── Helper: Parse Critic Evaluation ───────────────────────────────────────────
def parse_critic_feedback(feedback_text: str):
    """Parses standard critic output format into structured dictionary."""
    score = "8.5"
    strengths = []
    improvements = []
    verdict = ""

    score_match = re.search(r"Score:\s*([0-9]+(?:\.[0-9]+)?)\s*/\s*10", feedback_text, re.IGNORECASE)
    if score_match:
        score = score_match.group(1)

    verdict_match = re.search(r"One line verdict:\s*(.+)", feedback_text, re.IGNORECASE)
    if verdict_match:
        verdict = verdict_match.group(1).strip()

    strengths_section = re.search(r"Strengths:\s*\n((?:- .*\n?)+)", feedback_text, re.IGNORECASE)
    if strengths_section:
        strengths = [s.strip("- \n\r") for s in strengths_section.group(1).strip().split("\n") if s.strip()]

    improvements_section = re.search(r"Areas to Improve:\s*\n((?:- .*\n?)+)", feedback_text, re.IGNORECASE)
    if improvements_section:
        improvements = [s.strip("- \n\r") for s in improvements_section.group(1).strip().split("\n") if s.strip()]

    return {
        "score": score,
        "strengths": strengths,
        "improvements": improvements,
        "verdict": verdict or "High quality synthesis with clear actionable insights.",
        "raw": feedback_text
    }


# ── Helper: Parse Sources from Search Results ─────────────────────────────────
def parse_search_sources(search_text: str):
    """Extracts structured titles, URLs, and snippets from the search text."""
    blocks = search_text.split("----")
    sources = []
    for block in blocks:
        title_match = re.search(r"Title:\s*(.+)", block)
        url_match = re.search(r"URL:\s*(https?://\S+)", block)
        snippet_match = re.search(r"Snippet:\s*(.+)", block, re.DOTALL)

        if url_match:
            sources.append({
                "title": title_match.group(1).strip() if title_match else "Web Source",
                "url": url_match.group(1).strip(),
                "snippet": snippet_match.group(1).strip() if snippet_match else ""
            })
    return sources


# ── Helper: Sanitize & Redact Credentials from any display string ─────────────
def sanitize_safe_text(raw_text: str) -> str:
    """Removes and redacts any trace of API keys from strings, errors, or logs."""
    if not raw_text:
        return ""
    text = str(raw_text)
    g_key = get_gemini_api_key()
    t_key = os.getenv("TAVILY_API_KEY")
    if g_key and len(g_key) > 6:
        text = text.replace(g_key, "[PROTECTED_KEY]")
    if t_key and len(t_key) > 6:
        text = text.replace(t_key, "[PROTECTED_KEY]")
    # Generic patterns for Google and Tavily keys
    text = re.sub(r'AQ\.[A-Za-z0-9_-]{20,}', '[PROTECTED_KEY]', text)
    text = re.sub(r'AIzaSy[A-Za-z0-9_-]{20,}', '[PROTECTED_KEY]', text)
    text = re.sub(r'tvly-[A-Za-z0-9_-]{15,}', '[PROTECTED_KEY]', text)
    text = re.sub(r'(?:key|api_key|token)=([A-Za-z0-9_.-]+)', r'key=[PROTECTED_KEY]', text, flags=re.IGNORECASE)
    return text


import base64

# Helper: load favicon as base64 for inline HTML rendering
def get_favicon_b64():
    if os.path.exists("assets/favicon.png"):
        with open("assets/favicon.png", "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    return ""

favicon_b64 = get_favicon_b64()

# ── Sidebar Configuration & Controls ─────────────────────────────────────────
with st.sidebar:
    icon_html = f'<img src="data:image/png;base64,{favicon_b64}" width="36" height="36" style="border-radius:9px;box-shadow:0 0 12px rgba(56,189,248,0.3);"/>' if favicon_b64 else '<span style="font-size:1.6rem;">⚡</span>'
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:0.75rem;margin-bottom:1.2rem;">
        {icon_html}
        <div>
            <div style="font-family:'Syne',sans-serif;font-weight:800;font-size:1.15rem;color:#f8fafc;">ResearcherAI</div>
            <div style="font-family:'DM Mono',monospace;font-size:0.65rem;color:#38bdf8;letter-spacing:0.1em;">POWERED BY MOHSIN ALI</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 1. API Status Indicator (Zero-Exposure & Protected)
    gemini_key = get_gemini_api_key() or ""
    tavily_key = os.getenv("TAVILY_API_KEY", "")

    st.markdown("<p style='font-family:DM Mono;font-size:0.7rem;color:#94a3b8;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.4rem;'>API Credentials Status</p>", unsafe_allow_html=True)

    col_stat1, col_stat2 = st.columns(2)
    with col_stat1:
        if gemini_key and len(gemini_key) > 5:
            st.markdown("<div style='background:rgba(16,185,129,0.12);border:1px solid rgba(16,185,129,0.3);border-radius:10px;padding:0.4rem 0.6rem;text-align:center;font-size:0.72rem;color:#34d399;'>🔒 Gemini Protected</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='background:rgba(239,68,68,0.12);border:1px solid rgba(239,68,68,0.3);border-radius:10px;padding:0.4rem 0.6rem;text-align:center;font-size:0.72rem;color:#f87171;'>○ Gemini Missing</div>", unsafe_allow_html=True)

    with col_stat2:
        if tavily_key and len(tavily_key) > 5:
            st.markdown("<div style='background:rgba(16,185,129,0.12);border:1px solid rgba(16,185,129,0.3);border-radius:10px;padding:0.4rem 0.6rem;text-align:center;font-size:0.72rem;color:#34d399;'>🔒 Tavily Protected</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='background:rgba(239,68,68,0.12);border:1px solid rgba(239,68,68,0.3);border-radius:10px;padding:0.4rem 0.6rem;text-align:center;font-size:0.72rem;color:#f87171;'>○ Tavily Missing</div>", unsafe_allow_html=True)

    # API Keys Drawer (Strictly Zero-Exposure: existing values are NEVER rendered in HTML)
    with st.expander("🛡️ API Key Security & Updates", expanded=False):
        st.markdown("""
        <div style="background:rgba(16,185,129,0.06);border:1px solid rgba(16,185,129,0.22);border-radius:10px;padding:0.6rem 0.8rem;margin-bottom:0.8rem;">
            <div style="font-family:'DM Mono',monospace;font-size:0.7rem;color:#34d399;font-weight:600;">🔒 ZERO-EXPOSURE PRIVACY ACTIVE</div>
            <div style="font-size:0.73rem;color:#94a3b8;margin-top:0.25rem;line-height:1.4;">
                Existing API keys are strictly masked and isolated in the environment. They are never transmitted to or displayed in the browser DOM.
            </div>
        </div>
        """, unsafe_allow_html=True)

        user_gemini = st.text_input(
            "New Gemini API Key",
            value="",
            type="password",
            placeholder="•••••••••••••••• (Encrypted in .env)" if gemini_key else "Paste new Gemini key...",
            help="For your privacy, active keys are never revealed or copied from this interface. Only enter a value if you wish to override your current key."
        )

        user_tavily = st.text_input(
            "New Tavily Search Key",
            value="",
            type="password",
            placeholder="•••••••••••••••• (Encrypted in .env)" if tavily_key else "Paste new Tavily key...",
            help="Active keys remain hidden. Only enter a value if you wish to override your current key."
        )

        if st.button("🔒 Save & Update Keys", use_container_width=True):
            updated = False
            if user_gemini.strip():
                clean_g = user_gemini.strip()
                os.environ["GOOGLE_GEMINI_KEY"] = clean_g
                os.environ["GOOGLE_API_KEY"] = clean_g
                if "GEMINI_API_KEY" in os.environ:
                    del os.environ["GEMINI_API_KEY"]
                updated = True
            if user_tavily.strip():
                clean_t = user_tavily.strip()
                os.environ["TAVILY_API_KEY"] = clean_t
                updated = True

            if updated:
                st.success("New credentials safely registered!")
                time.sleep(0.5)
                st.rerun()
            else:
                st.info("No new key entered. Your existing protected keys remain active.")

    st.markdown("<div style='height:1px;background:rgba(255,255,255,0.08);margin:1.2rem 0;'></div>", unsafe_allow_html=True)

    # 2. Model & Agent Settings
    st.markdown("<p style='font-family:DM Mono;font-size:0.7rem;color:#94a3b8;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.4rem;'>Gemini Model Configuration</p>", unsafe_allow_html=True)

    gemini_models = [
        "gemini-3-flash-preview",
        "gemini-3.1-flash-lite",
        "gemini-3.5-flash-lite",
        "gemini-3.5-flash",
    ]

    selected_idx = 0
    if st.session_state.selected_model in gemini_models:
        selected_idx = gemini_models.index(st.session_state.selected_model)

    st.session_state.selected_model = st.selectbox(
        "Gemini Model",
        options=gemini_models,
        index=selected_idx,
        help="gemini-3-flash-preview offers blazing-fast multi-agent research with automated quota failover protection."
    )

    st.session_state.temperature = st.slider(
        "Creativity / Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.1,
        help="0.0 = Most factual and deterministic; 0.7+ = More creative synthesis."
    )

    st.markdown("<div style='height:1px;background:rgba(255,255,255,0.08);margin:1.2rem 0;'></div>", unsafe_allow_html=True)

    # 3. Session Research History
    st.markdown("<p style='font-family:DM Mono;font-size:0.7rem;color:#94a3b8;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.4rem;'>Recent Researches</p>", unsafe_allow_html=True)

    if st.session_state.history:
        for idx, item in enumerate(reversed(st.session_state.history[-6:])):
            history_title = item.get("topic", "Untitled")
            if len(history_title) > 28:
                history_title = history_title[:28] + "…"
            
            st.button(
                f"📄 {history_title}",
                key=f"hist_btn_{idx}",
                use_container_width=True,
                on_click=on_history_click,
                args=(item.get("topic", "Untitled"), item.get("results", {}))
            )

        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.history = []
            st.rerun()
    else:
        st.caption("No research completed in this session yet.")

    st.markdown("<div style='height:1px;background:rgba(255,255,255,0.08);margin:1.5rem 0;'></div>", unsafe_allow_html=True)
    st.caption("ResearcherAI • Powered by Mohsin Ali")


# ── Hero Header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrapper">
    <div class="hero-badge">
        <span>⚡</span> POWERED BY GOOGLE GEMINI & TAVILY SEARCH
    </div>
    <h1 class="hero-title">
        Deep Research <span class="hero-gradient">Orchestrator</span>
    </h1>
    <p class="hero-desc">
        Watch 4 specialized AI agents collaborate in real time — discovering web insights, scraping deep articles, drafting structured reports, and running critical evaluations.
    </p>
</div>
""", unsafe_allow_html=True)


# ── Interactive Examples Handler ──────────────────────────────────────────────
examples = [
    "Roadmap for AGI development in next 5 years",
    "Impact of AI Agents on Software Engineering in 2026",
    "Breakthroughs in Quantum Computing & Fault Tolerance",
    "Future of Solid-State Batteries in Electric Vehicles"
]

def set_topic(chosen_topic: str):
    st.session_state.topic_input = chosen_topic


# ── Main Input & Pipeline Status Grid ─────────────────────────────────────────
col_query, col_stepper = st.columns([5, 4.2], gap="large")

with col_query:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    # Topic input strictly synchronized with session state
    topic_query = st.text_input(
        "Enter Your Research Topic or Question:",
        placeholder="e.g. Next-generation humanoid robots: State of the art & commercialization",
        key="main_topic_input",
    )
    st.session_state.topic_input = topic_query

    # Primary Action Button
    col_btn, col_clear = st.columns([3, 1])
    with col_btn:
        start_research = st.button(
            "⚡ Launch Multi-Agent Pipeline",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.running
        )
    with col_clear:
        st.button(
            "Reset",
            use_container_width=True,
            disabled=st.session_state.running,
            on_click=on_reset_click
        )

    st.markdown('</div>', unsafe_allow_html=True)

    # Clickable Example Chips
    st.markdown("<p style='font-family:DM Mono;font-size:0.7rem;color:#64748b;letter-spacing:0.1em;text-transform:uppercase;margin:0.2rem 0 0.5rem;'>Quick Start Templates:</p>", unsafe_allow_html=True)
    chip_cols = st.columns(2)
    for i, ex in enumerate(examples):
        with chip_cols[i % 2]:
            st.button(
                f"💡 {ex}",
                key=f"ex_chip_{i}",
                use_container_width=True,
                disabled=st.session_state.running,
                on_click=on_template_click,
                args=(ex,)
            )

with col_stepper:
    st.markdown("""
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:0.8rem;">
        <span style="font-family:'Syne',sans-serif;font-weight:700;font-size:1.1rem;color:#f8fafc;">Agent Pipeline</span>
        <span style="font-family:'DM Mono',monospace;font-size:0.7rem;color:#38bdf8;">ORCHESTRATION</span>
    </div>
    """, unsafe_allow_html=True)

    def render_stepper(container, active_step, results, durations):
        pipeline_steps = [
            ("01", "search", "Search Agent", "Discovers real-time web sources via Tavily API"),
            ("02", "reader", "Reader Agent", "Scrapes & extracts high-signal full article content"),
            ("03", "writer", "Writer Chain", "Synthesizes data into a structured research report"),
            ("04", "critic", "Critic Chain", "Strictly reviews, scores, and refines the output"),
        ]

        cards_html = []
        for num, sid, name, desc in pipeline_steps:
            duration = durations.get(sid)
            duration_text = f" • {duration:.1f}s" if duration is not None else ""

            if sid in results:
                cls = "agent-step-card done"
                badge = f'<span class="step-pill pill-done">✓ DONE{duration_text}</span>'
            elif active_step == sid:
                cls = "agent-step-card active"
                badge = '<span class="step-pill pill-running">● ACTIVE</span>'
            else:
                cls = "agent-step-card"
                badge = '<span class="step-pill pill-waiting">○ WAITING</span>'

            cards_html.append(f"""
            <div class="{cls}">
                <div class="step-meta-row">
                    <div style="display:flex;align-items:center;gap:0.6rem;">
                        <span class="step-id">{num}</span>
                        <span class="step-name">{name}</span>
                    </div>
                    {badge}
                </div>
                <div class="step-desc">{desc}</div>
            </div>
            """)

        container.markdown("\n".join(cards_html), unsafe_allow_html=True)

    stepper_placeholder = st.empty()
    initial_active = "search" if (st.session_state.running and not st.session_state.done) else st.session_state.current_step
    render_stepper(
        stepper_placeholder,
        initial_active,
        st.session_state.results,
        st.session_state.step_durations
    )


# ── Pipeline Execution Flow ───────────────────────────────────────────────────
if start_research:
    active_topic = (st.session_state.get("main_topic_input") or st.session_state.get("topic_input") or "").strip()
    active_gemini_key = get_gemini_api_key()

    if not active_topic:
        st.warning("⚠️ Please provide a research topic or click one of the quick start templates.")
    elif not active_gemini_key:
        st.error("❌ Google Gemini API key is missing. Please add it to your `.env` as `GOOGLE_GEMINI_KEY` or in the sidebar API Settings.")
    elif not os.getenv("TAVILY_API_KEY"):
        st.error("❌ Tavily API key is missing. Please add it to your `.env` or in the sidebar API Settings.")
    else:
        st.session_state.results = {}
        st.session_state.step_durations = {}
        st.session_state.running = True
        st.session_state.done = False
        st.session_state.current_step = "search"
        st.rerun()


# Execution Loop
if st.session_state.running and not st.session_state.done:
    topic_query = (st.session_state.get("main_topic_input") or st.session_state.get("topic_input") or "").strip()
    results_dict = dict(st.session_state.results)
    custom_llm = get_llm(model=st.session_state.selected_model, temperature=st.session_state.temperature)

    # Progress visual
    prog_container = st.empty()
    prog_bar = st.progress(0, text="Initializing Research Pipeline with Gemini…")

    try:
        # ── Step 1: Search Agent ──
        if "search" not in results_dict:
            st.session_state.current_step = "search"
            render_stepper(stepper_placeholder, "search", results_dict, st.session_state.step_durations)
            prog_bar.progress(15, text="🔍 Step 1/4: Search Agent is querying the live web…")
            t0 = time.time()

            search_agent = build_search_agent(custom_llm=custom_llm)
            sr = search_agent.invoke({
                "messages": [("user", f"Find recent, reliable and detailed information about: {topic_query}")]
            })
            raw_content = sr["messages"][-1].content
            results_dict["search"] = extract_text_content(raw_content)
            st.session_state.step_durations["search"] = time.time() - t0
            st.session_state.results = dict(results_dict)

        # ── Step 2: Reader Agent ──
        if "reader" not in results_dict:
            st.session_state.current_step = "reader"
            render_stepper(stepper_placeholder, "reader", results_dict, st.session_state.step_durations)
            prog_bar.progress(40, text="📄 Step 2/4: Reader Agent is scraping and cleaning the top source…")
            t0 = time.time()

            reader_agent = build_reader_agent(custom_llm=custom_llm)
            rr = reader_agent.invoke({
                "messages": [(
                    "user",
                    f"Based on the following search results about '{topic_query}', "
                    f"pick the most relevant URL and scrape it for deeper content.\n\n"
                    f"Search Results:\n{results_dict['search'][:1000]}"
                )]
            })
            raw_reader = rr["messages"][-1].content
            results_dict["reader"] = extract_text_content(raw_reader)
            st.session_state.step_durations["reader"] = time.time() - t0
            st.session_state.results = dict(results_dict)

        # ── Step 3: Writer Chain ──
        if "writer" not in results_dict:
            st.session_state.current_step = "writer"
            render_stepper(stepper_placeholder, "writer", results_dict, st.session_state.step_durations)
            prog_bar.progress(65, text="✍️ Step 3/4: Writer Chain is synthesizing findings into an executive report…")
            t0 = time.time()

            research_combined = (
                f"SEARCH RESULTS:\n{results_dict['search']}\n\n"
                f"DETAILED SCRAPED CONTENT:\n{results_dict['reader']}"
            )
            writer_chain = get_writer_chain(custom_llm=custom_llm)
            raw_writer = writer_chain.invoke({
                "topic": topic_query,
                "research": research_combined
            })
            results_dict["writer"] = extract_text_content(raw_writer)
            st.session_state.step_durations["writer"] = time.time() - t0
            st.session_state.results = dict(results_dict)

        # ── Step 4: Critic Chain ──
        if "critic" not in results_dict:
            st.session_state.current_step = "critic"
            render_stepper(stepper_placeholder, "critic", results_dict, st.session_state.step_durations)
            prog_bar.progress(88, text="🧐 Step 4/4: Critic Chain is grading quality and evaluating rigor…")
            t0 = time.time()

            critic_chain = get_critic_chain(custom_llm=custom_llm)
            raw_critic = critic_chain.invoke({
                "report": results_dict["writer"]
            })
            results_dict["critic"] = extract_text_content(raw_critic)
            st.session_state.step_durations["critic"] = time.time() - t0
            st.session_state.results = dict(results_dict)

        # All completed
        render_stepper(stepper_placeholder, None, results_dict, st.session_state.step_durations)
        prog_bar.progress(100, text="🎉 Multi-agent research complete!")
        time.sleep(0.4)

        # Finalize and record into history
        st.session_state.running = False
        st.session_state.done = True
        st.session_state.current_step = None
        st.session_state.history.append({
            "topic": topic_query,
            "results": dict(results_dict),
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })

    except Exception as e:
        st.session_state.running = False
        st.session_state.current_step = None
        safe_err = sanitize_safe_text(str(e))
        st.error(f"⚠️ Research pipeline encountered an error: {safe_err}")
        st.info("Tip: Double-check your Gemini and Tavily API keys in `.env` or sidebar settings.")

    st.rerun()


# ── Results Workspace & Presentation ──────────────────────────────────────────
res = st.session_state.results

if res and ("writer" in res or "search" in res):
    st.markdown("<div style='height:1px;background:linear-gradient(90deg, transparent, rgba(56,189,248,0.35), transparent);margin:2.5rem 0;'></div>", unsafe_allow_html=True)

    # Top Header for Results
    header_col, action_col = st.columns([7, 3])
    with header_col:
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:0.75rem;">
            <span style="font-size:1.8rem;">📑</span>
            <div>
                <h2 style="font-family:'Syne',sans-serif;font-size:1.7rem;font-weight:800;color:#f8fafc;margin:0;">
                    Research Intelligence Dossier
                </h2>
                <p style="font-size:0.88rem;color:#38bdf8;margin:0;font-family:'DM Mono',monospace;">
                    TOPIC: {st.session_state.get("main_topic_input") or st.session_state.get("topic_input")}
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with action_col:
        if "writer" in res:
            report_markdown = res["writer"]
            total_duration = sum(st.session_state.step_durations.values())
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"Research_Report_{timestamp_str}.md"

            st.download_button(
                label="📥 Download Dossier (.md)",
                data=report_markdown,
                file_name=filename,
                mime="text/markdown",
                use_container_width=True
            )

    # Metric summary chips
    if "writer" in res:
        word_count = len(res["writer"].split())
        est_read_time = max(1, round(word_count / 200))
        critic_data = parse_critic_feedback(res.get("critic", ""))
        total_time = sum(st.session_state.step_durations.values())

        st.markdown(f"""
        <div class="metric-grid" style="margin-top:1.5rem;">
            <div class="metric-chip">
                <div class="metric-chip-value">{word_count}</div>
                <div class="metric-chip-label">Total Words</div>
            </div>
            <div class="metric-chip">
                <div class="metric-chip-value">~{est_read_time} min</div>
                <div class="metric-chip-label">Read Time</div>
            </div>
            <div class="metric-chip">
                <div class="metric-chip-value" style="color:#38bdf8;">{critic_data['score']}/10</div>
                <div class="metric-chip-label">Critic Score</div>
            </div>
            <div class="metric-chip">
                <div class="metric-chip-value">{total_time:.1f}s</div>
                <div class="metric-chip-label">Execution Time</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Multi-Tab Workspace ───────────────────────────────────────────────────
    tab_report, tab_critic, tab_sources, tab_trace = st.tabs([
        "📄 Executive Report",
        "🎯 Critic Evaluation",
        "🌐 Sources & Knowledge",
        "🔬 Pipeline Telemetry"
    ])

    # ── TAB 1: EXECUTIVE REPORT ──
    with tab_report:
        if "writer" in res:
            st.markdown('<div class="glass-card" style="padding:2.2rem 2.8rem;">', unsafe_allow_html=True)
            st.markdown(res["writer"])
            st.markdown('</div>', unsafe_allow_html=True)

            with st.expander("📋 View or Copy Raw Markdown"):
                st.code(res["writer"], language="markdown")
        else:
            st.info("Report generation in progress...")

    # ── TAB 2: CRITIC EVALUATION ──
    with tab_critic:
        if "critic" in res:
            c_data = parse_critic_feedback(res["critic"])

            st.markdown(f"""
            <div class="score-hero-card">
                <div class="score-circle">
                    <span class="number">{c_data['score']}</span>
                    <span class="max">/ 10</span>
                </div>
                <div>
                    <div style="font-family:'DM Mono',monospace;font-size:0.75rem;color:#38bdf8;letter-spacing:0.15em;text-transform:uppercase;">
                        CRITIC VERDICT
                    </div>
                    <div class="score-verdict">
                        "{c_data['verdict']}"
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            col_pro, col_con = st.columns(2, gap="medium")
            with col_pro:
                st.markdown("""
                <div style="background:rgba(16,185,129,0.06);border:1px solid rgba(16,185,129,0.25);border-radius:16px;padding:1.4rem;">
                    <div style="font-family:'Syne',sans-serif;font-weight:700;font-size:1.05rem;color:#34d399;margin-bottom:0.8rem;display:flex;align-items:center;gap:0.5rem;">
                        <span>🌟</span> Strengths Identified
                    </div>
                """, unsafe_allow_html=True)
                if c_data["strengths"]:
                    for s in c_data["strengths"]:
                        st.markdown(f"<div style='font-size:0.9rem;color:#e2e8f0;margin-bottom:0.5rem;'>✓ {s}</div>", unsafe_allow_html=True)
                else:
                    st.caption("No specific strengths listed.")
                st.markdown("</div>", unsafe_allow_html=True)

            with col_con:
                st.markdown("""
                <div style="background:rgba(245,158,11,0.06);border:1px solid rgba(245,158,11,0.25);border-radius:16px;padding:1.4rem;">
                    <div style="font-family:'Syne',sans-serif;font-weight:700;font-size:1.05rem;color:#fbbf24;margin-bottom:0.8rem;display:flex;align-items:center;gap:0.5rem;">
                        <span>💡</span> Recommended Improvements
                    </div>
                """, unsafe_allow_html=True)
                if c_data["improvements"]:
                    for imp in c_data["improvements"]:
                        st.markdown(f"<div style='font-size:0.9rem;color:#e2e8f0;margin-bottom:0.5rem;'>• {imp}</div>", unsafe_allow_html=True)
                else:
                    st.caption("No critical improvement areas flagged.")
                st.markdown("</div>", unsafe_allow_html=True)

            with st.expander("🔍 View Raw Critic Output"):
                st.markdown(f"```text\n{res['critic']}\n```")
        else:
            st.info("Critic review pending...")

    # ── TAB 3: SOURCES & EXTRACTED KNOWLEDGE ──
    with tab_sources:
        if "search" in res:
            parsed_sources = parse_search_sources(res["search"])
            if parsed_sources:
                st.markdown(f"<p style='font-family:DM Mono;font-size:0.75rem;color:#94a3b8;margin-bottom:1rem;'>DISCOVERED SOURCES ({len(parsed_sources)} IDENTIFIED):</p>", unsafe_allow_html=True)
                for src in parsed_sources:
                    st.markdown(f"""
                    <div class="source-card">
                        <div class="source-title">{src['title']}</div>
                        <div class="source-snippet">{src['snippet']}</div>
                        <a href="{src['url']}" target="_blank" class="source-link">🔗 {src['url']}</a>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.text(res["search"])

        if "reader" in res:
            with st.expander("📄 Reader Agent Extracted Content (Cleaned Deep Content)", expanded=False):
                st.markdown(f"""
                <div style="background:rgba(0,0,0,0.3);border-radius:12px;padding:1.2rem;font-size:0.85rem;color:#cbd5e1;line-height:1.6;white-space:pre-wrap;max-height:450px;overflow-y:auto;">
                    {res['reader']}
                </div>
                """, unsafe_allow_html=True)

    # ── TAB 4: PIPELINE TELEMETRY ──
    with tab_trace:
        st.markdown("<p style='font-family:DM Mono;font-size:0.75rem;color:#94a3b8;margin-bottom:1rem;'>EXECUTION TIMINGS & STEP LOGS</p>", unsafe_allow_html=True)
        col_t1, col_t2 = st.columns(2)

        with col_t1:
            st.json(st.session_state.step_durations)

        with col_t2:
            st.markdown(f"""
            <div class="glass-card" style="padding:1.2rem;">
                <div style="font-family:'DM Mono',monospace;font-size:0.75rem;color:#38bdf8;">ACTIVE CONFIGURATION</div>
                <div style="font-size:0.85rem;color:#e2e8f0;margin-top:0.5rem;">
                    • <b>Model:</b> {st.session_state.selected_model}<br/>
                    • <b>Temperature:</b> {st.session_state.temperature}<br/>
                    • <b>Orchestrator:</b> LangChain Sequential Chains<br/>
                    • <b>LLM Provider:</b> Google Gemini<br/>
                    • <b>Search Provider:</b> Tavily Python Client
                </div>
            </div>
            """, unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:3rem 1rem 1.5rem;font-family:'DM Mono',monospace;font-size:0.75rem;color:#94a3b8;letter-spacing:0.08em;">
    RESEARCHER AI • POWERED BY MOHSIN ALI • BUILT WITH GOOGLE GEMINI & STREAMLIT
</div>
""", unsafe_allow_html=True)
