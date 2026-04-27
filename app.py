# =============================================================================
#  app.py — Smart Exam Proctoring Dashboard (Display Only)
#  Author: Sidak Raj Virdi | Roll: 1024240043 | Batch: 2X12
#  Thapar Institute of Engineering & Technology, Patiala
#
#  THIS FILE DOES ZERO AI COMPUTATION.
#  It only reads shared_state.json + shared_frame.jpg written by backend.py.
#  Result: Streamlit runs at full UI speed, backend runs at full AI speed.
#
#  HOW TO RUN:
#    Terminal 1:  python backend.py
#    Terminal 2:  streamlit run app.py
# =============================================================================

import streamlit as st
import json
import os
import time
from datetime import datetime
from collections import deque

st.set_page_config(
    page_title="AI Proctor — TIET",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Shared file paths (must match backend.py) ─────────────────────────────────
STATE_FILE = "shared_state.json"
FRAME_FILE = "shared_frame.jpg"
LOG_FILE   = "logs.txt"

# ── Full dark CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Syne:wght@400;600;800&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #080c10 !important;
    color: #c8d6e8 !important;
    font-family: 'Syne', sans-serif !important;
}
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] { display: none !important; }

.block-container { padding: 0 1rem 1rem !important; max-width: 100% !important; }

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: #1e3a5f; border-radius: 2px; }

/* ── Header ── */
.proctor-header {
    background: linear-gradient(90deg,#0a0f16,#0d1829,#0a0f16);
    border-bottom: 1px solid #1a3a5c;
    padding: 10px 24px;
    display: flex; align-items: center; justify-content: space-between;
    font-family: 'JetBrains Mono', monospace;
    position: sticky; top:0; z-index:999;
}
.brand { font-family:'Syne',sans-serif; font-weight:800; font-size:1.05rem;
         letter-spacing:.08em; color:#e8f4ff; }
.brand span { color:#2d9cdb; }
.session-info { font-size:.68rem; color:#4a7fa5; letter-spacing:.12em;
                text-transform:uppercase; }
.rec-dot { display:inline-block; width:7px; height:7px; background:#e74c3c;
           border-radius:50%; animation:recblink 1.2s ease-in-out infinite;
           margin-right:5px; vertical-align:middle; }
@keyframes recblink{0%,100%{opacity:1}50%{opacity:.2}}

/* ── Metric cards ── */
.metric-card {
    background:#0d1520; border:1px solid #1a2d42; border-radius:8px;
    padding:12px 14px; position:relative; overflow:hidden;
}
.metric-card::before { content:''; position:absolute; top:0; left:0; right:0;
    height:2px; background:var(--acc,#2d9cdb); }
.mc-ok   { --acc:#27ae60; }
.mc-warn { --acc:#f39c12; }
.mc-high { --acc:#e74c3c; animation:cardpulse 1s ease-in-out infinite; }
@keyframes cardpulse{0%,100%{border-color:#1a2d42}50%{border-color:#e74c3c55}}
.metric-label { font-family:'JetBrains Mono',monospace; font-size:.6rem;
                color:#4a7fa5; letter-spacing:.15em; text-transform:uppercase;
                margin-bottom:5px; }
.metric-value { font-family:'JetBrains Mono',monospace; font-size:1.3rem;
                font-weight:700; color:var(--acc,#2d9cdb); }
.metric-sub   { font-size:.6rem; color:#3a5f7a; margin-top:2px;
                font-family:'JetBrains Mono',monospace; }

/* ── Alert banner ── */
.alert-banner { border-radius:6px; padding:8px 16px;
    font-family:'JetBrains Mono',monospace; font-size:.72rem;
    letter-spacing:.06em; margin-bottom:8px; border-left:3px solid;
    display:flex; align-items:center; gap:10px; }
.al-ok       { background:#0a1f12; border-color:#27ae60; color:#52d68a; }
.al-low      { background:#1a1a0a; border-color:#f39c12; color:#f6c35a; }
.al-medium   { background:#1a100a; border-color:#e67e22; color:#f0934a; }
.al-high     { background:#1a0a0a; border-color:#e74c3c; color:#f07070;
               animation:alertglow .8s ease-in-out infinite; }
.al-critical { background:#200005; border-color:#c0392b; color:#ff5555;
               animation:alertglow .5s ease-in-out infinite; }
@keyframes alertglow{0%,100%{opacity:1}50%{opacity:.7}}
.badge { background:currentColor; color:#080c10; padding:1px 7px;
         border-radius:3px; font-size:.6rem; font-weight:700; white-space:nowrap; }

/* ── Section label ── */
.sec-label { font-family:'JetBrains Mono',monospace; font-size:.58rem;
    color:#2d6a8a; letter-spacing:.2em; text-transform:uppercase;
    padding:5px 0 4px; border-bottom:1px solid #0f2030; margin-bottom:7px; }

/* ── Module panels ── */
.mod-panel { background:#0d1520; border:1px solid #1a2d42;
    border-radius:7px; padding:9px 12px; margin-bottom:7px; }
.mod-header { display:flex; justify-content:space-between;
    align-items:center; margin-bottom:4px; }
.mod-name { font-family:'JetBrains Mono',monospace; font-size:.62rem;
    color:#7fa8c0; letter-spacing:.07em; }
.mod-detail { font-family:'JetBrains Mono',monospace; font-size:.6rem;
    color:#3a5f7a; }
.b-ok   { background:#0a2012; color:#27ae60; border:1px solid #27ae6040;
          font-family:'JetBrains Mono',monospace; font-size:.58rem;
          padding:2px 8px; border-radius:3px; font-weight:700; }
.b-warn { background:#1a1200; color:#f39c12; border:1px solid #f39c1240;
          font-family:'JetBrains Mono',monospace; font-size:.58rem;
          padding:2px 8px; border-radius:3px; font-weight:700; }
.b-alert{ background:#200a0a; color:#e74c3c; border:1px solid #e74c3c40;
          font-family:'JetBrains Mono',monospace; font-size:.58rem;
          padding:2px 8px; border-radius:3px; font-weight:700; }
.b-off  { background:#0d1520; color:#3a5f7a; border:1px solid #1a2d42;
          font-family:'JetBrains Mono',monospace; font-size:.58rem;
          padding:2px 8px; border-radius:3px; font-weight:700; }

/* ── Risk bar ── */
.risk-wrap { background:#0d1520; border:1px solid #1a2d42;
    border-radius:7px; padding:9px 13px; margin-bottom:7px; }
.risk-labels { display:flex; justify-content:space-between;
    font-family:'JetBrains Mono',monospace; font-size:.6rem;
    color:#4a7fa5; margin-bottom:5px; }
.risk-track { background:#0a0f16; border-radius:3px; height:5px;
    border:1px solid #1a2d42; overflow:hidden; }
.risk-fill  { height:100%; border-radius:3px;
    background:var(--bc,#2d9cdb); transition:width .4s,background .4s; }

/* ── Stats grid ── */
.stats-grid { display:grid; grid-template-columns:1fr 1fr; gap:5px; }
.stat-cell  { background:#0a0f16; border:1px solid #1a2d42;
    border-radius:5px; padding:6px 9px;
    font-family:'JetBrains Mono',monospace; }
.stat-lbl { font-size:.56rem; color:#2d6a8a; letter-spacing:.1em;
    text-transform:uppercase; }
.stat-val { font-size:.82rem; font-weight:700; color:#7fa8c0; margin-top:1px; }

/* ── Log ── */
.log-wrap { background:#080c10; border:1px solid #1a2d42; border-radius:7px;
    padding:8px 11px; height:200px; overflow-y:auto;
    font-family:'JetBrains Mono',monospace; font-size:.62rem; }
.log-row { display:flex; gap:9px; padding:2px 0;
    border-bottom:1px solid #0d1520; }
.log-row:last-child { border-bottom:none; }
.lt { color:#2d6a8a; flex-shrink:0; width:58px; }
.ll-ok       { color:#27ae60; flex-shrink:0; width:52px; }
.ll-low      { color:#f39c12; flex-shrink:0; width:52px; }
.ll-medium   { color:#e67e22; flex-shrink:0; width:52px; }
.ll-high     { color:#e74c3c; flex-shrink:0; width:52px; }
.ll-critical { color:#c0392b; flex-shrink:0; width:52px; }
.lm { color:#7fa8c0; }

/* ── FPS badge ── */
.fps-badge { display:inline-block; background:#0a1f12; border:1px solid #27ae6040;
    color:#27ae60; font-family:'JetBrains Mono',monospace; font-size:.65rem;
    padding:2px 10px; border-radius:4px; margin-bottom:6px; }

/* Streamlit button overrides */
.stButton > button {
    background:#0d1829 !important; border:1px solid #1a3a5c !important;
    color:#7fa8c0 !important; font-family:'JetBrains Mono',monospace !important;
    font-size:.68rem !important; letter-spacing:.1em !important;
    border-radius:5px !important; padding:5px 14px !important;
}
.stButton > button:hover {
    background:#1a2d42 !important; color:#c8d6e8 !important;
    border-color:#2d9cdb !important; }
</style>
""", unsafe_allow_html=True)


# ── Session log (in-memory for this Streamlit session) ───────────────────────
if "ui_log"        not in st.session_state: st.session_state.ui_log        = deque(maxlen=50)
if "last_level"    not in st.session_state: st.session_state.last_level    = "OK"
if "score_history" not in st.session_state: st.session_state.score_history = deque(maxlen=60)


# ── Helpers ───────────────────────────────────────────────────────────────────

def read_state() -> dict | None:
    """Read latest detection results from backend. Returns None if not ready."""
    try:
        if not os.path.exists(STATE_FILE):
            return None
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    except Exception:
        return None


def fmt_elapsed(sec: int) -> str:
    return f"{sec//60:02d}:{sec%60:02d}"


def level_css(level: str) -> str:
    return {"OK":"ok","LOW":"low","MEDIUM":"medium",
            "HIGH":"high","CRITICAL":"critical"}.get(level, "ok")


def badge_cls(status: str) -> str:
    return {"OK":"b-ok","WARN":"b-warn","ALERT":"b-alert","OFF":"b-off"}.get(status,"b-off")


def render_log(entries) -> str:
    rows = ""
    for e in list(entries)[:20]:
        ll = f"ll-{e['level'].lower()}"
        rows += (f'<div class="log-row">'
                 f'<span class="lt">{e["time"]}</span>'
                 f'<span class="{ll}">{e["level"]}</span>'
                 f'<span class="lm">{e["msg"]}</span></div>')
    empty = '<span style="color:#2d6a8a">— no events yet —</span>'
    return f'<div class="log-wrap">{rows if rows else empty}</div>'


# ── Header ────────────────────────────────────────────────────────────────────
now_str = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
st.markdown(f"""
<div class="proctor-header">
  <div>
    <div class="brand">AI<span>PROCTOR</span> &nbsp;·&nbsp; TIET Patiala</div>
    <div class="session-info">Sidak Raj Virdi · Roll 1024240043 · Batch 2X12</div>
  </div>
  <div style="text-align:center">
    <div class="session-info"><span class="rec-dot"></span>LIVE SESSION</div>
    <div class="session-info" style="margin-top:2px">{now_str}</div>
  </div>
  <div style="text-align:right">
    <div class="session-info">Edge Device: Jetson Nano</div>
    <div class="session-info" style="margin-top:2px">Modules: GAZE · HEAD · IDENTITY · YOLO</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ── Metric bar ────────────────────────────────────────────────────────────────
m1, m2, m3, m4, m5 = st.columns(5)
mc = {
    "time":  m1.empty(),
    "frames":m2.empty(),
    "low":   m3.empty(),
    "high":  m4.empty(),
    "shots": m5.empty(),
}

# ── Main columns ──────────────────────────────────────────────────────────────
col_feed, col_side = st.columns([3, 1.1], gap="small")

with col_feed:
    st.markdown('<div class="sec-label">◈  Live Camera Feed</div>', unsafe_allow_html=True)

    # Info message
    backend_msg = st.empty()

    fps_ph    = st.empty()
    feed_ph   = st.empty()
    alert_ph  = st.empty()

    # Export button
    if st.button("⬇  Export Logs"):
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE) as f:
                content = f.read()
            st.download_button("💾 Download", content,
                               file_name=f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                               mime="text/plain")

with col_side:
    st.markdown('<div class="sec-label">◈  Module Status</div>', unsafe_allow_html=True)
    mod_gaze_ph = st.empty()
    mod_head_ph = st.empty()
    mod_id_ph   = st.empty()
    mod_yolo_ph = st.empty()

    st.markdown('<div class="sec-label" style="margin-top:8px">◈  Suspicion Score</div>', unsafe_allow_html=True)
    risk_ph = st.empty()

    st.markdown('<div class="sec-label" style="margin-top:8px">◈  Session Stats</div>', unsafe_allow_html=True)
    stats_ph = st.empty()

    st.markdown('<div class="sec-label" style="margin-top:8px">◈  Event Log</div>', unsafe_allow_html=True)
    log_ph = st.empty()


def render_module(ph, name, status, detail):
    bc = badge_cls(status)
    ph.markdown(f"""
<div class="mod-panel">
  <div class="mod-header">
    <span class="mod-name">{name}</span>
    <span class="{bc}">{status}</span>
  </div>
  <div class="mod-detail">{detail}</div>
</div>""", unsafe_allow_html=True)


def render_risk(ph, score, peak):
    ratio  = min(score / 12.0, 1.0) * 100
    col    = "#27ae60" if ratio < 35 else "#f39c12" if ratio < 65 else "#e74c3c"
    ph.markdown(f"""
<div class="risk-wrap">
  <div class="risk-labels">
    <span>SUSPICION SCORE</span>
    <span style="color:{col};font-weight:700">{score:.1f}</span>
  </div>
  <div class="risk-track">
    <div class="risk-fill" style="width:{ratio:.1f}%;--bc:{col}"></div>
  </div>
  <div class="risk-labels" style="margin-top:4px">
    <span>SESSION PEAK</span><span>{peak:.1f}</span>
  </div>
</div>""", unsafe_allow_html=True)


def render_stats(ph, counts, shots):
    ph.markdown(f"""
<div class="stats-grid">
  <div class="stat-cell">
    <div class="stat-lbl">LOW</div>
    <div class="stat-val" style="color:#f39c12">{counts.get('LOW',0)}</div>
  </div>
  <div class="stat-cell">
    <div class="stat-lbl">MEDIUM</div>
    <div class="stat-val" style="color:#e67e22">{counts.get('MEDIUM',0)}</div>
  </div>
  <div class="stat-cell">
    <div class="stat-lbl">HIGH</div>
    <div class="stat-val" style="color:#e74c3c">{counts.get('HIGH',0)}</div>
  </div>
  <div class="stat-cell">
    <div class="stat-lbl">CRITICAL</div>
    <div class="stat-val" style="color:#c0392b">{counts.get('CRITICAL',0)}</div>
  </div>
</div>""", unsafe_allow_html=True)


# ── Render idle placeholders ──────────────────────────────────────────────────
render_module(mod_gaze_ph, "M1 · EYE GAZE",     "OFF", "mediapipe facemesh · iris tracking")
render_module(mod_head_ph, "M2 · HEAD POSE",     "OFF", "dlib 68-pt · solvepnp euler")
render_module(mod_id_ph,   "M3 · IDENTITY",      "OFF", "arcface · deepsort tracking")
render_module(mod_yolo_ph, "M4 · OBJECT DETECT", "OFF", "yolov8n · coco pretrained")
render_risk(risk_ph, 0.0, 0.0)
render_stats(stats_ph, {}, 0)
log_ph.markdown(render_log(st.session_state.ui_log), unsafe_allow_html=True)

peak_score = 0.0

# ── Display loop — reads shared state, ZERO AI work ──────────────────────────
while True:
    state = read_state()

    if state is None:
        backend_msg.markdown("""
<div style="background:#0d1520;border:1px solid #1a3a5c;border-radius:7px;
padding:20px;text-align:center;font-family:JetBrains Mono,monospace;">
  <div style="color:#2d9cdb;font-size:1rem;margin-bottom:8px">⏳ Waiting for backend...</div>
  <div style="color:#4a7fa5;font-size:0.65rem">Run: <span style="color:#7fa8c0">python backend.py</span> in a separate terminal</div>
</div>""", unsafe_allow_html=True)
        time.sleep(0.5)
        continue
    else:
        backend_msg.empty()

    level     = state.get("alert_level", "OK")
    msg       = state.get("alert_msg",   "System nominal")
    counts    = state.get("counts",      {})
    fps_val   = state.get("fps",         0.0)
    elapsed   = state.get("elapsed_sec", 0)
    frames    = state.get("frame_count", 0)
    direction = state.get("direction",   "CENTER")
    gaze      = state.get("gaze",        "CENTER")
    faces     = state.get("person_count",1)
    shots     = state.get("screenshots", 0)
    objects   = state.get("objects",     [])

    # Track suspicion score (rolling weighted)
    score_map = {"OK":0,"LOW":1,"MEDIUM":2,"HIGH":3,"CRITICAL":4}
    score_raw = score_map.get(level, 0) * 3.0
    st.session_state.score_history.append(score_raw)
    curr_score = sum(st.session_state.score_history) / max(len(st.session_state.score_history),1)
    if curr_score > peak_score:
        peak_score = curr_score

    # Add to UI event log (dedup consecutive same-level events)
    if level != "OK" and level != st.session_state.last_level:
        st.session_state.ui_log.appendleft({
            "time":  state.get("timestamp","--:--:--"),
            "level": level,
            "msg":   msg,
        })
        st.session_state.last_level = level
    elif level == "OK":
        st.session_state.last_level = "OK"

    # ── Metric cards ──────────────────────────────────────────────────────────
    high_cls = "mc-high" if counts.get("HIGH",0)+counts.get("CRITICAL",0) > 0 else "mc-ok"
    low_cls  = "mc-warn" if counts.get("LOW",0)+counts.get("MEDIUM",0) > 0 else "mc-ok"

    mc["time"].markdown(f"""
<div class="metric-card mc-ok">
  <div class="metric-label">Session Time</div>
  <div class="metric-value">{fmt_elapsed(elapsed)}</div>
  <div class="metric-sub">running</div>
</div>""", unsafe_allow_html=True)

    mc["frames"].markdown(f"""
<div class="metric-card mc-ok">
  <div class="metric-label">Frames Analysed</div>
  <div class="metric-value">{frames:,}</div>
  <div class="metric-sub">total processed</div>
</div>""", unsafe_allow_html=True)

    mc["low"].markdown(f"""
<div class="metric-card {low_cls}">
  <div class="metric-label">Low / Medium</div>
  <div class="metric-value">{counts.get('LOW',0)+counts.get('MEDIUM',0)}</div>
  <div class="metric-sub">gaze · head pose</div>
</div>""", unsafe_allow_html=True)

    mc["high"].markdown(f"""
<div class="metric-card {high_cls}">
  <div class="metric-label">High / Critical</div>
  <div class="metric-value">{counts.get('HIGH',0)+counts.get('CRITICAL',0)}</div>
  <div class="metric-sub">object · identity</div>
</div>""", unsafe_allow_html=True)

    mc["shots"].markdown(f"""
<div class="metric-card {'mc-warn' if shots>0 else 'mc-ok'}">
  <div class="metric-label">Evidence Saved</div>
  <div class="metric-value">{shots}</div>
  <div class="metric-sub">screenshots</div>
</div>""", unsafe_allow_html=True)

    # ── FPS badge ─────────────────────────────────────────────────────────────
    fps_ph.markdown(f'<div class="fps-badge">◉ LIVE  &nbsp;·&nbsp;  {fps_val} FPS  &nbsp;·&nbsp;  backend processing</div>',
                    unsafe_allow_html=True)

    # ── Camera frame ──────────────────────────────────────────────────────────
    if os.path.exists(FRAME_FILE):
        feed_ph.image(FRAME_FILE, use_container_width=True)

    # ── Alert banner ──────────────────────────────────────────────────────────
    css_l = level_css(level)
    icons = {"OK":"●","LOW":"◐","MEDIUM":"◑","HIGH":"◉","CRITICAL":"⬤"}
    alert_ph.markdown(f"""
<div class="alert-banner al-{css_l}">
  <span class="badge">{level}</span>
  <span>{icons.get(level,'●')}  {msg}</span>
</div>""", unsafe_allow_html=True)

    # ── Module status panels ───────────────────────────────────────────────────
    gaze_st   = "WARN"  if gaze not in ("CENTER","NO FACE","NO_FACE") else "OK"
    head_st   = "ALERT" if direction not in ("CENTER","NO FACE","NO_FACE") else "OK"
    id_st     = "ALERT" if faces > 1 else "OK"
    yolo_st   = "ALERT" if objects else "OK"

    render_module(mod_gaze_ph, "M1 · EYE GAZE",
                  gaze_st, f"gaze: {gaze}")
    render_module(mod_head_ph, "M2 · HEAD POSE",
                  head_st, f"direction: {direction}")
    render_module(mod_id_ph,   "M3 · IDENTITY",
                  id_st,   f"faces in frame: {faces}")
    render_module(mod_yolo_ph, "M4 · OBJECT DETECT",
                  yolo_st, f"detected: {', '.join(objects) if objects else 'none'}")

    render_risk(risk_ph, round(curr_score, 1), round(peak_score, 1))
    render_stats(stats_ph, counts, shots)
    log_ph.markdown(render_log(st.session_state.ui_log), unsafe_allow_html=True)

    # Poll every 100ms — fast enough for smooth display, light on CPU
    time.sleep(0.1)