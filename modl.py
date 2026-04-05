import streamlit as st
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from PIL import Image
import os
import time
import random
from datetime import datetime, timedelta
import pandas as pd

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SecureWatch — AI Industrial Surveillance",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

  /* Base */
  html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #080c0e;
    color: #c8d8e0;
  }
  .main { background-color: #080c0e; }
  .block-container { padding-top: 1.5rem !important; }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background: #0d1417 !important;
    border-right: 1px solid #1e2d35;
  }
  section[data-testid="stSidebar"] * { color: #c8d8e0 !important; }

  /* Metric cards */
  div[data-testid="metric-container"] {
    background: #111a1f;
    border: 1px solid #1e2d35;
    border-radius: 6px;
    padding: 16px 20px !important;
  }
  div[data-testid="metric-container"] label {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: .65rem !important;
    letter-spacing: 2px !important;
    color: #5a7080 !important;
    text-transform: uppercase !important;
  }
  div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 2.4rem !important;
    letter-spacing: 2px !important;
    color: #00e5ff !important;
  }

  /* Headers */
  h1 { font-family: 'Bebas Neue', sans-serif !important;
       letter-spacing: 4px !important; color: #eef5f8 !important; }
  h2 { font-family: 'Bebas Neue', sans-serif !important;
       letter-spacing: 3px !important; color: #eef5f8 !important; }
  h3 { font-family: 'IBM Plex Mono', monospace !important;
       font-size: .8rem !important; letter-spacing: 2px !important;
       color: #00e5ff !important; text-transform: uppercase; }

  /* Buttons */
  div.stButton > button {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: .75rem !important; letter-spacing: 2px !important;
    text-transform: uppercase !important;
    background: transparent !important;
    color: #00e5ff !important;
    border: 1px solid #00e5ff !important;
    border-radius: 3px !important;
    padding: 10px 24px !important;
    transition: all .2s !important;
  }
  div.stButton > button:hover {
    background: rgba(0,229,255,.1) !important;
  }

  /* Tabs */
  div[data-testid="stTabs"] button {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: .72rem !important; letter-spacing: 2px !important;
    text-transform: uppercase !important; color: #5a7080 !important;
  }
  div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #00e5ff !important;
    border-bottom: 2px solid #00e5ff !important;
  }

  /* Divider */
  hr { border-color: #1e2d35 !important; }

  /* Sliders */
  div[data-testid="stSlider"] label {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: .68rem !important; letter-spacing: 1px !important;
    color: #5a7080 !important; text-transform: uppercase !important;
  }

  /* Selectbox / radio */
  div[data-testid="stRadio"] label {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: .72rem !important; letter-spacing: 1px !important;
    text-transform: uppercase !important; color: #c8d8e0 !important;
  }

  /* Alert / info boxes */
  div[data-testid="stAlert"] {
    border-radius: 4px !important;
    border-left-width: 3px !important;
  }
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
OUTPUT_DIR = "/tmp/yolo_output"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

def hex_to_bgr(hex_color: str):
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i+2], 16) for i in (0, 2, 4))
    return (b, g, r)

@st.cache_resource
def load_model(model_path="yolov8n.pt"):
    return YOLO(model_path)

def _draw_box(frame, x1, y1, x2, y2, conf, idx, color, corner=14, th=2):
    for cx, cy, dx, dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
        cv2.line(frame, (cx,cy), (cx+dx*corner, cy), color, th)
        cv2.line(frame, (cx,cy), (cx, cy+dy*corner), color, th)
    label = f"#{idx}  {conf:.0%}"
    font, fs, ft = cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1
    (lw, lh), _ = cv2.getTextSize(label, font, fs, ft)
    cv2.rectangle(frame, (x1, y1-lh-10), (x1+lw+10, y1), color, -1)
    cv2.putText(frame, label, (x1+5, y1-4), font, fs, (255,255,255), ft, cv2.LINE_AA)

def _draw_hud(frame, count, color, conf_threshold):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (16,16), (220,96), (15,15,15), -1)
    frame[:] = cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)
    cv2.rectangle(frame, (16,16), (220,96), color, 1)
    cv2.putText(frame, "HEADS DETECTED", (26,38), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
    cv2.putText(frame, str(count), (26,88), cv2.FONT_HERSHEY_DUPLEX, 1.8, color, 2, cv2.LINE_AA)
    cv2.rectangle(frame, (0,h-26), (w,h), (15,15,15), -1)
    cv2.putText(frame, f"  YOLOv8  |  conf={conf_threshold:.0%}  |  {count} head(s)",
                (8,h-8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (130,130,130), 1, cv2.LINE_AA)

def detect_heads(frame, conf_threshold, iou_threshold, head_ratio, box_color_hex):
    model = load_model()
    color = hex_to_bgr(box_color_hex)
    results = model.predict(source=frame, conf=conf_threshold, iou=iou_threshold,
                            classes=[0], verbose=False)
    head_count = 0
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            body_h = y2 - y1
            hy2 = y1 + int(body_h * head_ratio)
            head_count += 1
            _draw_box(frame, x1, y1, x2, hy2, conf, head_count, color)
    _draw_hud(frame, head_count, color, conf_threshold)
    return frame, head_count

# ── Sidebar navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='font-family:"Bebas Neue",sans-serif;font-size:1.8rem;
                letter-spacing:4px;color:#00e5ff;margin-bottom:4px;'>
      🔒 SECUREWATCH
    </div>
    <div style='font-family:"IBM Plex Mono",monospace;font-size:.62rem;
                letter-spacing:2px;color:#5a7080;margin-bottom:28px;'>
      AI INDUSTRIAL SURVEILLANCE
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "NAVIGATION",
        ["🏠  Home", "📊  Dashboard", "🎯  Detection", "📞  Contact"],
        label_visibility="visible"
    )

    st.markdown("---")
    st.markdown("### ⚙️ Detection Settings")

    conf_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.40, 0.05)
    iou_threshold  = st.slider("IOU Threshold",        0.1, 1.0, 0.45, 0.05)
    head_ratio     = st.slider("Head Ratio",            0.1, 0.6, 0.28, 0.01)
    box_color_hex  = st.color_picker("Bounding Box Color", "#00C8FF")

    st.markdown("---")
    now = datetime.now()
    st.markdown(f"""
    <div style='font-family:"IBM Plex Mono",monospace;font-size:.62rem;color:#5a7080;'>
      🕐 {now.strftime('%d %b %Y · %H:%M:%S')}<br>
      <span style='color:#00ff88'>● SYSTEM ONLINE</span>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════════════
if "Home" in page:
    st.markdown("""
    <div style='padding:48px 0 24px;'>
      <div style='font-family:"IBM Plex Mono",monospace;font-size:.7rem;
                  letter-spacing:4px;color:#00e5ff;text-transform:uppercase;
                  margin-bottom:16px;'>
        ⬡ AI Industrial Surveillance System
      </div>
      <div style='font-family:"Bebas Neue",sans-serif;font-size:clamp(3rem,6vw,5.5rem);
                  letter-spacing:4px;color:#eef5f8;text-transform:uppercase;
                  line-height:.95;margin-bottom:20px;'>
        PROTECT YOUR <span style='color:#00e5ff'>SITE</span> 24 / 7
      </div>
      <div style='font-size:1rem;font-weight:300;color:#5a7080;
                  max-width:600px;line-height:1.7;margin-bottom:40px;'>
        Real-time intrusion detection, worker safety monitoring and automated
        emergency response — powered by YOLOv8 and computer vision.
      </div>
    </div>
    """, unsafe_allow_html=True)

    # KPI stats
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Detection Accuracy", "99.2%")
    with c2:
        st.metric("Response Latency", "< 80 ms")
    with c3:
        st.metric("Uptime", "24 / 7")
    with c4:
        st.metric("Camera Feeds", "8+")

    st.markdown("---")
    st.markdown("### // CORE CAPABILITIES")

    col1, col2, col3 = st.columns(3)

    features = [
        ("🔍", "Intrusion Detection",
         "Computer vision detects unauthorised persons on site after hours. "
         "Instant alerts sent to the security centre and smartphones.",
         "YOLOv8 · OpenCV"),
        ("⛑️", "PPE Verification",
         "Verifies helmets, high-visibility vests and other protective equipment "
         "in real time. Flags non-compliant workers automatically.",
         "Object Classification"),
        ("🫸", "Fall Detection",
         "Pose estimation models detect falls and hazardous postures on the "
         "worksite, triggering emergency protocols instantly.",
         "MediaPipe · OpenPose"),
        ("🚨", "Auto Emergency Call",
         "When an incident is confirmed, the system contacts emergency services "
         "with the exact GPS location of the event.",
         "Multi-channel Alerts"),
        ("📊", "Real-Time Dashboard",
         "Live video feeds with detection overlays, KPI counters, event timeline, "
         "and per-camera status — all in one interface.",
         "Streamlit · Python"),
        ("📈", "Daily Reports",
         "Automated safety reports summarising incidents, PPE compliance rates, "
         "and camera uptime — delivered by email.",
         "SMS · Email · Web"),
    ]

    cols = [col1, col2, col3, col1, col2, col3]
    for col, (icon, title, desc, tag) in zip(cols, features):
        with col:
            st.markdown(f"""
            <div style='background:#111a1f;border:1px solid #1e2d35;border-radius:6px;
                        padding:28px 24px;margin-bottom:16px;min-height:180px;'>
              <div style='font-size:1.6rem;margin-bottom:12px;'>{icon}</div>
              <div style='font-family:"Bebas Neue",sans-serif;font-size:1.2rem;
                          letter-spacing:2px;color:#eef5f8;text-transform:uppercase;
                          margin-bottom:10px;'>{title}</div>
              <div style='font-size:.83rem;color:#5a7080;line-height:1.6;
                          margin-bottom:12px;'>{desc}</div>
              <span style='font-family:"IBM Plex Mono",monospace;font-size:.6rem;
                           letter-spacing:1.5px;text-transform:uppercase;
                           background:rgba(0,229,255,.08);color:#00e5ff;
                           border:1px solid rgba(0,229,255,.2);border-radius:3px;
                           padding:3px 8px;'>{tag}</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### // TECHNOLOGY STACK")

    techs = ["YOLOv8","OpenCV","MediaPipe","Python","Streamlit",
             "PyTorch","Ultralytics","NumPy","Twilio API","RTSP Streams","Edge AI"]
    chips_html = "".join([
        f"""<span style='font-family:"IBM Plex Mono",monospace;font-size:.72rem;
                        border:1px solid #1e2d35;color:#c8d8e0;
                        padding:8px 16px;border-radius:3px;background:#111a1f;
                        display:inline-block;margin:4px;'>{t}</span>"""
        for t in techs
    ])
    st.markdown(f"<div style='margin-top:12px;'>{chips_html}</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ══════════════════════════════════════════════════════════════════
elif "Dashboard" in page:
    col_title, col_btn = st.columns([4, 1])
    with col_title:
        st.markdown("""
        <h1 style='margin:0;padding:0;'>LIVE SECURITY DASHBOARD</h1>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div style='font-family:"IBM Plex Mono",monospace;font-size:.68rem;
                    color:#5a7080;letter-spacing:1px;margin-bottom:16px;'>
          {datetime.now().strftime('%d %b %Y  ·  %H:%M:%S')}
        </div>
        """, unsafe_allow_html=True)
    with col_btn:
        simulate = st.button("⚠ Simulate Alert")

    if simulate:
        st.error(f"🚨 **NEW INTRUSION ALERT** — CAM-0{random.randint(1,3)} / "
                 f"{datetime.now().strftime('%H:%M:%S')} — CONF {random.randint(82,97)}%")

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    with k1: st.metric("🚨 Active Alerts",    "2",  "+2 last 15 min")
    with k2: st.metric("⛑️ PPE Violations",  "1",  "Sector B · Zone 3")
    with k3: st.metric("👷 Workers On-Site",  "12", "All zones")
    with k4: st.metric("📷 Cameras Active",  "7/8","CAM-04 offline")

    st.markdown("---")

    # Camera feeds
    st.markdown("### // CAMERA FEEDS")

    cam_col1, cam_col2 = st.columns(2)

    def cam_card(label, status_color, status_text, detail, alert_text=None):
        border = f"border:2px solid {status_color};"
        inner = ""
        if alert_text:
            inner = f"""<div style='position:absolute;top:50%;left:50%;
                          transform:translate(-50%,-50%);
                          font-family:"Bebas Neue",sans-serif;font-size:1rem;
                          letter-spacing:3px;color:{status_color};
                          border:2px solid {status_color};padding:6px 16px;
                          background:rgba(255,68,68,.1);'>{alert_text}</div>"""
        return f"""
        <div style='background:#0a1318;{border}border-radius:6px;
                    padding:16px;margin-bottom:12px;position:relative;
                    min-height:160px;'>
          <div style='font-family:"IBM Plex Mono",monospace;font-size:.65rem;
                      letter-spacing:2px;color:{status_color};
                      margin-bottom:8px;'>{label}</div>
          <div style='font-family:"IBM Plex Mono",monospace;font-size:.6rem;
                      color:#5a7080;margin-bottom:16px;'>{detail}</div>
          <div style='height:80px;background:linear-gradient(180deg,#091520,#0a1318);
                      border-radius:4px;display:flex;align-items:center;
                      justify-content:center;position:relative;overflow:hidden;'>
            <div style='font-size:1.8rem;opacity:.3;'>📷</div>
            {inner}
          </div>
          <div style='margin-top:8px;font-family:"IBM Plex Mono",monospace;
                      font-size:.6rem;color:{status_color};'>{status_text}</div>
        </div>"""

    with cam_col1:
        st.markdown(cam_card(
            "CAM-01 / GATE NORTH", "#ff4444",
            "⚠ INTRUSION DETECTED", "NIGHT MODE · CONF 94%",
            "⚠ INTRUSION"
        ), unsafe_allow_html=True)
        st.markdown(cam_card(
            "CAM-03 / SECTOR B", "#f0b429",
            "⚠ PPE VIOLATION — NO HELMET", "CONF 87%"
        ), unsafe_allow_html=True)

    with cam_col2:
        st.markdown(cam_card(
            "CAM-02 / SECTOR A", "#00ff88",
            "● ONLINE — 2 WORKERS PPE COMPLIANT", "CONF 97%"
        ), unsafe_allow_html=True)
        st.markdown(cam_card(
            "CAM-04 / WAREHOUSE", "#5a7080",
            "● OFFLINE — MAINTENANCE", "—"
        ), unsafe_allow_html=True)

    st.markdown("---")

    # Alerts + Timeline
    left, right = st.columns([2, 1])

    with left:
        st.markdown("### // INCIDENT TIMELINE — LAST 12H")
        now_h = datetime.now()
        hours = [(now_h - timedelta(hours=11-i)).strftime("%H:00") for i in range(12)]
        vals  = [0, 0, 1, 0, 0, 2, 0, 1, 0, 0, 1, 3]
        df_chart = pd.DataFrame({"Hour": hours, "Incidents": vals})
        st.bar_chart(df_chart.set_index("Hour"), color="#00e5ff", height=220)

    with right:
        st.markdown("### // CAMERA STATUS")
        cams = [
            ("CAM-01 / Gate North",  "🔴 ALERT"),
            ("CAM-02 / Sector A",    "🟢 ONLINE"),
            ("CAM-03 / Sector B",    "🟡 WARNING"),
            ("CAM-04 / Warehouse",   "⚫ OFFLINE"),
            ("CAM-05 / Entry",       "🟢 ONLINE"),
            ("CAM-06 / Roof",        "🟢 ONLINE"),
            ("CAM-07 / Parking",     "🟢 ONLINE"),
            ("CAM-08 / Storage",     "🟢 ONLINE"),
        ]
        for name, status in cams:
            st.markdown(f"""
            <div style='display:flex;justify-content:space-between;
                        padding:8px 0;border-bottom:1px solid #1e2d35;
                        font-family:"IBM Plex Mono",monospace;font-size:.68rem;'>
              <span style='color:#c8d8e0;'>{name}</span>
              <span>{status}</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### // RECENT ALERTS")

    alerts = [
        ("🚨", "critical", "#ff4444", "Intrusion Detected — Gate North",
         "CAM-01 · 02:47:11 · CONF 94% · SECURITY NOTIFIED"),
        ("⛑️", "warning",  "#f0b429", "PPE Violation — No Helmet",
         "CAM-03 · Sector B · 02:44:50 · CONF 87%"),
        ("👥", "info",     "#00e5ff", "Worker Count Change",
         "CAM-02 · Sector A · 02:30:00 · 2 → 3 workers"),
        ("🔒", "info",     "#00e5ff", "Night Mode Activated",
         "ALL CAMERAS · 20:00:00 · AUTO"),
    ]

    for icon, level, color, title, meta in alerts:
        st.markdown(f"""
        <div style='display:flex;gap:14px;align-items:flex-start;
                    padding:14px;border-radius:4px;background:#111a1f;
                    border-left:3px solid {color};margin-bottom:10px;'>
          <div style='font-size:1.1rem;margin-top:2px;'>{icon}</div>
          <div>
            <div style='font-size:.85rem;font-weight:600;color:#eef5f8;
                        margin-bottom:4px;'>{title}</div>
            <div style='font-family:"IBM Plex Mono",monospace;font-size:.6rem;
                        color:#5a7080;letter-spacing:1px;'>{meta}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# PAGE: DETECTION (YOLOv8)
# ══════════════════════════════════════════════════════════════════
elif "Detection" in page:
    st.markdown("<h1>AI HEAD DETECTION</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-family:"IBM Plex Mono",monospace;font-size:.7rem;
                letter-spacing:2px;color:#5a7080;margin-bottom:24px;'>
      YOLOV8 · REAL-TIME · COMPUTER VISION
    </div>
    """, unsafe_allow_html=True)

    mode = st.radio("Input Source", ["📁 Upload Image", "📷 Webcam Snapshot"], horizontal=True)
    st.markdown("---")

    if mode == "📁 Upload Image":
        uploaded_file = st.file_uploader("Choose an image", type=["jpg","jpeg","png","webp"])
        if uploaded_file:
            pil_image = Image.open(uploaded_file).convert("RGB")
            image_np  = np.array(pil_image)
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### ORIGINAL")
                st.image(pil_image, use_container_width=True)
            with col2:
                st.markdown("### RESULT")
                with st.spinner("Running YOLOv8…"):
                    result_bgr, count = detect_heads(
                        image_bgr, conf_threshold, iou_threshold,
                        head_ratio, box_color_hex
                    )
                result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
                st.image(result_rgb, use_container_width=True)

            st.metric("👤 Heads Detected", count)

            out_path = os.path.join(OUTPUT_DIR, f"result_{uploaded_file.name}")
            cv2.imwrite(out_path, result_bgr)
            with open(out_path, "rb") as f:
                st.download_button("⬇ Download Result", f,
                                   file_name=f"result_{uploaded_file.name}")

    elif mode == "📷 Webcam Snapshot":
        camera_image = st.camera_input("Take a photo")
        if camera_image:
            pil_image = Image.open(camera_image).convert("RGB")
            image_np  = np.array(pil_image)
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

            with st.spinner("Running YOLOv8…"):
                result_bgr, count = detect_heads(
                    image_bgr, conf_threshold, iou_threshold,
                    head_ratio, box_color_hex
                )
            result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
            st.image(result_rgb, use_container_width=True)
            st.metric("👤 Heads Detected", count)


# ══════════════════════════════════════════════════════════════════
# PAGE: CONTACT
# ══════════════════════════════════════════════════════════════════
elif "Contact" in page:
    st.markdown("""
    <div style='padding:32px 0 48px;'>
      <div style='font-family:"IBM Plex Mono",monospace;font-size:.7rem;
                  letter-spacing:4px;color:#00e5ff;text-transform:uppercase;
                  margin-bottom:12px;'>// SecureWatch — Project Team 2026</div>
      <div style='font-family:"Bebas Neue",sans-serif;
                  font-size:clamp(2.5rem,5vw,4.5rem);
                  letter-spacing:4px;color:#eef5f8;line-height:.9;'>
        MEET THE <span style='color:#00e5ff;'>TEAM</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Team cards
    t1, t2, t3 = st.columns(3)
    team = [
        ("01", "Lead Developer",             "Doha\nZilaoui",
         "System architecture, AI pipeline design and project coordination. "
         "Specialist in deep learning and industrial computer vision."),
        ("02", "Computer Vision Engineer",   "Soukaina\nHachmoud",
         "YOLOv8 model training, PPE classification and fall detection "
         "implementation using pose estimation frameworks."),
        ("03", "Backend & Alerts Engineer",  "Sara\nFadil",
         "Real-time alert pipeline, SMS/email notification system "
         "and multi-camera stream management."),
    ]

    for col, (idx, role, name, desc) in zip([t1, t2, t3], team):
        with col:
            name_lines = name.replace("\n","<br>")
            st.markdown(f"""
            <div style='background:#111a1f;border:1px solid #1e2d35;
                        border-radius:6px;padding:36px 30px;
                        border-top:2px solid #00e5ff;margin-bottom:24px;
                        min-height:260px;position:relative;'>
              <div style='font-family:"IBM Plex Mono",monospace;font-size:.62rem;
                          letter-spacing:3px;color:#00e5ff;text-transform:uppercase;
                          margin-bottom:14px;'>// {role}</div>
              <div style='font-family:"Bebas Neue",sans-serif;font-size:1.8rem;
                          letter-spacing:2px;color:#eef5f8;text-transform:uppercase;
                          line-height:1.0;'>{name_lines}</div>
              <div style='width:32px;height:2px;background:#00e5ff;margin:16px 0;'></div>
              <div style='font-size:.82rem;color:#5a7080;line-height:1.65;'>{desc}</div>
              <div style='position:absolute;bottom:-8px;right:4px;
                          font-family:"Bebas Neue",sans-serif;font-size:5rem;
                          color:rgba(0,229,255,.04);pointer-events:none;'>{idx}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Contact form + info
    form_col, info_col = st.columns([1, 1])

    with form_col:
        st.markdown("### SEND A MESSAGE")
        with st.form("contact_form"):
            name_input  = st.text_input("Full Name",      placeholder="Your name")
            email_input = st.text_input("Email Address",  placeholder="you@example.com")
            subj_input  = st.text_input("Subject",        placeholder="e.g. Integration request")
            msg_input   = st.text_area("Message",         placeholder="Describe your enquiry…", height=140)
            submitted   = st.form_submit_button("Send Message →")
            if submitted:
                if name_input and email_input and msg_input:
                    st.success(f"✅ Message sent! We'll get back to you at **{email_input}**.")
                else:
                    st.warning("⚠ Please fill in all required fields.")

    with info_col:
        st.markdown("### PROJECT INFO")
        info_blocks = [
            ("Project Name",  "SecureWatch — AI Industrial Surveillance"),
            ("Technologies",  "YOLOv8 · OpenCV · MediaPipe · Streamlit · Python"),
            ("Institution",   "Engineering Project — 2026"),
            ("Team Members",  "Doha Zilaoui · Soukaina Hachmoud · Sara Fadil"),
        ]
        for label, val in info_blocks:
            st.markdown(f"""
            <div style='background:#111a1f;border:1px solid #1e2d35;
                        border-radius:6px;padding:20px;margin-bottom:12px;'>
              <div style='font-family:"IBM Plex Mono",monospace;font-size:.6rem;
                          letter-spacing:2px;color:#00e5ff;text-transform:uppercase;
                          margin-bottom:6px;'>{label}</div>
              <div style='font-size:.85rem;color:#c8d8e0;'>{val}</div>
            </div>
            """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='font-family:"IBM Plex Mono",monospace;font-size:.62rem;
            color:#5a7080;letter-spacing:1px;text-align:center;padding:8px 0;'>
  <span style='color:#00e5ff;'>SECUREWATCH</span> — AI Industrial Surveillance
  &nbsp;·&nbsp; Doha Zilaoui · Soukaina Hachmoud · Sara Fadil &nbsp;·&nbsp; 2026
</div>
""", unsafe_allow_html=True)