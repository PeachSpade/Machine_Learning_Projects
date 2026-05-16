import os
import warnings

# tensorflow is very noisy by default, kill all that output before importing anything
os.environ["TF_CPP_MIN_LOG_LEVEL"]   = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"]  = "0"
os.environ["ABSL_MIN_LOG_LEVEL"]     = "3"
os.environ["GRPC_VERBOSITY"]         = "ERROR"
os.environ["GLOG_minloglevel"]       = "3"
os.environ["TF_AUTOGRAPH_VERBOSITY"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"]   = "-1"
warnings.filterwarnings("ignore")

import logging
logging.disable(logging.WARNING)

import base64
import threading
import time
from collections import deque

import cv2
import numpy as np
import pandas as pd
from flask import Flask, jsonify, make_response, request
from deepface import DeepFace

# flask also likes to log every request, which we don't need
logging.getLogger("werkzeug").setLevel(logging.ERROR)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32mb cap on incoming frames

# flips to True once the warmup call finishes so the frontend knows it's safe to start
_detector_ready = False


def _warmup():
    # runs deepface once on a blank image at startup so the tensorflow model is
    # already loaded in memory when the first real frame comes in
    global _detector_ready
    try:
        dummy = np.zeros((120, 120, 3), dtype=np.uint8)
        DeepFace.analyze(
            dummy,
            actions=["emotion"],
            enforce_detection=False,
            detector_backend="opencv",
            silent=True,
        )
        _detector_ready = True
        print("[MoodSense] detector ready")
    except Exception as e:
        print(f"[MoodSense] warmup error: {e}")


# run warmup in the background so the server starts immediately
threading.Thread(target=_warmup, daemon=True).start()


# everything that gets tracked during a recording session lives here
class Session:
    def __init__(self):
        # lock because deepface runs on a background thread and flask serves on another
        self._lock = threading.Lock()
        self.reset()

    def reset(self):
        with self._lock:
            self._history  = deque(maxlen=3600)  # about an hour at typical analysis rates
            self._counts   = {}                  # how many times each emotion has been dominant
            self._all      = {}                  # full emotion scores from the most recent frame
            self._str_em   = None                # which emotion we are currently on a streak for
            self._str_n    = 0                   # how many frames in a row that emotion has appeared
            self._longest  = {"emotion": None, "count": 0}  # best streak of the session
            self._start    = time.time()
            self._analyzed = 0                   # total frames sent through deepface
            self._last     = {"faces": [], "ts": 0}  # the most recent analysis result

    def update(self, faces):
        with self._lock:
            self._analyzed += 1
            self._last = {"faces": faces, "ts": time.time()}
            if not faces:
                return

            f   = faces[0]
            dom = f["dominant"]
            self._all = f.get("all_emotions", {})

            self._history.append({
                "t": time.time(),
                "e": dom,
                "c": f["confidence"],
                "n": len(faces),
            })

            self._counts[dom] = self._counts.get(dom, 0) + 1

            # streak tracking, reset when the emotion changes
            if dom == self._str_em:
                self._str_n += 1
            else:
                self._str_em = dom
                self._str_n  = 1

            if self._str_n > self._longest["count"]:
                self._longest = {"emotion": dom, "count": self._str_n}

    def last_result(self):
        with self._lock:
            return dict(self._last)

    def stats(self):
        with self._lock:
            counts = dict(self._counts)
            total  = max(sum(counts.values()), 1)
            dom    = max(counts, key=counts.get) if counts else None

            # angry, fear, and disgust together make up the stress signal
            sp = sum(counts.get(e, 0) for e in ("angry", "fear", "disgust")) / total * 100

            h = list(self._history)
            if len(h) >= 10:
                ems  = [x["e"] for x in h[-100:]]
                # count how often the dominant emotion switches, fewer switches = more stable
                chg  = sum(1 for i in range(1, len(ems)) if ems[i] != ems[i - 1])
                stab = 1.0 - chg / max(len(ems) - 1, 1)
            else:
                stab = None

            return {
                "counts":     counts,
                "dominant":   dom,
                "dom_pct":    counts.get(dom, 0) / total * 100 if dom else 0,
                "stress_pct": sp,
                "stress":     "HIGH" if sp > 35 else "MEDIUM" if sp > 15 else "LOW",
                "stability":  stab,
                "streak_em":  self._str_em,
                "streak_n":   self._str_n,
                "longest":    dict(self._longest),
                "elapsed":    int(time.time() - self._start),
                "analyzed":   self._analyzed,
                # how many stress-signal frames appeared in the last 20 readings
                "rs": sum(1 for x in h[-20:] if x["e"] in ("angry", "fear", "disgust")),
                "history": [{"t": x["t"], "e": x["e"], "c": x["c"]} for x in h[-300:]],
                "all":   dict(self._all),
                "ready": _detector_ready,
            }

    def csv(self):
        with self._lock:
            h = list(self._history)
        if not h:
            return None
        df = pd.DataFrame([{
            "datetime":   time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(x["t"])),
            "emotion":    x["e"],
            "confidence": round(x["c"], 2),
            "faces":      x["n"],
        } for x in h])
        return df.to_csv(index=False)


STATE    = Session()
_busy    = False           # stops us firing multiple deepface calls at the same time
_busy_lk = threading.Lock()


def _run_analysis(img_rgb):
    global _busy
    try:
        raw = DeepFace.analyze(
            img_rgb,
            actions=["emotion"],
            enforce_detection=False,
            detector_backend="opencv",  # opencv is the fastest backend for live video
            silent=True,
        )

        if not isinstance(raw, list):
            raw = [raw]

        faces = []
        for p in raw:
            r      = p.get("region", {})
            em_raw = p.get("emotion", {})
            dom    = p.get("dominant_emotion", "neutral")

            # deepface already gives us percentages in 0-100 range
            em = {k: float(v) for k, v in em_raw.items()}

            faces.append({
                "x":            int(r.get("x", 0)),
                "y":            int(r.get("y", 0)),
                "w":            int(r.get("w", 0)),
                "h":            int(r.get("h", 0)),
                "dominant":     dom,
                "confidence":   em.get(dom, 0.0),
                "all_emotions": em,
            })

        STATE.update(faces)
    except Exception:
        STATE.update([])
    finally:
        _busy = False


# routes

@app.route("/")
def index():
    return HTML


@app.route("/api/analyze", methods=["POST"])
def analyze():
    global _busy
    data = request.get_json(force=True)
    try:
        img_bytes = base64.b64decode(data.get("frame", ""))
        arr       = np.frombuffer(img_bytes, np.uint8)
        bgr       = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        # deepface expects RGB, opencv gives us BGR by default
        rgb       = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except Exception:
        return jsonify(STATE.last_result())

    with _busy_lk:
        if not _busy:
            _busy = True
            threading.Thread(target=_run_analysis, args=(rgb,), daemon=True).start()

    # return whatever the last completed analysis gave us while the new one runs
    return jsonify(STATE.last_result())


@app.route("/api/stats")
def stats():
    return jsonify(STATE.stats())


@app.route("/api/reset", methods=["POST"])
def reset():
    STATE.reset()
    return jsonify({"ok": True})


@app.route("/api/export")
def export():
    data = STATE.csv()
    if not data:
        return jsonify({"error": "no data"}), 404
    r = make_response(data)
    r.headers["Content-Type"] = "text/csv"
    r.headers["Content-Disposition"] = (
        f"attachment; filename=moodsense_{int(time.time())}.csv"
    )
    return r


HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MoodSense</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#080810;--s1:rgba(255,255,255,.042);--s2:rgba(255,255,255,.072);
  --bd:rgba(255,255,255,.09);--txt:#e2e8f0;--sub:#64748b;--p:#6366f1;--r:12px;
  --happy:#fbbf24;--sad:#60a5fa;--angry:#f87171;--fear:#a78bfa;
  --surprise:#fb923c;--neutral:#94a3b8;--disgust:#34d399;
}
body{background:var(--bg);color:var(--txt);font-family:'DM Sans',sans-serif;min-height:100vh;overflow-x:hidden}
::-webkit-scrollbar{width:5px}::-webkit-scrollbar-thumb{background:rgba(255,255,255,.1);border-radius:3px}

header{
  display:flex;align-items:center;justify-content:space-between;
  padding:13px 24px;border-bottom:1px solid var(--bd);
  background:rgba(8,8,16,.9);backdrop-filter:blur(14px);
  position:sticky;top:0;z-index:100;gap:12px;
}
.logo{
  font-family:'Bebas Neue',sans-serif;font-size:26px;letter-spacing:1px;
  background:linear-gradient(90deg,#6366f1,#22d3ee);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
  background-clip:text;white-space:nowrap;
}
.header-mid{display:flex;align-items:center;gap:14px;flex:1;justify-content:center}
.status-pill{
  display:flex;align-items:center;gap:7px;background:var(--s1);
  border:1px solid var(--bd);border-radius:20px;padding:5px 14px;
  font-family:'JetBrains Mono',monospace;font-size:12px;color:var(--sub);
}
.dot{width:7px;height:7px;border-radius:50%;background:var(--sub);flex-shrink:0}
.dot.live{background:#22c55e;animation:blink 1.4s ease-in-out infinite}
.dot.loading{background:#fbbf24;animation:blink .8s ease-in-out infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.28}}
.timer{font-family:'JetBrains Mono',monospace;font-size:13px;color:var(--sub)}
.header-right{display:flex;align-items:center;gap:8px;flex-shrink:0}

.btn{
  display:flex;align-items:center;gap:6px;padding:8px 15px;border-radius:8px;
  font-family:'DM Sans',sans-serif;font-size:13px;font-weight:500;
  border:1px solid var(--bd);background:var(--s1);color:var(--txt);
  cursor:pointer;transition:all .15s ease;white-space:nowrap;
}
.btn:hover{background:var(--s2);border-color:rgba(255,255,255,.16)}
.btn.primary{background:var(--p);border-color:var(--p);color:#fff}
.btn.primary:hover{background:#4f46e5}
.btn.stop{background:rgba(248,113,113,.13);border-color:rgba(248,113,113,.32);color:#f87171}
.btn.stop:hover{background:rgba(248,113,113,.22)}

.page{padding:18px 22px;display:flex;flex-direction:column;gap:16px}
.top-row{display:grid;grid-template-columns:1fr 310px;gap:16px;align-items:start}

.cam-wrap{
  position:relative;border-radius:var(--r);overflow:hidden;
  background:#000;border:1px solid var(--bd);
}
.cam-wrap.scanning::after{
  content:'';position:absolute;inset:0;
  background:repeating-linear-gradient(
    transparent,transparent 3px,
    rgba(99,102,241,.022) 3px,rgba(99,102,241,.022) 4px
  );
  pointer-events:none;z-index:2;animation:scan 12s linear infinite;
}
@keyframes scan{0%{background-position:0 0}100%{background-position:0 100%}}
#video{width:100%;display:block;aspect-ratio:16/9;object-fit:cover}
#overlay{
  position:absolute;inset:0;width:100%;height:100%;
  pointer-events:none;z-index:3;
}
.cam-badge{
  position:absolute;top:10px;left:10px;z-index:4;
  display:flex;align-items:center;gap:6px;
  background:rgba(8,8,16,.78);backdrop-filter:blur(8px);
  border:1px solid var(--bd);border-radius:6px;
  padding:4px 10px;font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--sub);
}
.cam-fps{
  position:absolute;top:10px;right:10px;z-index:4;
  background:rgba(8,8,16,.78);backdrop-filter:blur(8px);
  border:1px solid var(--bd);border-radius:6px;
  padding:4px 10px;font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--sub);
}
.loading-overlay{
  position:absolute;inset:0;z-index:5;
  display:flex;flex-direction:column;align-items:center;justify-content:center;gap:12px;
  background:rgba(8,8,16,.85);backdrop-filter:blur(4px);
}
.spinner{
  width:36px;height:36px;border:3px solid rgba(255,255,255,.1);
  border-top-color:#6366f1;border-radius:50%;
  animation:spin .8s linear infinite;
}
@keyframes spin{to{transform:rotate(360deg)}}
.loading-txt{font-family:'JetBrains Mono',monospace;font-size:12px;color:var(--sub);letter-spacing:1px}
.offline-screen{
  aspect-ratio:16/9;display:flex;flex-direction:column;
  align-items:center;justify-content:center;gap:12px;background:var(--s1);
}
.offline-icon{font-size:48px;filter:grayscale(1);opacity:.28}
.offline-txt{font-family:'JetBrains Mono',monospace;font-size:11px;letter-spacing:3px;color:var(--sub);text-transform:uppercase}
.offline-hint{font-size:13px;color:rgba(100,116,139,.5)}

.card{
  background:var(--s1);border:1px solid var(--bd);border-radius:var(--r);
  padding:16px 18px;position:relative;overflow:hidden;
}
.card-glow{
  position:absolute;top:0;left:0;right:0;height:2px;
  border-radius:var(--r) var(--r) 0 0;transition:background .4s ease;
}
.clabel{
  font-family:'JetBrains Mono',monospace;font-size:9px;letter-spacing:2.5px;
  color:var(--sub);text-transform:uppercase;margin-bottom:10px;
}
.emotion-panel{display:flex;flex-direction:column;gap:11px}
.em-emoji{font-size:36px;line-height:1;margin-bottom:3px}
.em-name{
  font-family:'Bebas Neue',sans-serif;font-size:54px;line-height:.92;
  letter-spacing:1px;transition:color .35s ease;
}
.em-conf{font-family:'JetBrains Mono',monospace;font-size:12px;color:var(--sub);margin-top:5px}
.conf-bar{height:3px;border-radius:2px;background:rgba(255,255,255,.07);margin-top:10px;overflow:hidden}
.conf-fill{height:100%;border-radius:2px;transition:width .35s ease,background .35s ease}
.em-row{display:flex;align-items:center;gap:8px;margin-bottom:7px}
.em-row-name{font-size:12px;color:var(--sub);width:60px;flex-shrink:0;text-transform:capitalize}
.em-row-bar{flex:1;height:4px;background:rgba(255,255,255,.06);border-radius:2px;overflow:hidden}
.em-row-fill{height:100%;border-radius:2px;transition:width .3s ease}
.em-row-pct{font-family:'JetBrains Mono',monospace;font-size:10px;color:var(--sub);width:30px;text-align:right;flex-shrink:0}
.streak-big{font-family:'Bebas Neue',sans-serif;font-size:46px;line-height:1;transition:color .35s ease}
.streak-sub{font-size:12px;color:var(--sub);margin-top:4px}

.chart-card{background:var(--s1);border:1px solid var(--bd);border-radius:var(--r);padding:16px 18px}
.chart-title{font-family:'JetBrains Mono',monospace;font-size:9px;letter-spacing:2.5px;color:var(--sub);text-transform:uppercase;margin-bottom:12px}

.stats-row{display:grid;grid-template-columns:repeat(5,1fr);gap:11px}
.stat-card{background:var(--s1);border:1px solid var(--bd);border-radius:var(--r);padding:14px 16px;position:relative;overflow:hidden}
.stat-top{position:absolute;top:0;left:0;right:0;height:2px;border-radius:var(--r) var(--r) 0 0;transition:background .35s ease}
.stat-label{font-family:'JetBrains Mono',monospace;font-size:9px;letter-spacing:2px;color:var(--sub);text-transform:uppercase;margin-bottom:7px}
.stat-value{font-family:'JetBrains Mono',monospace;font-size:20px;font-weight:500;color:var(--txt);line-height:1;transition:color .35s ease}
.stat-sub{font-size:11px;color:var(--sub);margin-top:4px}

.alert{border-radius:10px;padding:11px 16px;font-size:13px}
.alert-stress{background:rgba(248,113,113,.1);border:1px solid rgba(248,113,113,.24);color:#fca5a5}
.alert-happy{background:rgba(34,197,94,.1);border:1px solid rgba(34,197,94,.24);color:#86efac}

.drawer-overlay{display:none;position:fixed;inset:0;background:rgba(0,0,0,.5);z-index:200}
.drawer-overlay.open{display:block}
.drawer{
  position:fixed;right:0;top:0;bottom:0;width:290px;background:#0c0c1c;
  border-left:1px solid var(--bd);z-index:201;padding:22px;
  transform:translateX(100%);transition:transform .22s ease;
  display:flex;flex-direction:column;gap:18px;overflow-y:auto;
}
.drawer.open{transform:translateX(0)}
.drawer-title{font-family:'Bebas Neue',sans-serif;font-size:22px;letter-spacing:1px}
.setting-label{font-family:'JetBrains Mono',monospace;font-size:9px;letter-spacing:2px;color:var(--sub);text-transform:uppercase;margin-bottom:8px}
.slider{width:100%;-webkit-appearance:none;height:4px;border-radius:2px;background:rgba(255,255,255,.1);outline:none}
.slider::-webkit-slider-thumb{-webkit-appearance:none;width:15px;height:15px;border-radius:50%;background:var(--p);cursor:pointer}
.toggle-row{display:flex;align-items:center;justify-content:space-between;margin-bottom:10px}
.toggle-lbl{font-size:13px;color:var(--txt)}
.tog{position:relative;width:34px;height:18px;cursor:pointer}
.tog input{opacity:0;width:0;height:0}
.tog-sl{position:absolute;inset:0;background:rgba(255,255,255,.1);border-radius:9px;transition:.2s}
.tog-sl::before{content:'';position:absolute;height:12px;width:12px;left:3px;bottom:3px;background:var(--sub);border-radius:50%;transition:.2s}
.tog input:checked+.tog-sl{background:var(--p)}
.tog input:checked+.tog-sl::before{transform:translateX(16px);background:#fff}
hr.dr{border:none;border-top:1px solid var(--bd)}

.toast{
  position:fixed;bottom:22px;right:22px;z-index:300;background:#14142a;
  border:1px solid var(--bd);border-radius:10px;padding:11px 18px;
  font-size:13px;color:var(--txt);box-shadow:0 8px 32px rgba(0,0,0,.4);
  transform:translateY(16px);opacity:0;transition:all .22s ease;pointer-events:none;
}
.toast.show{transform:translateY(0);opacity:1}

@media(max-width:900px){.top-row{grid-template-columns:1fr}.stats-row{grid-template-columns:repeat(3,1fr)}}
@media(max-width:580px){.stats-row{grid-template-columns:repeat(2,1fr)}.header-mid{display:none}}
</style>
</head>
<body>

<header>
  <div class="logo">MoodSense</div>
  <div class="header-mid">
    <div class="status-pill">
      <span class="dot" id="statusDot"></span>
      <span id="statusTxt" style="font-family:'JetBrains Mono',monospace;font-size:12px">Idle</span>
    </div>
    <div class="timer" id="timer">00:00</div>
  </div>
  <div class="header-right">
    <button class="btn" onclick="exportCSV()">
      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
      Export
    </button>
    <button class="btn" onclick="resetSession()">
      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="1 4 1 10 7 10"/><path d="M3.51 15a9 9 0 102.13-9.36L1 10"/></svg>
      Reset
    </button>
    <button class="btn" onclick="toggleDrawer()">
      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="3" y1="6" x2="21" y2="6"/><line x1="3" y1="12" x2="21" y2="12"/><line x1="3" y1="18" x2="21" y2="18"/></svg>
      Settings
    </button>
    <button class="btn primary" id="camBtn" onclick="toggleCamera()">
      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="23 7 16 12 23 17 23 7"/><rect x="1" y="5" width="15" height="14" rx="2" ry="2"/></svg>
      <span id="camBtnTxt">Start Camera</span>
    </button>
  </div>
</header>

<div class="page">
  <div class="top-row">

    <div class="cam-wrap scanning" id="camWrap">
      <div class="offline-screen" id="offlineScreen">
        <div class="offline-icon">&#128247;</div>
        <div class="offline-txt">Camera Offline</div>
        <div class="offline-hint">Press Start Camera to begin</div>
      </div>

      <!-- loading spinner shown while FER model warms up -->
      <div class="loading-overlay" id="loadingOverlay" style="display:none">
        <div class="spinner"></div>
        <div class="loading-txt">Loading detector...</div>
      </div>

      <video id="video" autoplay playsinline muted style="display:none"></video>
      <canvas id="overlay"></canvas>

      <div class="cam-badge" id="camBadge" style="display:none">
        <span class="dot live"></span>
        <span id="faceCount">0 faces</span>
      </div>
      <div class="cam-fps" id="camFps" style="display:none">
        <span id="fpsDisplay">&#8212; fps</span>
      </div>
    </div>

    <div class="emotion-panel">
      <div class="card">
        <div class="card-glow" id="emGlow"></div>
        <div class="clabel">Dominant Emotion</div>
        <div class="em-emoji" id="emEmoji">&#127917;</div>
        <div class="em-name" id="emName" style="color:var(--sub)">&#8212;</div>
        <div class="em-conf" id="emConf">Waiting for camera...</div>
        <div class="conf-bar"><div class="conf-fill" id="confFill" style="width:0%"></div></div>
      </div>
      <div class="card">
        <div class="clabel">All Emotions</div>
        <div id="emBars"></div>
      </div>
      <div class="card">
        <div class="card-glow" id="streakGlow"></div>
        <div class="clabel">Current Streak</div>
        <div class="streak-big" id="streakNum" style="color:var(--sub)">0</div>
        <div class="streak-sub" id="streakSub">No data yet</div>
      </div>
    </div>
  </div>

  <div id="alertArea"></div>

  <div class="chart-card" id="timelineCard">
    <div class="chart-title">Emotion Timeline &#8212; last 2 minutes</div>
    <canvas id="timelineChart" height="72"></canvas>
  </div>

  <div class="chart-card" id="distCard">
    <div class="chart-title">Session Distribution</div>
    <canvas id="distChart" height="50"></canvas>
  </div>

  <div class="stats-row" id="statsRow">
    <div class="stat-card">
      <div class="stat-label">Session</div>
      <div class="stat-value" id="statDur">00:00</div>
      <div class="stat-sub" id="statFrames">0 analyzed</div>
    </div>
    <div class="stat-card">
      <div class="stat-top" id="statDomTop"></div>
      <div class="stat-label">Dominant Mood</div>
      <div class="stat-value" id="statDom" style="font-size:15px">&#8212;</div>
      <div class="stat-sub" id="statDomPct">&#8212;</div>
    </div>
    <div class="stat-card">
      <div class="stat-top" id="statStressTop"></div>
      <div class="stat-label">Stress Level</div>
      <div class="stat-value" id="statStress">LOW</div>
      <div class="stat-sub" id="statStressPct">0% stress signals</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Mood Stability</div>
      <div class="stat-value" id="statStab">&#8212;</div>
      <div class="stat-sub">consistency score</div>
    </div>
    <div class="stat-card">
      <div class="stat-top" id="statLsTop"></div>
      <div class="stat-label">Longest Streak</div>
      <div class="stat-value" id="statLs">0</div>
      <div class="stat-sub" id="statLsSub">&#8212;</div>
    </div>
  </div>
</div>

<div class="drawer-overlay" id="drawerOverlay" onclick="toggleDrawer()"></div>
<div class="drawer" id="drawer">
  <div class="drawer-title">Settings</div>
  <div>
    <div class="setting-label">Analysis Interval</div>
    <input type="range" class="slider" min="80" max="1000" step="20" value="200" id="intervalSlider">
    <div style="display:flex;justify-content:space-between;margin-top:6px">
      <span style="font-size:11px;color:var(--sub)">Faster</span>
      <span style="font-family:'JetBrains Mono',monospace;font-size:12px;color:var(--txt)" id="intervalVal">200ms</span>
      <span style="font-size:11px;color:var(--sub)">Slower</span>
    </div>
  </div>
  <div>
    <div class="setting-label" style="margin-bottom:12px">Panels</div>
    <div class="toggle-row"><span class="toggle-lbl">Timeline Chart</span>
      <label class="tog"><input type="checkbox" checked onchange="togglePanel('timelineCard',this.checked)"><span class="tog-sl"></span></label></div>
    <div class="toggle-row"><span class="toggle-lbl">Distribution Chart</span>
      <label class="tog"><input type="checkbox" checked onchange="togglePanel('distCard',this.checked)"><span class="tog-sl"></span></label></div>
    <div class="toggle-row"><span class="toggle-lbl">Session Stats</span>
      <label class="tog"><input type="checkbox" checked onchange="togglePanel('statsRow',this.checked)"><span class="tog-sl"></span></label></div>
    <div class="toggle-row"><span class="toggle-lbl">Scan Lines</span>
      <label class="tog"><input type="checkbox" checked onchange="document.getElementById('camWrap').classList.toggle('scanning',this.checked)"><span class="tog-sl"></span></label></div>
  </div>
  <hr class="dr">
  <button class="btn stop" style="width:100%" onclick="resetSession()">Reset Session Data</button>
  <button class="btn" style="width:100%;margin-top:8px" onclick="exportCSV()">Export CSV</button>
</div>

<div class="toast" id="toast"></div>

<script>
const EC={happy:'#fbbf24',sad:'#60a5fa',angry:'#f87171',fear:'#a78bfa',surprise:'#fb923c',neutral:'#94a3b8',disgust:'#34d399'};
const EE={happy:'😊',sad:'😢',angry:'😠',fear:'😨',surprise:'😲',neutral:'😐',disgust:'🤢'};
const ALL_EM=['happy','sad','angry','fear','surprise','neutral','disgust'];

let cameraOn=false;
let videoEl=document.getElementById('video');
let overlayEl=document.getElementById('overlay');
let overlayCtx=overlayEl.getContext('2d');
let mediaStream=null,analyzeLoop=null,statsLoop=null,timerLoop=null,rafId=null;
let sessionStart=null,_analyzing=false;
let frameIdx=0,fpsArr=[],lastFpsTs=performance.now();

// holds the last set of faces from deepface so the render loop can keep drawing them
let cachedFaces=[];
let canvasW=640,canvasH=360;

// set up the two charts, timeline on top and session distribution below

const tCtx=document.getElementById('timelineChart').getContext('2d');
const dCtx=document.getElementById('distChart').getContext('2d');

const timelineChart=new Chart(tCtx,{
  type:'line',
  data:{labels:[],datasets:ALL_EM.map(em=>({
    label:em,data:[],borderColor:EC[em],backgroundColor:'transparent',
    borderWidth:1.5,pointRadius:0,tension:0.4,spanGaps:true
  }))},
  options:{responsive:true,animation:{duration:0},
    scales:{
      x:{display:false},
      y:{min:0,max:100,grid:{color:'rgba(255,255,255,.05)'},
         ticks:{color:'#64748b',font:{family:'JetBrains Mono',size:10},callback:v=>v+'%'}}
    },
    plugins:{legend:{display:true,position:'top',align:'start',
      labels:{boxWidth:10,color:'#64748b',font:{family:'DM Sans',size:11}}}}
  }
});

const distChart=new Chart(dCtx,{
  type:'bar',
  data:{
    labels:ALL_EM.map(e=>e.charAt(0).toUpperCase()+e.slice(1)),
    datasets:[{data:ALL_EM.map(()=>0),
      backgroundColor:ALL_EM.map(e=>EC[e]+'bb'),
      borderColor:ALL_EM.map(e=>EC[e]),borderWidth:1,borderRadius:4}]
  },
  options:{responsive:true,animation:{duration:400},indexAxis:'y',
    scales:{
      x:{min:0,max:100,grid:{color:'rgba(255,255,255,.05)'},
         ticks:{color:'#64748b',font:{family:'JetBrains Mono',size:10},callback:v=>v+'%'}},
      y:{grid:{display:false},ticks:{color:'#94a3b8',font:{family:'DM Sans',size:12}}}
    },
    plugins:{legend:{display:false}}
  }
});

// start and stop the webcam stream

async function toggleCamera(){cameraOn?stopCamera():await startCamera()}

async function startCamera(){
  try{
    mediaStream=await navigator.mediaDevices.getUserMedia({
      video:{width:{ideal:1280},height:{ideal:720},frameRate:{ideal:60}}
    });
    videoEl.srcObject=mediaStream;
    videoEl.style.display='block';
    document.getElementById('offlineScreen').style.display='none';
    document.getElementById('camBadge').style.display='flex';
    document.getElementById('camFps').style.display='block';
    document.getElementById('loadingOverlay').style.display='flex';
    await videoEl.play();

    // only set the canvas size here, doing it inside the draw loop causes a visible flash each frame
    canvasW=videoEl.offsetWidth||640;
    canvasH=videoEl.offsetHeight||360;
    overlayEl.width=canvasW;
    overlayEl.height=canvasH;

    cameraOn=true;sessionStart=Date.now();
    document.getElementById('camBtnTxt').textContent='Stop Camera';
    document.getElementById('camBtn').className='btn stop';
    setStatus('loading');buildEmBars();

    // kick off the render loop so face boxes stay on screen between analysis calls
    startRenderLoop();

    const ms=parseInt(document.getElementById('intervalSlider').value);
    analyzeLoop=setInterval(runAnalysis,ms);
    statsLoop=setInterval(fetchStats,1000);
    timerLoop=setInterval(tickTimer,1000);
  }catch(e){showToast('Camera error: '+e.message,true);}
}

function stopCamera(){
  if(mediaStream)mediaStream.getTracks().forEach(t=>t.stop());
  videoEl.srcObject=null;videoEl.style.display='none';
  document.getElementById('offlineScreen').style.display='flex';
  document.getElementById('camBadge').style.display='none';
  document.getElementById('camFps').style.display='none';
  document.getElementById('loadingOverlay').style.display='none';
  stopRenderLoop();
  overlayCtx.clearRect(0,0,canvasW,canvasH);
  cachedFaces=[];cameraOn=false;
  clearInterval(analyzeLoop);clearInterval(statsLoop);clearInterval(timerLoop);
  document.getElementById('camBtnTxt').textContent='Start Camera';
  document.getElementById('camBtn').className='btn primary';
  setStatus('idle');
}

// this runs at 60fps via requestAnimationFrame and just redraws whatever faces we last detected

function startRenderLoop(){
  function loop(){
    drawAnnotations(cachedFaces);
    rafId=requestAnimationFrame(loop);
  }
  rafId=requestAnimationFrame(loop);
}

function stopRenderLoop(){
  if(rafId){cancelAnimationFrame(rafId);rafId=null;}
}

// fires every N milliseconds, grabs a frame, sends it to the server, updates the ui

async function runAnalysis(){
  if(_analyzing||!cameraOn)return;_analyzing=true;

  // rough fps counter based on how often this function actually completes
  frameIdx++;
  const now=performance.now();
  fpsArr.push(1000/Math.max(now-lastFpsTs,1));lastFpsTs=now;
  if(fpsArr.length>20)fpsArr.shift();
  if(frameIdx%6===0){
    document.getElementById('fpsDisplay').textContent=
      Math.round(fpsArr.reduce((a,b)=>a+b,0)/fpsArr.length)+' fps';
  }

  // we downscale to 320x240 before sending, much faster to encode and deepface still detects fine
  const cap=document.createElement('canvas');cap.width=320;cap.height=240;
  cap.getContext('2d').drawImage(videoEl,0,0,320,240);

  try{
    const blob=await new Promise(r=>cap.toBlob(r,'image/jpeg',0.75));
    const buf=await blob.arrayBuffer();
    const b64=btoa(String.fromCharCode(...new Uint8Array(buf)));

    const resp=await fetch('/api/analyze',{
      method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({frame:b64})
    });
    const result=await resp.json();

    cachedFaces=result.faces||[];
    updateLivePanel(cachedFaces);
    updateFaceCount(cachedFaces.length);

    // once we get the first real result back we know the model is ready so we hide the spinner
    document.getElementById('loadingOverlay').style.display='none';
    setStatus('live');
  }catch(e){}finally{_analyzing=false;}
}

// draws all the face bounding boxes and emotion labels on top of the video

function drawAnnotations(faces){
  // always use clearRect here, never reset canvas.width/height or you get a flicker every frame
  overlayCtx.clearRect(0,0,canvasW,canvasH);
  if(!faces||!faces.length)return;

  // since we analyzed a smaller frame we need to scale the face coordinates back up
  const sx=canvasW/320, sy=canvasH/240;

  faces.forEach((f,i)=>{
    const x=f.x*sx, y=f.y*sy, w=f.w*sx, h=f.h*sy;
    const col=EC[f.dominant]||'#888', cf=f.confidence||0;

    // shadow gives a nice colored glow effect behind the box without any extra elements
    overlayCtx.save();
    overlayCtx.shadowColor=col;overlayCtx.shadowBlur=14;
    overlayCtx.strokeStyle=col;overlayCtx.lineWidth=1.5;
    overlayCtx.strokeRect(x,y,w,h);
    overlayCtx.restore();

    // four corner brackets look much better than a full rectangle around the face
    const cL=Math.min(22,w*.16,h*.16);
    overlayCtx.strokeStyle=col;overlayCtx.lineWidth=2.5;
    [
      [[x,y+cL],[x,y],[x+cL,y]],
      [[x+w-cL,y],[x+w,y],[x+w,y+cL]],
      [[x,y+h-cL],[x,y+h],[x+cL,y+h]],
      [[x+w-cL,y+h],[x+w,y+h],[x+w,y+h-cL]]
    ].forEach(pts=>{
      overlayCtx.beginPath();
      overlayCtx.moveTo(...pts[0]);overlayCtx.lineTo(...pts[1]);overlayCtx.lineTo(...pts[2]);
      overlayCtx.stroke();
    });

    // pill label above the face showing the emotion name and confidence
    const lbl=(EE[f.dominant]||'')+' '+f.dominant.toUpperCase()+'  '+cf.toFixed(0)+'%';
    overlayCtx.font='500 12px "DM Sans"';
    const tw=overlayCtx.measureText(lbl).width;
    const ph=26,px=9,ly=Math.max(0,y-ph-5);
    overlayCtx.fillStyle=col+'dd';
    rRect(overlayCtx,x,ly,tw+px*2,ph,5);overlayCtx.fill();
    overlayCtx.fillStyle='#08080f';
    overlayCtx.fillText(lbl,x+px,ly+ph-7);

    if(faces.length>1){
      overlayCtx.font='500 10px "DM Sans"';
      overlayCtx.fillStyle=col+'cc';
      rRect(overlayCtx,x+w-20,y+4,16,14,3);overlayCtx.fill();
      overlayCtx.fillStyle='#08080f';
      overlayCtx.fillText('#'+(i+1),x+w-18,y+14);
    }
  });
}

function rRect(ctx,x,y,w,h,r){
  ctx.beginPath();ctx.moveTo(x+r,y);ctx.lineTo(x+w-r,y);ctx.arcTo(x+w,y,x+w,y+r,r);
  ctx.lineTo(x+w,y+h-r);ctx.arcTo(x+w,y+h,x+w-r,y+h,r);
  ctx.lineTo(x+r,y+h);ctx.arcTo(x,y+h,x,y+h-r,r);ctx.lineTo(x,y+r);
  ctx.arcTo(x,y,x+r,y,r);ctx.closePath();
}

// updates the big emotion card and the seven emotion bars on the right

function updateLivePanel(faces){
  if(!faces||!faces.length){
    document.getElementById('emName').textContent='—';
    document.getElementById('emName').style.color='var(--sub)';
    document.getElementById('emEmoji').textContent='🎭';
    document.getElementById('emConf').textContent='No face detected';
    document.getElementById('confFill').style.width='0%';
    document.getElementById('emGlow').style.background='';
    return;
  }
  const f=faces[0],em=f.dominant||'neutral',cf=f.confidence||0,col=EC[em]||'#888';
  document.getElementById('emName').textContent=em.toUpperCase();
  document.getElementById('emName').style.color=col;
  document.getElementById('emEmoji').textContent=EE[em]||'😐';
  document.getElementById('emConf').textContent=cf.toFixed(1)+'% confidence';
  document.getElementById('confFill').style.width=cf+'%';
  document.getElementById('confFill').style.background=col;
  document.getElementById('emGlow').style.background=
    `linear-gradient(90deg,transparent,${col}80,transparent)`;
  if(f.all_emotions)updateEmBars(f.all_emotions);
}

function buildEmBars(){
  document.getElementById('emBars').innerHTML=ALL_EM.map(em=>`
    <div class="em-row">
      <div class="em-row-name">${em}</div>
      <div class="em-row-bar"><div class="em-row-fill" id="bar_${em}" style="width:0%;background:${EC[em]}"></div></div>
      <div class="em-row-pct" id="pct_${em}">0%</div>
    </div>`).join('');
}

function updateEmBars(all){
  ALL_EM.forEach(em=>{
    const v=parseFloat(all[em]||0);
    const b=document.getElementById('bar_'+em),p=document.getElementById('pct_'+em);
    if(b)b.style.width=v+'%';if(p)p.textContent=v.toFixed(0)+'%';
  });
}

// polls the server every second and updates all the dashboard numbers

async function fetchStats(){
  try{const r=await fetch('/api/stats');applyStats(await r.json());}catch(e){}
}

function applyStats(s){
  const dur=fmt(s.elapsed||0);
  document.getElementById('statDur').textContent=dur;
  document.getElementById('statFrames').textContent=(s.analyzed||0)+' analyzed';
  document.getElementById('timer').textContent=dur;

  const dom=s.dominant,dc=dom?(EC[dom]||'#888'):'#64748b',de=dom?(EE[dom]||''):'';
  document.getElementById('statDom').textContent=dom?de+' '+dom.toUpperCase():'—';
  document.getElementById('statDom').style.color=dc;
  document.getElementById('statDomPct').textContent=dom?(s.dom_pct||0).toFixed(0)+'% of session':'';
  document.getElementById('statDomTop').style.background=dc;

  const sl=s.stress||'LOW',sc=sl==='HIGH'?'#f87171':sl==='MEDIUM'?'#fbbf24':'#34d399';
  document.getElementById('statStress').textContent=sl;
  document.getElementById('statStress').style.color=sc;
  document.getElementById('statStressPct').textContent=(s.stress_pct||0).toFixed(0)+'% stress signals';
  document.getElementById('statStressTop').style.background=sc;

  const stab=s.stability!=null?(s.stability*100).toFixed(0)+'%':'—';
  const stc=s.stability!=null?(s.stability>.7?'#34d399':s.stability>.4?'#fbbf24':'#f87171'):'#64748b';
  document.getElementById('statStab').textContent=stab;
  document.getElementById('statStab').style.color=stc;

  const ls=s.longest||{},lc=ls.emotion?(EC[ls.emotion]||'#888'):'#64748b';
  document.getElementById('statLs').textContent=ls.count||0;
  document.getElementById('statLs').style.color=lc;
  document.getElementById('statLsSub').textContent=ls.emotion?(EE[ls.emotion]||'')+' '+ls.emotion:'—';
  document.getElementById('statLsTop').style.background=lc;

  const sc2=s.streak_em?(EC[s.streak_em]||'#888'):'#64748b';
  document.getElementById('streakNum').textContent=s.streak_n||0;
  document.getElementById('streakNum').style.color=sc2;
  document.getElementById('streakSub').textContent=
    s.streak_em?(EE[s.streak_em]||'')+' '+s.streak_em+' frames':'No data yet';
  document.getElementById('streakGlow').style.background=
    `linear-gradient(90deg,transparent,${sc2}70,transparent)`;

  const counts=s.counts||{},total=Math.max(Object.values(counts).reduce((a,b)=>a+b,0),1);
  distChart.data.datasets[0].data=ALL_EM.map(e=>((counts[e]||0)/total)*100);
  distChart.update('none');

  if(s.history&&s.history.length>1){
    const now=Date.now()/1000,hist=s.history.filter(h=>h.t>=now-120);
    timelineChart.data.labels=hist.map(()=>'');
    ALL_EM.forEach((em,i)=>{
      timelineChart.data.datasets[i].data=hist.map(h=>h.e===em?h.c:null);
    });
    timelineChart.update('none');
  }

  const alertArea=document.getElementById('alertArea');
  if((s.rs||0)>=13)
    alertArea.innerHTML='<div class="alert alert-stress">&#9888; Elevated stress detected in recent readings</div>';
  else if(s.streak_em==='happy'&&(s.streak_n||0)>=25)
    alertArea.innerHTML=`<div class="alert alert-happy">&#10003; Sustained positive mood &mdash; happy streak of ${s.streak_n} frames</div>`;
  else alertArea.innerHTML='';
}

// small functions used in a few different places

function tickTimer(){
  if(!sessionStart)return;
  document.getElementById('statDur').textContent=fmt(Math.floor((Date.now()-sessionStart)/1000));
}

function fmt(s){return String(Math.floor(s/60)).padStart(2,'0')+':'+String(s%60).padStart(2,'0')}
function updateFaceCount(n){document.getElementById('faceCount').textContent=n+(n===1?' face':' faces')}

function setStatus(state){
  const dot=document.getElementById('statusDot'),txt=document.getElementById('statusTxt');
  if(state==='live'){dot.className='dot live';txt.textContent='Live';txt.style.color='#22c55e';}
  else if(state==='loading'){dot.className='dot loading';txt.textContent='Loading...';txt.style.color='#fbbf24';}
  else{dot.className='dot';txt.textContent='Idle';txt.style.color='#64748b';}
}

function togglePanel(id,show){document.getElementById(id).style.display=show?'':'none'}
function toggleDrawer(){
  document.getElementById('drawer').classList.toggle('open');
  document.getElementById('drawerOverlay').classList.toggle('open');
}

document.getElementById('intervalSlider').addEventListener('input',function(){
  document.getElementById('intervalVal').textContent=this.value+'ms';
  if(cameraOn){clearInterval(analyzeLoop);analyzeLoop=setInterval(runAnalysis,parseInt(this.value));}
});

async function resetSession(){
  await fetch('/api/reset',{method:'POST'});applyStats({});showToast('Session cleared');
}

async function exportCSV(){
  try{
    const r=await fetch('/api/export');
    if(!r.ok){showToast('No data yet',true);return;}
    const blob=await r.blob(),url=URL.createObjectURL(blob),a=document.createElement('a');
    a.href=url;a.download='moodsense_'+Date.now()+'.csv';a.click();
    URL.revokeObjectURL(url);showToast('CSV exported');
  }catch(e){showToast('Export failed',true);}
}

function showToast(msg,err=false){
  const t=document.getElementById('toast');t.textContent=msg;
  t.style.borderColor=err?'rgba(248,113,113,.3)':'rgba(99,102,241,.3)';
  t.classList.add('show');setTimeout(()=>t.classList.remove('show'),2500);
}

// if the window is resized the canvas needs its dimensions updated to match
window.addEventListener('resize',()=>{
  if(cameraOn){
    canvasW=videoEl.offsetWidth||640;canvasH=videoEl.offsetHeight||360;
    overlayEl.width=canvasW;overlayEl.height=canvasH;
  }
});

buildEmBars();
</script>
</body>
</html>"""


if __name__ == "__main__":
    import webbrowser
    threading.Timer(1.5, lambda: webbrowser.open("http://localhost:5000")).start()
    print("MoodSense running at http://localhost:5000")
    app.run(debug=False, host="0.0.0.0", port=5000, threaded=True)