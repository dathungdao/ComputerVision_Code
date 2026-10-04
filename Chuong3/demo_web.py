import os
import re
import sys
import time
import base64
import socket
import webbrowser

import cv2
import numpy as np
import joblib
import uvicorn

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from skimage.feature import local_binary_pattern

from nhan_dang_rau_thom import (
    BASE_DIR, RESULTS_DIR, DEFAULT_DATASET, CLASS_NAMES, AUGMENTS, GROUPS,
    BoVW, build_X, extract_features, find_leaf, normalize_illumination,
    resize_long, shape_features, MODES,
)

# ============================================================
# GIAO DIỆN DEMO TRÊN LOCALHOST
#
#   python demo_web.py        → mở http://127.0.0.1:8501
#
# - Tải ảnh lên / chụp bằng webcam / chọn ảnh mẫu
# - Hiện kết quả + xác suất 4 loại + từng bước của pipeline
# - Nút biến đổi ảnh (mờ, nhiễu, sáng, tối, xoay) để thử độ bền
# - Tab kết quả thực nghiệm (5 kịch bản)
#
# Dùng model results/model.joblib do nhan_dang_rau_thom.py tạo ra.
# ============================================================

MODEL_PATH = os.path.join(RESULTS_DIR, "model.joblib")
HOST, PORT = "127.0.0.1", 8501

app = FastAPI(title="Nhận dạng rau thơm")
app.mount("/results", StaticFiles(directory=RESULTS_DIR), name="results")
app.mount("/dataset", StaticFiles(directory=DEFAULT_DATASET), name="dataset")

BUNDLE = None


def load_bundle():
    global BUNDLE
    if BUNDLE is None:
        BUNDLE = joblib.load(MODEL_PATH)
    return BUNDLE


def decode_image(data_url):
    raw = base64.b64decode(data_url.split(",", 1)[-1])
    buf = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_REDUCED_COLOR_4)
    if img is None or max(img.shape[:2]) < 800:
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return img


def to_data_url(img, quality=85):
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def pipeline_steps(img):
    """Ảnh minh họa từng bước (4 → 6) cho một ảnh."""

    roi, mask, info = find_leaf(img)
    vis = img.copy()
    if info["box"] is not None:
        t = max(2, img.shape[1] // 250)
        cv2.drawContours(vis, [info["box"].astype(np.int32)], -1, (0, 0, 255), t)

    roi_n = normalize_illumination(roi)
    gray = cv2.cvtColor(roi_n, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, 8, 1, method="uniform")
    lbp = (lbp / lbp.max() * 255).astype(np.uint8) * (mask > 0)

    shape_vis = roi.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        cv2.drawContours(shape_vis, [cv2.convexHull(cnt)], -1, (255, 0, 0), 2)
        cv2.drawContours(shape_vis, [cnt], -1, (0, 0, 255), 2)

    full_mask = info["mask"] if info["mask"] is not None else np.zeros(img.shape[:2], np.uint8)
    steps = [
        ("Ảnh vào + khung lá", vis),
        ("Bước 5: mask tách lá", full_mask),
        ("Lá xoay về trục chuẩn", roi),
        ("Bước 4: cân bằng sáng (CLAHE)", roi_n),
        ("Texture: LBP", lbp),
        ("Shape: contour + bao lồi", shape_vis),
    ]
    geo = shape_features(mask)
    shape_info = {
        "Tỉ lệ dài/rộng": round(float(geo[1]), 2),
        "Độ tròn": round(float(geo[0]), 2),
        "Độ đặc (solidity)": round(float(geo[3]), 2),
        "Số răng cưa": int(geo[6]),
    }
    return [{"title": t, "img": to_data_url(resize_long(s, 640))} for t, s in steps], \
        shape_info, info["method"]


@app.post("/api/predict")
def predict(payload: dict):
    if not os.path.exists(MODEL_PATH):
        return JSONResponse({"error": "Chưa có model. Hãy chạy nhan_dang_rau_thom.py trước."},
                            status_code=400)
    t0 = time.time()
    bundle = load_bundle()

    img = decode_image(payload["image"])
    if img is None:
        return JSONResponse({"error": "Không đọc được ảnh."}, status_code=400)
    img = resize_long(img)

    transform = payload.get("transform")
    if transform in AUGMENTS:
        img = AUGMENTS[transform](img, np.random.default_rng())

    f, _, info = extract_features(img, MODES[bundle["mode"]])
    one = {g: f[g][None] for g in GROUPS}
    one["orb"] = [f["orb"]]
    bovw = BoVW(kmeans=bundle["kmeans"]) if bundle["kmeans"] is not None else None
    X = build_X(one, bundle["groups"], None, bovw)

    model = bundle["model"]
    proba = model.predict_proba(X)[0]
    order = np.argsort(proba)[::-1]
    classes = bundle["classes"]
    probs = [{"key": classes[int(model.classes_[i])],
              "name": CLASS_NAMES[classes[int(model.classes_[i])]],
              "p": float(proba[i])} for i in order]

    steps, shape_info, method = pipeline_steps(img)
    return {
        "label": probs[0]["name"],
        "key": probs[0]["key"],
        "confidence": probs[0]["p"],
        "probs": probs,
        "steps": steps,
        "shape": shape_info,
        "leaf_found": method == "tach_la",
        "transform": transform if transform in AUGMENTS else None,
        "ms": int((time.time() - t0) * 1000),
    }


@app.get("/api/info")
def info():
    if not os.path.exists(MODEL_PATH):
        return {"ready": False}
    cfg = load_bundle()["config"]
    metrics = {}
    log = os.path.join(RESULTS_DIR, "log.txt")
    if os.path.exists(log):
        text = open(log, encoding="utf-8").read()
        for key, pat in [("f1", r"F1 group-CV\s*:\s*([\d.]+ ± [\d.]+)"),
                         ("acc", r"Accuracy gộp\s*:\s*([\d.]+)")]:
            m = re.search(pat, text)
            if m:
                metrics[key] = m.group(1)
    return {"ready": True, "features": cfg["features"], "clf": cfg["clf"],
            "augment": bool(cfg.get("augment")), "metrics": metrics,
            "transforms": list(AUGMENTS)}


@app.get("/api/samples")
def samples(per_class: int = 3):
    out = []
    for key in load_bundle()["classes"]:
        files = sorted(f for f in os.listdir(os.path.join(DEFAULT_DATASET, key))
                       if f.lower().endswith((".jpg", ".jpeg", ".png")))
        pick = [files[int(i)] for i in np.linspace(0, len(files) - 1, per_class)]
        out += [{"key": key, "name": CLASS_NAMES[key], "url": f"/dataset/{key}/{f}"}
                for f in pick]
    return out


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE


PAGE = r"""<!doctype html>
<html lang="vi">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Nhận dạng rau thơm</title>
<style>
  :root {
    --bg: #f4f7f2; --card: #ffffff; --ink: #1d2b1f; --muted: #5d6b60;
    --line: #dfe7dc; --accent: #2f7d4a; --accent-soft: #e3f1e7; --warn: #b4532a;
  }
  * { box-sizing: border-box; }
  body { margin: 0; font-family: "Segoe UI", system-ui, sans-serif; background: var(--bg);
         color: var(--ink); }
  header { background: var(--accent); color: #fff; padding: 18px 28px; }
  header h1 { margin: 0; font-size: 22px; }
  header p { margin: 4px 0 0; opacity: .85; font-size: 14px; }
  nav { display: flex; gap: 4px; padding: 0 28px; background: #276a3e; }
  nav button { background: none; border: 0; color: #d6ecdc; padding: 10px 16px;
               font-size: 14px; cursor: pointer; border-bottom: 3px solid transparent; }
  nav button.on { color: #fff; border-bottom-color: #fff; }
  main { max-width: 1280px; margin: 0 auto; padding: 22px 28px 40px; }
  .grid { display: grid; grid-template-columns: 380px 1fr; gap: 22px; align-items: start; }
  .card { background: var(--card); border: 1px solid var(--line); border-radius: 12px;
          padding: 18px; }
  .card h2 { margin: 0 0 12px; font-size: 16px; }
  .drop { border: 2px dashed #a9c7b1; border-radius: 10px; padding: 22px; text-align: center;
          color: var(--muted); cursor: pointer; transition: background .15s; }
  .drop.over, .drop:hover { background: var(--accent-soft); }
  .row { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 12px; }
  .btn { border: 1px solid var(--line); background: #fff; border-radius: 8px; padding: 7px 12px;
         font-size: 13px; cursor: pointer; color: var(--ink); }
  .btn:hover { border-color: var(--accent); }
  .btn.primary { background: var(--accent); color: #fff; border-color: var(--accent); }
  .btn.on { background: var(--accent-soft); border-color: var(--accent); color: var(--accent); }
  .btn:disabled { opacity: .45; cursor: default; }
  video, #preview { width: 100%; border-radius: 8px; margin-top: 12px; background: #000; }
  #preview { background: none; max-height: 260px; object-fit: contain; }
  .samples { display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px; }
  .samples img { width: 100%; aspect-ratio: 3/4; object-fit: cover; border-radius: 6px;
                 cursor: pointer; border: 2px solid transparent; }
  .samples img:hover { border-color: var(--accent); }
  .samples .cap { font-size: 12px; color: var(--muted); grid-column: 1 / -1; margin-top: 6px; }
  .label { font-size: 34px; font-weight: 700; color: var(--accent); margin: 0; }
  .sub { color: var(--muted); font-size: 13px; margin-top: 4px; }
  .bars { margin-top: 14px; }
  .bar { display: grid; grid-template-columns: 90px 1fr 54px; gap: 10px; align-items: center;
         margin: 7px 0; font-size: 14px; }
  .track { background: #edf2eb; border-radius: 6px; height: 14px; overflow: hidden; }
  .fill { background: #9cc5a8; height: 100%; border-radius: 6px; transition: width .4s; }
  .bar.top .fill { background: var(--accent); }
  .steps { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-top: 6px; }
  .steps figure { margin: 0; }
  .steps img { width: 100%; height: 170px; object-fit: contain; border-radius: 8px;
               border: 1px solid var(--line); background: #fff; }
  .steps figcaption { font-size: 12px; color: var(--muted); margin-top: 4px; }
  .chips { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 12px; }
  .chip { background: var(--accent-soft); border-radius: 999px; padding: 4px 10px; font-size: 12px; }
  .empty { color: var(--muted); text-align: center; padding: 60px 20px; }
  .warn { color: var(--warn); font-size: 13px; margin-top: 8px; }
  .res-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
  .res-grid img { width: 100%; border-radius: 8px; border: 1px solid var(--line); }
  .res-grid p { font-size: 13px; color: var(--muted); margin: 6px 0 0; }
  .spinner { display: inline-block; width: 16px; height: 16px; border: 2px solid #cfe3d5;
             border-top-color: var(--accent); border-radius: 50%; animation: s 1s linear infinite;
             vertical-align: middle; margin-right: 6px; }
  @keyframes s { to { transform: rotate(360deg); } }
  @media (max-width: 900px) { .grid, .res-grid { grid-template-columns: 1fr; }
                              .steps { grid-template-columns: repeat(2, 1fr); } }
</style>
</head>
<body>
<header>
  <h1>🌿 Nhận dạng 4 loại rau thơm từ ảnh lá</h1>
  <p id="modelInfo">Húng quế · Tía tô · Mùi tàu · Lá lốt — đang tải mô hình…</p>
</header>
<nav>
  <button class="on" data-tab="demo">Nhận dạng</button>
  <button data-tab="results">Kết quả thực nghiệm</button>
</nav>

<main>
<section id="tab-demo">
  <div class="grid">
    <div>
      <div class="card">
        <h2>1. Ảnh đầu vào</h2>
        <div class="drop" id="drop">Kéo thả ảnh vào đây<br>hoặc <b>bấm để chọn ảnh</b></div>
        <input type="file" id="file" accept="image/*" hidden>
        <div class="row">
          <button class="btn" id="camBtn">📷 Bật webcam</button>
          <button class="btn primary" id="snapBtn" disabled>Chụp</button>
        </div>
        <video id="video" autoplay playsinline hidden></video>
        <img id="preview" hidden alt="">
      </div>

      <div class="card" style="margin-top:16px">
        <h2>2. Thử biến đổi ảnh</h2>
        <div class="row" id="transforms" style="margin-top:0"></div>
        <div class="sub">Áp biến đổi lên ảnh rồi nhận dạng lại — kiểm tra độ bền của mô hình.</div>
      </div>

      <div class="card" style="margin-top:16px">
        <h2>Ảnh mẫu</h2>
        <div class="samples" id="samples"></div>
        <div class="sub">Lưu ý: ảnh mẫu nằm trong tập huấn luyện. Ảnh lá mới / webcam mới là phép thử thật.</div>
      </div>
    </div>

    <div>
      <div class="card" id="result">
        <div class="empty">Chọn một ảnh lá để bắt đầu.</div>
      </div>
      <div class="card" style="margin-top:16px">
        <h2>Các bước xử lý</h2>
        <div class="steps" id="steps"><div class="empty" style="grid-column:1/-1">—</div></div>
      </div>
    </div>
  </div>
</section>

<section id="tab-results" hidden>
  <div class="res-grid">
    <div class="card"><h2>Kịch bản 1 – Đặc trưng × bộ phân lớp</h2>
      <img src="/results/kich_ban1_heatmap.png" alt="">
      <p>F1 trung bình ± độ lệch chuẩn, group 5-fold cross-validation theo khối ảnh.</p></div>
    <div class="card"><h2>Kịch bản 5 – Độ bền với biến đổi ảnh</h2>
      <img src="/results/kich_ban5.png" alt="">
      <p>Tăng cường dữ liệu giúp mô hình chịu được ảnh mờ và nhiễu.</p></div>
    <div class="card"><h2>Kịch bản 4 – Chia ngẫu nhiên vs theo khối</h2>
      <img src="/results/kich_ban4.png" alt="">
      <p>Chia ngẫu nhiên để ảnh cùng một lá ở cả train và test → kết quả cao ảo.</p></div>
    <div class="card"><h2>Bước 12 – Confusion matrix</h2>
      <img src="/results/buoc12_confusion.png" alt="">
      <p>Dự đoán out-of-fold trên 198 ảnh.</p></div>
    <div class="card"><h2>Các phép biến đổi ảnh</h2>
      <img src="/results/buoc06_bien_doi_anh.png" alt=""></div>
    <div class="card"><h2>Ảnh bị đoán sai</h2>
      <img src="/results/buoc12_anh_sai.png" alt="">
      <p>Cả 10 ảnh sai thuộc cùng một chiếc lá tía tô tròn, bị nhầm thành lá lốt.</p></div>
  </div>
</section>
</main>

<script>
const $ = s => document.querySelector(s);
let current = null;          // dataURL ảnh đang dùng
let transform = null;
let stream = null;

// ---------- tabs ----------
document.querySelectorAll("nav button").forEach(b => b.onclick = () => {
  document.querySelectorAll("nav button").forEach(x => x.classList.toggle("on", x === b));
  $("#tab-demo").hidden = b.dataset.tab !== "demo";
  $("#tab-results").hidden = b.dataset.tab !== "results";
});

// ---------- thông tin mô hình ----------
fetch("/api/info").then(r => r.json()).then(info => {
  if (!info.ready) { $("#modelInfo").textContent = "Chưa có mô hình – hãy chạy nhan_dang_rau_thom.py"; return; }
  const m = info.metrics;
  $("#modelInfo").textContent = `Mô hình: ${info.features} + ${info.clf}` +
    (info.augment ? " (có tăng cường dữ liệu)" : "") +
    (m.f1 ? ` · F1 cross-validation ${m.f1}` : "") + (m.acc ? ` · Accuracy ${m.acc}` : "");
  const box = $("#transforms");
  ["Ảnh gốc", ...info.transforms].forEach((t, i) => {
    const b = document.createElement("button");
    b.className = "btn" + (i === 0 ? " on" : "");
    b.textContent = t;
    b.onclick = () => {
      transform = i === 0 ? null : t;
      box.querySelectorAll(".btn").forEach(x => x.classList.toggle("on", x === b));
      if (current) predict();
    };
    box.appendChild(b);
  });
  loadSamples();
});

function loadSamples() {
  fetch("/api/samples").then(r => r.json()).then(list => {
    const box = $("#samples");
    let last = null;
    list.forEach(s => {
      if (s.name !== last) {
        const c = document.createElement("div"); c.className = "cap"; c.textContent = s.name;
        box.appendChild(c); last = s.name;
      }
      const img = document.createElement("img");
      img.src = s.url; img.title = s.url.split("/").pop();
      img.onclick = () => fetch(s.url).then(r => r.blob()).then(useBlob);
      box.appendChild(img);
    });
  });
  // Mở sẵn một ảnh: /?anh=/dataset/tia_to/tia_to_30.jpg
  const anh = new URLSearchParams(location.search).get("anh");
  if (anh) fetch(anh).then(r => r.blob()).then(useBlob);
}

// ---------- nhận ảnh ----------
function useBlob(blob) {
  const url = URL.createObjectURL(blob);
  const im = new Image();
  im.onload = () => {
    // thu nhỏ phía trình duyệt cho nhanh (ảnh điện thoại 50MP)
    const s = Math.min(1, 1600 / Math.max(im.width, im.height));
    const c = document.createElement("canvas");
    c.width = Math.round(im.width * s); c.height = Math.round(im.height * s);
    c.getContext("2d").drawImage(im, 0, 0, c.width, c.height);
    setImage(c.toDataURL("image/jpeg", 0.92));
    URL.revokeObjectURL(url);
  };
  im.src = url;
}

function setImage(dataUrl) {
  current = dataUrl;
  $("#preview").src = dataUrl; $("#preview").hidden = false;
  predict();
}

const drop = $("#drop");
drop.onclick = () => $("#file").click();
$("#file").onchange = e => e.target.files[0] && useBlob(e.target.files[0]);
drop.ondragover = e => { e.preventDefault(); drop.classList.add("over"); };
drop.ondragleave = () => drop.classList.remove("over");
drop.ondrop = e => { e.preventDefault(); drop.classList.remove("over");
                     e.dataTransfer.files[0] && useBlob(e.dataTransfer.files[0]); };

$("#camBtn").onclick = async () => {
  if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null;
                $("#video").hidden = true; $("#snapBtn").disabled = true;
                $("#camBtn").textContent = "📷 Bật webcam"; return; }
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 } });
    $("#video").srcObject = stream; $("#video").hidden = false;
    $("#snapBtn").disabled = false; $("#camBtn").textContent = "Tắt webcam";
  } catch (e) { alert("Không mở được webcam: " + e.message); }
};
$("#snapBtn").onclick = () => {
  const v = $("#video"), c = document.createElement("canvas");
  c.width = v.videoWidth; c.height = v.videoHeight;
  c.getContext("2d").drawImage(v, 0, 0);
  setImage(c.toDataURL("image/jpeg", 0.92));
};

// ---------- gọi mô hình ----------
async function predict() {
  $("#result").innerHTML = '<div class="empty"><span class="spinner"></span>Đang nhận dạng…</div>';
  const r = await fetch("/api/predict", { method: "POST", headers: { "Content-Type": "application/json" },
                                          body: JSON.stringify({ image: current, transform }) });
  const d = await r.json();
  if (d.error) { $("#result").innerHTML = `<div class="empty">${d.error}</div>`; return; }

  const bars = d.probs.map((p, i) => `
    <div class="bar ${i === 0 ? "top" : ""}"><span>${p.name}</span>
      <div class="track"><div class="fill" style="width:${(p.p * 100).toFixed(1)}%"></div></div>
      <b>${(p.p * 100).toFixed(0)}%</b></div>`).join("");
  const chips = Object.entries(d.shape).map(([k, v]) => `<span class="chip">${k}: <b>${v}</b></span>`).join("");
  $("#result").innerHTML = `
    <div class="sub">Kết quả${d.transform ? ` (ảnh đã <b>${d.transform.toLowerCase()}</b>)` : ""}</div>
    <p class="label">${d.label}</p>
    <div class="sub">Độ tin cậy ${(d.confidence * 100).toFixed(0)}% · xử lý ${d.ms} ms</div>
    ${d.leaf_found ? "" : '<div class="warn">⚠ Không tách được lá – đang dùng cả ảnh. Hãy đặt một lá trên nền sáng, không chạm mép ảnh.</div>'}
    <div class="bars">${bars}</div>
    <div class="chips">${chips}</div>`;
  $("#steps").innerHTML = d.steps.map(s =>
    `<figure><img src="${s.img}" alt=""><figcaption>${s.title}</figcaption></figure>`).join("");
}
</script>
</body>
</html>
"""


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    if not os.path.exists(MODEL_PATH):
        print(f"Chưa có model: {MODEL_PATH}\nHãy chạy nhan_dang_rau_thom.py trước.")
        sys.exit(1)
    load_bundle()

    # Cổng 8501 bận (vd. đang chạy app khác) → tìm cổng trống tiếp theo
    port = PORT
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex((HOST, port)) != 0:
                break
        port += 1
    url = f"http://{HOST}:{port}"
    print(f"Mở trình duyệt: {url}   (Ctrl+C để dừng)")
    if "--no-browser" not in sys.argv:
        webbrowser.open(url)
    uvicorn.run(app, host=HOST, port=port, log_level="warning")
