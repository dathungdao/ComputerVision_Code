import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# ============================================================
# BÀI TẬP 12: XÂY DỰNG HỆ THỐNG NHẬN DẠNG ĐỐI TƯỢNG TRUYỀN THỐNG
#
# Pipeline:
# Dataset
#   ↓
# Pre-processing
#   ↓
# Segmentation
#   ↓
# Feature Extraction
#   ├── Color   (HSV Histogram + mean/std)
#   ├── Texture (GLCM + LBP)
#   └── Shape   (Hu Moments + Circularity + Aspect Ratio ...)
#   ↓
# Feature Vector
#   ↓
# SVM / KNN
#   ↓
# Classification
#
# Dataset: 4 loại trái cây
#   - tao   (táo đỏ, tròn, bề mặt trơn)
#   - cam   (cam, tròn, bề mặt sần)
#   - chuoi (chuối vàng, dài, cong)
#   - chanh (chanh xanh, hơi bầu dục)
#
# Cấu trúc thư mục dataset:
#   dataset/
#       tao/   *.png
#       cam/   *.png
#       chuoi/ *.png
#       chanh/ *.png
#
# Nếu chưa có thư mục dataset, chương trình sẽ TỰ SINH một bộ
# dữ liệu nhỏ (ảnh tổng hợp có nhiễu, xoay, đổi kích thước,
# đổi độ sáng) để có thể chạy ngay.
# Có thể thay bằng ảnh thật bằng cách chép ảnh vào đúng thư mục.
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

CLASSES = ["tao", "cam", "chuoi", "chanh"]
IMAGES_PER_CLASS = 60
IMG_SIZE = 128

rng = np.random.default_rng(42)


# ============================================================
# 1. CHUẨN BỊ DATASET
# ============================================================

def random_background(size):
    """Nền sáng, có gradient và nhiễu nhẹ (giống nền bàn/giấy)."""

    base = rng.integers(190, 245)
    bg = np.full((size, size, 3), base, dtype=np.float32)

    # Gradient ánh sáng
    gx = np.linspace(-15, 15, size, dtype=np.float32)
    bg += gx[None, :, None] * rng.uniform(-1, 1)

    # Nhiễu
    bg += rng.normal(0, 4, bg.shape)

    return np.clip(bg, 0, 255).astype(np.uint8)


def add_surface_texture(img, mask, strength, blur):
    """Thêm vân bề mặt (ví dụ vỏ cam sần) bên trong mask."""

    noise = rng.normal(0, strength, img.shape[:2]).astype(np.float32)
    if blur > 0:
        noise = cv2.GaussianBlur(noise, (0, 0), blur)

    out = img.astype(np.float32)
    out[mask > 0] += noise[mask > 0][:, None]

    return np.clip(out, 0, 255).astype(np.uint8)


def draw_fruit(cls):
    """Vẽ một ảnh trái cây tổng hợp (BGR) cho lớp cls."""

    s = IMG_SIZE
    img = random_background(s)
    mask = np.zeros((s, s), dtype=np.uint8)

    cx = s // 2 + rng.integers(-10, 11)
    cy = s // 2 + rng.integers(-10, 11)
    angle = rng.uniform(0, 180)

    if cls == "tao":
        # Táo: đỏ, gần tròn, bề mặt trơn
        r = rng.integers(30, 42)
        cv2.ellipse(mask, (cx, cy), (r, int(r * rng.uniform(0.88, 1.0))),
                    angle, 0, 360, 255, -1)
        hue = rng.integers(0, 6)
        sat = rng.integers(170, 230)
        val = rng.integers(150, 210)
        texture = (6, 3)

    elif cls == "cam":
        # Cam: cam, tròn, bề mặt sần
        r = rng.integers(30, 42)
        cv2.ellipse(mask, (cx, cy), (r, int(r * rng.uniform(0.92, 1.0))),
                    angle, 0, 360, 255, -1)
        hue = rng.integers(10, 17)
        sat = rng.integers(190, 250)
        val = rng.integers(200, 250)
        texture = (28, 0.7)

    elif cls == "chuoi":
        # Chuối: vàng, dài, cong (vẽ bằng một cung dày)
        # Cung tròn có điểm giữa tại (cx, cy), sau đó xoay quanh (cx, cy)
        R = int(rng.integers(50, 65))
        thick = int(rng.integers(14, 20))
        half = rng.uniform(35, 50)
        cv2.ellipse(mask, (int(cx), int(cy + R)), (R, R), 0,
                    270 - half, 270 + half, 255, thick)
        M = cv2.getRotationMatrix2D((float(cx), float(cy)), angle, 1.0)
        mask = cv2.warpAffine(mask, M, (s, s), flags=cv2.INTER_NEAREST)
        hue = rng.integers(23, 30)
        sat = rng.integers(160, 230)
        val = rng.integers(200, 250)
        texture = (8, 2)

    else:  # chanh
        # Chanh: xanh lá, bầu dục nhẹ
        a = rng.integers(26, 36)
        b = int(a * rng.uniform(0.72, 0.85))
        cv2.ellipse(mask, (cx, cy), (a, b), angle, 0, 360, 255, -1)
        hue = rng.integers(38, 55)
        sat = rng.integers(150, 220)
        val = rng.integers(130, 200)
        texture = (16, 1)

    # Tô màu đối tượng (có shading tối dần ra mép)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    if dist.max() > 0:
        dist = dist / dist.max()
    shade = 0.65 + 0.35 * np.sqrt(dist)

    hsv_obj = np.zeros((s, s, 3), dtype=np.float32)
    hsv_obj[..., 0] = hue
    hsv_obj[..., 1] = sat
    hsv_obj[..., 2] = val * shade
    obj_bgr = cv2.cvtColor(np.clip(hsv_obj, 0, 255).astype(np.uint8),
                           cv2.COLOR_HSV2BGR)

    img[mask > 0] = obj_bgr[mask > 0]
    img = add_surface_texture(img, mask, *texture)

    # Làm mềm biên + nhiễu toàn ảnh
    img = cv2.GaussianBlur(img, (3, 3), 0)
    noise = rng.normal(0, 3, img.shape)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return img


def prepare_dataset():
    """Tạo dataset tổng hợp nếu thư mục dataset chưa tồn tại."""

    if os.path.isdir(DATASET_DIR) and all(
        len(glob.glob(os.path.join(DATASET_DIR, c, "*"))) > 0 for c in CLASSES
    ):
        print(f"Đã có dataset tại: {DATASET_DIR}")
        return

    print("Chưa có dataset → đang sinh dataset tổng hợp ...")

    for cls in CLASSES:
        folder = os.path.join(DATASET_DIR, cls)
        os.makedirs(folder, exist_ok=True)
        for i in range(IMAGES_PER_CLASS):
            cv2.imwrite(os.path.join(folder, f"{cls}_{i:03d}.png"),
                        draw_fruit(cls))

    print(f"Đã tạo {IMAGES_PER_CLASS} ảnh / lớp tại: {DATASET_DIR}")


def load_dataset():
    images, labels, paths = [], [], []

    for label, cls in enumerate(CLASSES):
        files = sorted(glob.glob(os.path.join(DATASET_DIR, cls, "*")))
        for f in files:
            img = cv2.imread(f)
            if img is None:
                continue
            images.append(img)
            labels.append(label)
            paths.append(f)

    return images, np.array(labels), paths


# ============================================================
# 2. TIỀN XỬ LÝ ẢNH
# ============================================================

# - Resize về cùng kích thước để đặc trưng có cùng thang đo.
# - Lọc Gaussian để giảm nhiễu trước khi phân đoạn.

def preprocess(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    return img, blurred


# ============================================================
# 3. SEGMENTATION
# ============================================================

# Trái cây có màu bão hòa (S cao), nền thì xám/trắng (S thấp).
# → Ngưỡng Otsu trên kênh S của HSV để tách đối tượng.
# → Morphology (open/close) để làm sạch mask.
# → Giữ lại contour lớn nhất làm đối tượng chính.

def segment(blurred):
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    s = hsv[..., 1]

    _, mask = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)

    clean = np.zeros_like(mask)
    contour = None
    if contours:
        contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(clean, [contour], -1, 255, -1)

    return clean, contour


# ============================================================
# 4. TRÍCH XUẤT ĐẶC TRƯNG
# ============================================================

# ------------------------------------------------------------
# 4.1 COLOR: Histogram HSV (chỉ trong vùng đối tượng)
# ------------------------------------------------------------

def color_features(img, mask):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    h_hist = cv2.calcHist([hsv], [0], mask, [18], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], mask, [8], [0, 256]).flatten()
    v_hist = cv2.calcHist([hsv], [2], mask, [8], [0, 256]).flatten()

    # Chuẩn hóa để không phụ thuộc kích thước đối tượng
    h_hist /= h_hist.sum() + 1e-8
    s_hist /= s_hist.sum() + 1e-8
    v_hist /= v_hist.sum() + 1e-8

    mean, std = cv2.meanStdDev(hsv, mask=mask)

    return np.concatenate([h_hist, s_hist, v_hist,
                           mean.flatten(), std.flatten()])


# ------------------------------------------------------------
# 4.2 TEXTURE: GLCM + LBP
# ------------------------------------------------------------

LBP_P, LBP_R = 8, 1


def texture_features(img, mask):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Cắt theo bounding box để GLCM tập trung vào đối tượng
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return np.zeros(4 + LBP_P + 2)
    roi = gray[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    roi_mask = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    # Nền ngoài mask đặt = giá trị trung bình để giảm ảnh hưởng biên
    roi = roi.copy()
    roi[roi_mask == 0] = int(roi[roi_mask > 0].mean())

    # GLCM: lượng tử về 32 mức xám
    q = (roi // 8).astype(np.uint8)
    glcm = graycomatrix(q, distances=[1, 2],
                        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                        levels=32, symmetric=True, normed=True)
    glcm_feat = [graycoprops(glcm, p).mean()
                 for p in ["contrast", "homogeneity", "energy", "correlation"]]

    # LBP uniform: histogram P + 2 bin, chỉ trong mask
    lbp = local_binary_pattern(gray, LBP_P, LBP_R, method="uniform")
    lbp_hist, _ = np.histogram(lbp[mask > 0], bins=LBP_P + 2,
                               range=(0, LBP_P + 2))
    lbp_hist = lbp_hist / (lbp_hist.sum() + 1e-8)

    return np.concatenate([glcm_feat, lbp_hist])


# ------------------------------------------------------------
# 4.3 SHAPE: Hu Moments + đặc trưng hình học
# ------------------------------------------------------------

def shape_features(mask, contour):
    if contour is None:
        return np.zeros(7 + 5)

    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    # Circularity = 4πA / P²  (hình tròn = 1)
    circularity = 4 * np.pi * area / (perimeter ** 2 + 1e-8)

    # Aspect ratio từ minAreaRect (không phụ thuộc góc xoay)
    (_, _), (w, h), _ = cv2.minAreaRect(contour)
    aspect_ratio = max(w, h) / (min(w, h) + 1e-8)

    # Extent = diện tích / diện tích hình chữ nhật bao
    extent = area / (w * h + 1e-8)

    # Solidity = diện tích / diện tích bao lồi (chuối cong → thấp)
    hull = cv2.convexHull(contour)
    solidity = area / (cv2.contourArea(hull) + 1e-8)

    # Eccentricity từ ellipse khớp
    if len(contour) >= 5:
        (_, _), (ma, MA), _ = cv2.fitEllipse(contour)
        a, b = max(ma, MA) / 2, min(ma, MA) / 2
        eccentricity = np.sqrt(1 - (b / (a + 1e-8)) ** 2)
    else:
        eccentricity = 0

    # Hu moments (log-scale) — bất biến tịnh tiến, tỉ lệ, xoay
    hu = cv2.HuMoments(cv2.moments(mask, binaryImage=True)).flatten()
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)

    return np.concatenate([hu, [circularity, aspect_ratio, extent,
                                solidity, eccentricity]])


# ============================================================
# 5. KẾT HỢP ĐẶC TRƯNG → FEATURE VECTOR
# ============================================================

def extract_features(img):
    img, blurred = preprocess(img)
    mask, contour = segment(blurred)

    f_color = color_features(img, mask)
    f_texture = texture_features(img, mask)
    f_shape = shape_features(mask, contour)

    return f_color, f_texture, f_shape


def build_feature_matrix(images):
    color, texture, shape = [], [], []
    for img in images:
        c, t, s = extract_features(img)
        color.append(c)
        texture.append(t)
        shape.append(s)
    return np.array(color), np.array(texture), np.array(shape)


# ============================================================
# 6-7. HUẤN LUYỆN & ĐÁNH GIÁ
# ============================================================

def evaluate(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        "name": name,
        "y_pred": y_pred,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="macro",
                                     zero_division=0),
        "recall": recall_score(y_test, y_pred, average="macro",
                               zero_division=0),
        "f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
    }


def train_svm(X_train, y_train):
    pipe = Pipeline([("scaler", StandardScaler()),
                     ("clf", SVC(kernel="rbf"))])
    grid = GridSearchCV(pipe, {"clf__C": [0.1, 1, 10, 100],
                               "clf__gamma": ["scale", 0.01, 0.1]},
                        cv=5, scoring="f1_macro")
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_


def train_knn(X_train, y_train):
    pipe = Pipeline([("scaler", StandardScaler()),
                     ("clf", KNeighborsClassifier())])
    grid = GridSearchCV(pipe, {"clf__n_neighbors": [1, 3, 5, 7, 9],
                               "clf__weights": ["uniform", "distance"]},
                        cv=5, scoring="f1_macro")
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_


# ============================================================
# CHƯƠNG TRÌNH CHÍNH
# ============================================================

prepare_dataset()

images, labels, paths = load_dataset()
print(f"\nTổng số ảnh: {len(images)}")
for i, c in enumerate(CLASSES):
    print(f"  {c:6s}: {np.sum(labels == i)} ảnh")


# ------------------------------------------------------------
# Minh họa pipeline trên 1 ảnh mỗi lớp
# ------------------------------------------------------------

plt.figure("Bài 12 - Pipeline", figsize=(12, 3 * len(CLASSES)))

for row, cls_id in enumerate(range(len(CLASSES))):
    idx = np.where(labels == cls_id)[0][0]
    img, blurred = preprocess(images[idx])
    mask, contour = segment(blurred)

    obj = cv2.bitwise_and(img, img, mask=mask)
    vis = img.copy()
    if contour is not None:
        cv2.drawContours(vis, [contour], -1, (255, 0, 0), 2)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, LBP_P, LBP_R, method="uniform")

    panels = [
        (cv2.cvtColor(img, cv2.COLOR_BGR2RGB), "Ảnh gốc", None),
        (mask, "Segmentation mask", "gray"),
        (cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), "Contour (Shape)", None),
        (cv2.cvtColor(obj, cv2.COLOR_BGR2RGB), "Đối tượng (Color)", None),
        (lbp * (mask > 0), "LBP (Texture)", "gray"),
    ]

    for col, (p, title, cmap) in enumerate(panels):
        plt.subplot(len(CLASSES), len(panels), row * len(panels) + col + 1)
        plt.imshow(p, cmap=cmap)
        plt.title(f"{CLASSES[cls_id]} - {title}" if col == 0 else title,
                  fontsize=9)
        plt.axis("off")

plt.tight_layout()


# ------------------------------------------------------------
# Trích xuất đặc trưng
# ------------------------------------------------------------

print("\nĐang trích xuất đặc trưng ...")
F_color, F_texture, F_shape = build_feature_matrix(images)

print(f"  Color   : {F_color.shape[1]} chiều")
print(f"  Texture : {F_texture.shape[1]} chiều")
print(f"  Shape   : {F_shape.shape[1]} chiều")

feature_sets = {
    "Color": F_color,
    "Texture": F_texture,
    "Shape": F_shape,
    "Color + Texture": np.hstack([F_color, F_texture]),
    "Color + Shape": np.hstack([F_color, F_shape]),
    "Texture + Shape": np.hstack([F_texture, F_shape]),
    "Color + Texture + Shape": np.hstack([F_color, F_texture, F_shape]),
}

# Chia train/test cố định (stratify để cân bằng lớp)
idx_train, idx_test = train_test_split(np.arange(len(labels)),
                                       test_size=0.3, stratify=labels,
                                       random_state=42)
y_train, y_test = labels[idx_train], labels[idx_test]


# ------------------------------------------------------------
# So sánh các tổ hợp đặc trưng với SVM và KNN
# ------------------------------------------------------------

print("\n" + "=" * 78)
print("SO SÁNH CÁC TỔ HỢP ĐẶC TRƯNG (macro-average trên tập test)")
print("=" * 78)
print(f"{'Đặc trưng':26s} {'Model':5s} {'Acc':>7s} {'Prec':>7s} "
      f"{'Rec':>7s} {'F1':>7s}")
print("-" * 78)

all_results = {}

for fname, X in feature_sets.items():
    X_train, X_test = X[idx_train], X[idx_test]

    svm, svm_params = train_svm(X_train, y_train)
    knn, knn_params = train_knn(X_train, y_train)

    for mname, model, params in [("SVM", svm, svm_params),
                                 ("KNN", knn, knn_params)]:
        r = evaluate(mname, model, X_test, y_test)
        r["params"] = params
        all_results[(fname, mname)] = r
        print(f"{fname:26s} {mname:5s} {r['accuracy']:7.3f} "
              f"{r['precision']:7.3f} {r['recall']:7.3f} {r['f1']:7.3f}")


# ------------------------------------------------------------
# Kết quả chi tiết với vector đặc trưng đầy đủ
# ------------------------------------------------------------

FULL = "Color + Texture + Shape"

for mname in ["SVM", "KNN"]:
    r = all_results[(FULL, mname)]
    print("\n" + "=" * 78)
    print(f"{mname} - {FULL}")
    print(f"Tham số tốt nhất: {r['params']}")
    print("=" * 78)
    print(classification_report(y_test, r["y_pred"],
                                target_names=CLASSES, digits=3,
                                zero_division=0))


# ------------------------------------------------------------
# Confusion matrix + biểu đồ so sánh
# ------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), num="Bài 12 - Confusion Matrix")
for ax, mname in zip(axes, ["SVM", "KNN"]):
    cm = confusion_matrix(y_test, all_results[(FULL, mname)]["y_pred"])
    ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(
        ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{mname} ({FULL})")
plt.tight_layout()

plt.figure("Bài 12 - So sánh F1", figsize=(11, 4.5))
names = list(feature_sets.keys())
x = np.arange(len(names))
f1_svm = [all_results[(n, "SVM")]["f1"] for n in names]
f1_knn = [all_results[(n, "KNN")]["f1"] for n in names]
plt.bar(x - 0.2, f1_svm, 0.4, label="SVM")
plt.bar(x + 0.2, f1_knn, 0.4, label="KNN")
plt.xticks(x, names, rotation=20, ha="right")
plt.ylabel("F1-score (macro)")
plt.ylim(0, 1.05)
plt.title("F1-score theo tổ hợp đặc trưng")
plt.legend()
plt.tight_layout()


# ------------------------------------------------------------
# Hiển thị một số dự đoán sai (nếu có)
# ------------------------------------------------------------

y_pred = all_results[(FULL, "SVM")]["y_pred"]
wrong = np.where(y_pred != y_test)[0]
print(f"\nSố mẫu SVM dự đoán sai: {len(wrong)} / {len(y_test)}")
for w in wrong[:10]:
    print(f"  {os.path.basename(paths[idx_test[w]])}: "
          f"thật = {CLASSES[y_test[w]]}, dự đoán = {CLASSES[y_pred[w]]}")


# ============================================================
# 8. PHÂN TÍCH ƯU / NHƯỢC ĐIỂM
# ============================================================

print("""
======================================================================
PHÂN TÍCH ƯU / NHƯỢC ĐIỂM
======================================================================

ƯU ĐIỂM
  - Cần ít dữ liệu (vài chục ảnh/lớp), huấn luyện nhanh trên CPU.
  - Đặc trưng có ý nghĩa, dễ giải thích:
      + Color  : phân biệt táo (đỏ) / cam (cam) / chuối (vàng) / chanh (xanh).
      + Texture: phân biệt vỏ sần (cam) với vỏ trơn (táo).
      + Shape  : chuối dài, cong (aspect ratio cao, solidity thấp)
                 so với táo/cam tròn (circularity ≈ 1).
  - Kết hợp nhiều nhóm đặc trưng bổ sung cho nhau → kết quả ổn định hơn
    so với dùng từng nhóm riêng lẻ.
  - SVM hoạt động tốt với dữ liệu ít, số chiều vừa phải;
    KNN đơn giản, không cần huấn luyện.

NHƯỢC ĐIỂM
  - Phụ thuộc mạnh vào bước segmentation: nền phức tạp, nền cùng màu,
    bóng đổ hoặc nhiều đối tượng chồng lấn → mask sai → đặc trưng sai.
  - Đặc trưng màu nhạy với điều kiện ánh sáng / cân bằng trắng.
  - Đặc trưng được thiết kế thủ công: phải chọn tham số (số bin, GLCM,
    LBP, ngưỡng...) theo kinh nghiệm, khó tổng quát cho bài toán mới.
  - Khó phân biệt các lớp giống nhau về màu + hình (vd: cam vs quýt).
  - KNN nhạy với việc chuẩn hóa và đặc trưng nhiễu; chậm khi dữ liệu lớn.
  - Kém hơn CNN / Deep Learning khi dữ liệu lớn và đa dạng.
""")

plt.show()
