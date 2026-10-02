import os
import sys
import csv
import glob
import time
import argparse
import warnings

import cv2
import numpy as np
import matplotlib.pyplot as plt
import joblib

from skimage.feature import (hog, local_binary_pattern,
                             graycomatrix, graycoprops)

from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     ParameterGrid)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ============================================================
# BÀI TẬP CHƯƠNG 3: NHẬN DẠNG 5 LOẠI RAU THƠM TỪ ẢNH LÁ
#   Húng quế - Tía tô - Rau răm - Mùi tàu - Lá lốt
#
# Ứng dụng: cân tự tính tiền ở siêu thị (rau không có mã vạch),
#           app hỗ trợ người đi chợ phân biệt các loại rau dễ nhầm.
#
# Áp dụng đủ 12 bước của quy trình nhận dạng truyền thống:
#
#   1. Xác định bài toán và lớp đối tượng
#   2. Thu thập và tổ chức dữ liệu
#   3. Gán nhãn dữ liệu
#   4. Tiền xử lý ảnh
#   5. Xác định vùng đối tượng (tách lá khỏi nền, xoay về trục chuẩn)
#   6. Trích chọn đặc trưng (Color, Texture, Shape, HOG, ORB)
#   7. Kết hợp các đặc trưng
#   8. Chuẩn hóa và giảm chiều (Standard / MinMax / PCA)
#   9. Chia tập dữ liệu (Train / Validation / Test)
#  10. Huấn luyện với bộ phân lớp (KNN, SVM, DT, RF, NB, LR)
#  11. Dự đoán dữ liệu mới
#  12. Đánh giá hệ thống (Accuracy, Precision, Recall, F1)
#
# Cấu trúc dữ liệu (tự chụp: mỗi ảnh MỘT lá đặt trên giấy trắng):
#   Chuong3/
#       dataset/        ← bộ chính (train / val / test)
#           hung_que/ tia_to/ rau_ram/ mui_tau/ la_lot/
#       dataset_hard/   ← bộ test "khó" (lá héo/rách, nền khác, tối...)
#           hung_que/ ...
#
# Chạy:
#   python nhan_dang_rau_thom.py          # kết quả lưu trong results/
#   python nhan_dang_rau_thom.py --show   # hiện thêm cửa sổ biểu đồ
#   python du_doan.py anh1.jpg ...        # dự đoán ảnh mới bằng model đã lưu
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATASET = os.path.join(BASE_DIR, "dataset")
DEFAULT_HARD = os.path.join(BASE_DIR, "dataset_hard")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Tên thư mục không dấu (tránh lỗi đường dẫn), tên hiển thị có dấu
CLASSES = ["hung_que", "tia_to", "rau_ram", "mui_tau", "la_lot"]
CLASS_NAMES = {"hung_que": "Húng quế", "tia_to": "Tía tô", "rau_ram": "Rau răm",
               "mui_tau": "Mùi tàu", "la_lot": "Lá lốt"}
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

MAX_SIDE = 800              # ảnh điện thoại rất lớn → thu nhỏ khi đọc
ROI_W, ROI_H = 384, 192     # khung chứa lá sau khi xoay (trục dài nằm ngang)
MIN_PER_CLASS = 60          # khuyến nghị tối thiểu mỗi lớp
MIN_TO_USE = 10             # lớp có ít hơn số này sẽ bị bỏ qua
FEATURE_VERSION = 2         # tăng khi đổi cách trích đặc trưng → bỏ cache
SEED = 42


class Tee:
    """Vừa in ra màn hình vừa ghi vào file log (dùng cho báo cáo)."""

    def __init__(self, path):
        self.file = open(path, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, s):
        self.stdout.write(s)
        self.file.write(s)

    def flush(self):
        self.stdout.flush()
        self.file.flush()


def rel(path):
    """Đường dẫn tương đối so với Chuong3 (nếu khác ổ đĩa thì giữ nguyên)."""

    try:
        return os.path.relpath(path, BASE_DIR)
    except ValueError:
        return path


def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def save_csv(path, headers, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)


# ============================================================
# BƯỚC 2. THU THẬP VÀ TỔ CHỨC DỮ LIỆU
# ============================================================

def imread_unicode(path):
    """cv2.imread không đọc được đường dẫn có dấu tiếng Việt trên Windows."""

    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path, img):
    ok, buf = cv2.imencode(os.path.splitext(path)[1], img)
    if ok:
        buf.tofile(path)


def resize_long(img, max_side=MAX_SIDE):
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s >= 1:
        return img
    return cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


def list_images(root, classes):
    """Trả về {lớp: [file...]} và danh sách file/thư mục bị bỏ qua."""

    files, skipped = {}, []
    for c in classes:
        folder = os.path.join(root, c)
        all_files = sorted(glob.glob(os.path.join(folder, "*")))
        files[c] = [f for f in all_files if f.lower().endswith(IMG_EXTS)]
        skipped += [f for f in all_files
                    if os.path.isfile(f) and not f.lower().endswith(IMG_EXTS)]

    if os.path.isdir(root):
        for d in os.listdir(root):
            if os.path.isdir(os.path.join(root, d)) and d not in classes:
                skipped.append(os.path.join(root, d) + os.sep)

    return files, skipped


def load_images(files, classes):
    images, labels, paths = [], [], []
    for label, c in enumerate(classes):
        for f in files.get(c, []):
            img = imread_unicode(f)
            if img is None:
                print(f"  [!] Không đọc được: {f}")
                continue
            images.append(resize_long(img))
            labels.append(label)
            paths.append(f)
    return images, np.array(labels, dtype=int), paths


# ============================================================
# BƯỚC 4. TIỀN XỬ LÝ ẢNH
# ============================================================

# - Resize: ảnh gốc thu nhỏ về cạnh dài 800px khi đọc (nhanh hơn),
#           lá sau khi tách được đặt vào khung cố định 384x192.
# - Lọc Gaussian trước khi tách lá để giảm nhiễu (vân giấy, bụi).
# - CLAHE trên kênh L (Lab): cân bằng histogram cục bộ, giảm ảnh hưởng
#   của ánh sáng mạnh/yếu mà không làm đổi màu (màu nằm ở kênh a, b).
# - Chuyển ảnh xám cho các đặc trưng Texture / HOG / ORB.

CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))


def normalize_illumination(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[..., 0] = CLAHE.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# ============================================================
# BƯỚC 5. XÁC ĐỊNH VÙNG ĐỐI TƯỢNG (ROI = CHIẾC LÁ)
# ============================================================

# 1. Ước lượng màu nền từ viền ảnh (giấy trắng), tính khoảng cách màu
#    Lab của từng pixel tới màu nền. Kênh L (độ sáng) được nhân trọng số
#    nhỏ → bóng đổ của lá trên giấy (chỉ tối hơn, không đổi màu) ít bị
#    nhận nhầm là lá.
# 2. Ngưỡng Otsu + morphology → giữ vùng liên thông lớn nhất, lấp lỗ
#    (vết sáng phản chiếu trên lá bóng như lá lốt).
# 3. Chuẩn hóa tư thế: PCA trên pixel của lá → xoay để trục dài nằm
#    ngang; lật để nửa có diện tích lớn hơn luôn nằm bên trái, bên trên
#    → mọi lá về cùng một tư thế chuẩn (quan trọng cho HOG).
# 4. Đặt lá vào khung 384x192 giữ nguyên tỉ lệ, nền ngoài lá tô trắng.
# Không tách được → dùng cả ảnh ("toan_anh").

L_WEIGHT = 0.4


def leaf_mask(blur):
    lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB).astype(np.float32)
    b = 10
    border = np.concatenate([lab[:b].reshape(-1, 3), lab[-b:].reshape(-1, 3),
                             lab[:, :b].reshape(-1, 3), lab[:, -b:].reshape(-1, 3)])
    bg = np.median(border, axis=0)

    diff = (lab - bg) * np.array([L_WEIGHT, 1.0, 1.0], np.float32)
    dist = np.linalg.norm(diff, axis=2)
    dist = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(dist, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    return mask


def largest_component(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(mask)
    if not contours:
        return filled, None
    cnt = max(contours, key=cv2.contourArea)
    cv2.drawContours(filled, [cnt], -1, 255, -1)      # lấp lỗ bên trong
    return filled, cnt


def rotate_bound(img, angle, center, border):
    """Xoay quanh center, mở rộng khung để không bị cắt mất phần nào."""

    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos) + 2, int(h * cos + w * sin) + 2
    M[0, 2] += nw / 2 - center[0]
    M[1, 2] += nh / 2 - center[1]
    return cv2.warpAffine(img, M, (nw, nh), borderValue=border)


def letterbox(img, mask):
    h, w = mask.shape
    s = min(ROI_W / w, ROI_H / h)
    nw, nh = max(1, int(w * s)), max(1, int(h * s))
    canvas = np.full((ROI_H, ROI_W, 3), 255, np.uint8)
    cmask = np.zeros((ROI_H, ROI_W), np.uint8)
    x, y = (ROI_W - nw) // 2, (ROI_H - nh) // 2
    canvas[y:y + nh, x:x + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    cmask[y:y + nh, x:x + nw] = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
    return canvas, cmask


def align_leaf(img, mask):
    ys, xs = np.nonzero(mask)
    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    mean = pts.mean(axis=0)
    evals, evecs = np.linalg.eigh(np.cov((pts - mean).T))
    major = evecs[:, np.argmax(evals)]
    angle = np.degrees(np.arctan2(major[1], major[0]))

    center = (float(mean[0]), float(mean[1]))
    img_r = rotate_bound(img, angle, center, (255, 255, 255))
    mask_r = rotate_bound(mask, angle, center, 0)
    mask_r = (mask_r > 127).astype(np.uint8) * 255

    ys, xs = np.nonzero(mask_r)
    img_r = img_r[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    mask_r = mask_r[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    h, w = mask_r.shape
    if mask_r[:, w - w // 2:].sum() > mask_r[:, :w // 2].sum():
        img_r, mask_r = img_r[:, ::-1], mask_r[:, ::-1]
    if mask_r[h - h // 2:].sum() > mask_r[:h // 2].sum():
        img_r, mask_r = img_r[::-1], mask_r[::-1]

    return letterbox(np.ascontiguousarray(img_r), np.ascontiguousarray(mask_r))


def find_leaf(img):
    """Trả về (roi 384x192, mask của roi, info)."""

    h, w = img.shape[:2]
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    mask, cnt = largest_component(leaf_mask(blur))

    frac = cv2.contourArea(cnt) / (h * w) if cnt is not None else 0
    if 0.003 <= frac <= 0.95:
        roi, roi_mask = align_leaf(img, mask)
        roi[roi_mask == 0] = 255
        box = cv2.boxPoints(cv2.minAreaRect(cnt))
        return roi, roi_mask, {"method": "tach_la", "box": box, "mask": mask}

    roi = cv2.resize(img, (ROI_W, ROI_H), interpolation=cv2.INTER_AREA)
    return roi, np.full((ROI_H, ROI_W), 255, np.uint8), \
        {"method": "toan_anh", "box": None, "mask": None}


# ============================================================
# BƯỚC 6. TRÍCH CHỌN ĐẶC TRƯNG
# ============================================================

# ------------------------------------------------------------
# 6.1 COLOR: Histogram HSV + Color moments (chỉ trên pixel của lá)
#     → Tía tô tím, húng quế xanh đậm, mùi tàu xanh nhạt,
#       rau răm có vệt nâu giữa lá, lá lốt xanh thẫm bóng.
# ------------------------------------------------------------

def color_features(roi, mask):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    h_hist = cv2.calcHist([hsv], [0], mask, [18], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], mask, [8], [0, 256]).flatten()
    v_hist = cv2.calcHist([hsv], [2], mask, [8], [0, 256]).flatten()
    h_hist /= h_hist.sum() + 1e-8
    s_hist /= s_hist.sum() + 1e-8
    v_hist /= v_hist.sum() + 1e-8

    px = hsv[mask > 0].astype(np.float32)
    if len(px) == 0:
        px = hsv.reshape(-1, 3).astype(np.float32)
    mean = px.mean(axis=0)
    std = px.std(axis=0)
    skew = np.cbrt(((px - mean) ** 3).mean(axis=0))

    return np.concatenate([h_hist, s_hist, v_hist, mean, std, skew])


# ------------------------------------------------------------
# 6.2 TEXTURE: LBP 2 bán kính + GLCM (bên trong lá)
#     → Gân lá nổi (lá lốt), lông tơ (tía tô), bề mặt bóng (húng quế).
#     Mask được co lại vài pixel để bỏ đường biên lá - nền.
# ------------------------------------------------------------

def texture_features(gray, mask):
    inner = cv2.erode(mask, np.ones((7, 7), np.uint8))
    if inner.sum() == 0:
        inner = mask

    lbp1 = local_binary_pattern(gray, 8, 1, method="uniform")
    h1, _ = np.histogram(lbp1[inner > 0], bins=10, range=(0, 10))
    lbp2 = local_binary_pattern(gray, 16, 2, method="uniform")
    h2, _ = np.histogram(lbp2[inner > 0], bins=18, range=(0, 18))
    h1 = h1 / (h1.sum() + 1e-8)
    h2 = h2 / (h2.sum() + 1e-8)

    # GLCM trên khung bao của lá, phần nền gán bằng mức xám trung bình
    ys, xs = np.nonzero(inner)
    box = gray[ys.min():ys.max() + 1, xs.min():xs.max() + 1].copy()
    box_m = inner[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    box[box_m == 0] = int(box[box_m > 0].mean())
    q = (box // 8).astype(np.uint8)          # 32 mức xám
    glcm = graycomatrix(q, distances=[1, 3],
                        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                        levels=32, symmetric=True, normed=True)
    g = [graycoprops(glcm, p).mean() for p in
         ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]]

    return np.concatenate([h1, h2, g])


# ------------------------------------------------------------
# 6.3 SHAPE: hình học + răng cưa + Hu moments + Fourier descriptors
#     → Lá lốt hình tim (tròn, đặc), rau răm / mùi tàu dài hẹp,
#       tía tô / mùi tàu mép răng cưa rõ (nhiều chỗ lõm, chu vi gồ ghề).
# ------------------------------------------------------------

N_FOURIER = 10
N_SHAPE = 7 + 7 + N_FOURIER


def fourier_descriptors(cnt, n=N_FOURIER, samples=128):
    """Biên độ hệ số Fourier của đường biên - bất biến tịnh tiến, tỉ lệ,
    xoay và điểm bắt đầu. Hệ số bậc thấp mô tả dáng tổng thể, bậc cao
    mô tả chi tiết mép lá (răng cưa)."""

    pts = cnt[:, 0, :].astype(np.float64)
    seg = np.sqrt((np.diff(pts, axis=0, append=pts[:1]) ** 2).sum(axis=1))
    s = np.concatenate([[0], np.cumsum(seg)[:-1]])
    total = s[-1] + seg[-1]
    t = np.linspace(0, total, samples, endpoint=False)
    x = np.interp(t, s, pts[:, 0])
    y = np.interp(t, s, pts[:, 1])

    mag = np.abs(np.fft.fft(x + 1j * y))
    return mag[2:2 + n] / (mag[1] + 1e-8)


def shape_features(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.zeros(N_SHAPE)
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    if area < 50 or len(cnt) < 5:
        return np.zeros(N_SHAPE)

    perimeter = cv2.arcLength(cnt, True)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    hull_perim = cv2.arcLength(hull, True)
    (_, _), (rw, rh), _ = cv2.minAreaRect(cnt)
    length = max(rw, rh)

    circularity = 4 * np.pi * area / (perimeter ** 2 + 1e-8)
    aspect_ratio = length / (min(rw, rh) + 1e-8)
    extent = area / (rw * rh + 1e-8)
    solidity = area / (hull_area + 1e-8)
    convexity = hull_perim / (perimeter + 1e-8)      # mép răng cưa → nhỏ

    (_, _), (ma, MA), _ = cv2.fitEllipse(cnt)
    a, b = max(ma, MA) / 2, min(ma, MA) / 2
    eccentricity = np.sqrt(max(0.0, 1 - (b / (a + 1e-8)) ** 2))

    # Số chỗ lõm đáng kể dọc mép lá (sâu > 1% chiều dài lá)
    n_teeth = 0
    try:
        defects = cv2.convexityDefects(cnt, cv2.convexHull(cnt, returnPoints=False))
        if defects is not None:
            depth = defects.reshape(-1, 4)[:, 3] / 256.0
            n_teeth = int(np.sum(depth > 0.01 * length))
    except cv2.error:
        pass

    hu = cv2.HuMoments(cv2.moments(mask, binaryImage=True)).flatten()
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)

    geo = [circularity, aspect_ratio, extent, solidity, convexity,
           eccentricity, n_teeth]
    return np.concatenate([geo, hu, fourier_descriptors(cnt)])


# ------------------------------------------------------------
# 6.4 GRADIENT: HOG trên lá đã xoay về trục chuẩn (128x64)
#     → Hướng gân lá, dáng mép lá theo từng vùng.
# ------------------------------------------------------------

HOG_SIZE = (128, 64)


def hog_features(gray, visualize=False):
    small = cv2.resize(gray, HOG_SIZE, interpolation=cv2.INTER_AREA)
    return hog(small, orientations=9, pixels_per_cell=(16, 16),
               cells_per_block=(2, 2), block_norm="L2-Hys",
               feature_vector=True, visualize=visualize)


# ------------------------------------------------------------
# 6.5 LOCAL FEATURES: ORB + Bag of Visual Words
#     → Bất biến với xoay, chịu được lá bị rách / che một phần.
#     Mỗi ảnh có số điểm ORB khác nhau → gom cụm (k-means) thành
#     "từ điển" K từ, mỗi ảnh biểu diễn bằng histogram K chiều.
#     Từ điển chỉ được học trên tập TRAIN (tránh rò rỉ dữ liệu).
# ------------------------------------------------------------

ORB = cv2.ORB_create(nfeatures=300, edgeThreshold=15, patchSize=15)


def orb_descriptors(gray):
    _, des = ORB.detectAndCompute(gray, None)
    if des is None:
        des = np.zeros((0, 32), np.uint8)
    return des


class BoVW:
    def __init__(self, k=100, kmeans=None):
        self.k = k
        self.kmeans = kmeans

    @staticmethod
    def _bits(d):
        # ORB là mô tả nhị phân 256 bit → tách bit để k-means (Euclid)
        # xấp xỉ khoảng cách Hamming.
        return np.unpackbits(d, axis=1).astype(np.float32)

    def fit(self, desc_list, max_desc=30000):
        non_empty = [d for d in desc_list if len(d)]
        if not non_empty:
            self.kmeans = None
            return self
        all_d = np.vstack(non_empty)
        rng = np.random.default_rng(SEED)
        if len(all_d) > max_desc:
            all_d = all_d[rng.choice(len(all_d), max_desc, replace=False)]
        k = min(self.k, len(all_d))
        self.kmeans = MiniBatchKMeans(n_clusters=k, random_state=SEED,
                                      n_init=3, batch_size=2048)
        self.kmeans.fit(self._bits(all_d))
        return self

    def transform(self, desc_list):
        if self.kmeans is None:
            return np.zeros((len(desc_list), 1), np.float32)
        k = self.kmeans.n_clusters
        out = np.zeros((len(desc_list), k), np.float32)
        for i, d in enumerate(desc_list):
            if len(d) == 0:
                continue
            words = self.kmeans.predict(self._bits(d))
            hist = np.bincount(words, minlength=k).astype(np.float32)
            out[i] = hist / hist.sum()
        return out


GROUPS = ["color", "texture", "shape", "hog"]     # + "orb" (xử lý riêng)


def extract_features(img, use_roi=True):
    """Bước 4 → 6 cho MỘT ảnh. Trả về (đặc trưng, roi, info)."""

    if use_roi:
        roi, mask, info = find_leaf(img)
    else:
        roi = cv2.resize(img, (ROI_W, ROI_H), interpolation=cv2.INTER_AREA)
        mask = np.full((ROI_H, ROI_W), 255, np.uint8)
        info = {"method": "toan_anh", "box": None, "mask": None}

    roi_n = normalize_illumination(roi)
    gray = cv2.cvtColor(roi_n, cv2.COLOR_BGR2GRAY)

    feats = {
        "color": color_features(roi_n, mask),
        "texture": texture_features(gray, mask),
        "shape": shape_features(mask),
        "hog": hog_features(gray),
        "orb": orb_descriptors(gray),
    }
    return feats, roi, info


MODES = {"roi": True, "full": False}


def compute_features(images, paths, cache_path, use_cache=True):
    """Trích đặc trưng cho mọi ảnh ở cả 2 chế độ (có tách lá / cả ảnh).

    Kết quả được cache lại: lần chạy sau không phải trích lại nếu ảnh
    không thay đổi.
    """

    key = [FEATURE_VERSION] + [(p, os.path.getsize(p), os.path.getmtime(p))
                               for p in paths]
    if use_cache and os.path.exists(cache_path):
        try:
            data = joblib.load(cache_path)
            if data.get("key") == key:
                print(f"  Dùng đặc trưng đã cache: {os.path.basename(cache_path)}")
                return data["feats"]
        except Exception:
            pass

    feats = {}
    t0 = time.time()
    for mode, use_roi in MODES.items():
        acc = {g: [] for g in GROUPS + ["orb", "method", "thumb"]}
        for i, img in enumerate(images):
            f, roi, info = extract_features(img, use_roi)
            for g in GROUPS + ["orb"]:
                acc[g].append(f[g])
            acc["method"].append(info["method"])
            acc["thumb"].append(cv2.resize(roi, (96, 48)))
            if (i + 1) % 100 == 0:
                print(f"    [{mode}] {i + 1}/{len(images)} ảnh")
        for g in GROUPS:
            acc[g] = np.array(acc[g], dtype=np.float32)
        feats[mode] = acc
    print(f"  Trích đặc trưng xong trong {time.time() - t0:.1f}s")

    joblib.dump({"key": key, "feats": feats}, cache_path, compress=3)
    return feats


# ============================================================
# BƯỚC 7. KẾT HỢP CÁC ĐẶC TRƯNG
# ============================================================

FEATURE_SETS = {
    "Color": ["color"],
    "Texture": ["texture"],
    "Shape": ["shape"],
    "HOG": ["hog"],
    "ORB-BoVW": ["orb"],
    "Color+Shape": ["color", "shape"],
    "Texture+Shape": ["texture", "shape"],
    "Color+Texture+Shape": ["color", "texture", "shape"],
    "Color+Tex+Shape+HOG": ["color", "texture", "shape", "hog"],
    "Tất cả": ["color", "texture", "shape", "hog", "orb"],
}


def build_X(feats, groups, idx=None, bovw=None):
    parts = []
    for g in groups:
        if g == "orb":
            d = feats["orb"] if idx is None else [feats["orb"][i] for i in idx]
            parts.append(bovw.transform(d))
        else:
            parts.append(feats[g] if idx is None else feats[g][idx])
    return np.hstack(parts)


def fit_bovw(feats, groups, idx):
    if "orb" not in groups:
        return None
    return BoVW().fit([feats["orb"][i] for i in idx])


# ============================================================
# BƯỚC 8 + 10. CHUẨN HÓA / GIẢM CHIỀU + BỘ PHÂN LỚP
# ============================================================

# Chuẩn hóa và PCA nằm trong sklearn Pipeline → chỉ "học" (fit) trên
# tập train, sau đó áp dụng y hệt cho val / test / ảnh mới.

SCALERS = {"standard": StandardScaler, "minmax": MinMaxScaler}

CLASSIFIERS = {
    "KNN": (KNeighborsClassifier(),
            {"n_neighbors": [1, 3, 5, 7], "weights": ["uniform", "distance"]}),
    "SVM": (SVC(kernel="rbf"),
            {"C": [1, 10, 100], "gamma": ["scale", 0.001]}),
    "Decision Tree": (DecisionTreeClassifier(random_state=SEED),
                      {"max_depth": [None, 10, 20]}),
    "Random Forest": (RandomForestClassifier(n_estimators=200,
                                             random_state=SEED, n_jobs=-1),
                      {"max_features": ["sqrt", 0.2]}),
    "Naive Bayes": (GaussianNB(),
                    {"var_smoothing": [1e-9, 1e-6, 1e-3]}),
    "Logistic Regression": (LogisticRegression(max_iter=3000),
                            {"C": [0.1, 1, 10]}),
}


def make_model(clf_name, params, scaler="standard", use_pca=False):
    clf = clone(CLASSIFIERS[clf_name][0]).set_params(**params)
    return Pipeline([
        ("scaler", SCALERS[scaler]() if scaler in SCALERS else "passthrough"),
        ("pca", PCA(n_components=0.95, random_state=SEED) if use_pca else "passthrough"),
        ("clf", clf),
    ])


def metrics(y_true, y_pred):
    return {
        "acc": accuracy_score(y_true, y_pred),
        "prec": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "rec": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def tune(X, y, idx_tr, idx_val, clf_name, scaler="standard", use_pca=False):
    """Chọn siêu tham số bằng tập VALIDATION (đúng vai trò ở bước 9)."""

    best = None
    for params in ParameterGrid(CLASSIFIERS[clf_name][1]):
        model = make_model(clf_name, params, scaler, use_pca)
        model.fit(X[idx_tr], y[idx_tr])
        f = f1_score(y[idx_val], model.predict(X[idx_val]),
                     average="macro", zero_division=0)
        if best is None or f > best["val_f1"]:
            best = {"val_f1": f, "params": params, "model": model}
    return best


def cv_f1(feats, groups, y, idx, cfg, params, n_splits=5):
    """K-fold CV; từ điển BoVW được học lại trong từng fold."""

    n_splits = min(n_splits, np.bincount(y[idx]).min())
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    scores = []
    for tr, te in skf.split(idx, y[idx]):
        i_tr, i_te = idx[tr], idx[te]
        bovw = fit_bovw(feats, groups, i_tr)
        model = make_model(cfg["clf"], params, cfg["scaler"], cfg["pca"])
        model.fit(build_X(feats, groups, i_tr, bovw), y[i_tr])
        pred = model.predict(build_X(feats, groups, i_te, bovw))
        scores.append(f1_score(y[i_te], pred, average="macro", zero_division=0))
    return float(np.mean(scores)), float(np.std(scores))


def final_fit(feats, groups, y, idx, cfg):
    bovw = fit_bovw(feats, groups, idx)
    model = make_model(cfg["clf"], cfg["params"], cfg["scaler"], cfg["pca"])
    model.fit(build_X(feats, groups, idx, bovw), y[idx])
    return model, bovw


# ============================================================
# BƯỚC 11. DỰ ĐOÁN DỮ LIỆU MỚI
# ============================================================

# Ảnh mới phải đi qua ĐÚNG pipeline như lúc huấn luyện:
#   resize → tách lá, xoay → CLAHE → đặc trưng → (scaler, PCA) → classifier

def predict_image(img, bundle):
    """Trả về (tên lớp, độ tin cậy hoặc None, info, roi)."""

    img = resize_long(img)
    f, roi, info = extract_features(img, MODES[bundle["mode"]])

    one = {g: f[g][None] for g in GROUPS}
    one["orb"] = [f["orb"]]
    bovw = BoVW(kmeans=bundle["kmeans"]) if bundle["kmeans"] is not None else None
    X = build_X(one, bundle["groups"], None, bovw)

    model = bundle["model"]
    pred = int(model.predict(X)[0])
    conf = None
    if hasattr(model, "predict_proba"):
        try:
            conf = float(model.predict_proba(X)[0].max())
        except AttributeError:
            conf = None
    return bundle["classes"][pred], conf, info, roi


# ============================================================
# HÌNH MINH HỌA
# ============================================================

def rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def names(classes):
    return [CLASS_NAMES.get(c, c) for c in classes]


def savefig(fig, path):
    fig.savefig(path, dpi=120, bbox_inches="tight")
    print(f"  → Đã lưu {rel(path)}")


def plot_counts(classes, labels, hard_labels, out):
    fig = plt.figure("Bước 2 - Số ảnh mỗi lớp", figsize=(8, 4))
    x = np.arange(len(classes))
    plt.bar(x - 0.2, np.bincount(labels, minlength=len(classes)), 0.4,
            label="dataset (train/val/test)")
    if hard_labels is not None and len(hard_labels):
        plt.bar(x + 0.2, np.bincount(hard_labels, minlength=len(classes)), 0.4,
                label="dataset_hard")
    plt.axhline(MIN_PER_CLASS, color="red", ls="--", lw=1,
                label=f"khuyến nghị ≥ {MIN_PER_CLASS}")
    plt.xticks(x, names(classes))
    plt.ylabel("Số ảnh")
    plt.title("Phân bố số ảnh theo loại rau")
    plt.legend()
    savefig(fig, out)


def plot_roi_demo(images, labels, classes, out):
    rows = [np.where(labels == c)[0][0] for c in range(len(classes))]
    fig, axes = plt.subplots(len(rows), 4, figsize=(13, 2.4 * len(rows)),
                             num="Bước 4-5 - Tiền xử lý và tách lá", squeeze=False)
    for r, idx in enumerate(rows):
        img = images[idx]
        roi, _, info = find_leaf(img)
        vis = img.copy()
        if info["box"] is not None:
            cv2.drawContours(vis, [info["box"].astype(np.int32)], -1, (0, 0, 255), 3)
        mask = info["mask"] if info["mask"] is not None else np.zeros(img.shape[:2])
        panels = [(rgb(vis), f"{names(classes)[labels[idx]]} - ảnh gốc"),
                  (mask, f"Mask ({info['method']})"),
                  (rgb(roi), "Lá đã xoay về trục chuẩn"),
                  (rgb(normalize_illumination(roi)), "Sau CLAHE")]
        for c, (p, t) in enumerate(panels):
            axes[r, c].imshow(p, cmap="gray" if p.ndim == 2 else None)
            axes[r, c].set_title(t, fontsize=9)
            axes[r, c].axis("off")
    fig.tight_layout()
    savefig(fig, out)


METHOD_COLORS = {"tach_la": (0, 200, 0), "toan_anh": (0, 0, 255)}


def save_roi_montages(thumbs, methods, labels, classes, out_dir, cols=10):
    """Lưới lá đã tách của từng lớp để kiểm tra bằng mắt bước 5.

    Viền xanh = tách được lá, viền đỏ = không tách được (dùng cả ảnh).
    """

    os.makedirs(out_dir, exist_ok=True)
    for c, name in enumerate(classes):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            continue
        tiles = [cv2.copyMakeBorder(thumbs[i], 3, 3, 3, 3, cv2.BORDER_CONSTANT,
                                    value=METHOD_COLORS[methods[i]]) for i in idx]
        blank = np.zeros_like(tiles[0])
        while len(tiles) % cols:
            tiles.append(blank)
        grid = np.vstack([np.hstack(tiles[i:i + cols])
                          for i in range(0, len(tiles), cols)])
        imwrite_unicode(os.path.join(out_dir, f"roi_{name}.jpg"), grid)
    print(f"  → Đã lưu lưới lá đã tách từng lớp trong {rel(out_dir)}/")


def plot_feature_demo(img, title, out):
    roi, mask, _ = find_leaf(img)
    roi_n = normalize_illumination(roi)
    gray = cv2.cvtColor(roi_n, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(roi_n, cv2.COLOR_BGR2HSV)

    # Shape: contour, bao lồi và các chỗ lõm (răng cưa)
    shape_vis = roi.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        cv2.drawContours(shape_vis, [cnt], -1, (0, 0, 255), 1)
        cv2.drawContours(shape_vis, [cv2.convexHull(cnt)], -1, (255, 0, 0), 1)

    _, hog_img = hog_features(gray, visualize=True)
    lbp = local_binary_pattern(gray, 8, 1, method="uniform")
    kps = ORB.detect(gray, None)
    orb_vis = cv2.drawKeypoints(roi_n, kps, None, color=(0, 0, 255))

    fig, axes = plt.subplots(2, 3, figsize=(13, 5.5), num="Bước 6 - Đặc trưng")
    axes[0, 0].imshow(rgb(roi_n))
    axes[0, 0].set_title(f"{title} - lá đã chuẩn hóa")
    axes[0, 1].bar(range(18), cv2.calcHist([hsv], [0], mask, [18], [0, 180]).ravel(),
                   color=[plt.cm.hsv(i / 18) for i in range(18)])
    axes[0, 1].set_title("Color: histogram Hue trong lá")
    axes[0, 2].imshow(lbp * (mask > 0), cmap="gray")
    axes[0, 2].set_title("Texture: LBP (P=8, R=1)")
    axes[1, 0].imshow(rgb(shape_vis))
    axes[1, 0].set_title("Shape: contour (đỏ) + bao lồi (xanh)")
    axes[1, 1].imshow(hog_img, cmap="gray")
    axes[1, 1].set_title("Gradient: HOG 128x64")
    axes[1, 2].imshow(rgb(orb_vis))
    axes[1, 2].set_title(f"Local: {len(kps)} điểm ORB")
    for ax in [axes[0, 0], axes[0, 2], axes[1, 0], axes[1, 1], axes[1, 2]]:
        ax.axis("off")
    fig.tight_layout()
    savefig(fig, out)


def plot_heatmaps(s1, fs_names, clf_names, out):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), num="Kịch bản 1")
    for ax, key, title in [(axes[0], "val_f1", "F1 trên VALIDATION (dùng để chọn)"),
                           (axes[1], "test_f1", "F1 trên TEST (tham khảo)")]:
        M = np.array([[s1[(fs, c)][key] for c in clf_names] for fs in fs_names])
        im = ax.imshow(M, cmap="YlGn", vmin=0, vmax=1)
        ax.set_xticks(range(len(clf_names)), clf_names, rotation=25, ha="right")
        ax.set_yticks(range(len(fs_names)), fs_names)
        for i in range(len(fs_names)):
            for j in range(len(clf_names)):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8)
        ax.set_title(title)
    fig.colorbar(im, ax=axes, shrink=0.8)
    savefig(fig, out)


def plot_confusion(y_true, y_pred, classes, title, out):
    fig, ax = plt.subplots(figsize=(6.5, 5.5), num=title)
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    ConfusionMatrixDisplay(cm, display_labels=names(classes)).plot(
        ax=ax, cmap="Greens", colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    savefig(fig, out)


def plot_predictions(images, idx, y_true, y_pred, classes, title, out, cols=4):
    if len(idx) == 0:
        return
    nm = names(classes)
    rows = int(np.ceil(len(idx) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 2.9 * rows),
                             num=title, squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, i, t, p in zip(axes.ravel(), idx, y_true, y_pred):
        ax.imshow(rgb(images[i]))
        ax.set_title(f"thật: {nm[t]} | đoán: {nm[p]}",
                     color="green" if t == p else "red", fontsize=9)
    fig.suptitle(title)
    fig.tight_layout()
    savefig(fig, out)


def plot_hard_compare(rows, out):
    labels = [r[0] for r in rows]
    x = np.arange(len(labels))
    fig = plt.figure("Kịch bản 5", figsize=(10, 4.5))
    plt.bar(x - 0.2, [r[1] for r in rows], 0.4, label="Test thường")
    plt.bar(x + 0.2, [r[2] for r in rows], 0.4, label="Test khó (dataset_hard)")
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylabel("F1-score (macro)")
    plt.ylim(0, 1.05)
    plt.title("Khả năng tổng quát hóa: test thường vs test khó")
    plt.legend()
    fig.tight_layout()
    savefig(fig, out)


# ============================================================
# CHƯƠNG TRÌNH CHÍNH
# ============================================================

def main():
    ap = argparse.ArgumentParser(description="Nhận dạng 5 loại rau thơm - 12 bước")
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--hard", default=DEFAULT_HARD)
    ap.add_argument("--results", default=RESULTS_DIR)
    ap.add_argument("--show", action="store_true", help="hiện cửa sổ biểu đồ")
    ap.add_argument("--no-cache", action="store_true", help="trích lại đặc trưng")
    args = ap.parse_args()

    if not args.show:
        plt.switch_backend("Agg")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

    out = args.results
    os.makedirs(out, exist_ok=True)
    sys.stdout = Tee(os.path.join(out, "log.txt"))
    P = lambda name: os.path.join(out, name)

    # ------------------------------------------------------------
    section("BƯỚC 1 - XÁC ĐỊNH BÀI TOÁN VÀ LỚP ĐỐI TƯỢNG")
    # ------------------------------------------------------------
    print("""  Đối tượng    : lá của 5 loại rau thơm phổ biến ở chợ Việt Nam
  Số lớp       : 5 - phân loại ĐA LỚP
      hung_que = Húng quế : lá bầu dục nhọn, xanh đậm, bóng, thân tím
      tia_to   = Tía tô   : lá tròn hơn, mép răng cưa, mặt dưới tím
      rau_ram  = Rau răm  : lá dài hẹp (mũi mác), có vệt nâu giữa lá
      mui_tau  = Mùi tàu  : lá dài thuôn, mép răng cưa có gai, xanh nhạt
      la_lot   = Lá lốt   : lá hình tim, to, bóng, gân nổi rõ
  Cặp dễ nhầm  : Húng quế ↔ Tía tô (dáng lá), Rau răm ↔ Mùi tàu (đều dài)
  Đầu vào      : ảnh đơn chụp bằng điện thoại, MỘT lá trên nền giấy trắng
  Loại bài toán: CLASSIFICATION (vị trí lá chỉ dùng để tách ROI ở bước 5)
  Ứng dụng     : cân tự tính tiền ở siêu thị, app hỗ trợ người đi chợ""")

    # ------------------------------------------------------------
    section("BƯỚC 2 - THU THẬP VÀ TỔ CHỨC DỮ LIỆU")
    # ------------------------------------------------------------
    files, skipped = list_images(args.dataset, CLASSES)
    counts = {c: len(files[c]) for c in CLASSES}
    print(f"  Thư mục: {args.dataset}")
    for c in CLASSES:
        print(f"    {c:9s} ({CLASS_NAMES[c]:8s}): {counts[c]:4d} ảnh")
    for s in skipped:
        print(f"  [!] Bỏ qua (không phải ảnh/không đúng tên lớp): {s}")

    classes = [c for c in CLASSES if counts[c] >= MIN_TO_USE]
    if len(classes) < 2:
        for c in CLASSES:
            os.makedirs(os.path.join(args.dataset, c), exist_ok=True)
            os.makedirs(os.path.join(args.hard, c), exist_ok=True)
        print(f"""
  Chưa đủ dữ liệu (cần ít nhất 2 lớp, mỗi lớp ≥ {MIN_TO_USE} ảnh).
  Đã tạo sẵn thư mục cho từng lớp:
    {args.dataset}{os.sep}<lớp>
    {args.hard}{os.sep}<lớp>
  Chép ảnh vào rồi chạy lại.""")
        return

    for c in CLASSES:
        if c not in classes:
            print(f"  [!] Lớp {c} có {counts[c]} ảnh (< {MIN_TO_USE}) → tạm bỏ qua")
        elif counts[c] < MIN_PER_CLASS:
            print(f"  [!] Lớp {c} mới có {counts[c]} ảnh, nên có ≥ {MIN_PER_CLASS}")

    images, y, paths = load_images(files, classes)
    n_cls = np.bincount(y, minlength=len(classes))
    if n_cls.max() > 1.5 * n_cls.min():
        print(f"  [!] Dữ liệu lệch lớp: ít nhất {n_cls.min()}, nhiều nhất {n_cls.max()}")

    hard_files, _ = list_images(args.hard, classes)
    h_images, h_y, h_paths = load_images(hard_files, classes)
    print(f"  Bộ chính: {len(images)} ảnh | Bộ test khó: {len(h_images)} ảnh")
    plot_counts(classes, y, h_y, P("buoc02_so_anh.png"))

    # ------------------------------------------------------------
    section("BƯỚC 3 - GÁN NHÃN DỮ LIỆU")
    # ------------------------------------------------------------
    print("  Nhãn lấy theo tên thư mục, mã hóa thành số:")
    for i, c in enumerate(classes):
        print(f"    {c:9s} ({CLASS_NAMES[c]}) = {i}")
    save_csv(P("buoc03_nhan.csv"), ["anh", "nhan", "ma_nhan", "bo_du_lieu"],
             [[rel(p), classes[l], l, "dataset"] for p, l in zip(paths, y)] +
             [[rel(p), classes[l], l, "dataset_hard"] for p, l in zip(h_paths, h_y)])
    print("  → Đã lưu buoc03_nhan.csv (mở bằng Excel để kiểm tra nhãn)")

    # ------------------------------------------------------------
    section("BƯỚC 4, 5 - TIỀN XỬ LÝ VÀ TÁCH LÁ (ROI)")
    # ------------------------------------------------------------
    plot_roi_demo(images, y, classes, P("buoc04_05_tien_xu_ly_roi.png"))

    feats = compute_features(images, paths, P("cache_dataset.joblib"),
                             not args.no_cache)
    h_feats = (compute_features(h_images, h_paths, P("cache_hard.joblib"),
                                not args.no_cache) if h_images else None)

    methods = np.array(feats["roi"]["method"])
    print("\n  Kết quả tách lá (tach_la = tách được, toan_anh = không tách được):")
    for i, c in enumerate(classes):
        m = methods[y == i]
        print(f"    {c:9s} tach_la = {np.sum(m == 'tach_la'):4d}   "
              f"toan_anh = {np.sum(m == 'toan_anh'):4d}")
    save_roi_montages(feats["roi"]["thumb"], methods, y, classes, P("roi"))
    if h_feats is not None:
        h_methods = np.array(h_feats["roi"]["method"])
        print(f"  Bộ test khó: tach_la = {np.sum(h_methods == 'tach_la')}, "
              f"toan_anh = {np.sum(h_methods == 'toan_anh')}")
        save_roi_montages(h_feats["roi"]["thumb"], h_methods, h_y, classes,
                          P("roi_hard"))

    # ------------------------------------------------------------
    section("BƯỚC 6, 7 - TRÍCH CHỌN VÀ KẾT HỢP ĐẶC TRƯNG")
    # ------------------------------------------------------------
    n_orb = [len(d) for d in feats["roi"]["orb"]]
    print(f"  Color   : {feats['roi']['color'].shape[1]} chiều (HSV hist + moments)")
    print(f"  Texture : {feats['roi']['texture'].shape[1]} chiều (LBP x2 + GLCM)")
    print(f"  Shape   : {feats['roi']['shape'].shape[1]} chiều (hình học + răng cưa"
          f" + Hu + Fourier)")
    print(f"  HOG     : {feats['roi']['hog'].shape[1]} chiều")
    print(f"  ORB     : trung bình {np.mean(n_orb):.0f} điểm/ảnh → BoVW 100 chiều")
    print(f"\n  Các tổ hợp đặc trưng thử nghiệm: {', '.join(FEATURE_SETS)}")
    for c in range(len(classes)):
        i = np.where(y == c)[0][0]
        plot_feature_demo(images[i], names(classes)[c],
                          P(f"buoc06_dac_trung_{classes[c]}.png"))
        if not args.show:
            plt.close("all")

    # ------------------------------------------------------------
    section("BƯỚC 9 - CHIA TẬP DỮ LIỆU (70% / 15% / 15%, stratify)")
    # ------------------------------------------------------------
    all_idx = np.arange(len(y))
    idx_trval, idx_test = train_test_split(all_idx, test_size=0.15,
                                           stratify=y, random_state=SEED)
    idx_tr, idx_val = train_test_split(idx_trval, test_size=0.15 / 0.85,
                                       stratify=y[idx_trval], random_state=SEED)
    print(f"  Train: {len(idx_tr)} | Validation: {len(idx_val)} | Test: {len(idx_test)}"
          f" | Test khó: {len(h_y)}")
    print("  Validation dùng để chọn đặc trưng, bộ phân lớp, siêu tham số;")
    print("  Test chỉ dùng để báo cáo, không dùng để chọn.")

    # ------------------------------------------------------------
    section("BƯỚC 10 - HUẤN LUYỆN: KỊCH BẢN 1 - ĐẶC TRƯNG x BỘ PHÂN LỚP")
    # ------------------------------------------------------------
    fs_names, clf_names = list(FEATURE_SETS), list(CLASSIFIERS)
    F = feats["roi"]
    bovw_tr = BoVW().fit([F["orb"][i] for i in idx_tr])

    print(f"  {'Đặc trưng':20s} {'Bộ phân lớp':20s} {'Chiều':>6s} "
          f"{'Val F1':>7s} {'Test F1':>8s}  Tham số")
    s1 = {}
    for fs in fs_names:
        X = build_X(F, FEATURE_SETS[fs], None, bovw_tr)
        for clf in clf_names:
            r = tune(X, y, idx_tr, idx_val, clf)
            r["test"] = metrics(y[idx_test], r["model"].predict(X[idx_test]))
            r["test_f1"] = r["test"]["f1"]
            r["dim"] = X.shape[1]
            s1[(fs, clf)] = r
            print(f"  {fs:20s} {clf:20s} {X.shape[1]:6d} {r['val_f1']:7.3f} "
                  f"{r['test_f1']:8.3f}  {r['params']}")
    save_csv(P("kich_ban1_dac_trung_x_phan_lop.csv"),
             ["dac_trung", "bo_phan_lop", "so_chieu", "val_f1", "test_acc",
              "test_f1", "tham_so"],
             [[fs, c, s1[(fs, c)]["dim"], round(s1[(fs, c)]["val_f1"], 4),
               round(s1[(fs, c)]["test"]["acc"], 4), round(s1[(fs, c)]["test_f1"], 4),
               s1[(fs, c)]["params"]] for fs in fs_names for c in clf_names])
    plot_heatmaps(s1, fs_names, clf_names, P("kich_ban1_heatmap.png"))

    # Validation nhỏ nên nhiều cấu hình có thể bằng điểm nhau → phá hòa
    # bằng 5-fold CV trên train+val (ổn định hơn một lần chia duy nhất).
    top = max(r["val_f1"] for r in s1.values())
    ties = [k for k in s1 if s1[k]["val_f1"] >= top - 1e-9]
    if len(ties) > 1:
        print(f"\n  Có {len(ties)} cấu hình cùng val F1 = {top:.3f} → phá hòa bằng "
              f"5-fold CV trên train+val:")
        for k in ties:
            cfg_k = {"clf": k[1], "scaler": "standard", "pca": False}
            m, s = cv_f1(F, FEATURE_SETS[k[0]], y, idx_trval, cfg_k, s1[k]["params"])
            s1[k]["cv"] = (m, s)
            print(f"    {k[0]:20s} {k[1]:20s} CV F1 = {m:.3f} ± {s:.3f}")
        best_fs, best_clf = max(ties, key=lambda k: (s1[k]["cv"][0], -s1[k]["cv"][1],
                                                     -s1[k]["dim"]))
    else:
        best_fs, best_clf = ties[0]
    groups = FEATURE_SETS[best_fs]
    print(f"\n  ⇒ Tốt nhất: {best_fs} + {best_clf} "
          f"(val F1 = {s1[(best_fs, best_clf)]['val_f1']:.3f})")

    # ------------------------------------------------------------
    section("BƯỚC 8 - KỊCH BẢN 2 - CHUẨN HÓA VÀ GIẢM CHIỀU")
    # ------------------------------------------------------------
    X = build_X(F, groups, None, bovw_tr)
    s2 = {}
    print(f"  ({best_fs} + {best_clf})")
    print(f"  {'Cách chuẩn hóa':22s} {'Chiều sau':>9s} {'Val F1':>7s} {'Test F1':>8s}")
    for scaler, use_pca, name in [("standard", False, "Standardization"),
                                  ("none", False, "Không chuẩn hóa"),
                                  ("minmax", False, "Min-Max"),
                                  ("standard", True, "Standard + PCA 95%")]:
        r = tune(X, y, idx_tr, idx_val, best_clf, scaler, use_pca)
        r["test_f1"] = f1_score(y[idx_test], r["model"].predict(X[idx_test]),
                                average="macro", zero_division=0)
        dim = (r["model"].named_steps["pca"].n_components_ if use_pca else X.shape[1])
        s2[name] = dict(r, scaler=scaler, pca=use_pca, dim=dim)
        print(f"  {name:22s} {dim:9d} {r['val_f1']:7.3f} {r['test_f1']:8.3f}")
    save_csv(P("kich_ban2_chuan_hoa.csv"), ["cach", "so_chieu", "val_f1", "test_f1"],
             [[n, r["dim"], round(r["val_f1"], 4), round(r["test_f1"], 4)]
              for n, r in s2.items()])
    best_norm = max(s2, key=lambda k: s2[k]["val_f1"])
    print(f"\n  ⇒ Chọn: {best_norm}")

    # ------------------------------------------------------------
    section("BƯỚC 5 - KỊCH BẢN 3 - CÓ TÁCH LÁ (ROI) vs DÙNG CẢ ẢNH")
    # ------------------------------------------------------------
    s3 = {}
    for mode, name in [("roi", "Tách lá + xoay chuẩn"), ("full", "Cả ảnh (không tách)")]:
        Fm = feats[mode]
        bovw_m = fit_bovw(Fm, groups, idx_tr)
        Xm = build_X(Fm, groups, None, bovw_m)
        r = tune(Xm, y, idx_tr, idx_val, best_clf,
                 s2[best_norm]["scaler"], s2[best_norm]["pca"])
        r["test_f1"] = f1_score(y[idx_test], r["model"].predict(Xm[idx_test]),
                                average="macro", zero_division=0)
        s3[mode] = dict(r, name=name)
        print(f"  {name:24s} Val F1 = {r['val_f1']:.3f}   Test F1 = {r['test_f1']:.3f}")
    print("  (Khi dùng cả ảnh, đặc trưng Shape không có ý nghĩa vì mask là cả khung.)")
    save_csv(P("kich_ban3_roi.csv"), ["che_do", "val_f1", "test_f1"],
             [[r["name"], round(r["val_f1"], 4), round(r["test_f1"], 4)]
              for r in s3.values()])
    best_mode = max(s3, key=lambda k: s3[k]["val_f1"])

    cfg = {"mode": best_mode, "features": best_fs, "clf": best_clf,
           "params": s3[best_mode]["params"], "scaler": s2[best_norm]["scaler"],
           "pca": s2[best_norm]["pca"]}
    print(f"\n  ⇒ Cấu hình cuối: {cfg}")
    Fm = feats[cfg["mode"]]

    # ------------------------------------------------------------
    section("BƯỚC 9 - KỊCH BẢN 4 - CÁC CÁCH CHIA DỮ LIỆU")
    # ------------------------------------------------------------
    s4 = [["70/15/15 (chọn tham số bằng validation)", s3[best_mode]["test_f1"], ""]]

    idx80, idx20 = train_test_split(all_idx, test_size=0.2, stratify=y,
                                    random_state=SEED)
    best_p, best_cv = None, -1
    for params in ParameterGrid(CLASSIFIERS[cfg["clf"]][1]):
        m, _ = cv_f1(Fm, groups, y, idx80, cfg, params)
        if m > best_cv:
            best_cv, best_p = m, params
    model80, bovw80 = final_fit(Fm, groups, y, idx80, dict(cfg, params=best_p))
    f80 = f1_score(y[idx20], model80.predict(build_X(Fm, groups, idx20, bovw80)),
                   average="macro", zero_division=0)
    s4.append(["80/20 (chọn tham số bằng 5-fold CV trên 80%)", f80,
               f"CV F1 = {best_cv:.3f}, {best_p}"])

    m, s = cv_f1(Fm, groups, y, all_idx, cfg, cfg["params"])
    s4.append(["5-fold CV trên toàn bộ dữ liệu", m, f"± {s:.3f}"])

    for name, f, note in s4:
        print(f"  {name:48s} F1 = {f:.3f}  {note}")
    save_csv(P("kich_ban4_chia_du_lieu.csv"), ["cach_chia", "f1", "ghi_chu"],
             [[n, round(f, 4), note] for n, f, note in s4])

    # ------------------------------------------------------------
    section("BƯỚC 10 - HUẤN LUYỆN MÔ HÌNH CUỐI (train + validation)")
    # ------------------------------------------------------------
    model, bovw = final_fit(Fm, groups, y, idx_trval, cfg)
    bundle = {"classes": classes, "names": names(classes), "mode": cfg["mode"],
              "groups": groups, "model": model,
              "kmeans": bovw.kmeans if bovw else None,
              "config": cfg, "feature_version": FEATURE_VERSION}
    joblib.dump(bundle, P("model.joblib"))
    print(f"  Huấn luyện trên {len(idx_trval)} ảnh → đã lưu model.joblib")

    # ------------------------------------------------------------
    section("BƯỚC 11 - DỰ ĐOÁN DỮ LIỆU MỚI")
    # ------------------------------------------------------------
    rng = np.random.default_rng(SEED)
    demo = rng.choice(idx_test, size=min(8, len(idx_test)), replace=False)
    demo_pred = []
    for i in demo:
        label, conf, info, _ = predict_image(images[i], bundle)
        demo_pred.append(classes.index(label))
        c = f" ({conf:.0%})" if conf is not None else ""
        print(f"  {os.path.basename(paths[i]):28s} thật = {classes[y[i]]:9s} "
              f"đoán = {label:9s}{c}  [{info['method']}]")
    plot_predictions(images, demo, y[demo], demo_pred, classes,
                     "Bước 11 - Dự đoán ảnh test", P("buoc11_du_doan.png"))
    print("  (Ảnh mới bất kỳ: python du_doan.py <ảnh hoặc thư mục>)")

    # ------------------------------------------------------------
    section("BƯỚC 12 - ĐÁNH GIÁ HỆ THỐNG")
    # ------------------------------------------------------------
    y_pred = model.predict(build_X(Fm, groups, idx_test, bovw))
    mt = metrics(y[idx_test], y_pred)
    print(f"  TẬP TEST ({len(idx_test)} ảnh): Accuracy = {mt['acc']:.3f}  "
          f"Precision = {mt['prec']:.3f}  Recall = {mt['rec']:.3f}  F1 = {mt['f1']:.3f}\n")
    print(classification_report(y[idx_test], y_pred, labels=range(len(classes)),
                                target_names=names(classes), digits=3, zero_division=0))
    plot_confusion(y[idx_test], y_pred, classes, "Confusion matrix - Test",
                   P("buoc12_confusion_test.png"))
    wrong = idx_test[y_pred != y[idx_test]]
    plot_predictions(images, wrong[:12], y[wrong[:12]],
                     y_pred[y_pred != y[idx_test]][:12], classes,
                     "Ảnh test bị đoán sai", P("buoc12_sai_test.png"))

    mh = None
    if h_feats is not None:
        Hm = h_feats[cfg["mode"]]
        h_pred = model.predict(build_X(Hm, groups, None, bovw))
        mh = metrics(h_y, h_pred)
        print(f"\n  TẬP TEST KHÓ ({len(h_y)} ảnh): Accuracy = {mh['acc']:.3f}  "
              f"Precision = {mh['prec']:.3f}  Recall = {mh['rec']:.3f}  F1 = {mh['f1']:.3f}\n")
        print(classification_report(h_y, h_pred, labels=range(len(classes)),
                                    target_names=names(classes), digits=3,
                                    zero_division=0))
        plot_confusion(h_y, h_pred, classes, "Confusion matrix - Test khó",
                       P("buoc12_confusion_hard.png"))
        wrong_h = np.where(h_pred != h_y)[0]
        plot_predictions(h_images, wrong_h[:12], h_y[wrong_h[:12]],
                         h_pred[wrong_h[:12]], classes,
                         "Ảnh test khó bị đoán sai", P("buoc12_sai_hard.png"))

        # --------------------------------------------------------
        section("KỊCH BẢN 5 - TỔNG QUÁT HÓA: MỌI BỘ PHÂN LỚP TRÊN TEST KHÓ")
        # --------------------------------------------------------
        Xtv = build_X(Fm, groups, None, fit_bovw(Fm, groups, idx_tr))
        rows = []
        print(f"  ({best_fs}, {cfg['mode']}, {best_norm})")
        print(f"  {'Bộ phân lớp':20s} {'Test F1':>8s} {'Test khó F1':>12s} {'Chênh':>7s}")
        for clf in clf_names:
            r = tune(Xtv, y, idx_tr, idx_val, clf, cfg["scaler"], cfg["pca"])
            m_c, b_c = final_fit(Fm, groups, y, idx_trval,
                                 dict(cfg, clf=clf, params=r["params"]))
            ft = f1_score(y[idx_test], m_c.predict(build_X(Fm, groups, idx_test, b_c)),
                          average="macro", zero_division=0)
            fh = f1_score(h_y, m_c.predict(build_X(Hm, groups, None, b_c)),
                          average="macro", zero_division=0)
            rows.append([clf, ft, fh])
            print(f"  {clf:20s} {ft:8.3f} {fh:12.3f} {fh - ft:+7.3f}")
        save_csv(P("kich_ban5_test_kho.csv"), ["bo_phan_lop", "test_f1", "hard_f1"],
                 [[c, round(a, 4), round(b, 4)] for c, a, b in rows])
        plot_hard_compare(rows, P("kich_ban5_test_kho.png"))
    else:
        print(f"\n  [!] Chưa có {args.hard} → bỏ qua đánh giá test khó và kịch bản 5")

    # ------------------------------------------------------------
    section("TỔNG KẾT")
    # ------------------------------------------------------------
    print(f"  Cấu hình tốt nhất : {best_fs} | {best_clf} | {best_norm} | "
          f"{s3[best_mode]['name']}")
    print(f"  Tham số           : {cfg['params']}")
    print(f"  F1 test           : {mt['f1']:.3f}")
    if mh is not None:
        print(f"  F1 test khó       : {mh['f1']:.3f}")
    print(f"  Toàn bộ kết quả   : {rel(out)}{os.sep}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
