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
from sklearn.model_selection import StratifiedKFold, ParameterGrid
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
# BÀI TẬP CHƯƠNG 3: NHẬN DẠNG 4 LOẠI RAU THƠM TỪ ẢNH LÁ
#   Húng quế - Tía tô - Mùi tàu - Lá lốt
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
#   9. Chia dữ liệu: Group K-fold Cross-Validation theo khối ảnh
#      (ảnh của cùng một lá không nằm ở cả train và test)
#  10. Huấn luyện với bộ phân lớp (KNN, SVM, DT, RF, NB, LR)
#  11. Dự đoán dữ liệu mới
#  12. Đánh giá hệ thống (Accuracy, Precision, Recall, F1)
#
# Thêm: tăng cường dữ liệu (augmentation) bằng biến đổi ảnh - làm mờ,
#       thêm nhiễu, làm sáng, làm tối, xoay - và kiểm tra độ bền của mô
#       hình khi ảnh test bị biến đổi.
#
# Cấu trúc dữ liệu (tự chụp: mỗi ảnh MỘT lá đặt trên giấy trắng,
# tên file đánh số theo thứ tự chụp):
#   Chuong3/dataset/
#       hung_que/ hung_que_01.jpg ... hung_que_50.jpg
#       tia_to/   mui_tau/   la_lot/
#
# Chạy:
#   python nhan_dang_rau_thom.py          # kết quả lưu trong results/
#   python nhan_dang_rau_thom.py --show   # hiện thêm cửa sổ biểu đồ
#   python du_doan.py anh1.jpg ...        # dự đoán ảnh mới bằng model đã lưu
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATASET = os.path.join(BASE_DIR, "dataset")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Tên thư mục không dấu (tránh lỗi đường dẫn), tên hiển thị có dấu
CLASSES = ["hung_que", "tia_to", "mui_tau", "la_lot"]
CLASS_NAMES = {"hung_que": "Húng quế", "tia_to": "Tía tô", "mui_tau": "Mùi tàu",
               "la_lot": "Lá lốt"}
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

MAX_SIDE = 800              # ảnh điện thoại rất lớn → thu nhỏ khi đọc
ROI_W, ROI_H = 384, 192     # khung chứa lá sau khi xoay (trục dài nằm ngang)
MIN_PER_CLASS = 40          # khuyến nghị tối thiểu mỗi lớp
MIN_TO_USE = 10             # lớp có ít hơn số này sẽ bị bỏ qua
N_FOLDS = 5                 # số khối ảnh liên tiếp mỗi lớp = số fold CV
FEATURE_VERSION = 3         # tăng khi đổi cách trích đặc trưng → bỏ cache
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

def imread_unicode(path, max_side=MAX_SIDE):
    """Đọc ảnh (hỗ trợ đường dẫn tiếng Việt trên Windows).

    Ảnh điện thoại 50MP rất nặng: giải mã JPEG ở 1/4 độ phân giải (nhanh
    hơn nhiều) nếu vẫn lớn hơn max_side, vì ảnh sẽ bị thu nhỏ ngay sau đó.
    """

    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_REDUCED_COLOR_4)
    if img is None or max(img.shape[:2]) < max_side:
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img

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
#       lá lốt xanh thẫm bóng.
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
#     → Lá lốt hình tim (tròn, đặc), mùi tàu dài hẹp,
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


# ------------------------------------------------------------
# Biến đổi ảnh (data augmentation)
#   - Tăng cường dữ liệu huấn luyện: mỗi ảnh TRAIN sinh thêm 5 bản
#     biến đổi → mô hình học được lá trong nhiều điều kiện chụp hơn.
#   - Kiểm tra độ bền: áp các biến đổi (với tham số ngẫu nhiên KHÁC)
#     lên ảnh TEST → mô hình có còn nhận đúng khi ảnh xấu đi không.
# ------------------------------------------------------------

def aug_blur(img, rng):
    return cv2.GaussianBlur(img, (0, 0), rng.uniform(1.5, 3.0))


def aug_noise(img, rng):
    noise = rng.normal(0, rng.uniform(10, 20), img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)


def aug_bright(img, rng):
    return cv2.convertScaleAbs(img, alpha=rng.uniform(1.0, 1.15),
                               beta=rng.uniform(35, 60))


def aug_dark(img, rng):
    return cv2.convertScaleAbs(img, alpha=rng.uniform(0.45, 0.65), beta=0)


def aug_rotate(img, rng):
    # Lặp lại pixel viền thay vì tô đen góc ảnh → nền vẫn là giấy
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(15, 345), 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


AUGMENTS = {
    "Làm mờ": aug_blur,
    "Thêm nhiễu": aug_noise,
    "Làm sáng": aug_bright,
    "Làm tối": aug_dark,
    "Xoay": aug_rotate,
}


def extract_set(images, use_roi, transform=None, salt=0, keep_thumbs=False):
    acc = {g: [] for g in GROUPS + ["orb", "method", "thumb"]}
    for i, img in enumerate(images):
        if transform is not None:
            img = transform(img, np.random.default_rng([SEED, salt, i]))
        f, roi, info = extract_features(img, use_roi)
        for g in GROUPS + ["orb"]:
            acc[g].append(f[g])
        acc["method"].append(info["method"])
        if keep_thumbs:
            acc["thumb"].append(cv2.resize(roi, (96, 48)))
    for g in GROUPS:
        acc[g] = np.array(acc[g], dtype=np.float32)
    return acc


def compute_features(images, paths, cache_path, use_cache=True):
    """Trích trước đặc trưng cho mọi ảnh:
      - "orig": ảnh gốc, ở 2 chế độ (tách lá / cả ảnh)
      - "aug" : 5 bản biến đổi mỗi ảnh - dùng khi ảnh đó thuộc tập train
      - "pert": 5 bản biến đổi khác   - dùng khi ảnh đó thuộc tập test
    Ảnh nào là train / test do cross-validation quyết định sau.

    Kết quả được cache lại: lần chạy sau không phải trích lại nếu ảnh
    không thay đổi.
    """

    key = [FEATURE_VERSION, list(AUGMENTS)] + [
        (p, os.path.getsize(p), os.path.getmtime(p)) for p in paths]
    if use_cache and os.path.exists(cache_path):
        try:
            data = joblib.load(cache_path)
            if data.get("key") == key:
                print(f"  Dùng đặc trưng đã cache: {os.path.basename(cache_path)}")
                return data["feats"]
        except Exception:
            pass

    t0 = time.time()
    feats = {"orig": {}, "aug": {}, "pert": {}}
    for mode, use_roi in MODES.items():
        feats["orig"][mode] = extract_set(images, use_roi, keep_thumbs=(mode == "roi"))
    print(f"    ảnh gốc xong ({time.time() - t0:.0f}s)")

    for k, (kind, fn) in enumerate(AUGMENTS.items()):
        for part, salt in [("aug", 100 + k), ("pert", 200 + k)]:
            feats[part][kind] = {mode: extract_set(images, use_roi, fn, salt)
                                 for mode, use_roi in MODES.items()}
        print(f"    biến đổi '{kind}' xong ({time.time() - t0:.0f}s)")
    print(f"  Trích đặc trưng xong trong {time.time() - t0:.0f}s")

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


# ============================================================
# BƯỚC 9. CHIA DỮ LIỆU: GROUP K-FOLD CROSS-VALIDATION
# ============================================================

# Mỗi lá được chụp ~10 ảnh liên tiếp (xoay, lật mặt). Nếu chia train/test
# NGẪU NHIÊN theo ảnh, ảnh của cùng một lá nằm ở cả 2 phía → mô hình chỉ
# cần "nhớ" chiếc lá là đoán đúng → kết quả cao ảo.
#
# → Chia mỗi lớp thành N_FOLDS khối ảnh LIÊN TIẾP theo thứ tự chụp (tên
#   file đã đánh số theo thứ tự chụp). Fold k = khối k của mọi lớp.
#   Mỗi lần: 1 khối làm test, các khối còn lại làm train → K lần.
# → Siêu tham số được chọn bằng CV LỒNG (nested): trong tập train lại
#   chia theo khối để chọn; khối test của vòng ngoài không được dùng để
#   chọn → đánh giá không bị lạc quan.

def make_blocks(y):
    g = np.zeros(len(y), dtype=int)
    for c in np.unique(y):
        idx = np.where(y == c)[0]           # đã sắp theo tên file = thứ tự chụp
        g[idx] = np.arange(len(idx)) * N_FOLDS // len(idx)
    return g


def outer_splits(y, g, random_split=False):
    if random_split:
        skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
        return list(skf.split(np.zeros(len(y)), y))
    return [(np.where(g != b)[0], np.where(g == b)[0]) for b in range(N_FOLDS)]


def inner_splits(y, g, tr, random_split=False):
    if random_split:
        k = min(N_FOLDS - 1, np.bincount(y[tr]).min())
        skf = StratifiedKFold(k, shuffle=True, random_state=SEED)
        return [(tr[a], tr[b]) for a, b in skf.split(tr, y[tr])]
    return [(tr[g[tr] != b], tr[g[tr] == b]) for b in np.unique(g[tr])]


class Data:
    """Đặc trưng đã trích + nhãn + khối; dựng ma trận X cho từng tập con.

    Từ điển BoVW (ORB) phụ thuộc vào tập train → được học lại cho từng
    fold, rồi cache để các bộ phân lớp dùng chung.
    """

    def __init__(self, feats, y, g):
        self.feats, self.y, self.g = feats, y, g
        self._vocab, self._hist = {}, {}

    def source(self, mode, src):
        if src == "orig":
            return self.feats["orig"][mode]
        part, kind = src.split(":", 1)
        return self.feats[part][kind][mode]

    def vocab(self, mode, tr, use_aug=False):
        key = (mode, tr.tobytes(), use_aug)
        if key not in self._vocab:
            desc = [self.feats["orig"][mode]["orb"][i] for i in tr]
            if use_aug:
                for kind in AUGMENTS:
                    desc += [self.feats["aug"][kind][mode]["orb"][i] for i in tr]
            self._vocab[key] = BoVW().fit(desc)
        return key

    def hist(self, vkey, src):
        if (vkey, src) not in self._hist:
            self._hist[(vkey, src)] = self._vocab[vkey].transform(
                self.source(vkey[0], src)["orb"])
        return self._hist[(vkey, src)]

    def X(self, mode, groups, rows, src="orig", vkey=None):
        F = self.source(mode, src)
        return np.hstack([self.hist(vkey, src)[rows] if gr == "orb" else F[gr][rows]
                          for gr in groups])

    def dim(self, groups):
        F = self.feats["orig"]["roi"]
        return sum(100 if gr == "orb" else F[gr].shape[1] for gr in groups)


def fit_model(D, cfg, groups, params, tr, use_aug=False):
    mode = cfg["mode"]
    vkey = D.vocab(mode, tr, use_aug) if "orb" in groups else None
    X, y = [D.X(mode, groups, tr, "orig", vkey)], [D.y[tr]]
    if use_aug:
        for kind in AUGMENTS:
            X.append(D.X(mode, groups, tr, f"aug:{kind}", vkey))
            y.append(D.y[tr])
    model = make_model(cfg["clf"], params, cfg["scaler"], cfg["pca"])
    model.fit(np.vstack(X), np.concatenate(y))
    return model, vkey


def inner_tune(D, cfg, groups, tr, random_split=False):
    """Chọn siêu tham số bằng CV bên trong tập train (vòng trong)."""

    grid = list(ParameterGrid(CLASSIFIERS[cfg["clf"]][1]))
    if len(grid) == 1:
        return grid[0]

    mode, splits = cfg["mode"], []
    for a, b in inner_splits(D.y, D.g, tr, random_split):
        vkey = D.vocab(mode, a) if "orb" in groups else None
        splits.append((D.X(mode, groups, a, "orig", vkey), D.y[a],
                       D.X(mode, groups, b, "orig", vkey), D.y[b]))

    best, best_f = None, -1.0
    for params in grid:
        scores = []
        for Xa, ya, Xb, yb in splits:
            model = make_model(cfg["clf"], params, cfg["scaler"], cfg["pca"])
            pred = model.fit(Xa, ya).predict(Xb)
            scores.append(f1_score(yb, pred, average="macro", zero_division=0))
        if np.mean(scores) > best_f:
            best, best_f = params, float(np.mean(scores))
    return best


def run_cv(D, cfg, groups, random_split=False, use_aug=False, perturb=False):
    """Cross-validation vòng ngoài.

    Trả về F1 từng fold và dự đoán out-of-fold: mỗi ảnh được dự đoán bởi
    mô hình KHÔNG được học trên khối chứa ảnh đó.
    """

    y = D.y
    res = {"f1s": [], "params": [], "oof": np.full(len(y), -1),
           "pert": {k: np.full(len(y), -1) for k in AUGMENTS} if perturb else {}}
    for tr, te in outer_splits(y, D.g, random_split):
        params = inner_tune(D, cfg, groups, tr, random_split)
        model, vkey = fit_model(D, cfg, groups, params, tr, use_aug)
        pred = model.predict(D.X(cfg["mode"], groups, te, "orig", vkey))
        res["oof"][te] = pred
        res["f1s"].append(f1_score(y[te], pred, average="macro", zero_division=0))
        res["params"].append(params)
        for kind in res["pert"]:
            res["pert"][kind][te] = model.predict(
                D.X(cfg["mode"], groups, te, f"pert:{kind}", vkey))
    res["mean"], res["std"] = float(np.mean(res["f1s"])), float(np.std(res["f1s"]))
    return res


def common_params(res):
    """Bộ tham số được vòng trong chọn nhiều nhất qua các fold."""

    texts = [str(p) for p in res["params"]]
    return max(set(texts), key=texts.count)


def make_bundle(D, cfg, groups, model, vkey, classes):
    return {"classes": classes, "names": names(classes), "mode": cfg["mode"],
            "groups": groups, "model": model,
            "kmeans": D._vocab[vkey].kmeans if vkey else None,
            "config": cfg, "feature_version": FEATURE_VERSION}


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


def plot_counts(classes, labels, out):
    fig = plt.figure("Bước 2 - Số ảnh mỗi lớp", figsize=(7, 4))
    x = np.arange(len(classes))
    counts = np.bincount(labels, minlength=len(classes))
    plt.bar(x, counts, 0.6, color="seagreen")
    for i, c in enumerate(counts):
        plt.text(i, c + 0.5, str(c), ha="center")
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


def plot_heatmap(s1, fs_names, clf_names, out):
    M = np.array([[s1[(fs, c)]["mean"] for c in clf_names] for fs in fs_names])
    S = np.array([[s1[(fs, c)]["std"] for c in clf_names] for fs in fs_names])
    fig, ax = plt.subplots(figsize=(10, 6.5), num="Kịch bản 1")
    im = ax.imshow(M, cmap="YlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(clf_names)), clf_names, rotation=25, ha="right")
    ax.set_yticks(range(len(fs_names)), fs_names)
    for i in range(len(fs_names)):
        for j in range(len(clf_names)):
            ax.text(j, i, f"{M[i, j]:.2f}\n±{S[i, j]:.2f}", ha="center",
                    va="center", fontsize=7)
    ax.set_title(f"F1 (macro) - Group {N_FOLDS}-fold CV theo khối ảnh")
    fig.colorbar(im, ax=ax, shrink=0.8)
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


def plot_augment_demo(img, out):
    variants = [("Ảnh gốc", img)] + [
        (k, fn(img, np.random.default_rng(SEED))) for k, fn in AUGMENTS.items()]
    fig, axes = plt.subplots(2, len(variants), figsize=(3 * len(variants), 5),
                             num="Biến đổi ảnh")
    for c, (title, im) in enumerate(variants):
        roi, _, info = find_leaf(im)
        axes[0, c].imshow(rgb(im))
        axes[0, c].set_title(title)
        axes[1, c].imshow(rgb(roi))
        axes[1, c].set_title(f"Lá tách được ({info['method']})", fontsize=9)
        axes[0, c].axis("off")
        axes[1, c].axis("off")
    fig.tight_layout()
    savefig(fig, out)


def plot_bars(labels, series, title, out, num):
    """series: [(tên, giá trị, sai số hoặc None), ...]"""

    x = np.arange(len(labels))
    w = 0.8 / len(series)
    fig = plt.figure(num, figsize=(10, 4.5))
    for k, (name, vals, errs) in enumerate(series):
        plt.bar(x + (k - (len(series) - 1) / 2) * w, vals, w, yerr=errs,
                capsize=3, label=name)
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylabel("F1-score (macro)")
    plt.ylim(0, 1.05)
    plt.title(title)
    plt.legend()
    fig.tight_layout()
    savefig(fig, out)

# ============================================================
# CHƯƠNG TRÌNH CHÍNH
# ============================================================

def main():
    ap = argparse.ArgumentParser(description="Nhận dạng 4 loại rau thơm - 12 bước")
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
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
    t_start = time.time()

    # ------------------------------------------------------------
    section("BƯỚC 1 - XÁC ĐỊNH BÀI TOÁN VÀ LỚP ĐỐI TƯỢNG")
    # ------------------------------------------------------------
    print("""  Đối tượng    : lá của 4 loại rau thơm phổ biến ở chợ Việt Nam
  Số lớp       : 4 - phân loại ĐA LỚP
      hung_que = Húng quế : lá bầu dục nhọn, xanh đậm, bóng
      tia_to   = Tía tô   : lá tròn hơn, mép răng cưa, mặt dưới tím
      mui_tau  = Mùi tàu  : lá dài thuôn, mép răng cưa có gai, xanh nhạt
      la_lot   = Lá lốt   : lá hình tim, to, bóng, gân nổi rõ
  Cặp dễ nhầm  : Húng quế ↔ Tía tô (mặt trên tía tô cũng xanh, dáng bầu dục)
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
        print(f"""
  Chưa đủ dữ liệu (cần ít nhất 2 lớp, mỗi lớp ≥ {MIN_TO_USE} ảnh).
  Chép ảnh vào {args.dataset}{os.sep}<lớp> rồi chạy lại.""")
        return
    for c in CLASSES:
        if c not in classes:
            print(f"  [!] Lớp {c} có {counts[c]} ảnh (< {MIN_TO_USE}) → tạm bỏ qua")
        elif counts[c] < MIN_PER_CLASS:
            print(f"  [!] Lớp {c} mới có {counts[c]} ảnh, nên có ≥ {MIN_PER_CLASS}")

    t0 = time.time()
    images, y, paths = load_images(files, classes)
    print(f"  Đọc {len(images)} ảnh trong {time.time() - t0:.0f}s "
          f"(giải mã ở 1/4 độ phân giải, thu nhỏ về cạnh dài {MAX_SIDE}px)")
    plot_counts(classes, y, P("buoc02_so_anh.png"))

    # ------------------------------------------------------------
    section("BƯỚC 3 - GÁN NHÃN DỮ LIỆU")
    # ------------------------------------------------------------
    print("  Nhãn lấy theo tên thư mục, mã hóa thành số:")
    for i, c in enumerate(classes):
        print(f"    {c:9s} ({CLASS_NAMES[c]}) = {i}")

    g = make_blocks(y)
    print(f"\n  Mỗi lớp được chia thành {N_FOLDS} khối ảnh liên tiếp theo thứ tự chụp"
          f" (dùng cho bước 9):")
    for i, c in enumerate(classes):
        parts = []
        for b in range(N_FOLDS):
            idx = np.where((y == i) & (g == b))[0]
            n0 = os.path.splitext(os.path.basename(paths[idx[0]]))[0].split("_")[-1]
            n1 = os.path.splitext(os.path.basename(paths[idx[-1]]))[0].split("_")[-1]
            parts.append(f"K{b + 1}: {n0}-{n1}")
        print(f"    {c:9s} " + "  ".join(parts))
    save_csv(P("buoc03_nhan.csv"), ["anh", "nhan", "ma_nhan", "khoi"],
             [[rel(p), classes[l], l, b + 1] for p, l, b in zip(paths, y, g)])
    print("  → Đã lưu buoc03_nhan.csv")

    # ------------------------------------------------------------
    section("BƯỚC 4, 5 - TIỀN XỬ LÝ VÀ TÁCH LÁ (ROI)")
    # ------------------------------------------------------------
    plot_roi_demo(images, y, classes, P("buoc04_05_tien_xu_ly_roi.png"))
    feats = compute_features(images, paths, P("cache_dataset.joblib"),
                             not args.no_cache)

    methods = np.array(feats["orig"]["roi"]["method"])
    print("\n  Kết quả tách lá (tach_la = tách được, toan_anh = không tách được):")
    for i, c in enumerate(classes):
        m = methods[y == i]
        print(f"    {c:9s} tach_la = {np.sum(m == 'tach_la'):4d}   "
              f"toan_anh = {np.sum(m == 'toan_anh'):4d}")
    save_roi_montages(feats["orig"]["roi"]["thumb"], methods, y, classes, P("roi"))
    print("  Tách lá trên ảnh đã bị biến đổi:")
    for kind in AUGMENTS:
        m = np.array(feats["pert"][kind]["roi"]["method"])
        print(f"    {kind:11s}: {np.sum(m == 'tach_la')}/{len(m)} ảnh tách được lá")

    # ------------------------------------------------------------
    section("BƯỚC 6, 7 - TRÍCH CHỌN VÀ KẾT HỢP ĐẶC TRƯNG")
    # ------------------------------------------------------------
    F = feats["orig"]["roi"]
    n_orb = [len(d) for d in F["orb"]]
    print(f"  Color   : {F['color'].shape[1]} chiều (HSV hist + moments)")
    print(f"  Texture : {F['texture'].shape[1]} chiều (LBP x2 + GLCM)")
    print(f"  Shape   : {F['shape'].shape[1]} chiều (hình học + răng cưa + Hu + Fourier)")
    print(f"  HOG     : {F['hog'].shape[1]} chiều")
    print(f"  ORB     : trung bình {np.mean(n_orb):.0f} điểm/ảnh → BoVW 100 chiều")
    print(f"\n  Các tổ hợp đặc trưng thử nghiệm: {', '.join(FEATURE_SETS)}")
    for c in range(len(classes)):
        i = np.where(y == c)[0][0]
        plot_feature_demo(images[i], names(classes)[c],
                          P(f"buoc06_dac_trung_{classes[c]}.png"))
        if not args.show:
            plt.close("all")
    plot_augment_demo(images[np.where(y == 0)[0][0]], P("buoc06_bien_doi_anh.png"))

    # ------------------------------------------------------------
    section(f"BƯỚC 9 - CHIA DỮ LIỆU: GROUP {N_FOLDS}-FOLD CROSS-VALIDATION")
    # ------------------------------------------------------------
    D = Data(feats, y, g)
    print(f"""  Mỗi lá được chụp ~10 ảnh liên tiếp. Chia NGẪU NHIÊN theo ảnh sẽ để ảnh
  của cùng một lá ở cả train và test → mô hình "nhớ lá" → kết quả cao ảo.
  → Fold k: test = khối k của mọi lớp, train = {N_FOLDS - 1} khối còn lại.
  → Siêu tham số chọn bằng CV lồng bên trong tập train (theo khối).
  → Kết quả báo cáo = trung bình ± độ lệch chuẩn F1 (macro) qua {N_FOLDS} fold.""")
    for b, (tr, te) in enumerate(outer_splits(y, g)):
        print(f"    Fold {b + 1}: train = {len(tr):3d} ảnh, test = {len(te):3d} ảnh "
              f"({', '.join(str(n) for n in np.bincount(y[te], minlength=len(classes)))})")

    # ------------------------------------------------------------
    section("BƯỚC 10 - KỊCH BẢN 1 - ĐẶC TRƯNG x BỘ PHÂN LỚP")
    # ------------------------------------------------------------
    fs_names, clf_names = list(FEATURE_SETS), list(CLASSIFIERS)
    base = {"mode": "roi", "scaler": "standard", "pca": False}
    print(f"  {'Đặc trưng':20s} {'Bộ phân lớp':20s} {'Chiều':>6s} "
          f"{'F1 (TB ± std)':>15s}  Tham số chọn nhiều nhất")
    s1 = {}
    for fs in fs_names:
        for clf in clf_names:
            r = run_cv(D, dict(base, clf=clf), FEATURE_SETS[fs])
            r["dim"] = D.dim(FEATURE_SETS[fs])
            s1[(fs, clf)] = r
            print(f"  {fs:20s} {clf:20s} {r['dim']:6d} {r['mean']:8.3f} ± "
                  f"{r['std']:.3f}  {common_params(r)}")
    save_csv(P("kich_ban1_dac_trung_x_phan_lop.csv"),
             ["dac_trung", "bo_phan_lop", "so_chieu", "f1_tb", "f1_std",
              "f1_tung_fold", "tham_so"],
             [[fs, c, s1[(fs, c)]["dim"], round(s1[(fs, c)]["mean"], 4),
               round(s1[(fs, c)]["std"], 4),
               " ".join(f"{v:.3f}" for v in s1[(fs, c)]["f1s"]),
               common_params(s1[(fs, c)])] for fs in fs_names for c in clf_names])
    plot_heatmap(s1, fs_names, clf_names, P("kich_ban1_heatmap.png"))

    best_fs, best_clf = max(s1, key=lambda k: (round(s1[k]["mean"], 4),
                                               -s1[k]["std"], -s1[k]["dim"]))
    groups = FEATURE_SETS[best_fs]
    print(f"\n  ⇒ Tốt nhất: {best_fs} + {best_clf} "
          f"(F1 = {s1[(best_fs, best_clf)]['mean']:.3f} ± {s1[(best_fs, best_clf)]['std']:.3f})")

    # ------------------------------------------------------------
    section("BƯỚC 8 - KỊCH BẢN 2 - CHUẨN HÓA VÀ GIẢM CHIỀU")
    # ------------------------------------------------------------
    print(f"  ({best_fs} + {best_clf})")
    s2 = {}
    for scaler, use_pca, name in [("standard", False, "Standardization"),
                                  ("none", False, "Không chuẩn hóa"),
                                  ("minmax", False, "Min-Max"),
                                  ("standard", True, "Standard + PCA 95%")]:
        cfg = {"mode": "roi", "clf": best_clf, "scaler": scaler, "pca": use_pca}
        s2[name] = dict(run_cv(D, cfg, groups), cfg=cfg)
        print(f"  {name:22s} F1 = {s2[name]['mean']:.3f} ± {s2[name]['std']:.3f}")
    save_csv(P("kich_ban2_chuan_hoa.csv"), ["cach", "f1_tb", "f1_std"],
             [[n, round(r["mean"], 4), round(r["std"], 4)] for n, r in s2.items()])
    best_norm = max(s2, key=lambda k: round(s2[k]["mean"], 4))
    print(f"\n  ⇒ Chọn: {best_norm}")

    # ------------------------------------------------------------
    section("BƯỚC 5 - KỊCH BẢN 3 - CÓ TÁCH LÁ (ROI) vs DÙNG CẢ ẢNH")
    # ------------------------------------------------------------
    s3 = {}
    for mode, name in [("roi", "Tách lá + xoay chuẩn"), ("full", "Cả ảnh (không tách)")]:
        cfg = dict(s2[best_norm]["cfg"], mode=mode)
        s3[mode] = dict(run_cv(D, cfg, groups), name=name)
        print(f"  {name:24s} F1 = {s3[mode]['mean']:.3f} ± {s3[mode]['std']:.3f}")
    print("  (Khi dùng cả ảnh, đặc trưng Shape không có ý nghĩa vì mask là cả khung.)")
    save_csv(P("kich_ban3_roi.csv"), ["che_do", "f1_tb", "f1_std"],
             [[r["name"], round(r["mean"], 4), round(r["std"], 4)] for r in s3.values()])
    best_mode = max(s3, key=lambda k: round(s3[k]["mean"], 4))
    cfg = dict(s2[best_norm]["cfg"], mode=best_mode)
    print(f"\n  ⇒ Cấu hình: {best_fs} | {cfg}")

    # ------------------------------------------------------------
    section("BƯỚC 9 - KỊCH BẢN 4 - CHIA NGẪU NHIÊN vs CHIA THEO KHỐI ẢNH")
    # ------------------------------------------------------------
    print("  Cùng đặc trưng / chuẩn hóa, so sánh 2 cách chia 5-fold:")
    print(f"  {'Bộ phân lớp':20s} {'Ngẫu nhiên':>14s} {'Theo khối':>14s} {'Chênh':>7s}")
    s4 = []
    for clf in clf_names:
        c = dict(cfg, clf=clf)
        r_rand = run_cv(D, c, groups, random_split=True)
        r_grp = run_cv(D, c, groups)
        s4.append([clf, r_rand, r_grp])
        print(f"  {clf:20s} {r_rand['mean']:7.3f} ± {r_rand['std']:.3f} "
              f"{r_grp['mean']:7.3f} ± {r_grp['std']:.3f} "
              f"{r_rand['mean'] - r_grp['mean']:+7.3f}")
    print("  Chênh dương = chia ngẫu nhiên cho kết quả cao hơn thực tế (mô hình 'nhớ lá').")
    save_csv(P("kich_ban4_ngau_nhien_vs_khoi.csv"),
             ["bo_phan_lop", "ngau_nhien_tb", "ngau_nhien_std", "theo_khoi_tb",
              "theo_khoi_std"],
             [[c, round(a["mean"], 4), round(a["std"], 4), round(b["mean"], 4),
               round(b["std"], 4)] for c, a, b in s4])
    plot_bars(clf_names,
              [("Chia ngẫu nhiên theo ảnh", [a["mean"] for _, a, _ in s4],
                [a["std"] for _, a, _ in s4]),
               ("Chia theo khối ảnh (group)", [b["mean"] for _, _, b in s4],
                [b["std"] for _, _, b in s4])],
              "Rò rỉ dữ liệu: chia ngẫu nhiên vs chia theo khối", P("kich_ban4.png"),
              "Kịch bản 4")

    # ------------------------------------------------------------
    section("KỊCH BẢN 5 - TĂNG CƯỜNG DỮ LIỆU VÀ ĐỘ BỀN VỚI BIẾN ĐỔI ẢNH")
    # ------------------------------------------------------------
    print(f"""  Huấn luyện: (a) chỉ ảnh gốc  |  (b) ảnh gốc + 5 bản biến đổi mỗi ảnh train
  Kiểm tra  : ảnh test gốc và ảnh test bị biến đổi (tham số ngẫu nhiên khác
              với lúc tăng cường, để không "trùng đề").
  ({best_fs}, {best_clf}, {best_norm}, {cfg['mode']})""")
    r_plain = run_cv(D, cfg, groups, perturb=True)
    r_aug = run_cv(D, cfg, groups, use_aug=True, perturb=True)

    conds = ["Ảnh gốc"] + list(AUGMENTS)

    def f1_on(r, cond):
        pred = r["oof"] if cond == "Ảnh gốc" else r["pert"][cond]
        return f1_score(y, pred, average="macro", zero_division=0)

    rows5 = [[c, f1_on(r_plain, c), f1_on(r_aug, c)] for c in conds]
    print(f"\n  {'Ảnh test':12s} {'(a) không tăng cường':>21s} {'(b) có tăng cường':>18s}")
    for c, a, b in rows5:
        print(f"  {c:12s} {a:21.3f} {b:18.3f}")
    mean_a = np.mean([a for _, a, _ in rows5])
    mean_b = np.mean([b for _, _, b in rows5])
    print(f"  {'Trung bình':12s} {mean_a:21.3f} {mean_b:18.3f}")
    save_csv(P("kich_ban5_bien_doi_anh.csv"),
             ["anh_test", "khong_tang_cuong", "co_tang_cuong"],
             [[c, round(a, 4), round(b, 4)] for c, a, b in rows5])
    plot_bars(conds, [("(a) Không tăng cường", [a for _, a, _ in rows5], None),
                      ("(b) Có tăng cường", [b for _, _, b in rows5], None)],
              "Độ bền với biến đổi ảnh test", P("kich_ban5.png"), "Kịch bản 5")

    use_aug = bool(mean_b > mean_a)
    final = r_aug if use_aug else r_plain
    print(f"\n  ⇒ {'Dùng' if use_aug else 'Không dùng'} tăng cường dữ liệu cho mô hình cuối")

    # ------------------------------------------------------------
    section("BƯỚC 10 - HUẤN LUYỆN MÔ HÌNH CUỐI (toàn bộ dữ liệu)")
    # ------------------------------------------------------------
    all_idx = np.arange(len(y))
    cfg["params"] = inner_tune(D, cfg, groups, all_idx)
    cfg["features"], cfg["augment"] = best_fs, use_aug
    model, vkey = fit_model(D, cfg, groups, cfg["params"], all_idx, use_aug)
    joblib.dump(make_bundle(D, cfg, groups, model, vkey, classes), P("model.joblib"))
    n_train = len(all_idx) * (1 + len(AUGMENTS) * use_aug)
    print(f"  Cấu hình: {cfg}")
    print(f"  Huấn luyện trên {n_train} mẫu → đã lưu model.joblib")

    # ------------------------------------------------------------
    section("BƯỚC 11 - DỰ ĐOÁN DỮ LIỆU MỚI")
    # ------------------------------------------------------------
    # Mô hình học trên khối 2-5, dự đoán ảnh khối 1 (lá chưa thấy khi học)
    # qua đúng hàm predict_image mà du_doan.py dùng cho ảnh mới.
    tr, te = outer_splits(y, g)[0]
    m1, vk1 = fit_model(D, cfg, groups, inner_tune(D, cfg, groups, tr), tr, use_aug)
    bundle1 = make_bundle(D, cfg, groups, m1, vk1, classes)
    demo = np.random.default_rng(SEED).choice(te, size=min(8, len(te)), replace=False)
    demo_pred = []
    print("  Mô hình học trên khối 2-5, dự đoán ảnh của khối 1:")
    for i in demo:
        label, conf, info, _ = predict_image(images[i], bundle1)
        demo_pred.append(classes.index(label))
        c = f" ({conf:.0%})" if conf is not None else ""
        print(f"    {os.path.basename(paths[i]):18s} thật = {classes[y[i]]:9s} "
              f"đoán = {label:9s}{c}")
    plot_predictions(images, demo, y[demo], demo_pred, classes,
                     "Bước 11 - Dự đoán ảnh của lá chưa thấy khi huấn luyện",
                     P("buoc11_du_doan.png"))
    print("  (Ảnh mới bất kỳ: python du_doan.py <ảnh hoặc thư mục>)")

    # ------------------------------------------------------------
    section("BƯỚC 12 - ĐÁNH GIÁ HỆ THỐNG (dự đoán out-of-fold)")
    # ------------------------------------------------------------
    oof = final["oof"]
    mt = metrics(y, oof)
    print(f"  Mỗi ảnh được dự đoán bởi mô hình KHÔNG học khối chứa nó.")
    print(f"  F1 từng fold: {', '.join(f'{v:.3f}' for v in final['f1s'])}"
          f"  → {final['mean']:.3f} ± {final['std']:.3f}")
    print(f"\n  Gộp {len(y)} ảnh: Accuracy = {mt['acc']:.3f}  Precision = {mt['prec']:.3f}"
          f"  Recall = {mt['rec']:.3f}  F1 = {mt['f1']:.3f}\n")
    print(classification_report(y, oof, labels=range(len(classes)),
                                target_names=names(classes), digits=3, zero_division=0))
    plot_confusion(y, oof, classes, "Confusion matrix (out-of-fold)",
                   P("buoc12_confusion.png"))
    wrong = np.where(oof != y)[0]
    print(f"  Số ảnh đoán sai: {len(wrong)} / {len(y)}")
    for w in wrong[:15]:
        print(f"    {os.path.basename(paths[w]):18s} thật = {classes[y[w]]:9s} "
              f"đoán = {classes[oof[w]]}")
    plot_predictions(images, wrong[:12], y[wrong[:12]], oof[wrong[:12]], classes,
                     "Ảnh bị đoán sai", P("buoc12_anh_sai.png"))

    # ------------------------------------------------------------
    section("TỔNG KẾT")
    # ------------------------------------------------------------
    print(f"  Cấu hình tốt nhất : {best_fs} | {best_clf} | {best_norm} | "
          f"{s3[best_mode]['name']} | tăng cường: {'có' if use_aug else 'không'}")
    print(f"  Tham số           : {cfg['params']}")
    print(f"  F1 group-CV       : {final['mean']:.3f} ± {final['std']:.3f}")
    print(f"  Accuracy gộp      : {mt['acc']:.3f}")
    print(f"  Thời gian chạy    : {(time.time() - t_start) / 60:.1f} phút")
    print(f"  Toàn bộ kết quả   : {rel(out)}{os.sep}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
