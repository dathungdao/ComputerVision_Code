import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

from skimage.feature import graycomatrix, graycoprops, local_binary_pattern


# ============================================================
# CẤU HÌNH
# ============================================================

folder = "texture"

# Tham số LBP
R = 1
P = 8

# Tham số GLCM
distances = [1]
angles = [0]


# ============================================================
# LẤY DANH SÁCH ẢNH
# ============================================================

image_files = [
    f for f in os.listdir(folder)
    if f.lower().endswith((
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp"
    ))
]

image_files.sort()

if len(image_files) == 0:
    print("Không tìm thấy ảnh trong thư mục texture!")
    exit()


# ============================================================
# HÀM TÍNH GLCM
# ============================================================

def calculate_glcm_features(gray):

    # GLCM yêu cầu ảnh có giá trị nguyên
    # 0-255

    glcm = graycomatrix(
        gray,
        distances=distances,
        angles=angles,
        levels=256,
        symmetric=True,
        normed=True
    )

    contrast = graycoprops(
        glcm,
        "contrast"
    )[0, 0]

    energy = graycoprops(
        glcm,
        "energy"
    )[0, 0]

    homogeneity = graycoprops(
        glcm,
        "homogeneity"
    )[0, 0]

    return contrast, energy, homogeneity


# ============================================================
# HÀM TÍNH LBP
# ============================================================

def calculate_lbp(gray):

    lbp = local_binary_pattern(
        gray,
        P,
        R,
        method="uniform"
    )

    # Số bin đối với uniform LBP
    n_bins = P + 2

    histogram, _ = np.histogram(
        lbp.ravel(),
        bins=n_bins,
        range=(0, n_bins)
    )

    # Chuẩn hóa histogram
    histogram = histogram.astype(
        np.float32
    )

    histogram /= (
        histogram.sum() + 1e-7
    )

    return lbp, histogram


# ============================================================
# XỬ LÝ TỪNG ẢNH
# ============================================================

results = []

for filename in image_files:

    path = os.path.join(
        folder,
        filename
    )

    # --------------------------------------------------------
    # 1. ĐỌC ẢNH
    # --------------------------------------------------------

    img = cv2.imread(path)

    if img is None:
        print(
            f"Không thể đọc ảnh: {filename}"
        )
        continue

    # --------------------------------------------------------
    # 2. CHUYỂN SANG GRAYSCALE
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    # --------------------------------------------------------
    # 3. GLCM
    # --------------------------------------------------------

    contrast, energy, homogeneity = (
        calculate_glcm_features(gray)
    )

    # --------------------------------------------------------
    # 4. LBP
    # --------------------------------------------------------

    lbp, histogram = calculate_lbp(gray)

    # Lưu kết quả
    results.append({
        "filename": filename,
        "gray": gray,
        "lbp": lbp,
        "histogram": histogram,
        "contrast": contrast,
        "energy": energy,
        "homogeneity": homogeneity
    })


# ============================================================
# IN KẾT QUẢ GLCM
# ============================================================

print("=" * 80)
print("KẾT QUẢ GLCM")
print("=" * 80)

print(
    f"{'Ảnh':<25}"
    f"{'Contrast':<15}"
    f"{'Energy':<15}"
    f"{'Homogeneity':<15}"
)

print("-" * 80)

for result in results:

    print(
        f"{result['filename']:<25}"
        f"{result['contrast']:<15.4f}"
        f"{result['energy']:<15.4f}"
        f"{result['homogeneity']:<15.4f}"
    )


# ============================================================
# HIỂN THỊ LBP VÀ HISTOGRAM
# ============================================================

for result in results:

    filename = result["filename"]
    gray = result["gray"]
    lbp = result["lbp"]
    histogram = result["histogram"]

    # --------------------------------------------------------
    # Chuẩn hóa LBP để hiển thị
    # --------------------------------------------------------

    lbp_display = cv2.normalize(
        lbp,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    # --------------------------------------------------------
    # Hiển thị ảnh grayscale + LBP
    # --------------------------------------------------------

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)

    plt.imshow(
        gray,
        cmap="gray"
    )

    plt.title(
        f"Grayscale - {filename}"
    )

    plt.axis("off")


    plt.subplot(1, 2, 2)

    plt.imshow(
        lbp_display,
        cmap="gray"
    )

    plt.title(
        f"LBP - {filename}"
    )

    plt.axis("off")

    plt.tight_layout()
    plt.show()


    # --------------------------------------------------------
    # Histogram LBP
    # --------------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.bar(
        range(len(histogram)),
        histogram
    )

    plt.title(
        f"Histogram LBP - {filename}"
    )

    plt.xlabel("LBP Pattern")
    plt.ylabel("Normalized Frequency")

    plt.xticks(
        range(len(histogram))
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()
    plt.show()


# ============================================================
# SO SÁNH HISTOGRAM LBP GIỮA CÁC ẢNH
# ============================================================

plt.figure(figsize=(12, 6))

for result in results:

    plt.plot(
        range(len(result["histogram"])),
        result["histogram"],
        marker="o",
        label=result["filename"]
    )

plt.title(
    "So sánh Histogram LBP"
)

plt.xlabel("LBP Pattern")
plt.ylabel("Normalized Frequency")

plt.legend()
plt.grid()

plt.tight_layout()
plt.show()