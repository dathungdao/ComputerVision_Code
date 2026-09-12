import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


# ============================================================
# 1. CẤU HÌNH
# ============================================================

folder = "shapes"

image_files = [
    f for f in os.listdir(folder)
    if f.lower().endswith(
        (".jpg", ".jpeg", ".png", ".bmp")
    )
]

image_files.sort()

if len(image_files) == 0:
    print("Không tìm thấy ảnh trong thư mục shapes!")
    exit()


# ============================================================
# 2. THAM SỐ HOG
# ============================================================

# Kích thước ảnh đầu vào
IMAGE_SIZE = (128, 128)

# Kích thước cell
CELL_SIZE = (8, 8)

# Kích thước block
BLOCK_SIZE = (16, 16)

# Số cell trong một block
BLOCK_STRIDE = (8, 8)

# Số hướng gradient
NBINS = 9


# ============================================================
# 3. TẠO HOG DESCRIPTOR
# ============================================================

hog = cv2.HOGDescriptor(
    _winSize=IMAGE_SIZE,
    _blockSize=BLOCK_SIZE,
    _blockStride=BLOCK_STRIDE,
    _cellSize=CELL_SIZE,
    _nbins=NBINS
)


# ============================================================
# 4. HÀM TÍNH GRADIENT
# ============================================================

def calculate_gradient(gray):

    # Gradient theo chiều X
    gx = cv2.Sobel(
        gray,
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )

    # Gradient theo chiều Y
    gy = cv2.Sobel(
        gray,
        cv2.CV_32F,
        0,
        1,
        ksize=3
    )

    # Độ lớn gradient
    magnitude = cv2.magnitude(gx, gy)

    # Góc gradient
    angle = cv2.phase(
        gx,
        gy,
        angleInDegrees=True
    )

    return gx, gy, magnitude, angle


# ============================================================
# 5. TÍNH HISTOGRAM CHO MỘT CELL
# ============================================================

def calculate_cell_histogram(
    magnitude,
    angle,
    cell_x,
    cell_y,
    cell_size=8,
    nbins=9
):

    x1 = cell_x * cell_size
    y1 = cell_y * cell_size

    x2 = x1 + cell_size
    y2 = y1 + cell_size

    cell_mag = magnitude[y1:y2, x1:x2]
    cell_angle = angle[y1:y2, x1:x2]

    # Histogram
    histogram = np.zeros(nbins)

    bin_width = 180 / nbins

    # Duyệt từng pixel trong cell
    for i in range(cell_mag.shape[0]):

        for j in range(cell_mag.shape[1]):

            mag = cell_mag[i, j]
            ang = cell_angle[i, j]

            # HOG thường sử dụng hướng 0-180 độ
            ang = ang % 180

            bin_index = int(
                ang // bin_width
            )

            if bin_index >= nbins:
                bin_index = nbins - 1

            histogram[bin_index] += mag

    return histogram


# ============================================================
# 6. XỬ LÝ ẢNH
# ============================================================

results = []


for filename in image_files:

    path = os.path.join(folder, filename)

    img = cv2.imread(path)

    if img is None:
        print(
            f"Không thể đọc ảnh: {filename}"
        )
        continue

    # --------------------------------------------------------
    # Chuyển grayscale
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    # Resize để các ảnh có cùng kích thước
    gray = cv2.resize(
        gray,
        IMAGE_SIZE
    )

    # --------------------------------------------------------
    # Tính Gradient
    # --------------------------------------------------------

    gx, gy, magnitude, angle = \
        calculate_gradient(gray)

    # --------------------------------------------------------
    # Tính histogram cho cell đầu tiên
    # --------------------------------------------------------

    first_cell_histogram = \
        calculate_cell_histogram(
            magnitude,
            angle,
            0,
            0,
            CELL_SIZE[0],
            NBINS
        )

    # --------------------------------------------------------
    # Tính vector HOG
    # --------------------------------------------------------

    hog_vector = hog.compute(
        gray
    )

    hog_vector = hog_vector.flatten()

    results.append({

        "filename": filename,

        "gray": gray,

        "gx": gx,

        "gy": gy,

        "magnitude": magnitude,

        "angle": angle,

        "histogram": first_cell_histogram,

        "hog": hog_vector
    })


# ============================================================
# 7. HIỂN THỊ GRADIENT
# ============================================================

for result in results:

    plt.figure(figsize=(15, 5))

    # Ảnh grayscale
    plt.subplot(1, 3, 1)

    plt.imshow(
        result["gray"],
        cmap="gray"
    )

    plt.title(
        f"Grayscale - {result['filename']}"
    )

    plt.axis("off")


    # Gradient magnitude
    plt.subplot(1, 3, 2)

    plt.imshow(
        result["magnitude"],
        cmap="gray"
    )

    plt.title("Gradient Magnitude")

    plt.axis("off")


    # Gradient angle
    plt.subplot(1, 3, 3)

    plt.imshow(
        result["angle"],
        cmap="hsv"
    )

    plt.title("Gradient Direction")

    plt.axis("off")

    plt.tight_layout()

    plt.show()


# ============================================================
# 8. HIỂN THỊ HISTOGRAM HƯỚNG GRADIENT
# ============================================================

for result in results:

    histogram = result["histogram"]

    angles = np.arange(
        NBINS
    ) * (180 / NBINS)

    plt.figure(figsize=(8, 5))

    plt.bar(
        angles,
        histogram,
        width=15
    )

    plt.xlabel(
        "Hướng gradient (độ)"
    )

    plt.ylabel(
        "Tổng độ lớn gradient"
    )

    plt.title(
        f"Histogram HOG - {result['filename']}"
    )

    plt.xticks(
        angles,
        [f"{int(a)}°" for a in angles]
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    plt.show()


# ============================================================
# 9. PHÂN TÍCH VECTOR HOG
# ============================================================

print("\n" + "=" * 80)
print("KẾT QUẢ HOG")
print("=" * 80)


for result in results:

    vector = result["hog"]

    print(
        f"\nẢnh: {result['filename']}"
    )

    print(
        "Kích thước vector HOG:",
        vector.shape
    )

    print(
        "Số phần tử:",
        len(vector)
    )

    print(
        "Giá trị nhỏ nhất:",
        np.min(vector)
    )

    print(
        "Giá trị lớn nhất:",
        np.max(vector)
    )

    print(
        "Mean:",
        np.mean(vector)
    )


# ============================================================
# 10. HIỂN THỊ MỘT PHẦN VECTOR HOG
# ============================================================

for result in results:

    vector = result["hog"]

    print(
        f"\n{result['filename']} - "
        "20 phần tử đầu tiên:"
    )

    print(vector[:20])


# ============================================================
# 11. SO SÁNH VECTOR HOG
# ============================================================

print("\n" + "=" * 80)
print("KHOẢNG CÁCH GIỮA CÁC VECTOR HOG")
print("=" * 80)


for i in range(len(results)):

    for j in range(i + 1, len(results)):

        vector1 = results[i]["hog"]

        vector2 = results[j]["hog"]

        distance = np.linalg.norm(
            vector1 - vector2
        )

        print(
            f"{results[i]['filename']} "
            f"<-> "
            f"{results[j]['filename']}: "
            f"{distance:.6f}"
        )