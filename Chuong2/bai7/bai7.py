import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time


# ============================================================
# 1. CẤU HÌNH
# ============================================================

folder = "shapes"

image_files = [
    f for f in os.listdir(folder)
    if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
]

image_files.sort()

if len(image_files) == 0:
    print("Không tìm thấy ảnh trong thư mục shapes!")
    exit()


# ============================================================
# 2. XỬ LÝ TỪNG ẢNH
# ============================================================

results = []

for filename in image_files:

    path = os.path.join(folder, filename)

    # Đọc ảnh
    img = cv2.imread(path)

    if img is None:
        print(f"Không thể đọc ảnh: {filename}")
        continue

    # --------------------------------------------------------
    # Bước 1: Chuyển sang grayscale
    # --------------------------------------------------------

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --------------------------------------------------------
    # Bước 2: Phân đoạn đối tượng
    # --------------------------------------------------------

    # Làm mượt nhẹ để giảm nhiễu
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold Otsu
    _, binary = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Nếu đối tượng là màu đen trên nền trắng
    # thì đảo ảnh nhị phân
    white_pixels = np.sum(binary == 255)
    black_pixels = np.sum(binary == 0)

    if white_pixels > black_pixels:
        binary = cv2.bitwise_not(binary)

    # --------------------------------------------------------
    # Bước 3: Tìm contour
    # --------------------------------------------------------

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        print(f"Không tìm thấy contour: {filename}")
        continue

    # Chọn contour lớn nhất
    contour = max(contours, key=cv2.contourArea)

    # --------------------------------------------------------
    # Bước 4: Tính Hu Moments
    # --------------------------------------------------------

    moments = cv2.moments(contour)

    hu_moments = cv2.HuMoments(moments).flatten()

    # --------------------------------------------------------
    # Chuyển sang log scale
    # Giúp dễ so sánh vì Hu Moments có thể rất nhỏ
    # --------------------------------------------------------

    hu_log = []

    for h in hu_moments:
        if h != 0:
            hu_log.append(
                -np.sign(h) * np.log10(abs(h))
            )
        else:
            hu_log.append(0)

    hu_log = np.array(hu_log)

    # --------------------------------------------------------
    # Vẽ contour
    # --------------------------------------------------------

    contour_img = img.copy()

    cv2.drawContours(
        contour_img,
        [contour],
        -1,
        (0, 255, 0),
        3
    )

    # --------------------------------------------------------
    # Lưu kết quả
    # --------------------------------------------------------

    results.append({
        "filename": filename,
        "gray": gray,
        "binary": binary,
        "contour_img": contour_img,
        "contour": contour,
        "hu": hu_moments,
        "hu_log": hu_log
    })


# ============================================================
# 3. HIỂN THỊ HU MOMENTS
# ============================================================

print("\n" + "=" * 100)
print("KẾT QUẢ HU MOMENTS")
print("=" * 100)

for result in results:

    print(f"\nẢnh: {result['filename']}")

    print("-" * 70)

    print("Hu Moments gốc:")

    for i, value in enumerate(result["hu"]):
        print(f"Hu[{i + 1}] = {value:.10e}")

    print("\nHu Moments sau Log Transform:")

    for i, value in enumerate(result["hu_log"]):
        print(f"Hu[{i + 1}] = {value:.6f}")


# ============================================================
# 4. HIỂN THỊ ẢNH XỬ LÝ
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
        f"Grayscale\n{result['filename']}"
    )

    plt.axis("off")

    # Ảnh binary
    plt.subplot(1, 3, 2)

    plt.imshow(
        result["binary"],
        cmap="gray"
    )

    plt.title("Phân đoạn")
    plt.axis("off")

    # Contour
    plt.subplot(1, 3, 3)

    plt.imshow(
        cv2.cvtColor(
            result["contour_img"],
            cv2.COLOR_BGR2RGB
        )
    )

    plt.title("Contour")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


# ============================================================
# 5. TẠO BẢNG SO SÁNH VECTOR HU
# ============================================================

print("\n" + "=" * 100)
print("SO SÁNH VECTOR ĐẶC TRƯNG")
print("=" * 100)

header = (
    f"{'Ảnh':<20}"
    f"{'Hu1':<12}"
    f"{'Hu2':<12}"
    f"{'Hu3':<12}"
    f"{'Hu4':<12}"
    f"{'Hu5':<12}"
    f"{'Hu6':<12}"
    f"{'Hu7':<12}"
)

print(header)
print("-" * 100)

for result in results:

    values = result["hu_log"]

    print(
        f"{result['filename']:<20}"
        f"{values[0]:<12.4f}"
        f"{values[1]:<12.4f}"
        f"{values[2]:<12.4f}"
        f"{values[3]:<12.4f}"
        f"{values[4]:<12.4f}"
        f"{values[5]:<12.4f}"
        f"{values[6]:<12.4f}"
    )


# ============================================================
# 6. TÍNH KHOẢNG CÁCH GIỮA CÁC VECTOR HU
# ============================================================

print("\n" + "=" * 100)
print("KHOẢNG CÁCH GIỮA CÁC VECTOR HU")
print("=" * 100)

for i in range(len(results)):

    for j in range(i + 1, len(results)):

        vector1 = results[i]["hu_log"]
        vector2 = results[j]["hu_log"]

        distance = np.linalg.norm(
            vector1 - vector2
        )

        print(
            f"{results[i]['filename']} "
            f"<-> "
            f"{results[j]['filename']}: "
            f"{distance:.6f}"
        )


# ============================================================
# 7. VẼ BIỂU ĐỒ SO SÁNH
# ============================================================

plt.figure(figsize=(12, 6))

for result in results:

    plt.plot(
        range(1, 8),
        result["hu_log"],
        marker="o",
        label=result["filename"]
    )

plt.xlabel("Hu Moment")
plt.ylabel("Giá trị Log Transform")
plt.title("So sánh vector đặc trưng Hu Moments")

plt.xticks(range(1, 8))

plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()