import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

# =====================================================
# 1. ĐỌC ẢNH
# =====================================================

img = cv2.imread("image.png")

if img is None:
    print("Không thể đọc ảnh!")
    exit()

# Chuyển sang ảnh xám
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# =====================================================
# 2. HARRIS CORNER DETECTION
# =====================================================

start_time = time.perf_counter()

# Chuyển ảnh xám sang float32
gray_float = np.float32(gray)

# Harris Corner
harris = cv2.cornerHarris(
    gray_float,
    blockSize=2,
    ksize=3,
    k=0.04
)

# Làm giãn kết quả để dễ hiển thị
harris = cv2.dilate(harris, None)

# Ngưỡng phát hiện góc
threshold = 0.01 * harris.max()

harris_img = img.copy()

harris_points = []

for y in range(harris.shape[0]):
    for x in range(harris.shape[1]):
        if harris[y, x] > threshold:
            harris_points.append((x, y))
            cv2.circle(
                harris_img,
                (x, y),
                3,
                (0, 0, 255),
                -1
            )

harris_time = time.perf_counter() - start_time
harris_count = len(harris_points)


# =====================================================
# 3. SHI-TOMASI
# =====================================================

start_time = time.perf_counter()

shi_points = cv2.goodFeaturesToTrack(
    gray,
    maxCorners=500,
    qualityLevel=0.01,
    minDistance=10
)

shi_img = img.copy()

if shi_points is not None:
    shi_points = np.int32(shi_points)

    for point in shi_points:
        x, y = point.ravel()

        cv2.circle(
            shi_img,
            (x, y),
            4,
            (0, 255, 0),
            -1
        )

    shi_count = len(shi_points)
else:
    shi_count = 0

shi_time = time.perf_counter() - start_time


# =====================================================
# 4. FAST
# =====================================================

start_time = time.perf_counter()

fast = cv2.FastFeatureDetector_create(
    threshold=20,
    nonmaxSuppression=True
)

fast_keypoints = fast.detect(gray, None)

fast_img = img.copy()

for kp in fast_keypoints:
    x, y = np.int32(kp.pt)

    cv2.circle(
        fast_img,
        (x, y),
        4,
        (255, 0, 0),
        -1
    )

fast_count = len(fast_keypoints)

fast_time = time.perf_counter() - start_time


# =====================================================
# 5. IN KẾT QUẢ
# =====================================================

print("=" * 50)
print("KẾT QUẢ PHÁT HIỆN CORNER")
print("=" * 50)

print(f"Harris:")
print(f"  Số lượng keypoint: {harris_count}")
print(f"  Thời gian: {harris_time:.6f} giây")

print()

print(f"Shi-Tomasi:")
print(f"  Số lượng keypoint: {shi_count}")
print(f"  Thời gian: {shi_time:.6f} giây")

print()

print(f"FAST:")
print(f"  Số lượng keypoint: {fast_count}")
print(f"  Thời gian: {fast_time:.6f} giây")


# =====================================================
# 6. HIỂN THỊ KẾT QUẢ
# =====================================================

plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Ảnh gốc")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(cv2.cvtColor(harris_img, cv2.COLOR_BGR2RGB))
plt.title(f"Harris - {harris_count} keypoints")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(cv2.cvtColor(shi_img, cv2.COLOR_BGR2RGB))
plt.title(f"Shi-Tomasi - {shi_count} keypoints")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(cv2.cvtColor(fast_img, cv2.COLOR_BGR2RGB))
plt.title(f"FAST - {fast_count} keypoints")
plt.axis("off")

plt.tight_layout()
plt.show()