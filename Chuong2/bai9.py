import cv2
import matplotlib.pyplot as plt
import time

# ============================================================
# 1. ĐỌC ẢNH
# ============================================================

img = cv2.imread("image.png")

if img is None:
    print("Không thể đọc ảnh image.png!")
    exit()

# Chuyển sang grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# ============================================================
# 2. GLOBAL THRESHOLD
# ============================================================

start_time = time.perf_counter()

global_threshold_value = 127

_, global_thresh = cv2.threshold(
    gray,
    global_threshold_value,
    255,
    cv2.THRESH_BINARY
)

global_time = time.perf_counter() - start_time


# ============================================================
# 3. OTSU THRESHOLD
# ============================================================

start_time = time.perf_counter()

otsu_threshold_value, otsu = cv2.threshold(
    gray,
    0,
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

otsu_time = time.perf_counter() - start_time


# ============================================================
# 4. ADAPTIVE MEAN
# ============================================================

start_time = time.perf_counter()

adaptive_mean = cv2.adaptiveThreshold(
    gray,
    255,
    cv2.ADAPTIVE_THRESH_MEAN_C,
    cv2.THRESH_BINARY,
    11,
    2
)

adaptive_mean_time = time.perf_counter() - start_time


# ============================================================
# 5. ADAPTIVE GAUSSIAN
# ============================================================

start_time = time.perf_counter()

adaptive_gaussian = cv2.adaptiveThreshold(
    gray,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    11,
    2
)

adaptive_gaussian_time = time.perf_counter() - start_time


# ============================================================
# 6. IN KẾT QUẢ
# ============================================================

print("=" * 70)
print("SO SÁNH CÁC PHƯƠNG PHÁP THRESHOLDING")
print("=" * 70)

print(f"\nGlobal Threshold")
print(f"  Ngưỡng T: {global_threshold_value}")
print(f"  Thời gian: {global_time:.6f} giây")

print(f"\nOtsu Threshold")
print(f"  Ngưỡng tự động: {otsu_threshold_value:.2f}")
print(f"  Thời gian: {otsu_time:.6f} giây")

print(f"\nAdaptive Mean")
print(f"  Block Size: 11")
print(f"  C: 2")
print(f"  Thời gian: {adaptive_mean_time:.6f} giây")

print(f"\nAdaptive Gaussian")
print(f"  Block Size: 11")
print(f"  C: 2")
print(f"  Thời gian: {adaptive_gaussian_time:.6f} giây")


# ============================================================
# 7. HIỂN THỊ KẾT QUẢ
# ============================================================

plt.figure(figsize=(14, 10))

plt.subplot(2, 3, 1)
plt.imshow(gray, cmap="gray")
plt.title("Ảnh Grayscale")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(global_thresh, cmap="gray")
plt.title(f"Global Threshold\nT = {global_threshold_value}")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(otsu, cmap="gray")
plt.title(f"Otsu\nT = {otsu_threshold_value:.2f}")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(adaptive_mean, cmap="gray")
plt.title("Adaptive Mean")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(adaptive_gaussian, cmap="gray")
plt.title("Adaptive Gaussian")
plt.axis("off")

plt.tight_layout()
plt.show()


# ============================================================
# 8. HIỂN THỊ HISTOGRAM ẢNH GRAYSCALE
# ============================================================

plt.figure(figsize=(10, 5))

plt.hist(
    gray.ravel(),
    bins=256,
    range=(0, 256)
)

plt.axvline(
    global_threshold_value,
    linestyle="--",
    label=f"Global T = {global_threshold_value}"
)

plt.axvline(
    otsu_threshold_value,
    linestyle="--",
    label=f"Otsu T = {otsu_threshold_value:.2f}"
)

plt.title("Histogram ảnh Grayscale")
plt.xlabel("Mức xám")
plt.ylabel("Số lượng pixel")
plt.legend()
plt.grid(alpha=0.3)

plt.show()