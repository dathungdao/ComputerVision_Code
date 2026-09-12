import cv2
import matplotlib.pyplot as plt

# ==========================================
# 1. ĐỌC ẢNH MÀU
# ==========================================
img = cv2.imread("image.png")

if img is None:
    print("Không thể đọc ảnh!")
    exit()

# ==========================================
# 2. CHUYỂN ẢNH SANG RGB VÀ HSV
# ==========================================

# OpenCV đọc ảnh theo thứ tự BGR
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Chuyển từ BGR sang HSV
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


# ==========================================
# 3. TÍNH HISTOGRAM
# ==========================================

# Histogram ảnh RGB
hist_r = cv2.calcHist([img_rgb], [0], None, [256], [0, 256])
hist_g = cv2.calcHist([img_rgb], [1], None, [256], [0, 256])
hist_b = cv2.calcHist([img_rgb], [2], None, [256], [0, 256])

# Histogram ảnh HSV
hist_h = cv2.calcHist([img_hsv], [0], None, [180], [0, 180])
hist_s = cv2.calcHist([img_hsv], [1], None, [256], [0, 256])
hist_v = cv2.calcHist([img_hsv], [2], None, [256], [0, 256])


# ==========================================
# 4. HIỂN THỊ ẢNH
# ==========================================

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(img_rgb)
plt.title("Ảnh RGB")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(img_hsv)
plt.title("Ảnh HSV")
plt.axis("off")


# ==========================================
# HIỂN THỊ HISTOGRAM RGB
# ==========================================

plt.figure(figsize=(10, 5))

plt.plot(hist_r, label="Red")
plt.plot(hist_g, label="Green")
plt.plot(hist_b, label="Blue")

plt.title("Histogram trong không gian RGB")
plt.xlabel("Giá trị cường độ (0-255)")
plt.ylabel("Số lượng pixel")
plt.legend()
plt.grid()

plt.show()


# ==========================================
# HIỂN THỊ HISTOGRAM HSV
# ==========================================

plt.figure(figsize=(10, 5))

plt.plot(hist_h, label="Hue")
plt.plot(hist_s, label="Saturation")
plt.plot(hist_v, label="Value")

plt.title("Histogram trong không gian HSV")
plt.xlabel("Giá trị")
plt.ylabel("Số lượng pixel")
plt.legend()
plt.grid()

plt.show()