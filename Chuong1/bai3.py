import cv2
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# BÀI 3: SO SÁNH CÁC PHƯƠNG PHÁP KHỬ NHIỄU
# ============================================================

# 1. Đọc ảnh
img = cv2.imread("image.png")

if img is None:
    print("Không thể đọc ảnh!")
    exit()

# Chuyển BGR -> RGB để hiển thị bằng matplotlib
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ============================================================
# 2. TẠO GAUSSIAN NOISE
# ============================================================

# Chuyển ảnh sang float để thực hiện phép cộng nhiễu
img_float = img_rgb.astype(np.float32)

# Tạo nhiễu Gaussian
mean = 0
sigma = 25

gaussian_noise = np.random.normal(
    mean,
    sigma,
    img_float.shape
)

# Cộng nhiễu vào ảnh
gaussian_img = img_float + gaussian_noise

# Đưa giá trị pixel về khoảng 0-255
gaussian_img = np.clip(gaussian_img, 0, 255)

# Chuyển về uint8
gaussian_img = gaussian_img.astype(np.uint8)


# ============================================================
# 3. TẠO SALT-AND-PEPPER NOISE
# ============================================================

sp_img = img_rgb.copy()

# Tỷ lệ nhiễu
noise_probability = 0.05

# Tạo ma trận ngẫu nhiên
random_matrix = np.random.random(sp_img.shape[:2])

# Salt: pixel màu trắng
sp_img[random_matrix < noise_probability / 2] = 255

# Pepper: pixel màu đen
sp_img[
    (random_matrix >= noise_probability / 2) &
    (random_matrix < noise_probability)
] = 0


# ============================================================
# 4. ÁP DỤNG MEAN FILTER
# ============================================================

# Mean Filter = Averaging Filter
# Kích thước kernel 5x5

mean_gaussian = cv2.blur(gaussian_img, (5, 5))
mean_sp = cv2.blur(sp_img, (5, 5))


# ============================================================
# 5. ÁP DỤNG GAUSSIAN FILTER
# ============================================================

gaussian_filter_gaussian = cv2.GaussianBlur(
    gaussian_img,
    (5, 5),
    0
)

gaussian_filter_sp = cv2.GaussianBlur(
    sp_img,
    (5, 5),
    0
)


# ============================================================
# 6. ÁP DỤNG MEDIAN FILTER
# ============================================================

median_gaussian = cv2.medianBlur(
    gaussian_img,
    5
)

median_sp = cv2.medianBlur(
    sp_img,
    5
)


# ============================================================
# 7. HIỂN THỊ KẾT QUẢ
# ============================================================

plt.figure(figsize=(15, 10))


# ------------------------------------------------------------
# HÀNG 1: ẢNH GỐC
# ------------------------------------------------------------

plt.subplot(3, 4, 1)
plt.imshow(img_rgb)
plt.title("Ảnh gốc")
plt.axis("off")


# ------------------------------------------------------------
# HÀNG 1: GAUSSIAN NOISE
# ------------------------------------------------------------

plt.subplot(3, 4, 2)
plt.imshow(gaussian_img)
plt.title("Gaussian Noise")
plt.axis("off")


# ------------------------------------------------------------
# HÀNG 1: SALT-AND-PEPPER NOISE
# ------------------------------------------------------------

plt.subplot(3, 4, 3)
plt.imshow(sp_img)
plt.title("Salt-and-Pepper Noise")
plt.axis("off")


plt.subplot(3, 4, 4)
plt.axis("off")


# ============================================================
# GAUSSIAN NOISE
# ============================================================

plt.subplot(3, 4, 5)
plt.imshow(mean_gaussian)
plt.title("Mean - Gaussian Noise")
plt.axis("off")

plt.subplot(3, 4, 6)
plt.imshow(gaussian_filter_gaussian)
plt.title("Gaussian Filter")
plt.axis("off")

plt.subplot(3, 4, 7)
plt.imshow(median_gaussian)
plt.title("Median - Gaussian Noise")
plt.axis("off")

plt.subplot(3, 4, 8)
plt.imshow(img_rgb)
plt.title("Ảnh gốc")
plt.axis("off")


# ============================================================
# SALT-AND-PEPPER NOISE
# ============================================================

plt.subplot(3, 4, 9)
plt.imshow(mean_sp)
plt.title("Mean - Salt & Pepper")
plt.axis("off")

plt.subplot(3, 4, 10)
plt.imshow(gaussian_filter_sp)
plt.title("Gaussian - Salt & Pepper")
plt.axis("off")

plt.subplot(3, 4, 11)
plt.imshow(median_sp)
plt.title("Median - Salt & Pepper")
plt.axis("off")

plt.subplot(3, 4, 12)
plt.imshow(img_rgb)
plt.title("Ảnh gốc")
plt.axis("off")


plt.tight_layout()
plt.show()


# ============================================================
# NHẬN XÉT
# ============================================================

# 1. MEAN FILTER
#
# Mean Filter thay mỗi pixel bằng giá trị trung bình
# của các pixel xung quanh.
#
# Ưu điểm:
# - Đơn giản.
# - Dễ cài đặt.
# - Có khả năng làm giảm nhiễu.
#
# Nhược điểm:
# - Làm ảnh bị mờ.
# - Làm mất các cạnh và chi tiết.
# - Không phù hợp với Salt-and-Pepper Noise.
#
# Mean Filter có thể sử dụng để giảm Gaussian Noise,
# nhưng chất lượng thường không tốt bằng Gaussian Filter.


# ------------------------------------------------------------
# 2. GAUSSIAN FILTER
# ------------------------------------------------------------
#
# Gaussian Filter sử dụng kernel Gaussian để làm mượt ảnh.
#
# Ưu điểm:
# - Có hiệu quả tốt đối với Gaussian Noise.
# - Làm mượt ảnh tương đối tự nhiên.
# - Phù hợp khi nhiễu có phân bố gần Gaussian.
#
# Nhược điểm:
# - Có thể làm mất một số chi tiết nhỏ.
# - Không hiệu quả bằng Median Filter đối với
#   Salt-and-Pepper Noise.


# ------------------------------------------------------------
# 3. MEDIAN FILTER
# ------------------------------------------------------------
#
# Median Filter thay giá trị pixel bằng giá trị trung vị
# trong vùng lân cận.
#
# Ưu điểm:
# - Loại bỏ rất tốt Salt-and-Pepper Noise.
# - Giữ được biên tốt hơn Mean Filter và Gaussian Filter.
# - Hiệu quả với các điểm nhiễu đen/trắng đột biến.
#
# Nhược điểm:
# - Không phải lựa chọn tối ưu cho Gaussian Noise.
# - Kernel quá lớn có thể làm mất chi tiết.


# ============================================================
# CÂU HỎI:
# Xác định phương pháp phù hợp với từng loại nhiễu
# và giải thích.
# ============================================================

# TRẢ LỜI:
#
# 1. Đối với GAUSSIAN NOISE:
#
# Phương pháp phù hợp nhất:
# --> GAUSSIAN FILTER
#
# Lý do:
# Gaussian Noise có phân bố gần với phân phối Gaussian.
# Gaussian Filter được thiết kế để làm mượt ảnh và giảm
# các biến động ngẫu nhiên dạng Gaussian.
#
# Mean Filter cũng có thể giảm Gaussian Noise nhưng thường
# làm ảnh bị mờ nhiều hơn.
#
# Median Filter vẫn có thể làm giảm nhiễu nhưng không phải
# lựa chọn tối ưu cho loại nhiễu này.
#
#
# 2. Đối với SALT-AND-PEPPER NOISE:
#
# Phương pháp phù hợp nhất:
# --> MEDIAN FILTER
#
# Lý do:
# Salt-and-Pepper Noise tạo ra các pixel có giá trị rất cao
# (trắng) hoặc rất thấp (đen).
#
# Median Filter sử dụng giá trị trung vị trong vùng lân cận,
# do đó các pixel nhiễu đột biến thường bị loại bỏ.
#
# Đồng thời Median Filter giữ biên tốt hơn Mean Filter
# và Gaussian Filter.
#
#
# ============================================================
# KẾT LUẬN
# ============================================================
#
# +------------------------+----------------------+
# | Loại nhiễu             | Bộ lọc phù hợp       |
# +------------------------+----------------------+
# | Gaussian Noise         | Gaussian Filter     |
# | Salt-and-Pepper Noise  | Median Filter       |
# +------------------------+----------------------+
#
# Mean Filter có thể sử dụng cho Gaussian Noise nhưng
# thường làm ảnh bị mờ hơn.
#
# Vì vậy:
#
# Gaussian Noise
#       ↓
# Gaussian Filter
#
# Salt-and-Pepper Noise
#       ↓
# Median Filter
#
# Đây là lựa chọn phù hợp nhất trong bài toán này.