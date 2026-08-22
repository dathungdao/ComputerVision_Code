import cv2
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# BÀI 5: BIẾN ĐỔI HÌNH HỌC
# ============================================================

# 1. Đọc ảnh
img = cv2.imread("image.jpg")

if img is None:
    print("Không thể đọc ảnh!")
    exit()

# OpenCV đọc ảnh theo BGR
# Chuyển sang RGB để hiển thị bằng matplotlib
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ============================================================
# 2. RESIZE - THAY ĐỔI KÍCH THƯỚC
# ============================================================

# Lấy kích thước ảnh
height, width = img.shape[:2]

# Giảm kích thước còn 50%
resize_img = cv2.resize(
    img,
    (width // 2, height // 2),
    interpolation=cv2.INTER_AREA
)

resize_rgb = cv2.cvtColor(
    resize_img,
    cv2.COLOR_BGR2RGB
)


# ============================================================
# 3. ROTATE - XOAY ẢNH
# ============================================================

# Xoay ảnh 90 độ theo chiều kim đồng hồ
rotate_img = cv2.rotate(
    img,
    cv2.ROTATE_90_CLOCKWISE
)

rotate_rgb = cv2.cvtColor(
    rotate_img,
    cv2.COLOR_BGR2RGB
)


# ============================================================
# 4. FLIP - LẬT ẢNH
# ============================================================

# flipCode = 1:
# Lật theo chiều ngang
flip_img = cv2.flip(
    img,
    1
)

flip_rgb = cv2.cvtColor(
    flip_img,
    cv2.COLOR_BGR2RGB
)


# ============================================================
# 5. TRANSLATION - TỊNH TIẾN ẢNH
# ============================================================

# Tịnh tiến:
# x = 100 pixel sang phải
# y = 50 pixel xuống dưới

tx = 100
ty = 50

translation_matrix = np.float32([
    [1, 0, tx],
    [0, 1, ty]
])

translation_img = cv2.warpAffine(
    img,
    translation_matrix,
    (width, height)
)

translation_rgb = cv2.cvtColor(
    translation_img,
    cv2.COLOR_BGR2RGB
)


# ============================================================
# 6. CROP - CẮT ẢNH
# ============================================================

# Cắt phần trung tâm của ảnh
#
# Lấy khoảng:
# y: từ 25% đến 75% chiều cao
# x: từ 25% đến 75% chiều rộng

x1 = width // 4
x2 = 3 * width // 4

y1 = height // 4
y2 = 3 * height // 4

crop_img = img[y1:y2, x1:x2]

crop_rgb = cv2.cvtColor(
    crop_img,
    cv2.COLOR_BGR2RGB
)


# ============================================================
# 7. HIỂN THỊ KẾT QUẢ
# ============================================================

plt.figure(figsize=(15, 10))


# ------------------------------------------------------------
# ẢNH GỐC
# ------------------------------------------------------------

plt.subplot(2, 3, 1)
plt.imshow(img_rgb)
plt.title("Ảnh gốc")
plt.axis("off")


# ------------------------------------------------------------
# RESIZE
# ------------------------------------------------------------

plt.subplot(2, 3, 2)
plt.imshow(resize_rgb)
plt.title("Resize - 50%")
plt.axis("off")


# ------------------------------------------------------------
# ROTATE
# ------------------------------------------------------------

plt.subplot(2, 3, 3)
plt.imshow(rotate_rgb)
plt.title("Rotate - 90°")
plt.axis("off")


# ------------------------------------------------------------
# FLIP
# ------------------------------------------------------------

plt.subplot(2, 3, 4)
plt.imshow(flip_rgb)
plt.title("Flip - Ngang")
plt.axis("off")


# ------------------------------------------------------------
# TRANSLATION
# ------------------------------------------------------------

plt.subplot(2, 3, 5)
plt.imshow(translation_rgb)
plt.title("Translation - (100, 50)")
plt.axis("off")


# ------------------------------------------------------------
# CROP
# ------------------------------------------------------------

plt.subplot(2, 3, 6)
plt.imshow(crop_rgb)
plt.title("Crop - Trung tâm")
plt.axis("off")


plt.tight_layout()
plt.show()


# ============================================================
# 8. IN KÍCH THƯỚC ẢNH
# ============================================================

print("========================================")
print("KÍCH THƯỚC ẢNH")
print("========================================")

print("Ảnh gốc        :", img.shape)
print("Resize         :", resize_img.shape)
print("Rotate         :", rotate_img.shape)
print("Flip           :", flip_img.shape)
print("Translation    :", translation_img.shape)
print("Crop           :", crop_img.shape)


# ============================================================
# GIẢI THÍCH CÁC PHÉP BIẾN ĐỔI
# ============================================================

# ------------------------------------------------------------
# 1. RESIZE
# ------------------------------------------------------------
#
# Resize dùng để thay đổi kích thước ảnh.
#
# Trong bài này ảnh được giảm còn 50% kích thước.
#
# Ví dụ:
# 1920 x 1080
#      ↓
# 960 x 540
#
# Resize thường được sử dụng để:
# - Giảm dung lượng ảnh.
# - Giảm thời gian xử lý.
# - Đưa ảnh về kích thước đầu vào cố định cho mô hình AI.
#
# Khi resize ảnh xuống quá nhỏ:
# --> Có thể làm mất chi tiết.


# ------------------------------------------------------------
# 2. ROTATE
# ------------------------------------------------------------
#
# Rotate dùng để xoay ảnh quanh một tâm.
#
# Trong bài này:
# --> Xoay ảnh 90 độ theo chiều kim đồng hồ.
#
# OpenCV cung cấp hàm:
#
# cv2.rotate()
#
# Ngoài 90 độ, có thể xoay một góc bất kỳ bằng:
#
# cv2.getRotationMatrix2D()
# cv2.warpAffine()


# ------------------------------------------------------------
# 3. FLIP
# ------------------------------------------------------------
#
# Flip dùng để lật ảnh.
#
# flipCode = 1:
# --> Lật ngang.
#
# flipCode = 0:
# --> Lật dọc.
#
# flipCode = -1:
# --> Lật cả ngang và dọc.
#
# Hàm sử dụng:
#
# cv2.flip()


# ------------------------------------------------------------
# 4. TRANSLATION
# ------------------------------------------------------------
#
# Translation là phép tịnh tiến ảnh.
#
# Trong bài:
#
# tx = 100
# --> Dịch sang phải 100 pixel.
#
# ty = 50
# --> Dịch xuống dưới 50 pixel.
#
# Ma trận biến đổi:
#
# [ 1   0   tx ]
# [ 0   1   ty ]
#
# Sử dụng:
#
# cv2.warpAffine()


# ------------------------------------------------------------
# 5. CROP
# ------------------------------------------------------------
#
# Crop dùng để cắt một vùng của ảnh.
#
# Trong bài này lấy phần trung tâm của ảnh.
#
# Cú pháp:
#
# crop = img[y1:y2, x1:x2]
#
# Lưu ý:
# Trong NumPy:
#
# img[y, x]
#
# nên thứ tự là:
#
# [y1:y2, x1:x2]
#
# chứ không phải:
#
# [x1:x2, y1:y2]


# ============================================================
# NHẬN XÉT
# ============================================================

# Resize:
# --> Thay đổi kích thước ảnh nhưng vẫn giữ nội dung chính.
#
# Rotate:
# --> Thay đổi hướng của ảnh.
#
# Flip:
# --> Đảo ngược ảnh theo chiều ngang hoặc chiều dọc.
#
# Translation:
# --> Di chuyển toàn bộ ảnh sang vị trí mới.
#
# Crop:
# --> Loại bỏ một phần ảnh và giữ lại vùng quan tâm.


# ============================================================
# KẾT LUẬN
# ============================================================

# Các phép biến đổi hình học không nhất thiết làm thay đổi
# nội dung vật thể mà chủ yếu thay đổi:
#
# - Kích thước
# - Vị trí
# - Hướng
# - Vùng quan sát
#
# Những phép biến đổi này rất quan trọng trong xử lý ảnh
# và Computer Vision.
#
# Đặc biệt, Resize, Rotate, Flip và Crop thường được sử dụng
# trong Data Augmentation để tạo thêm dữ liệu huấn luyện
# cho các mô hình Machine Learning / Deep Learning.
