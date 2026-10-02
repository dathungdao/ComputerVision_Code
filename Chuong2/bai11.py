import cv2
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# BÀI TẬP 11: TÁCH ĐỐI TƯỢNG BẰNG MÀU
#
# Quy trình:
# RGB
#   ↓
# HSV
#   ↓
# Color Threshold
#   ↓
# Binary Mask
#   ↓
# Morphological Operation
#   ↓
# Contour
#   ↓
# Object
# ============================================================


# ============================================================
# 1. ĐỌC ẢNH
# ============================================================

img = cv2.imread("cat.png")

if img is None:
    print("Không thể đọc ảnh cat.png!")
    exit()

# OpenCV đọc ảnh theo thứ tự BGR.
# Khi hiển thị bằng Matplotlib cần chuyển BGR → RGB.

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ============================================================
# 2. CHUYỂN ẢNH SANG HSV
# ============================================================

# RGB/BGR biểu diễn màu theo 3 kênh màu.
# Tuy nhiên việc chọn một đối tượng theo màu trong RGB
# thường khó vì cả 3 kênh R, G, B đều thay đổi theo ánh sáng.
#
# HSV tách màu thành:
#
# H (Hue):
#     Biểu diễn loại màu.
#
# S (Saturation):
#     Độ bão hòa của màu.
#
# V (Value):
#     Độ sáng.
#
# Vì vậy HSV thường thuận tiện hơn RGB
# khi muốn tách đối tượng dựa trên màu.

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


# ============================================================
# 3. COLOR THRESHOLD
# ============================================================

# Ví dụ bên dưới dùng để tìm MÀU XANH LÁ.
#
# OpenCV sử dụng:
# H: 0 → 179
# S: 0 → 255
# V: 0 → 255
#
# Khoảng H của màu xanh lá thường nằm khoảng:
# 35 → 85
#
# Tuy nhiên đây chỉ là giá trị tham khảo.
# Với ảnh thực tế, cần điều chỉnh H, S, V
# dựa trên màu của đối tượng và điều kiện ánh sáng.

lower_green = np.array([35, 50, 50])
upper_green = np.array([85, 255, 255])


# cv2.inRange() kiểm tra từng pixel:
#
# Nếu pixel nằm trong khoảng màu xanh:
#       → 255 (trắng)
#
# Nếu pixel không nằm trong khoảng:
#       → 0 (đen)
#
# Kết quả chính là Binary Mask.

mask = cv2.inRange(
    hsv,
    lower_green,
    upper_green
)


# ============================================================
# 4. MORPHOLOGICAL OPERATION
# ============================================================

# Sau khi threshold, mask có thể xuất hiện:
# - Các điểm nhiễu nhỏ.
# - Lỗ nhỏ bên trong đối tượng.
# - Biên đối tượng không liên tục.
#
# Morphological Operation giúp làm sạch Binary Mask.

kernel = np.ones((5, 5), np.uint8)


# ------------------------------------------------------------
# MORPHOLOGICAL OPENING
# ------------------------------------------------------------

# Opening = Erosion → Dilation
#
# Công dụng:
# - Loại bỏ các vùng trắng nhỏ.
# - Giảm nhiễu.
#
# Đây là bước quan trọng nếu Binary Mask
# xuất hiện nhiều điểm trắng không mong muốn.

mask_open = cv2.morphologyEx(
    mask,
    cv2.MORPH_OPEN,
    kernel,
    iterations=2
)


# ------------------------------------------------------------
# MORPHOLOGICAL CLOSING
# ------------------------------------------------------------

# Closing = Dilation → Erosion
#
# Công dụng:
# - Lấp các lỗ nhỏ trong đối tượng.
# - Nối các vùng trắng gần nhau.
# - Làm contour của đối tượng liên tục hơn.

mask_clean = cv2.morphologyEx(
    mask_open,
    cv2.MORPH_CLOSE,
    kernel,
    iterations=2
)


# ============================================================
# 5. TÌM CONTOUR
# ============================================================

# Contour là đường biên của vùng trắng trong Binary Mask.
#
# RETR_EXTERNAL:
# chỉ lấy contour bên ngoài.
#
# CHAIN_APPROX_SIMPLE:
# loại bỏ các điểm dư thừa trên contour,
# giúp tiết kiệm bộ nhớ.

contours, hierarchy = cv2.findContours(
    mask_clean,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)


print("=" * 70)
print("KẾT QUẢ TÁCH ĐỐI TƯỢNG")
print("=" * 70)

print("Số contour tìm được:", len(contours))


# ============================================================
# 6. LỌC CONTOUR
# ============================================================

# Nếu Binary Mask có nhiều vùng nhiễu,
# findContours() sẽ tìm được rất nhiều contour.
#
# Ta có thể loại bỏ những contour quá nhỏ
# bằng diện tích.

MIN_AREA = 500

valid_contours = []

for contour in contours:

    area = cv2.contourArea(contour)

    if area >= MIN_AREA:
        valid_contours.append(contour)


print("Số contour sau khi lọc:", len(valid_contours))


# ============================================================
# 7. VẼ CONTOUR LÊN ẢNH GỐC
# ============================================================

contour_image = img.copy()

cv2.drawContours(
    contour_image,
    valid_contours,
    -1,
    (0, 0, 255),
    3
)

contour_image_rgb = cv2.cvtColor(
    contour_image,
    cv2.COLOR_BGR2RGB
)


# ============================================================
# 8. TÁCH ĐỐI TƯỢNG KHỎI NỀN
# ============================================================

# Tạo một mask mới chỉ chứa những contour hợp lệ.

object_mask = np.zeros_like(mask_clean)

cv2.drawContours(
    object_mask,
    valid_contours,
    -1,
    255,
    thickness=cv2.FILLED
)


# Dùng mask để giữ lại đối tượng,
# còn nền sẽ được chuyển thành màu đen.

object_only = cv2.bitwise_and(
    img,
    img,
    mask=object_mask
)

object_only_rgb = cv2.cvtColor(
    object_only,
    cv2.COLOR_BGR2RGB
)


# ============================================================
# 9. HIỂN THỊ CÁC KẾT QUẢ
# ============================================================

plt.figure(figsize=(16, 10))


# ------------------------------------------------------------
# ẢNH GỐC
# ------------------------------------------------------------

plt.subplot(2, 3, 1)

plt.imshow(img_rgb)

plt.title("1. Ảnh gốc")

plt.axis("off")


# ------------------------------------------------------------
# ẢNH HSV
# ------------------------------------------------------------

# HSV không nên hiển thị trực tiếp như RGB để đánh giá màu.
# Ở đây chủ yếu hiển thị để minh họa bước chuyển đổi.

hsv_display = cv2.cvtColor(
    hsv,
    cv2.COLOR_HSV2RGB
)

plt.subplot(2, 3, 2)

plt.imshow(hsv_display)

plt.title("2. Ảnh HSV")

plt.axis("off")


# ------------------------------------------------------------
# BINARY MASK
# ------------------------------------------------------------

plt.subplot(2, 3, 3)

plt.imshow(mask, cmap="gray")

plt.title("3. Binary Mask")

plt.axis("off")


# ------------------------------------------------------------
# MASK SAU MORPHOLOGY
# ------------------------------------------------------------

plt.subplot(2, 3, 4)

plt.imshow(mask_clean, cmap="gray")

plt.title("4. Mask sau Morphology")

plt.axis("off")


# ------------------------------------------------------------
# CONTOUR
# ------------------------------------------------------------

plt.subplot(2, 3, 5)

plt.imshow(contour_image_rgb)

plt.title("5. Contour")

plt.axis("off")


# ------------------------------------------------------------
# ĐỐI TƯỢNG SAU KHI TÁCH NỀN
# ------------------------------------------------------------

plt.subplot(2, 3, 6)

plt.imshow(object_only_rgb)

plt.title("6. Object sau khi tách nền")

plt.axis("off")


plt.tight_layout()

plt.show()


# ============================================================
# 10. IN THÔNG TIN ĐỐI TƯỢNG
# ============================================================

print("\n" + "=" * 70)
print("THÔNG TIN CÁC ĐỐI TƯỢNG")
print("=" * 70)

for i, contour in enumerate(valid_contours):

    area = cv2.contourArea(contour)

    x, y, w, h = cv2.boundingRect(contour)

    print(f"\nĐối tượng {i + 1}")
    print(f"  Diện tích: {area:.2f}")
    print(f"  Bounding Box:")
    print(f"    x = {x}")
    print(f"    y = {y}")
    print(f"    width = {w}")
    print(f"    height = {h}")