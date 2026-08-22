import cv2
import matplotlib.pyplot as plt


img = cv2.imread("image.png")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

plt.figure(figsize=(14, 8))
# Ảnh gốc
plt.subplot(2, 3, 1)
plt.imshow(img_rgb)
plt.title("Ảnh gốc - RGB")
plt.axis("off")

# Grayscale
plt.subplot(2, 3, 2)
plt.imshow(gray, cmap="gray")
plt.title("Grayscale")
plt.axis("off")

# HSV
plt.subplot(2, 3, 3)

plt.imshow(hsv[:, :, 0], cmap="hsv")
plt.title("HSV - Kênh H")
plt.axis("off")

# YCrCb
plt.subplot(2, 3, 4)

# Hiển thị kênh Y (độ sáng)
plt.imshow(ycrcb[:, :, 0], cmap="gray")
plt.title("YCrCb - Kênh Y")
plt.axis("off")

# Lab
plt.subplot(2, 3, 5)

# Hiển thị kênh L (độ sáng)
plt.imshow(lab[:, :, 0], cmap="gray")
plt.title("Lab - Kênh L")
plt.axis("off")

# Hiển thị ảnh HSV chuyển ngược về RGB
hsv_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

plt.subplot(2, 3, 6)
plt.imshow(hsv_rgb)
plt.title("HSV chuyển về RGB")
plt.axis("off")

plt.tight_layout()
plt.show()

print("Kích thước ảnh:", img.shape)
print("Kích thước Grayscale:", gray.shape)
print("Kích thước HSV:", hsv.shape)
print("Kích thước YCrCb:", ycrcb.shape)
print("Kích thước Lab:", lab.shape)

# Grayscale:
# - Chỉ có một kênh.
# - Mỗi pixel biểu diễn mức độ sáng từ tối đến sáng.
# - Không chứa thông tin màu.
# - Ưu điểm: đơn giản, ít dữ liệu, xử lý nhanh.
# - Thường được dùng trong phát hiện biên, threshold,
#   nhận dạng hình dạng,...

# HSV:
# - H (Hue): biểu diễn loại màu.
# - S (Saturation): độ bão hòa của màu.
# - V (Value): độ sáng.
# - HSV tách thông tin màu sắc khỏi độ sáng tốt hơn RGB.
# - Rất phù hợp với bài toán phân đoạn đối tượng theo màu,
#   ví dụ: phát hiện quả bóng đỏ, đèn giao thông,...

# YCrCb:
# - Y: thông tin độ sáng (Luminance).
# - Cr và Cb: thông tin màu.
# - Tách độ sáng khỏi thông tin màu.
# - Được sử dụng nhiều trong xử lý video, nén ảnh
#   và một số bài toán xử lý khuôn mặt.

# Lab:
# - L: độ sáng.
# - a: thành phần màu từ xanh lá đến đỏ.
# - b: thành phần màu từ xanh dương đến vàng.
# - Lab gần với cách con người cảm nhận màu sắc.
# - Có khả năng tách độ sáng và màu sắc khá tốt.
# - Có thể được sử dụng trong phân đoạn và so sánh màu.


# ============================================================
# CÂU HỎI:
# Không gian màu nào phù hợp nhất nếu cần phân đoạn
# đối tượng dựa trên màu sắc?
# ============================================================

# TRẢ LỜI:
#
# HSV là không gian màu phù hợp nhất trong trường hợp này.
#
# Lý do:
# 1. HSV tách thông tin màu sắc (Hue) khỏi độ sáng (Value).
# 2. Có thể dễ dàng xác định một khoảng Hue tương ứng
#    với màu cần tìm.
# 3. Ít bị ảnh hưởng bởi thay đổi độ sáng hơn RGB.
# 4. Rất thuận tiện để tạo mask phân đoạn màu.
#
# Ví dụ:
# Nếu cần tìm quả bóng màu đỏ:
#
# RGB
#   ↓
# HSV
#   ↓
# Chọn khoảng Hue của màu đỏ
#   ↓
# Tạo Mask
#   ↓
# Phân đoạn quả bóng
#
# Vì vậy:
# --> Nếu bài toán phân đoạn đối tượng dựa chủ yếu vào MÀU SẮC,
#     nên ưu tiên sử dụng HSV.
#
# YCrCb và Lab cũng có thể được sử dụng, nhưng HSV thường
# trực quan và thuận tiện hơn cho việc xác định màu bằng
# một khoảng giá trị Hue.