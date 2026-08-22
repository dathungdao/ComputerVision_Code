import cv2
import matplotlib.pyplot as plt


# ============================================================
# BÀI 4: TĂNG CƯỜNG ẢNH
# ============================================================

# 1. Đọc ảnh
img = cv2.imread("image.jpg")

if img is None:
    print("Không thể đọc ảnh!")
    exit()


# ============================================================
# 2. CHUYỂN ẢNH SANG GRAYSCALE
# ============================================================

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# ============================================================
# 3. HISTOGRAM EQUALIZATION
# ============================================================

# Cân bằng histogram toàn cục
equalized = cv2.equalizeHist(gray)


# ============================================================
# 4. CLAHE
# ============================================================

# Tạo đối tượng CLAHE
#
# clipLimit:
#   Giới hạn mức độ tăng cường tương phản.
#
# tileGridSize:
#   Chia ảnh thành các vùng nhỏ 8x8.
#
clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

# Áp dụng CLAHE
clahe_img = clahe.apply(gray)


# ============================================================
# 5. TÍNH HISTOGRAM
# ============================================================

hist_original = cv2.calcHist(
    [gray],
    [0],
    None,
    [256],
    [0, 256]
)

hist_equalized = cv2.calcHist(
    [equalized],
    [0],
    None,
    [256],
    [0, 256]
)

hist_clahe = cv2.calcHist(
    [clahe_img],
    [0],
    None,
    [256],
    [0, 256]
)


# ============================================================
# 6. HIỂN THỊ ẢNH
# ============================================================

plt.figure(figsize=(15, 10))


# -------------------------
# Ảnh gốc
# -------------------------

plt.subplot(2, 3, 1)
plt.imshow(gray, cmap="gray")
plt.title("Ảnh gốc")
plt.axis("off")


# -------------------------
# Histogram Equalization
# -------------------------

plt.subplot(2, 3, 2)
plt.imshow(equalized, cmap="gray")
plt.title("Histogram Equalization")
plt.axis("off")


# -------------------------
# CLAHE
# -------------------------

plt.subplot(2, 3, 3)
plt.imshow(clahe_img, cmap="gray")
plt.title("CLAHE")
plt.axis("off")


# ============================================================
# 7. HIỂN THỊ HISTOGRAM
# ============================================================

# Histogram ảnh gốc
plt.subplot(2, 3, 4)
plt.plot(hist_original)
plt.title("Histogram - Ảnh gốc")
plt.xlabel("Mức xám")
plt.ylabel("Số pixel")


# Histogram Equalization
plt.subplot(2, 3, 5)
plt.plot(hist_equalized)
plt.title("Histogram - Equalization")
plt.xlabel("Mức xám")
plt.ylabel("Số pixel")


# Histogram CLAHE
plt.subplot(2, 3, 6)
plt.plot(hist_clahe)
plt.title("Histogram - CLAHE")
plt.xlabel("Mức xám")
plt.ylabel("Số pixel")


plt.tight_layout()
plt.show()


# ============================================================
# 8. IN MỘT SỐ THÔNG TIN
# ============================================================

print("Kích thước ảnh:", gray.shape)

print("\nGiá trị trung bình:")
print("Ảnh gốc       :", gray.mean())
print("Equalization  :", equalized.mean())
print("CLAHE          :", clahe_img.mean())

print("\nĐộ lệch chuẩn:")
print("Ảnh gốc       :", gray.std())
print("Equalization  :", equalized.std())
print("CLAHE          :", clahe_img.std())


# ============================================================
# NHẬN XÉT
# ============================================================

# 1. ẢNH GỐC
#
# Ảnh có độ tương phản thấp thường có histogram tập trung
# trong một khoảng mức xám hẹp.
#
# Điều này khiến các vùng sáng và tối khó phân biệt,
# ảnh nhìn bị tối, nhạt hoặc thiếu chi tiết.


# ------------------------------------------------------------
# 2. HISTOGRAM EQUALIZATION
# ------------------------------------------------------------
#
# Histogram Equalization phân bố lại các mức xám trên
# toàn bộ ảnh.
#
# Mục đích:
# - Mở rộng phạm vi mức xám.
# - Tăng độ tương phản.
# - Làm các vùng sáng và tối dễ phân biệt hơn.
#
# Ưu điểm:
# - Đơn giản.
# - Dễ thực hiện.
# - Hiệu quả với ảnh có độ tương phản thấp.
#
# Nhược điểm:
# - Xử lý trên toàn bộ ảnh.
# - Có thể làm một số vùng bị tăng tương phản quá mức.
# - Có thể khuếch đại nhiễu.
#
# Histogram sau Equalization thường được phân bố rộng hơn
# so với histogram của ảnh ban đầu.


# ------------------------------------------------------------
# 3. CLAHE
# ------------------------------------------------------------
#
# CLAHE = Contrast Limited Adaptive Histogram Equalization.
#
# Khác với Histogram Equalization thông thường,
# CLAHE chia ảnh thành nhiều vùng nhỏ rồi thực hiện
# cân bằng histogram cục bộ.
#
# Đồng thời CLAHE giới hạn mức độ tăng cường tương phản
# bằng clipLimit.
#
# Ưu điểm:
# - Tăng cường tương phản cục bộ tốt.
# - Giữ được nhiều chi tiết ở các vùng khác nhau.
# - Hạn chế việc khuếch đại nhiễu quá mức.
# - Phù hợp với ảnh có ánh sáng không đồng đều.
#
# Nhược điểm:
# - Có nhiều tham số cần điều chỉnh.
# - Tính toán phức tạp hơn Histogram Equalization.


# ============================================================
# SO SÁNH
# ============================================================

# Histogram Equalization:
# --> Tăng tương phản TOÀN CỤC.
#
# CLAHE:
# --> Tăng tương phản CỤC BỘ.
#
# Nếu ảnh có ánh sáng tương đối đồng đều:
# --> Histogram Equalization có thể cho kết quả tốt.
#
# Nếu ảnh có vùng sáng/tối không đồng đều:
# --> CLAHE thường phù hợp hơn.


# ============================================================
# CÂU HỎI:
# Nhận xét sự thay đổi về độ tương phản và chất lượng ảnh.
# ============================================================

# TRẢ LỜI:
#
# Sau khi áp dụng Histogram Equalization và CLAHE,
# độ tương phản của ảnh được tăng lên so với ảnh ban đầu.
#
# Histogram Equalization làm mở rộng phạm vi mức xám trên
# toàn bộ ảnh, giúp các vùng sáng và tối được phân biệt rõ
# hơn. Tuy nhiên, nếu tăng cường quá mạnh, một số vùng có
# thể bị mất chi tiết hoặc nhiễu bị khuếch đại.
#
# CLAHE thực hiện tăng cường tương phản theo từng vùng nhỏ
# nên thường giữ được chi tiết cục bộ tốt hơn. CLAHE đặc biệt
# hiệu quả khi ảnh có độ sáng không đồng đều giữa các vùng.
#
# Vì vậy:
#
# - Ảnh gốc:
#     Độ tương phản thấp, chi tiết khó quan sát.
#
# - Histogram Equalization:
#     Tăng tương phản toàn ảnh, ảnh sáng/tối rõ hơn.
#
# - CLAHE:
#     Tăng tương phản cục bộ, thường giữ chi tiết tốt hơn
#     và phù hợp với ảnh có ánh sáng không đồng đều.
#
#
# ============================================================
# KẾT LUẬN
# ============================================================
#
# Histogram Equalization:
# --> Phù hợp khi cần tăng tương phản toàn cục.
#
# CLAHE:
# --> Phù hợp khi cần tăng tương phản cục bộ và ảnh có
#     điều kiện ánh sáng không đồng đều.
#
# Trong nhiều trường hợp ảnh thực tế, CLAHE cho kết quả
# tự nhiên và giữ chi tiết tốt hơn Histogram Equalization.