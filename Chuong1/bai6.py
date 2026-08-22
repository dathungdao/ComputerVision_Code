import cv2
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# BÀI 6: XÂY DỰNG QUY TRÌNH TIỀN XỬ LÝ ẢNH
# ============================================================
#
# Bài toán:
# Ảnh sản phẩm chụp bằng camera có các vấn đề:
# - Kích thước ảnh khác nhau
# - Một số ảnh bị nhiễu
# - Độ sáng không đồng đều
# - Không gian màu chưa phù hợp
#
# Quy trình đề xuất:
#
# Ảnh gốc
#    ↓
# Resize
#    ↓
# Khử nhiễu
#    ↓
# Tăng cường độ tương phản
#    ↓
# Chuyển sang HSV
#    ↓
# Ảnh sau tiền xử lý
#
# ============================================================


# ============================================================
# 1. ĐỌC ẢNH
# ============================================================

img = cv2.imread("image.jpg")

if img is None:
    print("Không thể đọc ảnh!")
    exit()


# OpenCV đọc ảnh theo BGR
# Chuyển sang RGB để hiển thị bằng Matplotlib
img_rgb = cv2.cvtColor(
    img,
    cv2.COLOR_BGR2RGB
)


# ============================================================
# 2. RESIZE
# ============================================================
#
# Đưa tất cả ảnh về cùng một kích thước.
#
# Ở đây chọn kích thước:
# 640 x 480
#
# Trong bài toán thực tế, kích thước cần phụ thuộc vào
# mô hình AI hoặc yêu cầu của hệ thống.


target_width = 640
target_height = 480

resized = cv2.resize(
    img,
    (target_width, target_height),
    interpolation=cv2.INTER_AREA
)


# ============================================================
# 3. KHỬ NHIỄU
# ============================================================
#
# Sử dụng Median Filter.
#
# Median Filter đặc biệt phù hợp với nhiễu dạng:
# Salt-and-Pepper Noise.
#
# Kernel 5x5.


denoised = cv2.medianBlur(
    resized,
    5
)


# ============================================================
# 4. TĂNG CƯỜNG ĐỘ TƯƠNG PHẢN BẰNG CLAHE
# ============================================================
#
# Vì độ sáng của ảnh không đồng đều nên sử dụng CLAHE.
#
# CLAHE hoạt động tốt hơn Histogram Equalization trong
# trường hợp các vùng ảnh có độ sáng khác nhau.


# Chuyển ảnh sang LAB
lab = cv2.cvtColor(
    denoised,
    cv2.COLOR_BGR2LAB
)

# Tách 3 kênh L, A, B
L, A, B = cv2.split(lab)

# Tạo CLAHE
clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

# Áp dụng CLAHE lên kênh L
L_enhanced = clahe.apply(L)

# Ghép các kênh trở lại
lab_enhanced = cv2.merge(
    (L_enhanced, A, B)
)

# LAB -> BGR
enhanced = cv2.cvtColor(
    lab_enhanced,
    cv2.COLOR_LAB2BGR
)


# ============================================================
# 5. CHUYỂN SANG HSV
# ============================================================
#
# Nếu bài toán cần phân đoạn sản phẩm dựa trên màu sắc,
# HSV phù hợp hơn RGB.
#
# HSV gồm:
# H = Hue       : loại màu
# S = Saturation : độ bão hòa
# V = Value      : độ sáng


hsv = cv2.cvtColor(
    enhanced,
    cv2.COLOR_BGR2HSV
)


# ============================================================
# 6. TẠO ẢNH HSV ĐỂ HIỂN THỊ
# ============================================================
#
# Không nên hiển thị trực tiếp HSV như RGB.
# Ta chuyển HSV trở lại RGB để xem kết quả.


hsv_rgb = cv2.cvtColor(
    hsv,
    cv2.COLOR_HSV2RGB
)


# ============================================================
# 7. CHUYỂN ẢNH SAU TIỀN XỬ LÝ VỀ RGB
# ============================================================

enhanced_rgb = cv2.cvtColor(
    enhanced,
    cv2.COLOR_BGR2RGB
)


# ============================================================
# 8. HIỂN THỊ TRƯỚC VÀ SAU TIỀN XỬ LÝ
# ============================================================

plt.figure(figsize=(15, 10))


# ------------------------------------------------------------
# ẢNH GỐC
# ------------------------------------------------------------

plt.subplot(2, 3, 1)

plt.imshow(img_rgb)

plt.title("1. Ảnh gốc")

plt.axis("off")


# ------------------------------------------------------------
# SAU RESIZE
# ------------------------------------------------------------

plt.subplot(2, 3, 2)

resized_rgb = cv2.cvtColor(
    resized,
    cv2.COLOR_BGR2RGB
)

plt.imshow(resized_rgb)

plt.title("2. Sau Resize")

plt.axis("off")


# ------------------------------------------------------------
# SAU KHỬ NHIỄU
# ------------------------------------------------------------

plt.subplot(2, 3, 3)

denoised_rgb = cv2.cvtColor(
    denoised,
    cv2.COLOR_BGR2RGB
)

plt.imshow(denoised_rgb)

plt.title("3. Sau khử nhiễu")

plt.axis("off")


# ------------------------------------------------------------
# SAU CLAHE
# ------------------------------------------------------------

plt.subplot(2, 3, 4)

plt.imshow(enhanced_rgb)

plt.title("4. Sau CLAHE")

plt.axis("off")


# ------------------------------------------------------------
# HSV
# ------------------------------------------------------------

plt.subplot(2, 3, 5)

plt.imshow(hsv_rgb)

plt.title("5. Không gian HSV")

plt.axis("off")


# ------------------------------------------------------------
# ẢNH KẾT QUẢ
# ------------------------------------------------------------

plt.subplot(2, 3, 6)

plt.imshow(enhanced_rgb)

plt.title("6. Ảnh sau tiền xử lý")

plt.axis("off")


plt.tight_layout()

plt.show()


# ============================================================
# 9. SO SÁNH TRƯỚC VÀ SAU TIỀN XỬ LÝ
# ============================================================

# Chuyển ảnh gốc và ảnh sau xử lý sang grayscale
gray_original = cv2.cvtColor(
    img,
    cv2.COLOR_BGR2GRAY
)

gray_processed = cv2.cvtColor(
    enhanced,
    cv2.COLOR_BGR2GRAY
)


print("==========================================")
print("SO SÁNH TRƯỚC VÀ SAU TIỀN XỬ LÝ")
print("==========================================")


# Kích thước
print("\nKích thước:")
print("Ảnh gốc       :", img.shape)
print("Ảnh sau resize:", resized.shape)


# Độ sáng trung bình
print("\nĐộ sáng trung bình:")
print(
    "Ảnh gốc       :",
    round(gray_original.mean(), 2)
)

print(
    "Ảnh xử lý     :",
    round(gray_processed.mean(), 2)
)


# Độ tương phản được ước lượng bằng độ lệch chuẩn
print("\nĐộ tương phản (Std):")

print(
    "Ảnh gốc       :",
    round(gray_original.std(), 2)
)

print(
    "Ảnh xử lý     :",
    round(gray_processed.std(), 2)
)


# ============================================================
# 10. HIỂN THỊ HISTOGRAM TRƯỚC VÀ SAU
# ============================================================

plt.figure(figsize=(12, 5))


plt.subplot(1, 2, 1)

plt.hist(
    gray_original.ravel(),
    bins=256,
    range=(0, 256)
)

plt.title("Histogram - Ảnh gốc")

plt.xlabel("Mức xám")

plt.ylabel("Số pixel")


plt.subplot(1, 2, 2)

plt.hist(
    gray_processed.ravel(),
    bins=256,
    range=(0, 256)
)

plt.title("Histogram - Sau tiền xử lý")

plt.xlabel("Mức xám")

plt.ylabel("Số pixel")


plt.tight_layout()

plt.show()


# ============================================================
# GIẢI THÍCH QUY TRÌNH
# ============================================================

# ------------------------------------------------------------
# BƯỚC 1: RESIZE
# ------------------------------------------------------------
#
# Mục đích:
# Đưa tất cả ảnh về cùng một kích thước.
#
# Ví dụ:
#
# 1920 x 1080
# 1280 x 720
# 800 x 600
#
#        ↓ Resize
#
# 640 x 480
#
# Điều này rất quan trọng khi đưa ảnh vào Machine Learning
# hoặc Deep Learning vì model thường yêu cầu input có kích
# thước cố định.


# ------------------------------------------------------------
# BƯỚC 2: KHỬ NHIỄU
# ------------------------------------------------------------
#
# Sử dụng Median Filter.
#
# Mục đích:
# - Loại bỏ nhiễu.
# - Giữ biên của vật thể tương đối tốt.
#
# Median Filter đặc biệt phù hợp với Salt-and-Pepper Noise.
#
# Nếu ảnh chứa Gaussian Noise nhiều thì có thể thay bằng:
#
# cv2.GaussianBlur()
#
# tùy thuộc vào loại nhiễu thực tế.


# ------------------------------------------------------------
# BƯỚC 3: TĂNG CƯỜNG ĐỘ TƯƠNG PHẢN
# ------------------------------------------------------------
#
# Sử dụng CLAHE.
#
# Mục đích:
# - Cải thiện ảnh có độ sáng không đồng đều.
# - Làm rõ các chi tiết ở vùng tối.
# - Tăng tương phản cục bộ.
#
# CLAHE tốt hơn Histogram Equalization trong nhiều trường
# hợp ảnh thực tế vì nó xử lý theo từng vùng nhỏ và giới hạn
# mức độ tăng tương phản.


# ------------------------------------------------------------
# BƯỚC 4: CHUYỂN SANG HSV
# ------------------------------------------------------------
#
# Nếu hệ thống cần nhận dạng/phân đoạn sản phẩm dựa vào
# màu sắc thì HSV phù hợp hơn RGB.
#
# HSV:
#
# H -> loại màu
# S -> độ bão hòa
# V -> độ sáng
#
# Ví dụ:
#
# Cần phát hiện sản phẩm màu đỏ
#          ↓
# RGB -> HSV
#          ↓
# Xác định khoảng Hue màu đỏ
#          ↓
# Tạo Mask
#          ↓
# Phân đoạn sản phẩm


# ============================================================
# CÂU HỎI 1:
# Đề xuất quy trình tiền xử lý.
# ============================================================

# TRẢ LỜI:
#
# Quy trình đề xuất:
#
# Đọc ảnh
#    ↓
# Resize
#    ↓
# Median Filter
#    ↓
# CLAHE
#    ↓
# Chuyển sang HSV
#    ↓
# Phân đoạn / nhận dạng
#
# Trong đó:
#
# Resize       -> xử lý kích thước
# Median       -> giảm nhiễu
# CLAHE        -> cải thiện độ tương phản
# HSV          -> xử lý màu sắc


# ============================================================
# CÂU HỎI 2:
# Giải thích thứ tự các bước.
# ============================================================

# TRẢ LỜI:
#
# Resize được thực hiện trước để tất cả ảnh có cùng kích thước.
#
# Sau đó khử nhiễu để giảm các pixel bất thường trước khi
# thực hiện các bước tăng cường ảnh.
#
# Tiếp theo sử dụng CLAHE để cải thiện độ tương phản và
# xử lý vấn đề ánh sáng không đồng đều.
#
# Cuối cùng chuyển sang HSV nếu bài toán cần sử dụng
# thông tin màu sắc để phân đoạn hoặc nhận dạng sản phẩm.
#
# Thứ tự này giúp giảm ảnh hưởng của nhiễu trước khi tăng
# cường tương phản, vì nếu tăng tương phản trước thì nhiễu
# cũng có thể bị khuếch đại.


# ============================================================
# CÂU HỎI 3:
# Cài đặt bằng OpenCV.
# ============================================================

# Các hàm OpenCV quan trọng được sử dụng:
#
# cv2.imread()
#       -> Đọc ảnh
#
# cv2.resize()
#       -> Thay đổi kích thước
#
# cv2.medianBlur()
#       -> Khử nhiễu
#
# cv2.cvtColor()
#       -> Chuyển đổi không gian màu
#
# cv2.createCLAHE()
#       -> Tạo bộ tăng cường tương phản CLAHE
#
# clahe.apply()
#       -> Áp dụng CLAHE
#
# cv2.split()
#       -> Tách các kênh màu
#
# cv2.merge()
#       -> Ghép các kênh màu


# ============================================================
# CÂU HỎI 4:
# So sánh ảnh trước và sau tiền xử lý.
# ============================================================

# TRẢ LỜI:
#
# Ảnh sau tiền xử lý có:
#
# - Kích thước đồng nhất.
# - Ít nhiễu hơn.
# - Độ tương phản được cải thiện.
# - Các vùng sáng/tối được thể hiện rõ hơn.
# - Có thể sử dụng HSV để phân đoạn dựa trên màu sắc.
#
# Tuy nhiên cần chú ý:
#
# Nếu khử nhiễu quá mạnh:
# -> Mất chi tiết.
#
# Nếu CLAHE quá mạnh:
# -> Có thể làm ảnh bị giả hoặc khuếch đại nhiễu.
#
# Vì vậy cần lựa chọn tham số phù hợp.


# ============================================================
# CÂU HỎI 5:
# Đánh giá ảnh hưởng của tiền xử lý đến chất lượng ảnh.
# ============================================================

# TRẢ LỜI:
#
# Tiền xử lý giúp ảnh phù hợp hơn với bước nhận dạng/phân
# đoạn tiếp theo.
#
# Cụ thể:
#
# Resize
# -> Đảm bảo kích thước đầu vào thống nhất.
#
# Median Filter
# -> Giảm nhiễu.
#
# CLAHE
# -> Tăng độ tương phản và cải thiện chi tiết.
#
# HSV
# -> Giúp tách thông tin màu sắc khỏi độ sáng.
#
# Nhờ đó hệ thống có thể nhận dạng/phân đoạn sản phẩm
# ổn định hơn.
#
# Tuy nhiên tiền xử lý quá mức có thể làm mất thông tin
# quan trọng hoặc làm thay đổi đặc trưng của ảnh.
#
# Vì vậy:
#
# "Tiền xử lý phải phù hợp với vấn đề của dữ liệu,
# không phải càng nhiều bước càng tốt."


# ============================================================
# KẾT LUẬN
# ============================================================

# Quy trình hoàn chỉnh:
#
#                 ẢNH GỐC
#                    │
#                    ▼
#              ┌───────────┐
#              │  RESIZE    │
#              └─────┬─────┘
#                    ▼
#              ┌───────────┐
#              │  MEDIAN   │
#              │   FILTER  │
#              └─────┬─────┘
#                    ▼
#              ┌───────────┐
#              │   CLAHE    │
#              └─────┬─────┘
#                    ▼
#              ┌───────────┐
#              │    HSV     │
#              └─────┬─────┘
#                    ▼
#           PHÂN ĐOẠN / NHẬN DẠNG
#
# Đây là một pipeline tiền xử lý hợp lý cho ảnh sản phẩm
# có nhiều kích thước, nhiễu, ánh sáng không đồng đều và
# yêu cầu phân tích màu sắc.