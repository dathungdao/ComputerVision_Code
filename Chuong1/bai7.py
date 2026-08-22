# BÀI 7: ĐỀ XUẤT HỆ THỐNG THỊ GIÁC MÁY TÍNH
# ============================================================
#
# Đề tài:
# PHÁT HIỆN VÀ NHẬN DẠNG BIỂN BÁO GIAO THÔNG
#
#
# 1. BÀI TOÁN CẦN GIẢI QUYẾT
# ============================================================
#
# Xây dựng một hệ thống thị giác máy tính có khả năng:
#
# - Phát hiện biển báo giao thông trong ảnh hoặc video.
# - Xác định vị trí của biển báo.
# - Phân loại biển báo.
# - Trả về tên hoặc loại biển báo.
#
# Ví dụ:
#
# Ảnh giao thông
#       ↓
# Phát hiện biển báo
#       ↓
# Xác định vị trí
#       ↓
# Phân loại
#       ↓
# Kết quả:
# "Biển báo giới hạn tốc độ 40 km/h"
#
#
# 2. DỮ LIỆU ẢNH CẦN SỬ DỤNG
# ============================================================
#
# Có thể sử dụng:
#
# - Ảnh đường phố.
# - Ảnh chụp từ camera ô tô.
# - Video giao thông.
# - Dataset biển báo giao thông.
#
# Ảnh cần có nhiều điều kiện khác nhau:
#
# - Ban ngày.
# - Ban đêm.
# - Trời nắng.
# - Trời mưa.
# - Biển báo ở gần.
# - Biển báo ở xa.
# - Góc chụp khác nhau.
# - Có vật thể che khuất một phần biển báo.
#
# Nếu sử dụng bài toán Object Detection, dataset cần có:
#
# - Ảnh.
# - Bounding Box.
# - Label của từng biển báo.
#
# Ví dụ:
#
# image_001.jpg
#
# Bounding Box:
# x1 = 120
# y1 = 80
# x2 = 230
# y2 = 190
#
# Label:
# speed_limit_40
#
#
# 3. THIẾT BỊ THU NHẬN ẢNH
# ============================================================
#
# Có thể sử dụng:
#
# - Camera điện thoại.
# - Webcam.
# - Camera hành trình.
# - Camera IP.
# - Camera gắn trên ô tô.
#
# Đối với hệ thống thời gian thực:
#
# Camera
#    ↓
# Video Stream
#    ↓
# Computer Vision
#    ↓
# Kết quả nhận dạng
#
#
# 4. CÁC BƯỚC TIỀN XỬ LÝ
# ============================================================
#
# Các bước tiền xử lý đề xuất:
#
# Bước 1:
# Resize ảnh về kích thước phù hợp.
#
# Ví dụ:
#
# 1920 x 1080
#       ↓
# 640 x 480
#
# Mục đích:
# - Giảm thời gian xử lý.
# - Đảm bảo kích thước đầu vào đồng nhất.
#
#
# Bước 2:
# Khử nhiễu.
#
# Có thể sử dụng:
#
# - Gaussian Filter.
# - Median Filter.
#
# Nếu ảnh có Salt-and-Pepper Noise:
# → ưu tiên Median Filter.
#
#
# Bước 3:
# Tăng cường ảnh.
#
# Có thể sử dụng:
#
# - CLAHE.
# - Histogram Equalization.
#
# CLAHE phù hợp khi ánh sáng không đồng đều.
#
#
# Bước 4:
# Chuyển đổi không gian màu.
#
# Có thể chuyển:
#
# BGR → HSV
#
# HSV thuận tiện cho việc phân tích màu sắc
# của biển báo.
#
#
# 5. KHÔNG GIAN MÀU DỰ KIẾN SỬ DỤNG
# ============================================================
#
# Không gian màu chính:
#
# HSV
#
# HSV gồm:
#
# H = Hue
# → Loại màu.
#
# S = Saturation
# → Độ bão hòa.
#
# V = Value
# → Độ sáng.
#
# Biển báo giao thông thường có các màu đặc trưng
# như:
#
# - Đỏ.
# - Xanh.
# - Vàng.
#
# Do đó HSV có thể hỗ trợ xác định vùng màu
# của biển báo.
#
#
# Tuy nhiên, nếu sử dụng Deep Learning như YOLO,
# không nhất thiết phải chuyển toàn bộ ảnh sang HSV.
# Model có thể học trực tiếp đặc trưng từ ảnh RGB/BGR.
#
# Trong trường hợp này:
#
# Ảnh gốc
#    ↓
# Resize
#    ↓
# Normalize
#    ↓
# YOLO
#
#
# 6. CÔNG CỤ / THƯ VIỆN SỬ DỤNG
# ============================================================
#
# Có thể sử dụng:
#
# Python
#     ↓
# OpenCV
#     ↓
# NumPy
#     ↓
# Matplotlib
#     ↓
# YOLO / PyTorch
#
# Cụ thể:
#
# OpenCV:
# - Đọc ảnh.
# - Đọc video.
# - Resize.
# - Chuyển đổi màu.
# - Xử lý ảnh.
# - Hiển thị kết quả.
#
# NumPy:
# - Xử lý ma trận ảnh.
# - Thao tác pixel.
#
# Matplotlib:
# - Hiển thị và phân tích ảnh.
#
# YOLO:
# - Object Detection.
# - Phát hiện vị trí biển báo.
# - Phân loại biển báo.
#
#
# 7. QUY TRÌNH XỬ LÝ DỰ KIẾN
# ============================================================
#
# Pipeline tổng thể:
#
#              CAMERA
#                 ↓
#              ẢNH/VIDEO
#                 ↓
#           ĐỌC ẢNH BẰNG
#             OpenCV
#                 ↓
#              RESIZE
#                 ↓
#           TIỀN XỬ LÝ
#          ┌──────┼──────┐
#          ↓      ↓      ↓
#       Denoise  CLAHE  HSV
#          └──────┼──────┘
#                 ↓
#           OBJECT DETECTION
#               YOLO
#                 ↓
#        ┌────────┴────────┐
#        ↓                 ↓
#   Bounding Box         Label
#        ↓                 ↓
#        └────────┬────────┘
#                 ↓
#           HIỂN THỊ KẾT QUẢ
#                 ↓
#        Biển báo + vị trí
#        + độ tin cậy
#
#
# 8. KẾT QUẢ MONG MUỐN
# ============================================================
#
# Hệ thống cần trả về:
#
# - Vị trí biển báo.
# - Loại biển báo.
# - Độ tin cậy.
#
# Ví dụ:
#
# +------------------------------------------+
# |                                          |
# |       ┌──────────────────┐               |
# |       │  BIỂN BÁO        │               |
# |       │       40         │               |
# |       └──────────────────┘               |
# |                                          |
# +------------------------------------------+
#
# Kết quả:
#
# Class:
# Speed Limit 40
#
# Confidence:
# 95%
#
# Bounding Box:
# (x1, y1, x2, y2)
#
#
# ============================================================
# ĐÁNH GIÁ HỆ THỐNG
# ============================================================
#
# Có thể đánh giá bằng:
#
# 1. Accuracy
# → Tỷ lệ nhận dạng đúng.
#
# 2. Precision
# → Trong các biển báo hệ thống dự đoán,
#   bao nhiêu dự đoán là đúng.
#
# 3. Recall
# → Trong các biển báo thực tế,
#   hệ thống phát hiện được bao nhiêu.
#
# 4. F1-score
# → Cân bằng Precision và Recall.
#
# 5. mAP
# → Chỉ số rất quan trọng đối với Object Detection.
#
#
# ============================================================
# KẾT LUẬN
# ============================================================
#
# Hệ thống phát hiện biển báo giao thông gồm các bước:
#
# Camera
#   ↓
# Thu nhận ảnh
#   ↓
# Resize
#   ↓
# Khử nhiễu
#   ↓
# Tăng cường ảnh
#   ↓
# Object Detection
#   ↓
# Phân loại biển báo
#   ↓
# Hiển thị Bounding Box
#   ↓
# Kết quả
#
# Hệ thống có thể được triển khai trên ảnh tĩnh hoặc
# video thời gian thực.
#
# Nếu yêu cầu độ chính xác cao, có thể sử dụng YOLO
# được huấn luyện trên dataset biển báo giao thông.
#
# Nếu yêu cầu hệ thống đơn giản để minh họa các kỹ thuật
# xử lý ảnh, có thể kết hợp HSV + Threshold + Contour
# để phát hiện các vùng biển báo dựa trên màu sắc.