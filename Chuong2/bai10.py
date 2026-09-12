import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

# ============================================================
# BÀI TẬP 10: PHÂN ĐOẠN ẢNH BẰNG K-MEANS
# ============================================================

# ------------------------------------------------------------
# 1. ĐỌC ẢNH
# ------------------------------------------------------------

img = cv2.imread("image.png")

if img is None:
    print("Không thể đọc ảnh image.png!")
    exit()

# OpenCV đọc ảnh theo thứ tự BGR.
# Chúng ta giữ ảnh BGR để thực hiện K-means,
# sau đó chuyển sang RGB khi hiển thị bằng Matplotlib.
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ------------------------------------------------------------
# 2. CHUẨN BỊ DỮ LIỆU CHO K-MEANS
# ------------------------------------------------------------

# Mỗi pixel của ảnh màu có 3 giá trị:
# [B, G, R]
#
# K-means cần dữ liệu dạng:
# [x1]
# [x2]
# [x3]
# ...
#
# Vì vậy ta biến đổi ảnh từ:
# (height, width, 3)
#
# thành:
# (height * width, 3)

pixel_data = np.float32(
    img.reshape((-1, 3))
)


# ------------------------------------------------------------
# 3. THIẾT LẬP THAM SỐ K-MEANS
# ------------------------------------------------------------

# K là số cụm màu mà thuật toán sẽ tạo ra.
#
# K càng nhỏ:
# - Ảnh chỉ có ít nhóm màu.
# - Phân đoạn đơn giản.
# - Nhưng có thể làm mất nhiều chi tiết màu.
#
# K càng lớn:
# - Có nhiều nhóm màu hơn.
# - Giữ lại nhiều chi tiết màu hơn.
# - Nhưng ảnh có thể trở nên quá chi tiết
#   và mất ý nghĩa của việc phân đoạn.

K_values = [2, 3, 4, 6]


# ------------------------------------------------------------
# 4. CÁC THAM SỐ CỦA K-MEANS
# ------------------------------------------------------------

# Tiêu chí dừng:
# Khi centroid thay đổi rất nhỏ (< 1.0)
# hoặc thuật toán thực hiện đủ 100 lần lặp
# thì K-means sẽ dừng.

criteria = (
    cv2.TERM_CRITERIA_EPS +
    cv2.TERM_CRITERIA_MAX_ITER,
    100,
    1.0
)

results = []


# ------------------------------------------------------------
# 5. THỰC HIỆN K-MEANS VỚI TỪNG GIÁ TRỊ K
# ------------------------------------------------------------

for K in K_values:

    print("\n" + "=" * 60)
    print(f"K = {K}")
    print("=" * 60)

    start_time = time.perf_counter()

    # cv2.kmeans thực hiện phân cụm các pixel thành K nhóm.
    #
    # K-means hoạt động theo nguyên tắc:
    # 1. Chọn K centroid ban đầu.
    # 2. Gán mỗi pixel vào centroid gần nhất.
    # 3. Tính lại centroid.
    # 4. Lặp lại cho đến khi hội tụ.
    #
    # cv2.KMEANS_PP_CENTERS:
    # sử dụng K-means++ để chọn centroid ban đầu,
    # thường giúp kết quả ổn định hơn so với chọn ngẫu nhiên.

    compactness, labels, centers = cv2.kmeans(
        pixel_data,
        K,
        None,
        criteria,
        10,
        cv2.KMEANS_PP_CENTERS
    )

    processing_time = time.perf_counter() - start_time

    # --------------------------------------------------------
    # 6. CHUYỂN CENTROID VỀ KIỂU UINT8
    # --------------------------------------------------------

    # centers chứa màu đại diện của K cụm.
    #
    # Ví dụ:
    # K = 3
    # centers có 3 màu đại diện:
    #
    # [B1, G1, R1]
    # [B2, G2, R2]
    # [B3, G3, R3]

    centers = np.uint8(centers)


    # --------------------------------------------------------
    # 7. TẠO ẢNH SAU KHI PHÂN ĐOẠN
    # --------------------------------------------------------

    # Mỗi pixel sẽ được thay thế bằng màu của centroid
    # thuộc cụm mà pixel đó được phân loại vào.

    segmented = centers[labels.flatten()]

    # Đưa dữ liệu từ dạng:
    # (height * width, 3)
    #
    # trở lại:
    # (height, width, 3)

    segmented = segmented.reshape(img.shape)


    # --------------------------------------------------------
    # 8. LƯU KẾT QUẢ
    # --------------------------------------------------------

    results.append({
        "K": K,
        "segmented": segmented,
        "centers": centers,
        "compactness": compactness,
        "time": processing_time
    })

    print(f"Số cụm màu K: {K}")
    print(f"Compactness: {compactness:.2f}")
    print(f"Thời gian xử lý: {processing_time:.6f} giây")


# ============================================================
# 9. HIỂN THỊ ẢNH GỐC VÀ CÁC KẾT QUẢ
# ============================================================

plt.figure(figsize=(16, 10))

# ------------------------------------------------------------
# Ảnh gốc
# ------------------------------------------------------------

plt.subplot(2, 3, 1)

plt.imshow(img_rgb)

plt.title("Ảnh gốc")

plt.axis("off")


# ------------------------------------------------------------
# Hiển thị kết quả K-means
# ------------------------------------------------------------

for i, result in enumerate(results):

    plt.subplot(2, 3, i + 2)

    # OpenCV sử dụng BGR nên cần chuyển sang RGB
    # trước khi hiển thị bằng Matplotlib.

    segmented_rgb = cv2.cvtColor(
        result["segmented"],
        cv2.COLOR_BGR2RGB
    )

    plt.imshow(segmented_rgb)

    plt.title(
        f"K-means - K = {result['K']}"
    )

    plt.axis("off")


plt.tight_layout()
plt.show()


# ============================================================
# 10. HIỂN THỊ CÁC CENTROID MÀU
# ============================================================

for result in results:

    K = result["K"]
    centers = result["centers"]

    print("\n" + "-" * 60)
    print(f"Các màu đại diện khi K = {K}")
    print("-" * 60)

    for i, center in enumerate(centers):

        # center đang ở dạng BGR.
        # Chuyển sang RGB để dễ đọc.

        b, g, r = center

        print(
            f"Cụm {i + 1}: "
            f"B={b}, G={g}, R={r}"
        )


# ============================================================
# 11. SO SÁNH THỜI GIAN XỬ LÝ
# ============================================================

print("\n" + "=" * 70)
print("SO SÁNH KẾT QUẢ")
print("=" * 70)

print(
    f"{'K':<10}"
    f"{'Compactness':<20}"
    f"{'Thời gian (s)':<20}"
)

print("-" * 70)

for result in results:

    print(
        f"{result['K']:<10}"
        f"{result['compactness']:<20.2f}"
        f"{result['time']:<20.6f}"
    )


# ============================================================
# 12. NHẬN XÉT TỰ ĐỘNG
# ============================================================

print("\n" + "=" * 70)
print("NHẬN XÉT")
print("=" * 70)

print("""
1. ẢNH VỚI K NHỎ
   - Chỉ có một số ít cụm màu.
   - Ảnh được đơn giản hóa mạnh.
   - Các màu gần nhau có thể bị gom vào cùng một cụm.
   - Ưu điểm: ảnh đơn giản, ít dữ liệu.
   - Nhược điểm: mất nhiều chi tiết.

2. K TĂNG
   - Số lượng màu đại diện tăng.
   - Các vùng màu được phân biệt chi tiết hơn.
   - Ảnh sau phân đoạn gần với ảnh gốc hơn.
   - Tuy nhiên thời gian tính toán thường tăng.

3. K QUÁ NHỎ
   - Không đủ cụm để biểu diễn các màu khác nhau.
   - Nhiều vùng có màu khác nhau bị gộp lại.
   - Biên giữa các đối tượng có thể bị mất.
   - Kết quả phân đoạn bị quá đơn giản.

4. K QUÁ LỚN
   - Ảnh có quá nhiều cụm màu.
   - Những khác biệt màu rất nhỏ cũng có thể tạo thành
     các cụm riêng.
   - Kết quả chứa nhiều chi tiết không cần thiết.
   - Tăng thời gian tính toán.
   - Có thể làm mất mục tiêu "đơn giản hóa" của phân đoạn.

5. LỰA CHỌN K
   - Không có một giá trị K tốt nhất cho mọi ảnh.
   - K nên được lựa chọn dựa trên mục đích phân đoạn.
   - Có thể thử nhiều giá trị K rồi so sánh kết quả.
""")