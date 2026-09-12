import cv2
import matplotlib.pyplot as plt
import time

# =====================================================
# 1. ĐỌC ẢNH
# =====================================================

img = cv2.imread("image.png")

if img is None:
    print("Không thể đọc ảnh!")
    exit()

# Chuyển sang ảnh xám
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# =====================================================
# 2. SIFT - PHÁT HIỆN KEYPOINT + TẠO DESCRIPTOR
# =====================================================

start_time = time.perf_counter()

# Khởi tạo SIFT
sift = cv2.SIFT_create()

# Phát hiện keypoint và descriptor
sift_keypoints, sift_descriptors = sift.detectAndCompute(
    gray,
    None
)

sift_time = time.perf_counter() - start_time


# =====================================================
# 3. ORB - PHÁT HIỆN KEYPOINT + TẠO DESCRIPTOR
# =====================================================

start_time = time.perf_counter()

# Khởi tạo ORB
orb = cv2.ORB_create(
    nfeatures=500
)

# Phát hiện keypoint và descriptor
orb_keypoints, orb_descriptors = orb.detectAndCompute(
    gray,
    None
)

orb_time = time.perf_counter() - start_time


# =====================================================
# 4. HIỂN THỊ KEYPOINT
# =====================================================

# Vẽ keypoint SIFT
sift_img = cv2.drawKeypoints(
    img,
    sift_keypoints,
    None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

# Vẽ keypoint ORB
orb_img = cv2.drawKeypoints(
    img,
    orb_keypoints,
    None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)


# =====================================================
# 5. IN THÔNG TIN DESCRIPTOR
# =====================================================

print("=" * 60)
print("SO SÁNH SIFT VÀ ORB")
print("=" * 60)

print("\nSIFT")
print("-" * 30)
print("Số lượng keypoint:", len(sift_keypoints))

if sift_descriptors is not None:
    print("Kích thước descriptor:", sift_descriptors.shape)
    print("Kiểu dữ liệu:", sift_descriptors.dtype)
else:
    print("Không tạo được descriptor")

print(f"Thời gian xử lý: {sift_time:.6f} giây")


print("\nORB")
print("-" * 30)
print("Số lượng keypoint:", len(orb_keypoints))

if orb_descriptors is not None:
    print("Kích thước descriptor:", orb_descriptors.shape)
    print("Kiểu dữ liệu:", orb_descriptors.dtype)
else:
    print("Không tạo được descriptor")

print(f"Thời gian xử lý: {orb_time:.6f} giây")


# =====================================================
# 6. HIỂN THỊ KẾT QUẢ
# =====================================================

plt.figure(figsize=(15, 7))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(sift_img, cv2.COLOR_BGR2RGB))
plt.title(f"SIFT - {len(sift_keypoints)} keypoints")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(orb_img, cv2.COLOR_BGR2RGB))
plt.title(f"ORB - {len(orb_keypoints)} keypoints")
plt.axis("off")

plt.tight_layout()
plt.show()