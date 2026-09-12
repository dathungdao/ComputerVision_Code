import cv2
import matplotlib.pyplot as plt
import time

# ============================================================
# 1. ĐỌC HAI ẢNH
# ============================================================

img1 = cv2.imread("image1.png")
img2 = cv2.imread("image2.png")

if img1 is None:
    print("Không thể đọc image1.png!")
    exit()

if img2 is None:
    print("Không thể đọc image2.png!")
    exit()


# Chuyển sang ảnh xám
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)


# ============================================================
# 2. TẠO ORB
# ============================================================

start_time = time.perf_counter()

orb = cv2.ORB_create(
    nfeatures=1000
)

# Phát hiện keypoint + tạo descriptor
keypoints1, descriptors1 = orb.detectAndCompute(
    gray1, None
)

keypoints2, descriptors2 = orb.detectAndCompute(
    gray2, None
)

orb_time = time.perf_counter() - start_time


# ============================================================
# 3. HIỂN THỊ KEYPOINT
# ============================================================

img1_keypoints = cv2.drawKeypoints(
    img1,
    keypoints1,
    None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

img2_keypoints = cv2.drawKeypoints(
    img2,
    keypoints2,
    None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)


plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
plt.imshow(
    cv2.cvtColor(img1_keypoints, cv2.COLOR_BGR2RGB)
)
plt.title(f"ORB - Image 1: {len(keypoints1)} keypoints")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(
    cv2.cvtColor(img2_keypoints, cv2.COLOR_BGR2RGB)
)
plt.title(f"ORB - Image 2: {len(keypoints2)} keypoints")
plt.axis("off")

plt.tight_layout()
plt.show()


# ============================================================
# 4. KIỂM TRA DESCRIPTOR
# ============================================================

print("=" * 60)
print("THÔNG TIN ORB")
print("=" * 60)

print("Image 1:")
print("  Số keypoint:", len(keypoints1))
print("  Descriptor shape:", descriptors1.shape)
print("  Descriptor dtype:", descriptors1.dtype)

print("\nImage 2:")
print("  Số keypoint:", len(keypoints2))
print("  Descriptor shape:", descriptors2.shape)
print("  Descriptor dtype:", descriptors2.dtype)

print(f"\nThời gian ORB: {orb_time:.6f} giây")


# ============================================================
# 5. BF MATCHER + HAMMING DISTANCE
# ============================================================

bf = cv2.BFMatcher(
    cv2.NORM_HAMMING,
    crossCheck=True
)


# ============================================================
# 6. FEATURE MATCHING
# ============================================================

start_time = time.perf_counter()

matches = bf.match(
    descriptors1,
    descriptors2
)

matching_time = time.perf_counter() - start_time


# ============================================================
# 7. SẮP XẾP MATCH THEO DISTANCE
# ============================================================

matches = sorted(
    matches,
    key=lambda x: x.distance
)


# Lấy 20 match tốt nhất
best_matches = matches[:20]


print("\n" + "=" * 60)
print("KẾT QUẢ FEATURE MATCHING")
print("=" * 60)

print("Tổng số matches:", len(matches))
print("Số match tốt nhất được hiển thị:", len(best_matches))
print(f"Thời gian matching: {matching_time:.6f} giây")

print("\nDistance của 20 match tốt nhất:")

for i, match in enumerate(best_matches):
    print(
        f"Match {i + 1:2d}: "
        f"Distance = {match.distance:.2f}"
    )


# ============================================================
# 8. HIỂN THỊ 20 MATCH TỐT NHẤT
# ============================================================

result = cv2.drawMatches(
    img1,
    keypoints1,
    img2,
    keypoints2,
    best_matches,
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)


plt.figure(figsize=(18, 9))

plt.imshow(
    cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
)

plt.title(
    "ORB + Hamming Distance - 20 Best Matches"
)

plt.axis("off")
plt.tight_layout()
plt.show()