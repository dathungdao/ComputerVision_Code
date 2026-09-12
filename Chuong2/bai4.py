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
    print("Không thể đọc image2.pngppython bai!")
    exit()

# Chuyển sang ảnh xám
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)


# ============================================================
# 2. SIFT - PHÁT HIỆN KEYPOINT VÀ TẠO DESCRIPTOR
# ============================================================

start_time = time.perf_counter()

sift = cv2.SIFT_create()

keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

sift_time = time.perf_counter() - start_time


print("=" * 60)
print("THÔNG TIN SIFT")
print("=" * 60)

print("Ảnh 1:")
print("  Số keypoint:", len(keypoints1))
print("  Descriptor:", descriptors1.shape)

print("\nẢnh 2:")
print("  Số keypoint:", len(keypoints2))
print("  Descriptor:", descriptors2.shape)

print(f"\nThời gian SIFT: {sift_time:.6f} giây")


# ============================================================
# 3. HIỂN THỊ KEYPOINT CỦA HAI ẢNH
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
plt.imshow(cv2.cvtColor(img1_keypoints, cv2.COLOR_BGR2RGB))
plt.title(f"Image 1 - {len(keypoints1)} keypoints")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(img2_keypoints, cv2.COLOR_BGR2RGB))
plt.title(f"Image 2 - {len(keypoints2)} keypoints")
plt.axis("off")

plt.tight_layout()
plt.show()


# ============================================================
# 4. BF MATCHER
# ============================================================

bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)


# ============================================================
# 5. KNN MATCHING
# ============================================================

start_time = time.perf_counter()

matches_knn = bf.knnMatch(
    descriptors1,
    descriptors2,
    k=2
)

matching_time = time.perf_counter() - start_time


print("\n" + "=" * 60)
print("KẾT QUẢ KNN MATCHING")
print("=" * 60)

print("Tổng số nhóm matching:", len(matches_knn))
print(f"Thời gian matching: {matching_time:.6f} giây")


# ============================================================
# 6. HIỂN THỊ MATCHING TRƯỚC KHI RATIO TEST
# ============================================================

# Lấy match tốt nhất trong mỗi nhóm
matches_before = [m[0] for m in matches_knn if len(m) == 2]

img_matches_before = cv2.drawMatches(
    img1,
    keypoints1,
    img2,
    keypoints2,
    matches_before,
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

plt.figure(figsize=(18, 8))
plt.imshow(cv2.cvtColor(img_matches_before, cv2.COLOR_BGR2RGB))
plt.title(
    f"Feature Matching trước Ratio Test - "
    f"{len(matches_before)} matches"
)
plt.axis("off")
plt.show()


# ============================================================
# 7. RATIO TEST
# ============================================================

ratio_threshold = 0.75

good_matches = []

for m, n in matches_knn:

    if m.distance < ratio_threshold * n.distance:
        good_matches.append(m)


print("\n" + "=" * 60)
print("KẾT QUẢ SAU RATIO TEST")
print("=" * 60)

print("Ngưỡng Ratio Test:", ratio_threshold)
print("Số matching trước Ratio Test:", len(matches_before))
print("Số matching sau Ratio Test:", len(good_matches))


# ============================================================
# 8. HIỂN THỊ MATCHING SAU RATIO TEST
# ============================================================

img_matches_after = cv2.drawMatches(
    img1,
    keypoints1,
    img2,
    keypoints2,
    good_matches,
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

plt.figure(figsize=(18, 8))

plt.imshow(
    cv2.cvtColor(img_matches_after, cv2.COLOR_BGR2RGB)
)

plt.title(
    f"Feature Matching sau Ratio Test - "
    f"{len(good_matches)} good matches"
)

plt.axis("off")
plt.show()


# ============================================================
# 9. SO SÁNH KẾT QUẢ
# ============================================================

print("\n" + "=" * 60)
print("SO SÁNH TRƯỚC VÀ SAU RATIO TEST")
print("=" * 60)

print(f"Matching trước Ratio Test: {len(matches_before)}")
print(f"Matching sau Ratio Test:   {len(good_matches)}")

if len(matches_before) > 0:
    percentage = (
        len(good_matches) / len(matches_before)
    ) * 100

    print(
        f"Tỷ lệ matching giữ lại: {percentage:.2f}%"
    )