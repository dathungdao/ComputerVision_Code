import cv2
img = cv2.imread("image.png")
cv2.imshow("Anh goc", img)
print("Shape:", img.shape)
if len(img.shape) == 2:
    print("Số kênh màu: 1 (ảnh grayscale)")
else:
    print("Số kênh màu:", img.shape[2])
print("Kiểu dữ liệu:", img.dtype)
cv2.imwrite("image_new.png", img)
print("Đã lưu ảnh thành image_new.png")
cv2.waitKey(0)
cv2.destroyAllWindows()