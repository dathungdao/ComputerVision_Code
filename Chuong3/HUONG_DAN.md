# Hướng dẫn đọc hiểu project – Nhận dạng rau thơm từ ảnh lá

Tài liệu này dành cho người **chưa biết gì** về project (thậm chí chưa học xử lý ảnh hay học máy). Đọc từ trên xuống: phần 1–2 cho biết project làm gì và gồm những gì; phần 3 giải thích các khái niệm cần biết; phần 4–6 đi vào từng file code; phần 7–8 là cách chạy và câu hỏi thường gặp.

Muốn xem **kết quả và phân tích** thì đọc [BaoCao.md](BaoCao.md). Tài liệu này chỉ giải thích **project được tổ chức và hoạt động như thế nào**.

---

## Mục lục

1. [Project này làm gì?](#1-project-này-làm-gì)
2. [Bản đồ thư mục và file](#2-bản-đồ-thư-mục-và-file)
3. [Các khái niệm cần biết trước](#3-các-khái-niệm-cần-biết-trước)
4. [File chính: nhan_dang_rau_thom.py](#4-file-chính-nhan_dang_rau_thompy)
5. [File du_doan.py](#5-file-du_doanpy)
6. [File demo_web.py](#6-file-demo_webpy)
7. [Cách cài đặt và chạy](#7-cách-cài-đặt-và-chạy)
8. [Câu hỏi thường gặp](#8-câu-hỏi-thường-gặp)
9. [Bảng thuật ngữ](#9-bảng-thuật-ngữ)

---

## 1. Project này làm gì?

**Đầu vào:** một bức ảnh chụp **một chiếc lá** đặt trên tờ giấy trắng.
**Đầu ra:** tên loại rau: **Húng quế**, **Tía tô**, **Mùi tàu** hoặc **Lá lốt**, kèm độ tin cậy.

Ví dụ: đưa ảnh một chiếc lá dài, hẹp, mép có gai → chương trình trả lời "Mùi tàu (100%)".

Project là bài tập Chương 3 môn Thị giác máy tính. Đề bài yêu cầu làm theo **quy trình 12 bước của hệ thống nhận dạng truyền thống**. "Truyền thống" nghĩa là: con người tự thiết kế cách đo đặc điểm của lá (màu, hình dạng, bề mặt…), rồi đưa các con số đó cho một thuật toán học máy để nó học cách phân loại. Cách này khác với *deep learning*, nơi máy tự học cách nhìn ảnh.

Toàn bộ quá trình gồm hai giai đoạn:

```
GIAI ĐOẠN 1 – HUẤN LUYỆN (chạy 1 lần, ~16 phút)        file: nhan_dang_rau_thom.py

  198 ảnh lá      →  tách lá   →  đo đặc điểm  →  thử nhiều  →  chọn cách  →  lưu mô hình
  đã biết tên        khỏi nền     (ra các con số)   cách học     tốt nhất      model.joblib
                                                                               + biểu đồ, bảng
                                                                                 kết quả

GIAI ĐOẠN 2 – SỬ DỤNG (mỗi ảnh < 1 giây)                file: du_doan.py hoặc demo_web.py

  1 ảnh lá mới    →  tách lá   →  đo đặc điểm  →  mô hình đã lưu  →  "Tía tô (95%)"
```

---

## 2. Bản đồ thư mục và file

```
Chuong3/
├── nhan_dang_rau_thom.py   ← FILE CHÍNH: toàn bộ 12 bước, huấn luyện + đánh giá
├── du_doan.py              ← dự đoán ảnh mới bằng dòng lệnh
├── demo_web.py             ← giao diện web để demo (http://127.0.0.1:8501)
├── BaoCao.md               ← báo cáo kết quả nộp giảng viên
├── HUONG_DAN.md            ← tài liệu bạn đang đọc
├── .gitignore              ← danh sách file KHÔNG đưa lên GitHub
│
├── dataset/                ← DỮ LIỆU HUẤN LUYỆN (198 ảnh, đã thu nhỏ 800px, 12 MB)
│   ├── hung_que/   hung_que_01.jpg … hung_que_50.jpg
│   ├── tia_to/     tia_to_01.jpg   … tia_to_50.jpg
│   ├── mui_tau/    mui_tau_01.jpg  … mui_tau_48.jpg
│   └── la_lot/     la_lot_01.jpg   … la_lot_50.jpg
│
├── dataset_goc/            ← ảnh gốc 50MP từ điện thoại (2,3 GB) – CHỈ CÓ TRÊN MÁY, không lên GitHub
│   ├── (4 thư mục giống dataset/)
│   └── ten_anh_goc.csv     ← bảng đối chiếu tên mới ↔ tên gốc (tên gốc chứa giờ chụp)
│
└── results/                ← MỌI KẾT QUẢ do nhan_dang_rau_thom.py sinh ra
    ├── log.txt             ← toàn bộ nội dung in ra màn hình khi chạy (đọc file này trước)
    ├── model.joblib        ← MÔ HÌNH ĐÃ HUẤN LUYỆN (du_doan.py và demo_web.py dùng file này)
    ├── cache_dataset.joblib← đặc trưng đã tính sẵn để lần chạy sau nhanh hơn (không lên GitHub)
    ├── buoc02_…png         ← hình minh họa của từng bước (số = số bước)
    ├── buoc03_nhan.csv     ← bảng nhãn của từng ảnh
    ├── …
    ├── kich_ban1…5 (.csv, .png) ← bảng số liệu và biểu đồ của 5 kịch bản thử nghiệm
    └── roi/                ← lưới ảnh tất cả các lá đã được cắt ra (để kiểm tra bằng mắt)
```

### Chi tiết từng file trong `results/`

| File | Bước | Nội dung |
|---|---|---|
| `log.txt` | Tất cả | Bản ghi đầy đủ một lần chạy: số ảnh, kết quả từng kịch bản, cấu hình cuối, đánh giá |
| `buoc02_so_anh.png` | 2 | Biểu đồ số ảnh của mỗi loại |
| `buoc03_nhan.csv` | 3 | Mỗi dòng một ảnh: đường dẫn, tên loại, mã số loại, khối (dùng cho bước 9) |
| `buoc04_05_tien_xu_ly_roi.png` | 4–5 | Mỗi loại một hàng: ảnh gốc → mask → lá đã xoay → sau cân bằng sáng |
| `roi/roi_<loại>.jpg` | 5 | Tất cả lá đã tách của một loại. Viền xanh = tách được, viền đỏ = thất bại |
| `buoc06_dac_trung_<loại>.png` | 6 | Minh họa 5 nhóm đặc trưng trên một lá |
| `buoc06_bien_doi_anh.png` | KB5 | Một ảnh sau 5 phép biến đổi (mờ, nhiễu, sáng, tối, xoay) |
| `kich_ban1_dac_trung_x_phan_lop.csv` / `kich_ban1_heatmap.png` | 10 | Điểm F1 của 10 tổ hợp đặc trưng × 6 thuật toán |
| `kich_ban2_chuan_hoa.csv` | 8 | So sánh các cách chuẩn hóa |
| `kich_ban3_roi.csv` | 5 | Có tách lá so với dùng cả ảnh |
| `kich_ban4_ngau_nhien_vs_khoi.csv` / `kich_ban4.png` | 9 | Chia dữ liệu ngẫu nhiên so với chia theo khối |
| `kich_ban5_bien_doi_anh.csv` / `kich_ban5.png` | KB5 | Mô hình có/không tăng cường dữ liệu, test trên ảnh bị biến đổi |
| `buoc11_du_doan.png` | 11 | Ví dụ dự đoán trên các lá mô hình chưa thấy |
| `buoc12_confusion.png` | 12 | Ma trận nhầm lẫn |
| `buoc12_anh_sai.png` | 12 | Những ảnh bị đoán sai |
| `model.joblib` | 10 | Mô hình cuối (xem [mục 4.10](#410-mô-hình-được-lưu-thế-nào-modeljoblib)) |
| `cache_dataset.joblib` | 4–6 | Đặc trưng đã tính, để lần chạy sau khỏi tính lại |

### Vì sao có hai thư mục ảnh?

Ảnh điện thoại gốc rất nặng (mỗi ảnh ~12 MB, tổng 2,3 GB), không đưa lên GitHub được. Vì chương trình luôn thu nhỏ ảnh về cạnh dài 800px trước khi xử lý, ta lưu sẵn bản 800px vào `dataset/` (chỉ 12 MB) – kết quả giống hệt khi dùng ảnh gốc. Ảnh gốc giữ trong `dataset_goc/` để phòng khi cần.

### Vì sao tên ảnh đánh số 01, 02, …?

Số thứ tự là **thứ tự chụp**. Mỗi chiếc lá được chụp khoảng 10 tấm liên tiếp, nên các ảnh có số gần nhau thường là cùng một chiếc lá. Bước 9 dựa vào điều này để chia dữ liệu (xem [mục 3.6](#36-chia-dữ-liệu-và-cross-validation)).

---

## 3. Các khái niệm cần biết trước

### 3.1. Máy tính "nhìn" ảnh như thế nào?

Với máy tính, ảnh là một bảng số khổng lồ. Ảnh 800×600 điểm ảnh (pixel) có 480.000 pixel; mỗi pixel có 3 số từ 0 đến 255 cho ba màu Xanh dương – Xanh lá – Đỏ (OpenCV dùng thứ tự **BGR**). Ví dụ pixel `[30, 120, 40]` là màu xanh lá đậm.

Máy không tự hiểu "đây là chiếc lá". Ta phải viết code để biến bảng số đó thành thông tin có ý nghĩa.

### 3.2. Đặc trưng (feature) là gì?

**Đặc trưng** là một con số mô tả một đặc điểm của đối tượng. Ví dụ với lá:

- *tỉ lệ dài/rộng*: lá mùi tàu ≈ 6, lá lốt ≈ 1,1
- *độ tròn*: hình tròn hoàn hảo = 1, lá càng dài càng gần 0
- *tỉ lệ pixel màu tím*: tía tô mặt dưới cao, các lá khác ≈ 0

Gom nhiều đặc trưng lại thành một dãy số gọi là **vector đặc trưng**, ví dụ `[6.2, 0.23, 0.01, …]`. Mỗi ảnh lá trở thành một vector – thuật toán học máy làm việc với các vector này chứ không với ảnh.

### 3.3. Bộ phân lớp (classifier) là gì?

Là thuật toán học từ nhiều ví dụ **(vector đặc trưng, tên loại)** để sau này nhìn một vector mới thì đoán ra tên loại. Project thử 6 thuật toán:

| Thuật toán | Ý tưởng dễ hiểu |
|---|---|
| **KNN** (K láng giềng gần nhất) | Tìm K lá trong dữ liệu học giống lá mới nhất, lấy loại xuất hiện nhiều nhất |
| **SVM** | Tìm "đường ranh giới" tách các loại xa nhau nhất có thể |
| **Decision Tree** (cây quyết định) | Chuỗi câu hỏi có/không: "tỉ lệ dài/rộng > 3?" → "có răng cưa?" → … |
| **Random Forest** | 200 cây quyết định khác nhau cùng bỏ phiếu → ổn định hơn một cây |
| **Naive Bayes** | Tính xác suất mỗi loại dựa trên phân bố của từng đặc trưng |
| **Logistic Regression** | Mỗi loại một công thức cộng có trọng số các đặc trưng, loại nào điểm cao nhất thắng |

Mỗi thuật toán có vài **siêu tham số** (hyperparameter) – "núm vặn" phải chọn trước khi học, ví dụ K trong KNN. Chương trình tự thử nhiều giá trị và chọn giá trị tốt nhất.

### 3.4. Huấn luyện – kiểm tra (train – test)

- **Tập huấn luyện (train):** ảnh cho thuật toán học, có đáp án.
- **Tập kiểm tra (test):** ảnh giấu đi, chỉ dùng để chấm điểm sau khi học xong.

Không được dùng ảnh test để học – giống như cho học sinh xem trước đề thi, điểm cao nhưng không phản ánh năng lực thật.

### 3.5. Đo độ chính xác

- **Accuracy:** tỉ lệ ảnh đoán đúng. 188/198 đúng → 0,949.
- **Precision của loại X:** trong các ảnh *máy bảo là X*, bao nhiêu % đúng là X.
- **Recall của loại X:** trong các ảnh *thật sự là X*, máy tìm ra được bao nhiêu %.
- **F1:** trung bình (điều hòa) của precision và recall. "Macro F1" = trung bình F1 của 4 loại – đây là thước đo chính của project.
- **Confusion matrix (ma trận nhầm lẫn):** bảng 4×4, hàng = loại thật, cột = loại máy đoán. Đường chéo là đoán đúng; ô ngoài đường chéo cho biết loại nào hay bị nhầm thành loại nào.

### 3.6. Chia dữ liệu và cross-validation

**Cross-validation k-fold:** chia dữ liệu thành k phần; lần lượt lấy 1 phần làm test, k−1 phần còn lại làm train; lặp k lần rồi lấy trung bình. Mọi ảnh đều được làm test đúng một lần → kết quả ổn định hơn chia một lần.

**Vấn đề riêng của project:** mỗi chiếc lá có ~10 ảnh gần giống nhau. Nếu chia ngẫu nhiên theo ảnh, ảnh của cùng một lá sẽ nằm ở cả train và test → máy chỉ cần "nhớ mặt" chiếc lá là đoán đúng → điểm cao giả tạo.

**Giải pháp – group cross-validation:** chia mỗi loại thành **5 khối ảnh liên tiếp** (01–10, 11–20, …). Fold 1 lấy khối 1 của mọi loại làm test, v.v. Vì ảnh cùng một lá thường liên tiếp nhau, chúng nằm chung một khối → test gồm những lá máy chưa từng thấy.

**Nested CV (CV lồng):** việc chọn siêu tham số cũng làm bằng cross-validation, nhưng *chỉ bên trong tập train* của mỗi fold. Như vậy tập test không bị dùng vào việc chọn lựa nào.

### 3.7. Tăng cường dữ liệu (data augmentation)

Tạo thêm ảnh huấn luyện bằng cách biến đổi ảnh có sẵn: làm mờ, thêm nhiễu, làm sáng, làm tối, xoay. Máy được học lá trong nhiều điều kiện hơn → chịu được ảnh xấu tốt hơn. Ảnh biến đổi **chỉ dùng để học**, không dùng để chấm điểm.

---

## 4. File chính: `nhan_dang_rau_thom.py`

File dài nhất (~1.460 dòng), chứa toàn bộ 12 bước. Code được chia thành các khối, mỗi khối có tiêu đề dạng:

```python
# ============================================================
# BƯỚC 5. XÁC ĐỊNH VÙNG ĐỐI TƯỢNG (ROI = CHIẾC LÁ)
# ============================================================
```

Phần đầu file chỉ **định nghĩa các hàm** (công cụ). Hàm `main()` ở cuối file mới là nơi **gọi các hàm theo thứ tự 12 bước**. Dòng cuối:

```python
if __name__ == "__main__":
    main()
```

nghĩa là: chỉ chạy `main()` khi file được chạy trực tiếp (`python nhan_dang_rau_thom.py`). Khi file khác *import* để dùng lại các hàm (như `du_doan.py`, `demo_web.py`), `main()` không chạy.

### 4.1. Phần đầu: thư viện và hằng số

**Thư viện dùng:**

| Thư viện | Dùng để |
|---|---|
| `cv2` (OpenCV) | Đọc/ghi ảnh, đổi không gian màu, lọc, tìm đường viền, xoay ảnh, ORB |
| `numpy` | Tính toán trên mảng số (ảnh là mảng numpy) |
| `skimage` (scikit-image) | HOG, LBP, GLCM |
| `sklearn` (scikit-learn) | 6 bộ phân lớp, chuẩn hóa, PCA, k-means, cross-validation, các thước đo |
| `matplotlib` | Vẽ biểu đồ, lưu hình |
| `joblib` | Lưu/đọc mô hình và cache ra file |

**Hằng số quan trọng** (có thể sửa để thay đổi hành vi):

| Hằng số | Giá trị | Ý nghĩa |
|---|---|---|
| `CLASSES` | `["hung_que", "tia_to", "mui_tau", "la_lot"]` | Tên các loại = tên thư mục trong `dataset/` |
| `CLASS_NAMES` | `{"hung_que": "Húng quế", …}` | Tên có dấu để hiển thị |
| `MAX_SIDE` | 800 | Ảnh được thu nhỏ để cạnh dài nhất = 800px |
| `ROI_W, ROI_H` | 384, 192 | Kích thước khung chứa lá sau khi cắt và xoay |
| `MIN_PER_CLASS` | 40 | Dưới số này chương trình cảnh báo "ít ảnh" |
| `MIN_TO_USE` | 10 | Loại có ít hơn 10 ảnh bị bỏ qua |
| `N_FOLDS` | 5 | Số khối / số fold cross-validation |
| `FEATURE_VERSION` | 3 | Tăng số này khi sửa cách tính đặc trưng → cache cũ tự bị bỏ |
| `SEED` | 42 | Hạt giống ngẫu nhiên – mỗi lần chạy cho cùng kết quả |

### 4.2. Các hàm tiện ích

| Hàm / lớp | Làm gì |
|---|---|
| `Tee` | Vừa in ra màn hình vừa ghi vào `results/log.txt` |
| `rel(path)` | Rút gọn đường dẫn cho dễ đọc khi in |
| `section(title)` | In tiêu đề có gạch ngang ngăn cách các bước |
| `save_csv(path, headers, rows)` | Ghi bảng ra file CSV (mở được bằng Excel, hiển thị đúng tiếng Việt) |

### 4.3. Bước 2 – Đọc dữ liệu

| Hàm | Làm gì |
|---|---|
| `imread_unicode(path)` | Đọc ảnh. Trên Windows, `cv2.imread` thông thường lỗi khi đường dẫn có dấu tiếng Việt, nên đọc file thành byte rồi giải mã bằng `cv2.imdecode`. Với ảnh rất lớn, giải mã ở **1/4 độ phân giải** cho nhanh (đằng nào cũng thu nhỏ) |
| `imwrite_unicode(path, img)` | Ghi ảnh, cũng hỗ trợ đường dẫn có dấu |
| `resize_long(img)` | Thu nhỏ ảnh sao cho cạnh dài = 800px, giữ nguyên tỉ lệ |
| `list_images(root, classes)` | Duyệt `dataset/<loại>/`, lấy danh sách file ảnh của từng loại; báo file/thư mục lạ bị bỏ qua |
| `load_images(files, classes)` | Đọc tất cả ảnh vào bộ nhớ. Trả về 3 thứ: danh sách ảnh, mảng nhãn số `y` (0, 1, 2, 3), danh sách đường dẫn |

### 4.4. Bước 4 – Tiền xử lý

`normalize_illumination(img)`: đổi ảnh sang không gian màu **Lab** (L = độ sáng, a và b = màu), áp **CLAHE** (cân bằng histogram cục bộ – làm vùng tối sáng lên, vùng chói dịu xuống) **chỉ trên kênh L**, rồi đổi về BGR. Nhờ vậy ảnh chụp ở ánh sáng khác nhau trở nên giống nhau hơn mà màu lá không bị đổi.

Các thao tác tiền xử lý khác (thu nhỏ, lọc Gaussian, chuyển ảnh xám) nằm rải trong các hàm của bước 5–6.

### 4.5. Bước 5 – Tách lá khỏi nền

Mục tiêu: từ ảnh tờ giấy có chiếc lá → ảnh chỉ còn chiếc lá, đã xoay nằm ngang, cùng kích thước.

```
ảnh gốc ─→ leaf_mask ─→ largest_component ─→ align_leaf ─→ ROI 384×192
            (lá trắng,    (giữ vùng lớn nhất,   (xoay, lật,      + mask
             nền đen)      lấp lỗ)               đặt vào khung)
```

| Hàm | Làm gì, chi tiết |
|---|---|
| `leaf_mask(blur)` | 1) Lấy các pixel ở **viền ảnh** (viền 10px), tính màu trung vị → đó là màu giấy. 2) Với mỗi pixel, tính "khoảng cách màu" tới màu giấy trong không gian Lab; kênh độ sáng L nhân 0,4 để **bóng đổ** (chỉ tối hơn giấy, không khác màu) ít bị coi là lá. 3) **Ngưỡng Otsu**: tự tìm ngưỡng tách "gần màu giấy" và "khác màu giấy". 4) **Morphology** open (xóa chấm nhiễu nhỏ) và close (lấp khe hở nhỏ). Kết quả: ảnh trắng đen, lá màu trắng |
| `largest_component(mask)` | Tìm các đường viền (`findContours`), giữ **vùng lớn nhất** (chiếc lá), tô kín bên trong (lấp lỗ do vết phản chiếu trên lá bóng) |
| `rotate_bound(img, angle, …)` | Xoay ảnh mà **không bị cắt mất góc** (tự mở rộng khung) |
| `letterbox(img, mask)` | Thu nhỏ lá cho vừa khung 384×192 **giữ nguyên tỉ lệ**, phần thừa tô trắng |
| `align_leaf(img, mask)` | 1) **PCA** trên tọa độ các pixel của lá → tìm hướng lá dài nhất. 2) Xoay để hướng đó nằm ngang. 3) Cắt sát lá. 4) Lật ngang/dọc sao cho nửa có diện tích lớn hơn luôn ở bên trái/bên trên. → Mọi lá về **cùng một tư thế**, dù chụp nghiêng thế nào |
| `find_leaf(img)` | Hàm tổng hợp gọi các hàm trên. Nếu diện tích lá hợp lý (0,3%–95% ảnh) → trả về ROI, mask và thông tin `{"method": "tach_la", "box": khung, "mask": …}`. Nếu không tách được → dùng cả ảnh, `method = "toan_anh"` |

**Vì sao lại cần xoay về tư thế chuẩn?** Đặc trưng HOG đo hướng đường nét theo từng vùng ảnh. Nếu cùng một lá lúc nằm ngang, lúc dựng đứng thì HOG khác hẳn nhau; xoay về tư thế chuẩn giúp HOG chỉ còn phản ánh hình dạng thật của lá.

### 4.6. Bước 6 – Trích chọn đặc trưng

Từ ROI (ảnh lá 384×192) và mask, tính 5 nhóm đặc trưng:

**a) `color_features(roi, mask)` – Màu (43 số)**
- Đổi sang không gian **HSV** (H = sắc màu, S = độ đậm màu, V = độ sáng).
- **Histogram:** đếm có bao nhiêu pixel rơi vào từng khoảng màu: H chia 18 khoảng, S 8 khoảng, V 8 khoảng → 34 số, chia cho tổng để thành tỉ lệ.
- **Color moments:** trung bình, độ lệch chuẩn, độ lệch (skew) của H, S, V → 9 số.
- Chỉ tính trên pixel thuộc lá (nhờ mask), không tính nền giấy.

**b) `texture_features(gray, mask)` – Kết cấu bề mặt (33 số)**
- **LBP** (Local Binary Pattern): với mỗi pixel, so sánh với các pixel xung quanh (sáng hơn = 1, tối hơn = 0) → được một "mẫu" mô tả kết cấu nhỏ quanh pixel đó. Đếm tần suất các mẫu → histogram. Tính ở 2 bán kính (1 và 2 pixel) → 10 + 18 = 28 số.
- **GLCM** (ma trận đồng xuất hiện mức xám): đếm tần suất hai pixel cách nhau 1–3 pixel có cặp độ sáng nào, rồi tính 5 chỉ số: contrast, dissimilarity, homogeneity, energy, correlation → 5 số.
- Mask được co lại vài pixel (erode 7×7) để bỏ đường biên lá–giấy (biên không phải kết cấu bề mặt lá).
- Ý nghĩa: gân nổi lá lốt, lông tơ tía tô, bề mặt bóng húng quế tạo ra các mẫu LBP/GLCM khác nhau.

**c) `shape_features(mask)` – Hình dạng (24 số)**

| Số đo | Cách tính | Ý nghĩa |
|---|---|---|
| Độ tròn (circularity) | 4π × diện tích / chu vi² | Tròn = 1; lá lốt cao, mùi tàu thấp |
| Tỉ lệ dài/rộng | cạnh dài / cạnh ngắn của hình chữ nhật bao nhỏ nhất | Mùi tàu ≈ 6, lá lốt ≈ 1,1 |
| Extent | diện tích lá / diện tích hình chữ nhật bao | Lá lấp đầy khung đến đâu |
| Solidity | diện tích lá / diện tích **bao lồi** | Mép lõm nhiều → thấp |
| Convexity | chu vi bao lồi / chu vi lá | Mép răng cưa → chu vi lá dài hơn → thấp |
| Eccentricity | từ hình elip khớp nhất | Độ "dẹt" |
| Số răng cưa | số chỗ lõm vào sâu hơn 1% chiều dài lá (`convexityDefects`) | Tía tô, mùi tàu nhiều |
| Hu moments (7 số) | 7 công thức từ "mô men" của hình | Mô tả hình dạng, không đổi khi xoay/phóng to |
| Fourier descriptors (10 số) | Hàm `fourier_descriptors`: lấy 128 điểm đều nhau trên đường viền, coi mỗi điểm là số phức x + iy, biến đổi Fourier, lấy độ lớn các hệ số 2–11 chia cho hệ số 1 | Hệ số thấp = dáng tổng thể, hệ số cao = chi tiết mép lá |

*Bao lồi* là hình lồi nhỏ nhất bọc quanh lá – giống như căng một sợi dây thun quanh chiếc lá.

**d) `hog_features(gray)` – Hướng gradient (756 số)**
- Thu ảnh lá về 128×64, chia thành các ô 16×16 pixel.
- Trong mỗi ô, tính hướng thay đổi độ sáng (gradient) của từng pixel, gom thành histogram 9 hướng.
- Chuẩn hóa theo từng khối 2×2 ô → (8−1) × (4−1) khối × 4 ô × 9 hướng = 756 số.
- Mô tả "đường nét theo từng vùng": gân lá, mép lá ở đâu, chạy hướng nào.

**e) `orb_descriptors(gray)` + lớp `BoVW` – Điểm đặc trưng cục bộ (100 số)**
- **ORB** tìm tối đa 300 "điểm đáng chú ý" trên lá (góc cạnh, chỗ giao gân) và mô tả vùng quanh mỗi điểm bằng 256 bit.
- Vấn đề: mỗi ảnh có số điểm khác nhau, không ghép thành vector cố định được.
- **Bag of Visual Words (BoVW):** gom tất cả mô tả ORB của tập train thành 100 cụm bằng **k-means** – mỗi cụm là một "từ hình ảnh". Mỗi ảnh được biểu diễn bằng histogram: bao nhiêu điểm của nó thuộc từng "từ" → 100 số.
- Lớp `BoVW` có `fit()` (học 100 cụm) và `transform()` (biến danh sách điểm ORB của mỗi ảnh thành histogram). Từ điển chỉ được học trên tập train của mỗi fold.

**`extract_features(img, use_roi)`** gọi tất cả các bước trên cho một ảnh: tách lá (hoặc dùng cả ảnh nếu `use_roi=False`) → CLAHE → ảnh xám → 5 nhóm đặc trưng. Trả về một từ điển `{"color": …, "texture": …, "shape": …, "hog": …, "orb": …}`.

### 4.7. Biến đổi ảnh và tính đặc trưng cho cả bộ dữ liệu

**5 hàm biến đổi** (dùng cho tăng cường dữ liệu và kiểm tra độ bền):

| Hàm | Biến đổi | Tham số ngẫu nhiên |
|---|---|---|
| `aug_blur` | Làm mờ Gaussian | độ mờ σ từ 1,5 đến 3 |
| `aug_noise` | Cộng nhiễu ngẫu nhiên vào từng pixel | độ lệch chuẩn 10–20 |
| `aug_bright` | Làm sáng: nhân 1,0–1,15 rồi cộng 35–60 | |
| `aug_dark` | Làm tối: nhân 0,45–0,65 | |
| `aug_rotate` | Xoay quanh tâm; góc trống lấp bằng pixel viền (giữ nền giấy) | góc 15°–345° |

Chúng được gom vào từ điển `AUGMENTS = {"Làm mờ": aug_blur, …}`.

**`compute_features(images, paths, cache_path)`** – tính sẵn đặc trưng cho **mọi ảnh**, gồm:
- `"orig"`: ảnh gốc, ở 2 chế độ (`"roi"` = có tách lá, `"full"` = dùng cả ảnh);
- `"aug"`: 5 bản biến đổi của mỗi ảnh – dùng làm dữ liệu tăng cường khi ảnh đó thuộc tập train;
- `"pert"`: 5 bản biến đổi khác (tham số ngẫu nhiên khác) – dùng để kiểm tra khi ảnh đó thuộc tập test.

Tổng cộng 198 × 22 lần trích đặc trưng, mất ~8 phút. Kết quả được lưu vào `results/cache_dataset.joblib`. Lần chạy sau, nếu ảnh không đổi (so tên, kích thước, thời gian sửa file) thì đọc lại cache, bỏ qua bước này.

Hàm phụ `extract_set` chạy `extract_features` cho cả danh sách ảnh (có thể kèm một phép biến đổi).

### 4.8. Bước 7, 8, 10 – Kết hợp đặc trưng, chuẩn hóa, bộ phân lớp

- **`FEATURE_SETS`** (bước 7): 10 tổ hợp đặc trưng được thử, ví dụ `"Texture+Shape": ["texture", "shape"]` → ghép 33 + 24 = 57 số thành một vector.
- **`build_X(feats, groups, …)`**: ghép các nhóm đặc trưng thành ma trận X (mỗi hàng một ảnh). Dùng khi dự đoán ảnh mới.
- **`SCALERS`** (bước 8): `StandardScaler` (đưa mỗi đặc trưng về trung bình 0, độ lệch chuẩn 1) và `MinMaxScaler` (đưa về khoảng 0–1). Cần vì các đặc trưng có thang đo rất khác nhau (độ tròn 0–1, số răng cưa 0–20…), với KNN/SVM đặc trưng số lớn sẽ lấn át.
- **`CLASSIFIERS`** (bước 10): 6 thuật toán, mỗi thuật toán kèm danh sách giá trị siêu tham số sẽ thử.
- **`make_model(clf_name, params, scaler, use_pca)`**: tạo một **Pipeline** của scikit-learn gồm 3 khâu nối tiếp: chuẩn hóa → (PCA, nếu bật) → bộ phân lớp. Pipeline đảm bảo chuẩn hóa/PCA chỉ "học" trên tập train rồi áp dụng nguyên xi cho tập test.
- **`metrics(y_true, y_pred)`**: tính accuracy, precision, recall, F1.

### 4.9. Bước 9 – Cross-validation

| Hàm / lớp | Làm gì |
|---|---|
| `make_blocks(y)` | Gán mỗi ảnh vào khối 0–4: trong mỗi loại, ảnh thứ i (theo thứ tự tên file) thuộc khối `i × 5 // số_ảnh` |
| `outer_splits(y, g, random_split)` | Tạo 5 cặp (train, test). Mặc định theo khối; `random_split=True` thì chia ngẫu nhiên theo ảnh (chỉ dùng ở kịch bản 4 để so sánh) |
| `inner_splits(…)` | Chia tiếp tập train thành các cặp nhỏ hơn để chọn siêu tham số (vòng trong của nested CV) |
| lớp `Data` | Giữ toàn bộ đặc trưng, nhãn `y`, khối `g`. Hàm `X(mode, groups, rows, src)` dựng ma trận đặc trưng cho một tập ảnh bất kỳ (ảnh gốc / ảnh tăng cường / ảnh biến đổi). Tự học và **nhớ** từ điển BoVW cho từng tập train, để 6 bộ phân lớp dùng chung, khỏi tính lại |
| `fit_model(D, cfg, groups, params, tr, use_aug)` | Huấn luyện một mô hình trên tập train `tr`; nếu `use_aug` thì thêm 5 bản biến đổi của mỗi ảnh train (train lớn gấp 6) |
| `inner_tune(D, cfg, groups, tr)` | Thử từng bộ siêu tham số, chấm bằng CV bên trong `tr`, trả về bộ tốt nhất |
| `run_cv(D, cfg, groups, …)` | **Hàm đánh giá trung tâm.** Với mỗi fold: chọn siêu tham số (`inner_tune`) → huấn luyện (`fit_model`) → dự đoán khối test. Trả về F1 từng fold, trung bình ± độ lệch chuẩn, và **dự đoán out-of-fold** (dự đoán cho từng ảnh bởi mô hình không học khối chứa nó). Nếu `perturb=True` thì dự đoán thêm trên ảnh test bị biến đổi |
| `common_params(res)` | Bộ siêu tham số được chọn nhiều nhất qua 5 fold (để in ra) |
| `make_bundle(…)` | Gói mô hình + thông tin cần thiết thành một từ điển để lưu file |

Một "cấu hình" (`cfg`) là từ điển như:
```python
{"mode": "roi", "clf": "Random Forest", "scaler": "standard", "pca": False}
```

### 4.10. Mô hình được lưu thế nào (`model.joblib`)?

`model.joblib` chứa một từ điển:

| Khóa | Nội dung |
|---|---|
| `classes` | `["hung_que", "tia_to", "mui_tau", "la_lot"]` – mã số 0..3 ứng với loại nào |
| `names` | Tên có dấu tương ứng |
| `mode` | `"roi"` (có tách lá) hay `"full"` |
| `groups` | Các nhóm đặc trưng dùng, ví dụ `["texture", "shape"]` |
| `model` | Pipeline đã huấn luyện (chuẩn hóa + Random Forest) |
| `kmeans` | Từ điển BoVW (chỉ có nếu dùng ORB; mô hình hiện tại không dùng → `None`) |
| `config` | Toàn bộ cấu hình, kể cả siêu tham số và có tăng cường dữ liệu hay không |
| `feature_version` | Phiên bản cách tính đặc trưng – để cảnh báo nếu code đã đổi |

### 4.11. Bước 11 – `predict_image(img, bundle)`

Hàm dự đoán **một ảnh mới**, dùng chung cho `du_doan.py`, `demo_web.py` và bước 11 trong `main()`:

1. Thu nhỏ ảnh (`resize_long`).
2. `extract_features` – tách lá, CLAHE, tính đặc trưng **giống hệt lúc huấn luyện**.
3. Ghép các nhóm đặc trưng mà mô hình dùng (`build_X`).
4. `model.predict` → mã loại; `model.predict_proba` → độ tin cậy.
5. Trả về (tên loại, độ tin cậy, thông tin tách lá, ảnh lá).

Nguyên tắc quan trọng (slide 27): **ảnh mới phải đi qua đúng pipeline như ảnh huấn luyện** – nếu lúc học có tách lá và CLAHE mà lúc dự đoán thì không, kết quả sẽ sai.

### 4.12. Các hàm vẽ hình

| Hàm | Vẽ gì |
|---|---|
| `plot_counts` | Biểu đồ cột số ảnh mỗi loại |
| `plot_roi_demo` | Các bước tách lá cho một ảnh mỗi loại |
| `save_roi_montages` | Lưới tất cả lá đã tách (viền xanh/đỏ) vào `results/roi/` |
| `plot_feature_demo` | Minh họa 5 nhóm đặc trưng trên một lá |
| `plot_augment_demo` | Một ảnh qua 5 phép biến đổi |
| `plot_heatmap` | Bảng màu F1 của kịch bản 1 |
| `plot_bars` | Biểu đồ cột so sánh (kịch bản 4, 5) |
| `plot_confusion` | Ma trận nhầm lẫn |
| `plot_predictions` | Lưới ảnh kèm "thật / đoán" (xanh = đúng, đỏ = sai) |

### 4.13. Hàm `main()` – chạy 12 bước theo thứ tự

| Thứ tự trong `main()` | Làm gì | Sinh ra file |
|---|---|---|
| Bước 1 | In mô tả bài toán | – |
| Bước 2 | Liệt kê và đọc ảnh, kiểm tra số lượng | `buoc02_so_anh.png` |
| Bước 3 | Gán nhãn số, chia khối | `buoc03_nhan.csv` |
| Bước 4–5 | Minh họa tách lá; tính đặc trưng mọi ảnh (hoặc đọc cache); thống kê tách lá | `buoc04_05_…png`, `roi/`, `cache_dataset.joblib` |
| Bước 6–7 | In số chiều đặc trưng; vẽ minh họa | `buoc06_…png` |
| Bước 9 | In cách chia 5 fold | – |
| Kịch bản 1 (bước 10) | `run_cv` cho 10 tổ hợp đặc trưng × 6 thuật toán; chọn cặp tốt nhất | `kich_ban1_…` |
| Kịch bản 2 (bước 8) | Với cặp tốt nhất, thử 4 cách chuẩn hóa | `kich_ban2_chuan_hoa.csv` |
| Kịch bản 3 (bước 5) | Tách lá so với dùng cả ảnh | `kich_ban3_roi.csv` |
| Kịch bản 4 (bước 9) | Chia ngẫu nhiên so với chia theo khối, cho cả 6 thuật toán | `kich_ban4_…` |
| Kịch bản 5 | Có/không tăng cường dữ liệu, kiểm tra trên ảnh bị biến đổi; quyết định có dùng tăng cường không | `kich_ban5_…` |
| Bước 10 | Huấn luyện mô hình cuối trên **toàn bộ** dữ liệu, lưu file | `model.joblib` |
| Bước 11 | Học trên khối 2–5, dự đoán vài ảnh khối 1 bằng `predict_image` | `buoc11_du_doan.png` |
| Bước 12 | Đánh giá bằng dự đoán out-of-fold: báo cáo từng loại, ma trận nhầm lẫn, liệt kê ảnh sai | `buoc12_…png` |
| Tổng kết | In cấu hình tốt nhất, F1, thời gian chạy | `log.txt` |

**Quy tắc chọn "tốt nhất":** F1 trung bình cao nhất; nếu bằng nhau thì độ lệch chuẩn thấp hơn, rồi ít chiều đặc trưng hơn.

---

## 5. File `du_doan.py`

Dự đoán ảnh mới **bằng dòng lệnh**, không cần giao diện. File ngắn (~85 dòng):

1. Đọc tham số dòng lệnh: danh sách ảnh hoặc thư mục, tùy chọn `--model` (mặc định `results/model.joblib`), `--show`.
2. `joblib.load` đọc mô hình; cảnh báo nếu `feature_version` khác code hiện tại.
3. Hàm `collect()` gom danh sách file (nếu đưa thư mục thì lấy mọi ảnh bên trong, kể cả thư mục con).
4. Với mỗi ảnh: `imread_unicode` → `predict_image` → in `tên_file → Tía tô (95%)`.
5. Nếu có `--show`: vẽ khung quanh lá và nhãn lên ảnh, mở cửa sổ; bấm phím bất kỳ sang ảnh tiếp, `q` hoặc `Esc` để thoát.

Mọi xử lý thực sự đều nằm trong `nhan_dang_rau_thom.py` – `du_doan.py` chỉ *import* và gọi lại.

```
python du_doan.py anh_moi.jpg
python du_doan.py thu_muc_anh/ --show
```

---

## 6. File `demo_web.py`

Giao diện web chạy trên máy (localhost) để demo. Dùng **FastAPI** (thư viện tạo web server bằng Python) và **uvicorn** (chương trình chạy server).

### 6.1. Cách một trang web hoạt động (rất ngắn gọn)

- **Server** (chạy bằng Python trên laptop) chờ yêu cầu ở địa chỉ `http://127.0.0.1:8501`. `127.0.0.1` nghĩa là "chính máy này"; người khác trên mạng không truy cập được.
- **Trình duyệt** gửi yêu cầu, server trả về trang HTML (giao diện) hoặc dữ liệu JSON (kết quả).
- Code JavaScript trong trang nhận ảnh từ người dùng, gửi lên server, nhận kết quả và vẽ lên màn hình.

### 6.2. Các "địa chỉ" (endpoint) của server

| Địa chỉ | Kiểu | Làm gì |
|---|---|---|
| `/` | GET | Trả về trang HTML (biến `PAGE` – chứa cả HTML, CSS và JavaScript) |
| `/api/info` | GET | Thông tin mô hình (đặc trưng, thuật toán, F1, accuracy đọc từ `log.txt`) và danh sách phép biến đổi |
| `/api/samples` | GET | Danh sách ảnh mẫu (3 ảnh mỗi loại, lấy đều trong `dataset/`) |
| `/api/predict` | POST | **Nhận dạng.** Nhận ảnh (dạng base64) và tên phép biến đổi (nếu có) → trả về kết quả |
| `/results/…` | GET | Phục vụ các file hình trong `results/` (tab "Kết quả thực nghiệm") |
| `/dataset/…` | GET | Phục vụ ảnh trong `dataset/` (ảnh mẫu) |

### 6.3. Điều gì xảy ra khi bạn chọn một ảnh?

```
Trình duyệt                                       Server (demo_web.py)
───────────                                       ────────────────────
1. Người dùng chọn file / chụp webcam
2. JavaScript thu nhỏ ảnh về ≤1600px
   (ảnh điện thoại 50MP gửi lên rất chậm)
3. Đổi ảnh thành chuỗi base64, gửi
   POST /api/predict {image, transform}  ───────→  4. decode_image: base64 → ảnh OpenCV
                                                    5. Nếu có transform: áp phép biến đổi
                                                    6. extract_features + build_X
                                                    7. model.predict_proba → xác suất 4 loại
                                                    8. pipeline_steps: vẽ 6 ảnh minh họa
                                                       + số đo hình dạng
                                         ←───────   9. Trả JSON: tên loại, xác suất,
                                                       ảnh minh họa (base64), thời gian xử lý
10. Vẽ kết quả, thanh xác suất,
    6 ảnh các bước lên trang
```

### 6.4. Các hàm Python trong file

| Hàm | Làm gì |
|---|---|
| `load_bundle()` | Đọc `model.joblib` **một lần** rồi giữ trong bộ nhớ |
| `decode_image(data_url)` | Chuỗi base64 → ảnh OpenCV (giải mã 1/4 độ phân giải nếu ảnh lớn) |
| `to_data_url(img)` | Ảnh OpenCV → chuỗi base64 JPEG để gửi về trình duyệt |
| `pipeline_steps(img)` | Gọi `find_leaf`, CLAHE, LBP, vẽ contour → 6 ảnh minh họa; tính số đo hình dạng (tỉ lệ dài/rộng, độ tròn, solidity, số răng cưa) |
| `predict(payload)` | Xử lý `/api/predict` (các bước 4–9 ở sơ đồ trên) |
| `info()`, `samples()`, `index()` | Xử lý `/api/info`, `/api/samples`, `/` |

### 6.5. Phần giao diện (biến `PAGE`)

- **CSS** (trong thẻ `<style>`): màu sắc, bố cục 2 cột, tự chuyển 1 cột trên màn hình hẹp.
- **HTML**: header, 2 tab, khung ảnh đầu vào, nút biến đổi, ảnh mẫu, khung kết quả, khung "Các bước xử lý", tab kết quả thực nghiệm.
- **JavaScript** (trong thẻ `<script>`):
  - chuyển tab; gọi `/api/info` để hiện thông tin mô hình và tạo các nút biến đổi;
  - `useBlob()`: nhận file → thu nhỏ bằng `<canvas>` → `setImage()`;
  - webcam: `navigator.mediaDevices.getUserMedia` mở camera, nút "Chụp" vẽ khung hình hiện tại lên canvas;
  - `predict()`: gửi ảnh lên server và vẽ kết quả;
  - tham số `?anh=`: mở sẵn một ảnh, ví dụ `http://127.0.0.1:8501/?anh=/dataset/tia_to/tia_to_30.jpg`.

### 6.6. Khởi động

Phần `if __name__ == "__main__":` cuối file: kiểm tra có `model.joblib` chưa → tìm cổng trống bắt đầu từ 8501 (cổng 8000 thường bị app khác chiếm) → mở trình duyệt → chạy server. Thêm `--no-browser` để không tự mở trình duyệt.

---

## 7. Cách cài đặt và chạy

### 7.1. Cài đặt (một lần)

Cần Python 3.10 trở lên. Cài thư viện:

```
pip install opencv-python numpy scikit-image scikit-learn matplotlib joblib fastapi uvicorn
```

### 7.2. Huấn luyện lại từ đầu

```
cd Chuong3
python nhan_dang_rau_thom.py
```

- Lần đầu ~16 phút (8 phút tính đặc trưng + 8 phút các kịch bản), chạy trên CPU.
- Kết quả ghi vào `results/`, đọc `results/log.txt` trước.
- Tùy chọn: `--show` (mở các cửa sổ biểu đồ), `--no-cache` (tính lại đặc trưng), `--dataset <thư mục>`, `--results <thư mục>`.

### 7.3. Dự đoán ảnh mới

```
python du_doan.py duong_dan_anh.jpg --show
```

### 7.4. Chạy giao diện demo

```
python demo_web.py
```

Trình duyệt tự mở `http://127.0.0.1:8501`. Bấm `Ctrl + C` trong cửa sổ dòng lệnh để tắt.

### 7.5. Muốn thêm ảnh hoặc thêm loại rau?

- **Thêm ảnh:** chép vào `dataset/<loại>/`, đặt tên tiếp số thứ tự theo thứ tự chụp, rồi chạy lại `nhan_dang_rau_thom.py` (cache tự phát hiện ảnh mới).
- **Thêm loại mới:** tạo thư mục `dataset/<ten_khong_dau>/`, rồi thêm tên vào `CLASSES` và `CLASS_NAMES` ở đầu `nhan_dang_rau_thom.py`.
- **Cách chụp:** mỗi ảnh một lá, nền giấy trắng, lá không chạm mép ảnh, không có vật khác (ngón tay…) trong khung.

---

## 8. Câu hỏi thường gặp

**Vì sao chạy lâu vậy?**
Mỗi ảnh được tính đặc trưng 22 lần (ảnh gốc × 2 chế độ + 10 bản biến đổi × 2 chế độ), và có 60 tổ hợp đặc trưng × thuật toán, mỗi tổ hợp đánh giá bằng CV lồng 5 × 4 lần. Lần chạy sau nhanh hơn nhờ cache.

**Vì sao nhiều cấu hình đạt F1 = 1,000? Mô hình hoàn hảo à?**
Không. Dữ liệu chỉ có 4–5 lá mỗi loại, chụp cùng điều kiện, nên khá "dễ". Bằng chứng: mô hình cuối vẫn nhầm cả 10 ảnh của một chiếc lá tía tô tròn. Xem phân tích trong [BaoCao.md](BaoCao.md).

**Vì sao F1 báo cáo là 0,933 ± 0,133 mà accuracy là 0,949?**
0,933 là trung bình F1 của 5 fold (một fold chỉ đạt 0,667 vì chứa chiếc lá tía tô bị nhầm); 0,949 là tỉ lệ đúng khi gộp cả 198 ảnh. Hai cách tính khác nhau nên ra số khác nhau.

**Mô hình cuối có dùng màu không?**
Không – chỉ Texture + Shape, vì đây là tổ hợp ít chiều nhất trong số các tổ hợp đạt điểm cao nhất. Báo cáo ghi nhận đây là điểm có thể cải thiện (dùng màu để nhận ra mặt tím của tía tô).

**Ảnh của tôi bị báo "không tách được lá"?**
Đảm bảo: nền sáng và đồng màu, lá không chạm mép ảnh, chỉ có một lá, không có vật khác. Chương trình lấy màu viền ảnh làm màu nền.

**Cache có cần đưa lên GitHub không?**
Không (đã ghi trong `.gitignore`). Ai clone về chạy lần đầu sẽ tự tạo lại.

**Đổi code tính đặc trưng thì cần làm gì?**
Tăng `FEATURE_VERSION` ở đầu `nhan_dang_rau_thom.py` để cache cũ bị bỏ, rồi chạy lại để có `model.joblib` mới (mô hình cũ không tương thích với đặc trưng mới).

---

## 9. Bảng thuật ngữ

| Thuật ngữ | Nghĩa |
|---|---|
| Pixel | Một điểm ảnh, gồm 3 số màu |
| BGR / HSV / Lab | Các cách biểu diễn màu. BGR: xanh dương–xanh lá–đỏ. HSV: sắc màu–độ đậm–độ sáng. Lab: độ sáng + 2 kênh màu |
| ROI (Region of Interest) | Vùng quan tâm – ở đây là chiếc lá |
| Mask | Ảnh trắng đen đánh dấu pixel nào thuộc lá (trắng) |
| Contour | Đường viền của vùng trắng trong mask |
| Bao lồi (convex hull) | Hình lồi nhỏ nhất bọc quanh đối tượng |
| Ngưỡng Otsu | Phương pháp tự tìm ngưỡng tách ảnh thành 2 nhóm |
| Morphology (open/close) | Phép xử lý hình thái: xóa chấm nhỏ / lấp khe nhỏ trên mask |
| CLAHE | Cân bằng histogram thích nghi cục bộ – cân bằng độ sáng từng vùng |
| PCA | Phân tích thành phần chính – tìm hướng dữ liệu trải rộng nhất; dùng để xoay lá (bước 5) và giảm chiều (bước 8) |
| Histogram | Biểu đồ đếm số lượng theo từng khoảng giá trị |
| LBP, GLCM | Hai cách đo kết cấu bề mặt |
| HOG | Histogram hướng gradient – đo hướng đường nét |
| ORB | Thuật toán tìm và mô tả điểm đặc trưng |
| k-means | Thuật toán gom dữ liệu thành k cụm |
| BoVW | Bag of Visual Words – biểu diễn ảnh bằng histogram các "từ hình ảnh" |
| Vector đặc trưng | Dãy số mô tả một ảnh |
| Classifier | Bộ phân lớp – thuật toán đoán loại từ vector đặc trưng |
| Siêu tham số | Thông số phải chọn trước khi huấn luyện (vd. K trong KNN) |
| Pipeline | Chuỗi các bước xử lý nối tiếp (chuẩn hóa → PCA → phân lớp) |
| Train / Validation / Test | Tập học / tập chọn tham số / tập chấm điểm |
| Cross-validation (k-fold) | Chia dữ liệu thành k phần, lần lượt mỗi phần làm test |
| Group CV | Cross-validation mà ảnh cùng nhóm (cùng lá) luôn nằm chung một phía |
| Nested CV | CV lồng: chọn tham số bằng CV bên trong tập train |
| Out-of-fold | Dự đoán cho một ảnh bởi mô hình không học ảnh đó |
| Overfitting (quá khớp) | Mô hình "học thuộc" dữ liệu huấn luyện, kém với dữ liệu mới |
| Data leakage (rò rỉ dữ liệu) | Thông tin tập test lọt vào lúc huấn luyện → điểm cao giả |
| Data augmentation | Tăng cường dữ liệu bằng biến đổi ảnh |
| Accuracy / Precision / Recall / F1 | Các thước đo độ chính xác (xem [mục 3.5](#35-đo-độ-chính-xác)) |
| Confusion matrix | Ma trận nhầm lẫn |
| base64 | Cách viết dữ liệu nhị phân (ảnh) thành chuỗi ký tự để gửi qua web |
| Endpoint / API | "Địa chỉ" trên server mà trình duyệt gọi để lấy dữ liệu |
| localhost / 127.0.0.1 | Chính máy tính đang dùng |
