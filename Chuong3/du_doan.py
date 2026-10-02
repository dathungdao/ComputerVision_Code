import os
import sys
import glob
import argparse

import cv2
import numpy as np
import joblib

from nhan_dang_rau_thom import (predict_image, imread_unicode, resize_long,
                                RESULTS_DIR, IMG_EXTS, FEATURE_VERSION)

# ============================================================
# BƯỚC 11 - DỰ ĐOÁN ẢNH MỚI BẰNG MODEL ĐÃ HUẤN LUYỆN
#
#   python du_doan.py anh.jpg
#   python du_doan.py thu_muc_anh/ --show
#
# Model (results/model.joblib) do nhan_dang_rau_thom.py tạo ra.
# Ảnh mới đi qua đúng pipeline lúc huấn luyện:
#   resize → tách lá, xoay → CLAHE → đặc trưng → chuẩn hóa → classifier
# ============================================================


def collect(inputs):
    files = []
    for p in inputs:
        if os.path.isdir(p):
            files += sorted(f for f in glob.glob(os.path.join(p, "**", "*"),
                                                 recursive=True)
                            if f.lower().endswith(IMG_EXTS))
        else:
            files.append(p)
    return files


def main():
    ap = argparse.ArgumentParser(description="Dự đoán loại rau thơm từ ảnh lá")
    ap.add_argument("inputs", nargs="+", help="ảnh hoặc thư mục ảnh")
    ap.add_argument("--model", default=os.path.join(RESULTS_DIR, "model.joblib"))
    ap.add_argument("--show", action="store_true", help="hiện ảnh kèm kết quả")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

    if not os.path.exists(args.model):
        print(f"Chưa có model: {args.model}\nHãy chạy nhan_dang_rau_thom.py trước.")
        return
    bundle = joblib.load(args.model)
    if bundle.get("feature_version") != FEATURE_VERSION:
        print("[!] Model được huấn luyện với phiên bản đặc trưng khác → nên huấn luyện lại.")
    print(f"Model: {bundle['config']}\n")

    for f in collect(args.inputs):
        img = imread_unicode(f)
        if img is None:
            print(f"{f}: không đọc được ảnh")
            continue

        label, conf, info, _ = predict_image(img, bundle)
        name = dict(zip(bundle["classes"], bundle["names"]))[label]
        c = f" ({conf:.0%})" if conf is not None else ""
        print(f"{os.path.basename(f):32s} → {name}{c}   [{info['method']}]")

        if args.show:
            vis = resize_long(img)
            if info["box"] is not None:
                cv2.drawContours(vis, [info["box"].astype(np.int32)], -1,
                                 (0, 255, 0), 3)
            cv2.putText(vis, f"{label}{c}", (15, 45), cv2.FONT_HERSHEY_SIMPLEX,
                        1.4, (0, 0, 255), 3)
            cv2.imshow("Du doan", vis)
            if cv2.waitKey(0) & 0xFF in (27, ord("q")):
                break

    if args.show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
