import cv2
import os
import numpy as np
import torch
from license_plate_cha import getChar
from detect import detect
from models.experimental import attempt_load
from utils_LP import crop_n_rotate_LP

# ====== Cài đặt ban đầu ======
save_dir = "captured_images"
os.makedirs(save_dir, exist_ok=True)

LP_weights = 'best_yolo7.pt'

# Kiểm tra thiết bị
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load mô hình YOLOv7 phát hiện biển số
model_LP = attempt_load(LP_weights, map_location=device)

# ====== Mở webcam ======
cap = cv2.VideoCapture(0)  # Camera thứ hai

# Cài đặt thông số camera
desired_width = 1280   # Độ phân giải HD
desired_height = 720
desired_fps = 30

# Set properties với error checking
cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)
cap.set(cv2.CAP_PROP_FPS, desired_fps)

if not cap.isOpened():
    print("Không thể mở camera!")
    exit()

image_count = 1
print("Nhấn 'c' để chụp và nhận diện biển số. Nhấn 'q' để thoát.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Không thể đọc frame từ camera.")
        break

    # Hiển thị preview
    preview = cv2.resize(frame, (640, 360))
    cv2.imshow("Camera Preview", preview)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        # ====== Lưu ảnh gốc ======
        filename = f"image_{image_count:03d}.jpg"
        filepath = os.path.join(save_dir, filename)
        cv2.imwrite(filepath, frame)
        print(f"Đã lưu ảnh: {filepath}")

        # ====== Xử lý nhận dạng biển số ======
        source_img = frame.copy()
        height, width, _ = source_img.shape
        Min_char = 0.001 * (height * width)
        Max_char = 0.1 * (height * width)

        pred, LP_detected_img = detect(model_LP, source_img, device, imgsz=640)

        # cv2.imshow('Input', cv2.resize(source_img, dsize=None, fx=0.5, fy=0.5))
        # cv2.imshow('LP Detected', cv2.resize(LP_detected_img, dsize=None, fx=0.5, fy=0.5))

        c = 0
        for *xyxy, conf, cls in reversed(pred):
            x1, y1, x2, y2 = map(int, xyxy)
            angle, rotate_thresh, LP_rotated = crop_n_rotate_LP(source_img, x1, y1, x2, y2)
            if rotate_thresh is None or LP_rotated is None:
                continue

            # cv2.imshow('LP Rotated', LP_rotated)
            save_path = f'captured_images/hehe.jpg'
            cv2.imwrite(save_path, LP_rotated)
            print(f"Đã lưu ảnh biển số: {save_path}")
            print(getChar(save_path))
            

    elif key == ord('q'):
        print("Thoát chương trình.")
        break

cap.release()
cv2.destroyAllWindows()
