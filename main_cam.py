import cv2
import os
import numpy as np
import torch

from detect import detect
from models.experimental import attempt_load
from src.char_classification.model import CNN_Model
from utils_LP import character_recog_CNN, crop_n_rotate_LP

# ====== Cài đặt ban đầu ======
save_dir = "captured_images"
os.makedirs(save_dir, exist_ok=True)

CHAR_CLASSIFICATION_WEIGHTS = './src/weights/weight.h5'
LP_weights = 'best_yolo7.pt'

# Load mô hình nhận diện ký tự
model_char = CNN_Model(trainable=False).model
model_char.load_weights(CHAR_CLASSIFICATION_WEIGHTS)

# Kiểm tra thiết bị
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load mô hình YOLOv7 phát hiện biển số
model_LP = attempt_load(LP_weights, map_location=device)

# ====== Mở webcam ======
cap = cv2.VideoCapture(0)  # Camera mặc định

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

        cv2.imshow('Input', cv2.resize(source_img, dsize=None, fx=0.5, fy=0.5))
        cv2.imshow('LP Detected', cv2.resize(LP_detected_img, dsize=None, fx=0.5, fy=0.5))

        c = 0
        for *xyxy, conf, cls in reversed(pred):
            x1, y1, x2, y2 = map(int, xyxy)
            angle, rotate_thresh, LP_rotated = crop_n_rotate_LP(source_img, x1, y1, x2, y2)
            if rotate_thresh is None or LP_rotated is None:
                continue

            cv2.imshow('LP Rotated', LP_rotated)
            cv2.imshow('Threshold', rotate_thresh)

            # Tìm các contour của ký tự
            LP_rotated_copy = LP_rotated.copy()
            cont, _ = cv2.findContours(rotate_thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            cont = sorted(cont, key=cv2.contourArea, reverse=True)[:17]
            cv2.drawContours(LP_rotated_copy, cont, -1, (100, 255, 255), 2)

            # Lọc và sắp xếp các ký tự
            char_x = []
            roiarea = LP_rotated.shape[0] * LP_rotated.shape[1]
            
            for cnt in cont:
                x, y, w, h = cv2.boundingRect(cnt)
                ratiochar = w / h
                char_area = w * h
                if (Min_char < char_area < Max_char) and (0.25 < ratiochar < 0.7):
                    char_x.append([x, y, w, h])

            if not char_x:
                continue

            char_x = np.array(char_x)
            threshold_12line = char_x[:, 1].min() + (char_x[:, 3].mean() / 2)
            char_x = sorted(char_x, key=lambda x: x[0])

            # Nhận dạng từng ký tự
            first_line = ""
            second_line = ""
            
            for i, char in enumerate(char_x):
                x, y, w, h = char
                cv2.rectangle(LP_rotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
                imgROI = rotate_thresh[y:y + h, x:x + w]
                cv2.imshow('Character ROI', imgROI)
                
                text = character_recog_CNN(model_char, imgROI)
                if text == 'Background':
                    text = ''
                
                if y < threshold_12line:
                    first_line += text
                else:
                    second_line += text

            plate_text = first_line + second_line
            print(f"Biển số xe {c + 1}: {plate_text}")
            cv2.putText(LP_detected_img, plate_text, (x1, y1 - 20), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 0), 2)
            cv2.imshow('Characters', LP_rotated_copy)
            cv2.imshow(f'License Plate {c + 1}', LP_rotated)
            c += 1

        cv2.imshow('Final Result', cv2.resize(LP_detected_img, dsize=None, fx=0.5, fy=0.5))
        image_count += 1

    elif key == ord('q'):
        print("Thoát chương trình.")
        break

cap.release()
cv2.destroyAllWindows()
