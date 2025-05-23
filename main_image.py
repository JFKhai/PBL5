import cv2
import numpy as np
import torch
from license_plate_cha import getChar
from detect import detect
from models.experimental import attempt_load
from src.char_classification.model import CNN_Model
from utils_LP import character_recog_CNN, crop_n_rotate_LP

# Điều chỉnh tham số nhận dạng ký tự
Min_char = 0.008  # Giảm xuống để bắt được ký tự nhỏ hơn
Max_char = 0.07   # Giảm xuống để loại bỏ nhiễu lớn
image_path = 'data/test/images/image_006.jpg'
# image_path = 'Camera_Roll/WIN_20250511_17_46_20_Pro.jpg'
CHAR_CLASSIFICATION_WEIGHTS = './src/weights/weight.h5'
# LP_weights = 'LP_detect_yolov7_500img.pt'
LP_weights = 'best_yolo7.pt'
#LP_weights = 'best.pt'

model_char = CNN_Model(trainable=False).model
model_char.load_weights(CHAR_CLASSIFICATION_WEIGHTS)

def auto_scale(height, width):
    global Min_char, Max_char
    Min_char = 0.001 * (height * width)
    Max_char = 0.1 * (height * width)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model_LP = attempt_load(LP_weights, map_location=device)

source_img = cv2.imread(image_path)
pred, LP_detected_img = detect(model_LP, source_img, device, imgsz=640)

cv2.imshow('input', cv2.resize(source_img, dsize=None, fx=0.5, fy=0.5))
cv2.imshow('LP_detected_img', cv2.resize(LP_detected_img, dsize=None, fx=0.5, fy=0.5))


c = 0
for *xyxy, conf, cls in reversed(pred):
    x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
    angle, rotate_thresh, LP_rotated = crop_n_rotate_LP(source_img, x1, y1, x2, y2)
    if (rotate_thresh is None) or (LP_rotated is None):
        continue
    cv2.imshow('LP_rotated', LP_rotated)
    
    # Lưu ảnh biển số đã cắt và xoay
    save_path = f'captured_images/hehe.jpg'
    cv2.imwrite(save_path, LP_rotated)
    print(f"Đã lưu ảnh biển số: {save_path}")
    print(getChar(save_path))
#     cv2.imshow('rotate_thresh', rotate_thresh)

#     #################### Prepocessing and Character segmentation ####################
#     LP_rotated_copy = LP_rotated.copy()
#     cont, hier = cv2.findContours(rotate_thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
#     cont = sorted(cont, key=cv2.contourArea, reverse=True)[:17]

#     cv2.imshow('rotate_thresh', rotate_thresh)
#     cv2.drawContours(LP_rotated_copy, cont, -1, (100, 255, 255), 2)  # Draw contours of characters in a LP
#     # cv2.imshow('rotate_img',rotate_img)

#     ##################### Filter out characters #################
#     char_x_ind = {}
#     char_x = []
#     height, width, _ = LP_rotated_copy.shape
#     roiarea = height * width

#     for ind, cnt in enumerate(cont):
#         (x, y, w, h) = cv2.boundingRect(cont[ind])
#         ratiochar = w / h
#         char_area = w * h
#         # Điều chỉnh tỷ lệ ký tự chặt chẽ hơn
#         if (Min_char * roiarea < char_area < Max_char * roiarea) and (0.2 < ratiochar < 0.8):
#             char_x.append([x, y, w, h])

#     if not char_x:
#         continue

#     char_x = np.array(char_x)
#     cv2.imshow('LP_rotated_copy', LP_rotated_copy)

#     ############ Character recognition ##########################

#     threshold_12line = char_x[:, 1].min() + (char_x[:, 3].mean() / 2)
#     char_x = sorted(char_x, key=lambda x: x[0], reverse=False)
#     strFinalString = ""
#     first_line = ""
#     second_line = ""

#     for i, char in enumerate(char_x):
#         x, y, w, h = char
#         cv2.rectangle(LP_rotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
#         imgROI = rotate_thresh[y:y + h, x:x + w]
#         cv2.imshow('imgROI', imgROI)
#         text = character_recog_CNN(model_char, imgROI)
#         if text == 'Background':
#             text = ''

#         if y < threshold_12line:
#             first_line += text
#         else:
#             second_line += text

#     strFinalString = first_line + second_line
#     cv2.putText(LP_detected_img, strFinalString, (x1, y1 - 20), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 0), 2)
#     cv2.imshow('charac', LP_rotated_copy)
#     cv2.imshow('LP_rotated_{}'.format(c), LP_rotated)
#     print('License Plate_{}:'.format(c), strFinalString)
#     c += 1

# cv2.imshow('final_result', cv2.resize(LP_detected_img, dsize=None, fx=0.5, fy=0.5))
print('Finally Done!')
cv2.waitKey(0)
