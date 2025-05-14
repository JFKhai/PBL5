from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import torch
import base64
from detect import detect
from models.experimental import attempt_load
from src.char_classification.model import CNN_Model
from utils_LP import character_recog_CNN, crop_n_rotate_LP

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Khởi tạo các model và thiết bị
CHAR_CLASSIFICATION_WEIGHTS = './src/weights/weight.h5'
LP_weights = 'best_yolo7.pt'

model_char = CNN_Model(trainable=False).model
model_char.load_weights(CHAR_CLASSIFICATION_WEIGHTS)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_LP = attempt_load(LP_weights, map_location=device)

def process_frame(frame):
    """Xử lý một frame và trả về kết quả nhận dạng"""
    Min_char = 0.001 * (frame.shape[0] * frame.shape[1])
    Max_char = 0.1 * (frame.shape[0] * frame.shape[1])
    
    pred, LP_detected_img = detect(model_LP, frame, device, imgsz=640)
    results = []
    
    for *xyxy, conf, cls in reversed(pred):
        x1, y1, x2, y2 = map(int, xyxy)
        angle, rotate_thresh, LP_rotated = crop_n_rotate_LP(frame, x1, y1, x2, y2)
        if rotate_thresh is None or LP_rotated is None:
            continue
            
        # Xử lý nhận dạng ký tự
        cont, _ = cv2.findContours(rotate_thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        cont = sorted(cont, key=cv2.contourArea, reverse=True)[:17]
        
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
        
        first_line = ""
        second_line = ""
        
        for char in char_x:
            x, y, w, h = char
            imgROI = rotate_thresh[y:y + h, x:x + w]
            text = character_recog_CNN(model_char, imgROI)
            if text == 'Background':
                continue
                
            if y < threshold_12line:
                first_line += text
            else:
                second_line += text
                
        plate_text = first_line + second_line
        results.append({
            'plate_number': plate_text,
            'confidence': float(conf),
            'bbox': [int(x) for x in [x1, y1, x2, y2]]
        })
        
    return results, LP_detected_img

@app.route('/detect_plate', methods=['POST'])
def detect_plate():
    """API endpoint để nhận dạng biển số từ frame video"""
    if 'frame' not in request.json:
        return jsonify({'error': 'No frame data provided'}), 400
        
    # Decode base64 frame
    frame_data = base64.b64decode(request.json['frame'])
    frame_arr = np.frombuffer(frame_data, dtype=np.uint8)
    frame = cv2.imdecode(frame_arr, cv2.IMREAD_COLOR)
    
    # Xử lý frame
    results, detected_img = process_frame(frame)
    
    # Encode kết quả
    _, buffer = cv2.imencode('.jpg', detected_img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return jsonify({
        'results': results,
        'image': img_base64
    })

@app.route('/health', methods=['GET'])
def health_check():
    """API endpoint để kiểm tra server hoạt động"""
    return jsonify({'status': 'healthy', 'message': 'Server is running'})

@app.route('/process_video', methods=['POST'])
def process_video():
    """API endpoint để xử lý video stream"""
    try:
        if 'frame' not in request.json:
            return jsonify({'error': 'No frame data provided'}), 400

        # Decode frame từ base64
        frame_data = request.json['frame']
        if ';base64,' in frame_data:
            frame_data = frame_data.split(';base64,')[1]
        
        frame_bytes = base64.b64decode(frame_data)
        frame_arr = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(frame_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({'error': 'Invalid frame data'}), 400

        # Xử lý frame
        results, detected_img = process_frame(frame)

        # Encode kết quả thành base64
        _, buffer = cv2.imencode('.jpg', detected_img)
        img_base64 = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"

        return jsonify({
            'status': 'success',
            'plates': results,
            'processed_image': img_base64
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Server starting...")
    print("Available endpoints:")
    print("  - POST /detect_plate : Nhận dạng biển số từ một frame")
    print("  - POST /process_video : Xử lý video stream")
    print("  - GET /health : Kiểm tra server")
    app.run(host='0.0.0.0', port=5000, debug=True)
