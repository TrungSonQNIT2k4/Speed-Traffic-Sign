import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from ultralytics import YOLO

# --- CẤU HÌNH ---
st.set_page_config(page_title="Nhận diện Biển báo", page_icon="🚦", layout="centered")

st.markdown("""
    <h1 style='text-align: center; color: #FF4B4B;'>🚦 AI Biển Báo Giao Thông</h1>
    <p style='text-align: center;'>Phiên bản Ổn định (HUD Mode) - Hỗ trợ mọi thiết bị</p>
    """, unsafe_allow_html=True)

# 1. Load Model (Cache để không load lại nhiều lần)
@st.cache_resource
def load_model():
    return YOLO('best.pt')

try:
    model = load_model()
except Exception as e:
    st.error(f"❌ Không tìm thấy model: {e}")
    st.stop()

# 2. Từ điển Cảnh báo (Nội dung sẽ hiện lên màn hình)
CLASS_MESSAGES = {
    "khu_vuc_dong_dan_cu": "KHU DONG DAN CU",
    "het_khu_vuc_dong_dan_cu": "HET KHU DONG DAN CU",
    "cam_quay_dau": "CAM QUAY DAU XE",
    "cam_di_nguoc_chieu": "NGUY HIEM! DI NGUOC CHIEU",
    "gioi_han_toc_do_50": "GIOI HAN TOC DO: 50KM/H",
    "gioi_han_toc_do_60": "GIOI HAN TOC DO: 60KM/H",
    "gioi_han_toc_do_80": "GIOI HAN TOC DO: 80KM/H",
    "cam_vuot": "CAM VUOT",
    # Bạn thêm các class khác vào đây (Viết không dấu cho an toàn font chữ)
}

# 3. Hàm vẽ Tiếng Việt/Cảnh báo lên khung hình (HUD)
def draw_warning(image, text):
    # Lấy kích thước ảnh
    h, w, _ = image.shape
    
    # Cấu hình khung cảnh báo (Màu đỏ, nền vàng)
    font_scale = 1.0 if w > 500 else 0.6 # Tự chỉnh cỡ chữ theo màn hình
    thickness = 2
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    
    # Vẽ hình chữ nhật nền (Background) ở dưới đáy ảnh
    cv2.rectangle(image, (0, h - 60), (w, h), (0, 0, 0), -1) # Nền đen
    cv2.rectangle(image, (0, h - 60), (w, h), (0, 255, 255), 2) # Viền vàng
    
    # Căn giữa chữ
    x_pos = (w - text_w) // 2
    y_pos = h - 20
    
    # Vẽ chữ cảnh báo
    cv2.putText(image, text, (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                font_scale, (0, 255, 255), thickness, cv2.LINE_AA)

# 4. Xử lý từng khung hình
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    
    # Xử lý nhận diện
    results = model.predict(img, conf=0.5, verbose=False)
    
    current_warning = ""
    
    # Vẽ khung bounding box của YOLO
    for r in results:
        img = r.plot() # Vẽ sẵn khung YOLO
        
        # Kiểm tra xem có biển báo nào cần cảnh báo không
        for box in r.boxes:
            cls_id = int(box.cls[0])
            name = model.names[cls_id]
            if name in CLASS_MESSAGES:
                current_warning = CLASS_MESSAGES[name]
                # Chỉ lấy biển báo đầu tiên thấy được để cảnh báo
                break 
    
    # Nếu có cảnh báo -> Vẽ đè lên video (Hiệu ứng HUD)
    if current_warning:
        draw_warning(img, current_warning)
        
    return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- GIAO DIỆN CHÍNH ---

# Cấu hình Server (Twilio + Google để xuyên tường lửa)
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:global.stun.twilio.com:3478"]}
    ]}
)

st.info("💡 Hướng dẫn: Mở bằng Chrome/Safari. Chọn thiết bị bên dưới và bấm START.")

# Chọn Camera
camera_type = st.radio("Chọn thiết bị:", ("Laptop", "Điện thoại (Cam sau)"), horizontal=True)

if "Điện thoại" in camera_type:
    # Bỏ 'exact', chỉ cần environment
    video_constraints = {"facingMode": "environment"}
else:
    video_constraints = {"facingMode": "user"}

# Khởi chạy WebRTC
webrtc_streamer(
    key="traffic-hud-stable", # Đổi key mới để reset worker cũ bị lỗi
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": video_constraints, "audio": False},
    video_frame_callback=video_frame_callback,
    async_processing=True,
)