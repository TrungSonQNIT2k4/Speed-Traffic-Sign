import av
import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from ultralytics import YOLO

# 1. Cấu hình giao diện Web
st.set_page_config(page_title="Nhận diện Biển báo", page_icon="🚦", layout="centered")
st.title("🚦 AI Nhận diện Biển báo Giao thông")
st.write("Hệ thống nhận diện biển báo Real-time (Hỗ trợ PC & Mobile)")

# 2. Tải Model (Cache để load nhanh hơn)
@st.cache_resource
def load_model():
    # Đảm bảo file best.pt nằm cùng thư mục với file app.py này
    return YOLO('best.pt')

try:
    model = load_model()
except Exception as e:
    st.error(f"❌ Lỗi không tìm thấy file model: {e}")
    st.stop()

# 3. Cấu hình WebRTC (Để chạy mượt trên mạng Internet)
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# 4. Giao diện chọn Camera
st.write("### 📸 Cấu hình Camera")
camera_type = st.radio(
    "Bạn đang dùng thiết bị gì?",
    ("Laptop/PC (Webcam Trước)", "Điện thoại (Camera Sau)"),
    horizontal=True
)

# Thiết lập tham số facingMode
# - 'user': Camera trước (Selfie/Webcam)
# - 'environment': Camera sau (Quay cảnh vật)
if camera_type == "Điện thoại (Camera Sau)":
    video_constraints = {"facingMode": {"exact": "environment"}}
else:
    video_constraints = {"facingMode": "user"}

# 5. Hàm xử lý từng khung hình (Core AI)
def video_frame_callback(frame):
    # Chuyển ảnh từ WebRTC sang định dạng OpenCV
    img = frame.to_ndarray(format="bgr24")

    # --- XỬ LÝ AI ---
    # Chạy YOLO với ngưỡng tự tin 0.45
    results = model.predict(img, conf=0.45, verbose=False)
    
    # Vẽ kết quả lên ảnh
    annotated_frame = results[0].plot()
    # ----------------

    # Trả ảnh về lại Web
    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# 6. Hiển thị màn hình Camera
st.write("---")
st.info("💡 Hướng dẫn: Bấm 'START' và chọn 'Allow' (Cho phép) để cấp quyền Camera.")

webrtc_streamer(
    key="traffic-sign-app",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": video_constraints, "audio": False},
    video_frame_callback=video_frame_callback,
    async_processing=True,
)