import av
import queue
import time
import streamlit as st
import streamlit.components.v1 as components # <--- QUAN TRỌNG: Thư viện này giúp né lỗi
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from ultralytics import YOLO

# --- CẤU HÌNH ---
st.set_page_config(page_title="Nhận diện Biển báo", page_icon="🚦", layout="centered")
st.title("🚦 AI Biển báo - Fix iOS")

# 1. Hàng đợi gửi tin nhắn
result_queue = queue.Queue()

# 2. Load Model
@st.cache_resource
def load_model():
    return YOLO('best.pt')

try:
    model = load_model()
except Exception as e:
    st.error(f"Lỗi model: {e}")
    st.stop()

# 3. Từ điển lời thoại
CLASS_MESSAGES = {
    "khu_vuc_dong_dan_cu": "Khu vực đông dân cư",
    "het_khu_vuc_dong_dan_cu": "Hết khu vực đông dân cư",
    "cam_quay_dau": "Cấm quay đầu",
    "cam_di_nguoc_chieu": "Đi ngược chiều",
    "gioi_han_toc_do_50": "Tốc độ 50",
    "gioi_han_toc_do_60": "Tốc độ 60",
    "cam_vuot": "Cấm vượt",
    # Thêm các lớp khác...
}

# 4. Xử lý AI
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    results = model.predict(img, conf=0.5, verbose=False)
    
    found_labels = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            name = model.names[cls_id]
            if name in CLASS_MESSAGES:
                found_labels.append(CLASS_MESSAGES[name])
    
    if found_labels:
        try:
            result_queue.put_nowait(found_labels[0])
        except queue.Full:
            pass

    annotated_frame = results[0].plot()
    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# --- GIAO DIỆN CHÍNH ---

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:global.stun.twilio.com:3478"]}
    ]}
)

camera_type = st.radio("Chọn thiết bị:", ("Laptop", "Điện thoại (Cam sau)"), horizontal=True)
if "Điện thoại" in camera_type:
    video_constraints = {"facingMode": "environment"}
else:
    video_constraints = {"facingMode": "user"}

ctx = webrtc_streamer(
    key="mobile-fix-v2",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": video_constraints, "audio": False},
    video_frame_callback=video_frame_callback,
    async_processing=True,
)

# --- PHẦN JS (ĐÃ SỬA ĐỂ KHÔNG BỊ LỖI TRÊN IPHONE CŨ) ---

status_placeholder = st.empty()
js_placeholder = st.empty()

# Nút kích hoạt (Dùng components.html để tránh lỗi Regex)
if st.button("🔊 KÍCH HOẠT LOA (Bấm 1 lần)"):
    components.html("""
    <script>
        window.speechSynthesis.cancel(); // Dừng các âm thanh cũ
        var msg = new SpeechSynthesisUtterance("Đã kích hoạt");
        msg.lang = 'vi-VN';
        window.speechSynthesis.speak(msg);
    </script>
    """, height=0, width=0)

if ctx.state.playing:
    while True:
        try:
            text_to_speak = result_queue.get(timeout=0.1)
            status_placeholder.warning(f"⚠️ Phát hiện: {text_to_speak}")
            
            # --- ĐÂY LÀ CHỖ SỬA QUAN TRỌNG ---
            # Dùng components.html thay vì st.write
            # Nó giúp bypass bộ lọc MathJax gây lỗi trên iOS cũ
            with js_placeholder:
                components.html(f"""
                    <script>
                        window.speechSynthesis.cancel(); 
                        var msg = new SpeechSynthesisUtterance("{text_to_speak}");
                        msg.lang = 'vi-VN';
                        msg.rate = 1.1;
                        window.speechSynthesis.speak(msg);
                    </script>
                """, height=0, width=0)
            
            time.sleep(2.5) # Đợi nói xong mới nhận tiếp
            js_placeholder.empty()
            
        except queue.Empty:
            time.sleep(0.1)