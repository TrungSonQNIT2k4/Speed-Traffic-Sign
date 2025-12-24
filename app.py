import av
import queue
import threading
import time
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from ultralytics import YOLO

# --- CẤU HÌNH ---
st.set_page_config(page_title="Nhận diện Biển báo (Mobile Voice)", page_icon="🚦")
st.title("🚦 AI Biển báo - Giọng nói Mobile")

# 1. Hàng đợi (Queue) để gửi tin nhắn từ AI (Thread phụ) sang Web (Thread chính)
# Đây là cầu nối quan trọng nhất!
result_queue = queue.Queue()

# 2. Cấu hình Model
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
    "cam_di_nguoc_chieu": "Nguy hiểm, đi ngược chiều",
    "gioi_han_toc_do_50": "Tốc độ 50",
    "gioi_han_toc_do_60": "Tốc độ 60",
    "cam_vuot": "Cấm vượt",
    # Thêm các lớp khác...
}

# 4. Hàm xử lý AI (Chạy ngầm)
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    
    # Nhận diện
    results = model.predict(img, conf=0.5, verbose=False)
    
    # Lấy kết quả gửi ra ngoài
    found_labels = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            name = model.names[cls_id]
            if name in CLASS_MESSAGES:
                found_labels.append(CLASS_MESSAGES[name])
    
    # Nếu thấy biển báo, gửi vào hàng đợi (chỉ lấy cái đầu tiên để đỡ spam)
    if found_labels:
        # Gửi tin nhắn về cho giao diện chính
        # Dùng `put_nowait` để không làm đơ video
        try:
            result_queue.put_nowait(found_labels[0])
        except queue.Full:
            pass

    annotated_frame = results[0].plot()
    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# --- GIAO DIỆN CHÍNH ---

# Cấu hình WebRTC (Thêm Twilio server cho mạnh)
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:global.stun.twilio.com:3478"]}
    ]}
)

# Chọn Camera
camera_type = st.radio("Chọn:", ("Laptop", "Điện thoại (Cam sau)"), horizontal=True)
constraints = {"facingMode": "environment"} if "Điện thoại" in camera_type else {"facingMode": "user"}

# Khởi tạo WebRTC
ctx = webrtc_streamer(
    key="mobile-voice",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": constraints, "audio": False},
    video_frame_callback=video_frame_callback,
    async_processing=True,
)

# --- PHẦN "VƯỢT QUYỀN" (JAVASCRIPT) ---
# Logic: Tạo một vùng rỗng, liên tục kiểm tra hàng đợi, nếu có tin thì chèn JS vào.

status_placeholder = st.empty() # Vùng hiển thị text
js_placeholder = st.empty()     # Vùng chèn code JS

# Nút kích hoạt âm thanh (BẮT BUỘC VỚI IPHONE/ANDROID)
# Trình duyệt chặn tự phát tiếng nếu người dùng không bấm gì đó trước.
if st.button("🔊 BẤM VÀO ĐÂY ĐỂ KÍCH HOẠT LOA (Quan trọng)"):
    js_placeholder.write(
        """<script>
        window.speechSynthesis.speak(new SpeechSynthesisUtterance("Đã kích hoạt giọng nói"));
        </script>""",
        unsafe_allow_html=True
    )

# Vòng lặp kiểm tra kết quả từ AI
if ctx.state.playing:
    while True:
        try:
            # Chờ lấy kết quả từ AI (timeout 0.1s để không đơ UI)
            text_to_speak = result_queue.get(timeout=0.1)
            
            # Hiển thị text lên màn hình
            status_placeholder.warning(f"⚠️ Phát hiện: {text_to_speak}")
            
            # CHÈN JAVASCRIPT ĐỂ ĐIỆN THOẠI NÓI
            # Đây là lệnh bắt trình duyệt đọc
            js_code = f"""
                <script>
                var msg = new SpeechSynthesisUtterance("{text_to_speak}");
                msg.lang = 'vi-VN'; // Chỉnh giọng tiếng Việt
                msg.rate = 1.2;     // Tốc độ nói
                window.speechSynthesis.speak(msg);
                </script>
            """
            js_placeholder.write(js_code, unsafe_allow_html=True)
            
            # Xóa message sau 1 giây để tránh chèn code liên tục
            time.sleep(2) 
            js_placeholder.empty()
            
        except queue.Empty:
            # Nếu không có biển báo nào thì lặp tiếp
            time.sleep(0.1)
            continue