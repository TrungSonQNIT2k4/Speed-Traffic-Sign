import av
import cv2
import time
import queue
import streamlit as st
import streamlit.components.v1 as components
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from ultralytics import YOLO

# --- CẤU HÌNH ---
st.set_page_config(page_title="Nhận diện Biển báo", page_icon="🚦", layout="centered")

# Dùng HTML thuần cho tiêu đề (tránh lỗi iOS cũ)
components.html("""
    <h2 style='text-align: center; color: #333; font-family: sans-serif;'>🚦 AI Biển Báo (HUD + Voice)</h2>
""", height=60)

# 1. Hàng đợi tin nhắn (Cầu nối giữa AI và Giao diện)
# Queue này giúp chuyển tin nhắn từ luồng xử lý ảnh sang luồng giao diện web
result_queue = queue.Queue()

# 2. Load Model
@st.cache_resource
def load_model():
    return YOLO('best.pt')

try:
    model = load_model()
except Exception as e:
    st.error(f"❌ Lỗi model: {e}")
    st.stop()

# 3. Từ điển & Cấu hình
CLASS_MESSAGES = {
    "khu_vuc_dong_dan_cu": "Khu vực đông dân cư",
    "het_khu_vuc_dong_dan_cu": "Hết khu vực đông dân cư",
    "cam_quay_dau": "Cấm quay đầu",
    "cam_di_nguoc_chieu": "Nguy hiểm đi ngược chiều",
    "gioi_han_toc_do_50": "Giới hạn tốc độ 50",
    "gioi_han_toc_do_60": "Giới hạn tốc độ 60",
    "cam_vuot": "Cấm vượt",
    # Thêm các lớp khác...
}

# Biến toàn cục để kiểm soát tần suất nói (tránh nói liên tục)
# Lưu ý: Trong Streamlit Cloud, biến global sẽ bị reset mỗi session, 
# nhưng với webrtc callback thì nó vẫn giữ được trong worker process.
last_spoken_time = {}
COOLDOWN = 5.0 # 5 giây mới nhắc lại 1 lần

# 4. Hàm vẽ HUD (Vẽ chữ lên video)
def draw_hud(image, text):
    h, w, _ = image.shape
    # Vẽ nền đen dưới đáy
    cv2.rectangle(image, (0, h-50), (w, h), (0, 0, 0), -1)
    # Vẽ chữ vàng
    font_scale = 0.8 if w < 500 else 1.2
    thickness = 2
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
    text_x = (w - text_size[0]) // 2
    cv2.putText(image, text, (text_x, h-15), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness)

# 5. Hàm xử lý AI
def video_frame_callback(frame):
    global last_spoken_time
    img = frame.to_ndarray(format="bgr24")
    
    # Nhận diện
    results = model.predict(img, conf=0.5, verbose=False)
    
    message_to_speak = None
    display_text = ""

    current_time = time.time()

    for r in results:
        img = r.plot() # Vẽ khung YOLO
        
        for box in r.boxes:
            cls_id = int(box.cls[0])
            name = model.names[cls_id]
            
            if name in CLASS_MESSAGES:
                raw_text = CLASS_MESSAGES[name]
                display_text = raw_text # Chữ hiện trên màn hình (có dấu)
                
                # Logic kiểm soát giọng nói (Cooldown)
                if (name not in last_spoken_time) or (current_time - last_spoken_time[name] > COOLDOWN):
                    last_spoken_time[name] = current_time
                    message_to_speak = raw_text # Chữ để đọc
                
                break # Chỉ lấy 1 biển báo ưu tiên nhất

    # 1. Vẽ HUD (Luôn làm)
    if display_text:
        # Chuyển tiếng Việt có dấu thành không dấu để vẽ CV2 không lỗi font (nếu cần)
        # Ở đây vẽ demo, nếu lỗi font trên server thì chấp nhận hoặc dùng PIL
        draw_hud(img, display_text) 

    # 2. Gửi lệnh nói vào hàng đợi (Nếu hết cooldown)
    if message_to_speak:
        try:
            result_queue.put_nowait(message_to_speak)
        except queue.Full:
            pass

    return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- GIAO DIỆN CHÍNH ---

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:global.stun.twilio.com:3478"]}
    ]}
)

# Nút kích hoạt âm thanh cho iOS (QUAN TRỌNG)
# iOS bắt buộc người dùng phải tương tác 1 lần thì web mới được quyền phát tiếng
st.warning("👇 Bắt buộc: Bấm nút dưới để bật loa trên iPhone/Safari")
if st.button("🔊 KÍCH HOẠT LOA IPHONE"):
    components.html("""
    <script>
        window.speechSynthesis.cancel();
        var msg = new SpeechSynthesisUtterance("Đã kích hoạt loa thành công");
        msg.lang = 'vi-VN';
        window.speechSynthesis.speak(msg);
    </script>
    """, height=0)

# Chọn thiết bị
camera_type = st.radio("Chọn:", ("Laptop", "Điện thoại (Cam sau)"), horizontal=True)
constraints = {"facingMode": "environment"} if "Điện thoại" in camera_type else {"facingMode": "user"}

# WebRTC Streamer
ctx = webrtc_streamer(
    key="hud-voice-final",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": constraints, "audio": False},
    video_frame_callback=video_frame_callback,
    async_processing=True,
)

# --- VÒNG LẶP XỬ LÝ GIỌNG NÓI (JAVASCRIPT INJECTION) ---
# Dùng placeholder để chèn JS mà không render lại toàn bộ trang
js_placeholder = st.empty()

if ctx.state.playing:
    while True:
        # Kiểm tra nếu Stream dừng thì thoát vòng lặp ngay (TRÁNH SẬP APP)
        if not ctx.state.playing:
            break

        try:
            # Lấy tin nhắn từ hàng đợi (chờ tối đa 0.5s)
            text = result_queue.get(timeout=0.5)
            
            # Bơm JavaScript vào để điện thoại đọc
            # Dùng components.html để bypass mọi lỗi regex của iOS cũ
            with js_placeholder:
                components.html(f"""
                <script>
                    window.speechSynthesis.cancel(); 
                    var msg = new SpeechSynthesisUtterance("{text}");
                    msg.lang = 'vi-VN'; 
                    msg.rate = 1.1;
                    window.speechSynthesis.speak(msg);
                </script>
                """, height=0, width=0)
            
        except queue.Empty:
            pass
        
        # Ngủ nhẹ để giảm tải CPU
        time.sleep(0.1)