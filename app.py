import av
import cv2
import time
import queue
import streamlit as st
import streamlit.components.v1 as components
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from ultralytics import YOLO

# --- CẤU HÌNH TRANG (Ẩn menu cho giống App xịn) ---
st.set_page_config(page_title="AI Traffic Sign", page_icon="🚦", layout="centered", initial_sidebar_state="collapsed")

# CSS Hack: Ẩn bớt các nút rườm rà của Streamlit để giao diện sạch
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        .stApp {margin-top: -80px;}
    </style>
    """, unsafe_allow_html=True)

# Tiêu đề đơn giản
components.html("""
    <h2 style='text-align: center; color: #333; font-family: sans-serif; margin-bottom: 0;'>🚦 AI AUTO-PILOT</h2>
""", height=50)

# --- HÀM FIX FONT ---
def remove_accents(input_str):
    s1 = u'ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝàáâãèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯưẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ'
    s0 = u'AAAAEEEIIOOOUUYaaaaeeeiiooouuyAaDdIiUuOoUuAaAaAaAaAaAaAaAaAaAaAaAaEeEeEeEeEeEeEeEeIiIiOoOoOoOoOoOoOoOoOoOoOoOoUuUuUuUuUuUuUuYyYyYyYy'
    s = ''
    for c in input_str:
        if c in s1:
            s += s0[s1.index(c)]
        else:
            s += c
    return s.upper()

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    return YOLO('best.pt')

try:
    model = load_model()
except Exception as e:
    st.error(f"Lỗi: {e}")
    st.stop()

# --- TỪ ĐIỂN ---
CLASS_MESSAGES = {
    "khu_vuc_dong_dan_cu": "Khu vực đông dân cư",
    "het_khu_vuc_dong_dan_cu": "Hết khu vực đông dân cư",
    "cam_quay_dau": "Cấm quay đầu",
    "cam_di_nguoc_chieu": "Nguy hiểm đi ngược chiều",
    "gioi_han_toc_do_50": "Giới hạn tốc độ 50",
    "gioi_han_toc_do_60": "Giới hạn tốc độ 60",
    "cam_vuot": "Cấm vượt",
    # Thêm class khác...
}

# --- XỬ LÝ ---
result_queue = queue.Queue()
last_spoken_time = {}
COOLDOWN = 5.0 

def draw_hud(image, text):
    clean_text = remove_accents(text)
    h, w, _ = image.shape
    cv2.rectangle(image, (0, h-60), (w, h), (0, 0, 0), -1)
    cv2.rectangle(image, (0, h-60), (w, h), (0, 255, 255), 2)
    font_scale = 0.9 if w < 500 else 1.3
    thickness = 2
    text_size = cv2.getTextSize(clean_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
    text_x = (w - text_size[0]) // 2
    cv2.putText(image, clean_text, (text_x, h-20), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness)

def video_frame_callback(frame):
    global last_spoken_time
    img = frame.to_ndarray(format="bgr24")
    results = model.predict(img, conf=0.5, verbose=False)
    
    display_text = ""
    message_to_speak = None
    current_time = time.time()

    for r in results:
        img = r.plot()
        for box in r.boxes:
            cls_id = int(box.cls[0])
            name = model.names[cls_id]
            if name in CLASS_MESSAGES:
                raw_text = CLASS_MESSAGES[name]
                display_text = raw_text
                if (name not in last_spoken_time) or (current_time - last_spoken_time[name] > COOLDOWN):
                    last_spoken_time[name] = current_time
                    message_to_speak = raw_text
                break

    if display_text:
        draw_hud(img, display_text) 
    if message_to_speak:
        try:
            result_queue.put_nowait(message_to_speak)
        except queue.Full:
            pass
    return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- AUTO CONFIGURATION (TỰ ĐỘNG HÓA) ---

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:global.stun.twilio.com:3478"]}
    ]}
)

# CẤU HÌNH THÔNG MINH (Smart Constraints):
# "environment": 
# - Điện thoại: Hiểu là Cam sau.
# - Laptop: Không có cam sau -> Nó tự fallback về cam duy nhất (Webcam trước).
# - KHÔNG DÙNG 'exact' để tránh sập laptop.
video_constraints = {
    "facingMode": "environment", 
    "width": {"ideal": 1280}, 
    "height": {"ideal": 720}
}

# AUTO START: desired_playing_state=True
ctx = webrtc_streamer(
    key="autopilot-v1",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": video_constraints, "audio": False},
    video_frame_callback=video_frame_callback,
    async_processing=True,
    desired_playing_state=True, # <--- DÒNG NÀY ĐỂ TỰ CHẠY KHÔNG CẦN BẤM START
)

# --- AUTO VOICE INJECTION ---
js_placeholder = st.empty()

if ctx.state.playing:
    while True:
        if not ctx.state.playing: break
        try:
            text = result_queue.get(timeout=1.0)
            with js_placeholder:
                # Script tự động đọc, bỏ qua các bước check rườm rà
                components.html(f"""
                <script>
                    var msg = new SpeechSynthesisUtterance("{text}");
                    msg.lang = 'vi-VN'; 
                    msg.rate = 1.1;
                    window.speechSynthesis.speak(msg);
                </script>
                """, height=0, width=0)
            time.sleep(3.0) 
        except queue.Empty:
            time.sleep(0.1)