import av
import cv2
import time
import queue
import streamlit as st
import streamlit.components.v1 as components
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from ultralytics import YOLO

# --- 1. CẤU HÌNH CƠ BẢN ---
st.set_page_config(page_title="Nhận diện Biển báo", page_icon="🚦", layout="centered")

# Tiêu đề HTML
components.html("""
    <h2 style='text-align: center; color: #333; font-family: sans-serif;'>🚦 AI Biển Báo (Stable Version)</h2>
""", height=60)

# Queue tin nhắn
result_queue = queue.Queue()

# --- 2. LOAD MODEL ---
@st.cache_resource
def load_model():
    return YOLO('best.pt')

try:
    model = load_model()
except Exception as e:
    st.error(f"❌ Lỗi model: {e}")
    st.stop()

# --- 3. HÀM XỬ LÝ FONT (QUAN TRỌNG CHO ANDROID/IPHONE) ---
def remove_accents(input_str):
    """Chuyển tiếng Việt có dấu thành KHÔNG DẤU IN HOA để vẽ lên HUD không lỗi"""
    s1 = u'ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝàáâãèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯưẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ'
    s0 = u'AAAAEEEIIOOOUUYaaaaeeeiiooouuyAaDdIiUuOoUuAaAaAaAaAaAaAaAaAaAaAaAaEeEeEeEeEeEeEeEeIiIiOoOoOoOoOoOoOoOoOoOoOoOoUuUuUuUuUuUuUuYyYyYyYy'
    s = ''
    for c in input_str:
        if c in s1:
            s += s0[s1.index(c)]
        else:
            s += c
    return s.upper()

# Từ điển
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

last_spoken_time = {}
COOLDOWN = 5.0 

# --- 4. VẼ HUD (ĐÃ FIX FONT) ---
def draw_hud(image, text):
    # Bước chuyển đổi quan trọng:
    clean_text = remove_accents(text) 
    
    h, w, _ = image.shape
    # Vẽ nền
    cv2.rectangle(image, (0, h-60), (w, h), (0, 0, 0), -1)
    cv2.rectangle(image, (0, h-60), (w, h), (0, 255, 255), 2)
    
    # Font dynamic
    font_scale = 0.8 if w < 500 else 1.2
    thickness = 2
    text_size = cv2.getTextSize(clean_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
    text_x = (w - text_size[0]) // 2
    
    # Vẽ chữ KHÔNG DẤU
    cv2.putText(image, clean_text, (text_x, h-20), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness)

# --- 5. CORE AI ---
def video_frame_callback(frame):
    global last_spoken_time
    img = frame.to_ndarray(format="bgr24")
    
    # Tắt verbose để giảm lag log
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
        draw_hud(img, display_text) # Vẽ HUD không dấu
        
    if message_to_speak:
        try:
            result_queue.put_nowait(message_to_speak)
        except queue.Full:
            pass

    return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- 6. GIAO DIỆN & CAMERA ---

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:global.stun.twilio.com:3478"]}
    ]}
)

# Nút kích hoạt loa (BẮT BUỘC VỚI IPHONE)
st.warning("📱 Lưu ý: Trên điện thoại, hãy bấm nút KÍCH HOẠT LOA trước khi bấm Start.")
if st.button("🔊 KÍCH HOẠT LOA"):
    components.html("""
    <script>
        if ('speechSynthesis' in window) {
            window.speechSynthesis.cancel();
            var msg = new SpeechSynthesisUtterance("Sẵn sàng");
            msg.lang = 'vi-VN';
            window.speechSynthesis.speak(msg);
        }
    </script>
    """, height=0)

# Chọn thiết bị
camera_mode = st.radio("Chọn thiết bị:", ("Laptop/PC", "Điện thoại (Cam sau)"), horizontal=True)

if camera_mode == "Laptop/PC":
    video_constraints = {"facingMode": "user"}
else:
    # Cấu hình "Mềm" cho điện thoại: 
    # environment + độ phân giải HD -> Giúp iPhone tự ưu tiên cam sau mà không bị lỗi 'exact'
    video_constraints = {
        "facingMode": "environment",
        "width": {"ideal": 1280},
        "height": {"ideal": 720}
    }

# Streamer
ctx = webrtc_streamer(
    key="stable-final-v9",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": video_constraints, "audio": False},
    video_frame_callback=video_frame_callback,
    async_processing=True,
)

# --- 7. XỬ LÝ VOICE (BẢN NHẸ NHÀNG CHO MOBILE) ---
js_placeholder = st.empty()

if ctx.state.playing:
    while True:
        # 1. THOÁT NGAY NẾU STOP: Quan trọng để không treo Android
        if not ctx.state.playing:
            break

        try:
            # 2. Timeout dài hơn (0.5s) để giảm tải CPU
            text = result_queue.get(timeout=0.5)
            
            with js_placeholder:
                components.html(f"""
                <script>
                    if ('speechSynthesis' in window) {{
                        window.speechSynthesis.cancel(); 
                        var msg = new SpeechSynthesisUtterance("{text}");
                        msg.lang = 'vi-VN'; 
                        msg.rate = 1.1;
                        window.speechSynthesis.speak(msg);
                    }}
                </script>
                """, height=0, width=0)
            
            # 3. Nghỉ lâu hơn (2s) sau khi nói
            time.sleep(2.0) 
            
        except queue.Empty:
            # 4. Nếu không có tin nhắn, nghỉ nhẹ 0.2s để nhường CPU xử lý video
            time.sleep(0.2)