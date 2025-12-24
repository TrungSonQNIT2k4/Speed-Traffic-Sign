import av
import cv2
import time
import queue
import streamlit as st
import streamlit.components.v1 as components
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from ultralytics import YOLO

# --- 1. HÀM XỬ LÝ FONT TIẾNG VIỆT (FIX LỖI Ô VUÔNG) ---
def remove_accents(input_str):
    """
    Chuyển đổi tiếng Việt có dấu thành không dấu in hoa.
    Ví dụ: "Giới hạn tốc độ" -> "GIOI HAN TOC DO"
    """
    s1 = u'ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝàáâãèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯưẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ'
    s0 = u'AAAAEEEIIOOOUUYaaaaeeeiiooouuyAaDdIiUuOoUuAaAaAaAaAaAaAaAaAaAaAaAaEeEeEeEeEeEeEeEeIiIiOoOoOoOoOoOoOoOoOoOoOoOoUuUuUuUuUuUuUuYyYyYyYy'
    s = ''
    for c in input_str:
        if c in s1:
            s += s0[s1.index(c)]
        else:
            s += c
    return s.upper()

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="Nhận diện Biển báo", page_icon="🚦", layout="centered")

# Tiêu đề HTML thuần
components.html("""
    <h2 style='text-align: center; color: #333; font-family: sans-serif;'>🚦 AI Biển Báo (Final Release)</h2>
""", height=60)

# 2. Hàng đợi tin nhắn
result_queue = queue.Queue()

# 3. Load Model
@st.cache_resource
def load_model():
    return YOLO('best.pt')

try:
    model = load_model()
except Exception as e:
    st.error(f"❌ Lỗi model: {e}")
    st.stop()

# 4. Từ điển (Cứ viết tiếng Việt có dấu bình thường)
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

# Biến kiểm soát giọng nói
last_spoken_time = {}
COOLDOWN = 5.0 

# 5. Hàm vẽ HUD (Đã tích hợp xóa dấu)
def draw_hud(image, text):
    # Chuyển thành KHÔNG DẤU trước khi vẽ
    clean_text = remove_accents(text)
    
    h, w, _ = image.shape
    # Vẽ nền đen dưới đáy
    cv2.rectangle(image, (0, h-60), (w, h), (0, 0, 0), -1)
    # Vẽ viền vàng cho nổi
    cv2.rectangle(image, (0, h-60), (w, h), (0, 255, 255), 2)
    
    # Cấu hình font
    font_scale = 0.9 if w < 500 else 1.3
    thickness = 2
    
    # Tính vị trí căn giữa
    text_size = cv2.getTextSize(clean_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
    text_x = (w - text_size[0]) // 2
    
    # Vẽ chữ
    cv2.putText(image, clean_text, (text_x, h-20), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness)

# 6. Hàm xử lý AI
def video_frame_callback(frame):
    global last_spoken_time
    img = frame.to_ndarray(format="bgr24")
    
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
                display_text = raw_text # Lưu text gốc để hiển thị
                
                # Logic Cooldown
                if (name not in last_spoken_time) or (current_time - last_spoken_time[name] > COOLDOWN):
                    last_spoken_time[name] = current_time
                    message_to_speak = raw_text # Lưu text để đọc
                
                break

    # 1. Vẽ HUD (Dùng hàm đã fix font)
    if display_text:
        draw_hud(img, display_text) 

    # 2. Gửi lệnh nói vào hàng đợi
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

st.warning("👇 Dành cho iPhone: Bấm nút dưới để kích hoạt loa")
if st.button("🔊 KÍCH HOẠT LOA IPHONE"):
    components.html("""
    <script>
        window.speechSynthesis.cancel();
        var msg = new SpeechSynthesisUtterance("Đã kết nối loa");
        msg.lang = 'vi-VN';
        window.speechSynthesis.speak(msg);
    </script>
    """, height=0)

# Chọn thiết bị
camera_type = st.radio("Chọn:", ("Laptop", "Điện thoại (Cam sau)"), horizontal=True)

if "Điện thoại" in camera_type:
    # CẤU HÌNH MẠNH CHO IPHONE:
    # 1. facingMode: exact environment -> Ép buộc cam sau
    # 2. width/height: ideal -> Yêu cầu độ phân giải cao (cam sau thường nét hơn)
    video_constraints = {
        "facingMode": {"exact": "environment"},
        "width": {"ideal": 1280},
        "height": {"ideal": 720}
    }
else:
    video_constraints = {"facingMode": "user"}

# WebRTC Streamer
ctx = webrtc_streamer(
    key="final-hud-v5", # Đổi key để reset sạch sẽ
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": video_constraints, "audio": False},
    video_frame_callback=video_frame_callback,
    async_processing=True,
)

# --- XỬ LÝ GIỌNG NÓI (TỐI ƯU CHO IPHONE) ---
js_placeholder = st.empty()

if ctx.state.playing:
    while True:
        if not ctx.state.playing:
            break

        try:
            # Tăng timeout lên 1s để giảm tải vòng lặp
            text = result_queue.get(timeout=1.0)
            
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
            
            # QUAN TRỌNG: Ngủ 3 giây sau khi nói để iPhone không bị "sốc nhiệt"
            time.sleep(3.0) 
            
        except queue.Empty:
            time.sleep(0.1)