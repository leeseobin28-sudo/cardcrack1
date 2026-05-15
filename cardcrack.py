# cardcrack.py
# 콘크리트 균열 자동 진단 V6.2 - Streamlit Cloud 배포용
# 라이브 카메라 가이드박스로 카드 크기를 맞춰 거리 보정

import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import cv2
from PIL import Image, ExifTags
from ultralytics import YOLO
import base64
import io

st.set_page_config(page_title="균열 자동 진단 V6.2", layout="wide")
st.title("🔍 콘크리트 균열 자동 진단 V6.2")
st.caption("💳 라이브 카메라 가이드박스로 카드 크기를 맞춰 촬영합니다.")

# ════════════════════════════════════════════════════════════════
# 상수
# ════════════════════════════════════════════════════════════════
CARD_W_MM = 85.60
CARD_H_MM = 53.98
SENSOR_W_MM = 36.0
DEFAULT_FOCAL_35MM = 26.0
DISTANCE_OPTIONS = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]

# ════════════════════════════════════════════════════════════════
# YOLO 모델
# ════════════════════════════════════════════════════════════════
@st.cache_resource
def load_yolo():
    return YOLO("bestcrack.pt")

# ════════════════════════════════════════════════════════════════
# EXIF
# ════════════════════════════════════════════════════════════════
def get_exif(pil_img):
    info = {"focal_35mm": None, "make": None, "model": None}
    try:
        exif = pil_img._getexif()
        if exif is None:
            return info
        tags = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}
        if "FocalLengthIn35mmFilm" in tags:
            info["focal_35mm"] = float(tags["FocalLengthIn35mmFilm"])
        info["make"] = tags.get("Make", None)
        info["model"] = tags.get("Model", None)
    except Exception:
        pass
    return info

# ════════════════════════════════════════════════════════════════
# 픽셀/거리 계산
# ════════════════════════════════════════════════════════════════
def card_pixel_ratio(dist_m, focal_35mm):
    """이미지 너비 대비 카드가 차지해야 할 비율 (0~1)"""
    f = focal_35mm or DEFAULT_FOCAL_35MM
    dist_mm = dist_m * 1000.0
    return (CARD_W_MM * f) / (dist_mm * SENSOR_W_MM)

def mm_per_pixel(dist_m, focal_35mm, image_width_px):
    f = focal_35mm or DEFAULT_FOCAL_35MM
    return (dist_m * 1000.0 * SENSOR_W_MM) / (f * image_width_px)

# ════════════════════════════════════════════════════════════════
# 라이브 카메라 + 가이드박스 HTML 컴포넌트
# ════════════════════════════════════════════════════════════════
def live_camera_with_guide(card_ratio, key="cam"):
    """
    card_ratio: 카메라 뷰 너비 대비 카드 가이드박스 너비 비율 (0~1)
    """
    html_code = f"""
    <div style="font-family: -apple-system, sans-serif;">
        <div id="cam-wrap-{key}" style="
            position: relative;
            width: 100%;
            max-width: 500px;
            margin: 0 auto;
            background: #000;
            border-radius: 12px;
            overflow: hidden;
        ">
            <video id="video-{key}" autoplay playsinline muted
                   style="width:100%; display:block;"></video>

            <!-- 가이드박스 오버레이 -->
            <div id="guide-{key}" style="
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                width: {card_ratio * 100}%;
                aspect-ratio: {CARD_W_MM} / {CARD_H_MM};
                border: 3px dashed #00ff66;
                box-shadow: 0 0 0 9999px rgba(0,0,0,0.35);
                box-sizing: border-box;
                pointer-events: none;
            ">
                <div style="
                    position: absolute;
                    top: -28px; left: 50%;
                    transform: translateX(-50%);
                    background: #00ff66;
                    color: #000;
                    padding: 2px 8px;
                    border-radius: 4px;
                    font-size: 12px;
                    font-weight: bold;
                    white-space: nowrap;
                ">📐 여기에 카드를 정확히 맞추세요</div>

                <div style="
                    position: absolute;
                    top: 50%; left: 50%;
                    width: 30px; height: 2px;
                    background: red;
                    transform: translate(-50%, -50%);
                "></div>
                <div style="
                    position: absolute;
                    top: 50%; left: 50%;
                    width: 2px; height: 30px;
                    background: red;
                    transform: translate(-50%, -50%);
                "></div>

                <div style="position:absolute;top:-2px;left:-2px;width:20px;height:20px;border-top:5px solid #00ff66;border-left:5px solid #00ff66;"></div>
                <div style="position:absolute;top:-2px;right:-2px;width:20px;height:20px;border-top:5px solid #00ff66;border-right:5px solid #00ff66;"></div>
                <div style="position:absolute;bottom:-2px;left:-2px;width:20px;height:20px;border-bottom:5px solid #00ff66;border-left:5px solid #00ff66;"></div>
                <div style="position:absolute;bottom:-2px;right:-2px;width:20px;height:20px;border-bottom:5px solid #00ff66;border-right:5px solid #00ff66;"></div>
            </div>
        </div>

        <div style="text-align:center; margin-top:12px;">
            <button id="snap-{key}" style="
                background: #ff4b4b;
                color: white;
                border: none;
                padding: 12px 32px;
                font-size: 16px;
                font-weight: bold;
                border-radius: 8px;
                cursor: pointer;
                box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            ">📸 촬영</button>
            <button id="switch-{key}" style="
                background: #555;
                color: white;
                border: none;
                padding: 12px 16px;
                font-size: 14px;
                border-radius: 8px;
                cursor: pointer;
                margin-left: 8px;
            ">🔄 카메라 전환</button>
        </div>

        <div id="result-{key}" style="margin-top:16px; text-align:center;"></div>
        <canvas id="canvas-{key}" style="display:none;"></canvas>
    </div>

    <script>
    (function() {{
        const video = document.getElementById('video-{key}');
        const canvas = document.getElementById('canvas-{key}');
        const snapBtn = document.getElementById('snap-{key}');
        const switchBtn = document.getElementById('switch-{key}');
        const resultDiv = document.getElementById('result-{key}');

        let currentStream = null;
        let useBackCamera = true;

        async function startCamera() {{
            if (currentStream) {{
                currentStream.getTracks().forEach(t => t.stop());
            }}
            try {{
                const constraints = {{
                    video: {{
                        facingMode: useBackCamera ? {{ exact: 'environment' }} : 'user',
                        width: {{ ideal: 1920 }},
                        height: {{ ideal: 1080 }}
                    }}
                }};
                currentStream = await navigator.mediaDevices.getUserMedia(constraints);
                video.srcObject = currentStream;
            }} catch (e) {{
                try {{
                    const fallback = {{
                        video: {{
                            facingMode: useBackCamera ? 'environment' : 'user'
                        }}
                    }};
                    currentStream = await navigator.mediaDevices.getUserMedia(fallback);
                    video.srcObject = currentStream;
                }} catch (err) {{
                    resultDiv.innerHTML = '<span style="color:red;">❌ 카메라 접근 실패: ' + err.message + '</span>';
                }}
            }}
        }}

        switchBtn.addEventListener('click', () => {{
            useBackCamera = !useBackCamera;
            startCamera();
        }});

        snapBtn.addEventListener('click', () => {{
            const w = video.videoWidth;
            const h = video.videoHeight;
            canvas.width = w;
            canvas.height = h;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(video, 0, 0, w, h);

            const dataUrl = canvas.toDataURL('image/jpeg', 0.92);

            resultDiv.innerHTML = `
                <div style="margin-top:10px;">
                    <p style="color:#00aa00;font-weight:bold;">✅ 촬영 완료! 아래 박스를 복사해 Streamlit에 붙여넣으세요.</p>
                    <img src="${{dataUrl}}" style="max-width:200px;border:2px solid #00aa00;border-radius:8px;"/>
                    <br/><br/>
                    <textarea id="dataout-{key}" readonly style="
                        width:100%;
                        height:80px;
                        font-size:10px;
                        font-family:monospace;
                    ">${{dataUrl}}</textarea>
                    <br/>
                    <button onclick="
                        navigator.clipboard.writeText(document.getElementById('dataout-{key}').value);
                        this.innerText='✅ 복사됨!';
                    " style="
                        background:#0088ff;color:white;border:none;
                        padding:8px 16px;border-radius:6px;cursor:pointer;
                        margin-top:8px;font-weight:bold;
                    ">📋 이미지 데이터 복사</button>
                </div>
            `;
        }});

        startCamera();
    }})();
    </script>
    """
    components.html(html_code, height=750)

# ════════════════════════════════════════════════════════════════
# base64 → PIL
# ════════════════════════════════════════════════════════════════
def b64_to_pil(b64_str):
    try:
        if "," in b64_str:
            b64_str = b64_str.split(",", 1)[1]
        img_bytes = base64.b64decode(b64_str)
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        st.error(f"이미지 디코딩 실패: {e}")
        return None

# ════════════════════════════════════════════════════════════════
# 세션 상태
# ════════════════════════════════════════════════════════════════
if "selected_dist" not in st.session_state:
    st.session_state.selected_dist = None
if "focal_35mm" not in st.session_state:
    st.session_state.focal_35mm = None

# ════════════════════════════════════════════════════════════════
# 사이드바
# ════════════════════════════════════════════════════════════════
st.sidebar.header("⚙️ 보정 옵션")
manual_focal = st.sidebar.number_input(
    "초점거리 (35mm 환산, mm)",
    min_value=0.0, max_value=200.0, value=26.0, step=1.0,
    help="일반 스마트폰 메인카메라: 약 24~28mm"
)
conf_thres = st.sidebar.slider("YOLO 신뢰도 임계값", 0.05, 0.9, 0.25, 0.05)

if st.sidebar.button("🔄 처음부터 다시 시작"):
    for k in ["selected_dist", "focal_35mm"]:
        st.session_state.pop(k, None)
    st.rerun()

# ════════════════════════════════════════════════════════════════
# STEP 1: 거리 선택
# ════════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("📐 STEP 1: 촬영 거리 선택")

cols = st.columns(len(DISTANCE_OPTIONS))
for i, d in enumerate(DISTANCE_OPTIONS):
    with cols[i]:
        label = f"📏 {d} m"
        if st.session_state.selected_dist == d:
            label = f"✅ {d} m"
        if st.button(label, key=f"dist_{d}", use_container_width=True):
            st.session_state.selected_dist = d
            st.rerun()

if st.session_state.selected_dist is None:
    st.info("👆 위에서 거리를 선택하세요.")
    st.stop()

st.success(f"선택된 거리: **{st.session_state.selected_dist} m**")

# ════════════════════════════════════════════════════════════════
# STEP 2: 라이브 카메라로 카드 맞추기
# ════════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("💳 STEP 2: 라이브 카메라로 카드 크기 맞추기")

card_ratio = card_pixel_ratio(st.session_state.selected_dist, manual_focal)

st.markdown(f"""
**📋 사용 방법:**
1. 균열 부위 **중앙에 신용카드(또는 체크카드)** 를 댑니다.
2. 아래 라이브 카메라에서 **녹색 가이드박스에 카드가 정확히 맞도록 거리를 조절**합니다.
3. 맞으면 **📸 촬영** 버튼을 누릅니다.
4. 촬영 후 카드를 빼고, 폰을 그대로 둔 채 STEP 3에서 균열만 촬영합니다.

> 💡 거리 **{st.session_state.selected_dist}m** 에서 카드는 화면 너비의 약 **{card_ratio*100:.1f}%** 를 차지합니다.
""")

live_camera_with_guide(card_ratio, key="card")

st.markdown("**📥 위에서 촬영한 카드 사진 데이터를 아래에 붙여넣으세요 (선택사항):**")

card_b64 = st.text_area(
    "카드 사진 데이터 (base64)",
    height=80,
    key="card_b64",
    label_visibility="collapsed"
)

if card_b64.strip():
    card_pil = b64_to_pil(card_b64)
    if card_pil:
        st.image(card_pil, caption="📷 촬영된 카드 사진", width=300)
        ex = get_exif(card_pil)
        if ex["focal_35mm"]:
            st.info(f"📷 EXIF 초점거리 감지: {ex['focal_35mm']} mm")
            st.session_state.focal_35mm = ex["focal_35mm"]
        else:
            st.session_state.focal_35mm = manual_focal
else:
    st.session_state.focal_35mm = manual_focal

st.session_state.focal_35mm = st.session_state.focal_35mm or manual_focal

# ════════════════════════════════════════════════════════════════
# STEP 3: 균열 촬영
# ════════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("🔍 STEP 3: 카드 제거 후, 같은 자리에서 균열 촬영")

st.warning(
    f"⚠️ **폰을 움직이지 마세요!** 카드만 치우고 그 자리에서 다시 촬영하세요.\n\n"
    f"📐 거리: **{st.session_state.selected_dist} m** | "
    f"🔭 초점거리: **{st.session_state.focal_35mm:.1f} mm**"
)

st.markdown("**라이브 카메라 (가이드박스는 참고용)**")
live_camera_with_guide(card_ratio, key="crack")

st.markdown("**📥 균열 사진 데이터를 붙여넣거나 파일로 업로드:**")

tab1, tab2 = st.tabs(["📋 base64 붙여넣기", "📁 파일 업로드"])
with tab1:
    crack_b64 = st.text_area(
        "균열 사진 데이터 (base64)",
        height=80,
        key="crack_b64",
        label_visibility="collapsed"
    )
with tab2:
    crack_upload = st.file_uploader(
        "균열 사진 업로드 (JPG/PNG만 지원, HEIC 비지원)",
        type=["jpg", "jpeg", "png"],
        key="crack_upload"
    )

pil_img = None
if crack_b64.strip():
    pil_img = b64_to_pil(crack_b64)
elif crack_upload is not None:
    try:
        pil_img = Image.open(crack_upload).convert("RGB")
    except Exception as e:
        st.error(f"❌ 이미지 로드 실패: {e}")

if pil_img is None:
    st.info("👆 균열 사진을 촬영하거나 업로드하세요.")
    st.stop()

# ════════════════════════════════════════════════════════════════
# STEP 4: 분석
# ════════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("📊 STEP 4: 자동 진단 결과")

img_np = np.array(pil_img)
H, W = img_np.shape[:2]

col1, col2 = st.columns(2)
with col1:
    st.markdown("**📷 입력 이미지**")
    st.image(pil_img, use_container_width=True)
with col2:
    st.markdown("**📋 측정 조건**")
    st.write(f"📐 거리: **{st.session_state.selected_dist} m**")
    st.write(f"🔭 초점거리(35mm): **{st.session_state.focal_35mm:.1f} mm**")
    st.write(f"🖼️ 이미지 크기: **{W} × {H} px**")

dist = st.session_state.selected_dist
focal = st.session_state.focal_35mm

with st.spinner("🔍 균열 탐지 중..."):
    yolo = load_yolo()
    results = yolo.predict(img_np, conf=conf_thres, verbose=False)

if not results or results[0].masks is None:
    st.error("❌ 균열을 찾지 못했습니다. 사이드바에서 신뢰도 임계값을 낮춰보세요.")
    st.stop()

masks = results[0].masks.data.cpu().numpy()
full_mask = np.zeros((H, W), dtype=np.uint8)
for m in masks:
    mr = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
    full_mask = np.maximum(full_mask, (mr > 0.5).astype(np.uint8))

if full_mask.sum() == 0:
    st.error("❌ 균열 마스크가 비어있습니다.")
    st.stop()

scale = mm_per_pixel(dist, focal, W)
pixel_cnt = int(full_mask.sum())
area_cm2 = (pixel_cnt * scale * scale) / 100.0
dt = cv2.distanceTransform(full_mask, cv2.DIST_L2, 5)
max_width_mm = 2 * float(dt.max()) * scale

st.markdown("### 🎯 측정 결과")
c1, c2, c3, c4 = st.columns(4)
c1.metric("촬영거리", f"{dist:.2f} m")
c2.metric("mm/pixel", f"{scale:.4f}")
c3.metric("균열 면적", f"{area_cm2:.2f} cm²")
c4.metric("최대 균열 폭", f"{max_width_mm:.2f} mm")

overlay = img_np.copy()
overlay[full_mask > 0] = [255, 50, 50]
blended = cv2.addWeighted(img_np, 0.55, overlay, 0.45, 0)
st.image(blended, caption="🎯 균열 탐지 결과", use_container_width=True)

st.success(
    f"✅ 카드 기반 거리 보정으로 측정 완료.\n\n"
    f"📏 1 픽셀 = {scale:.4f} mm | 총 {pixel_cnt:,} 픽셀 검출"
)
