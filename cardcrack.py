# cardcrack.py
# 콘크리트 균열 자동 진단 V7.0 - 원스텝 자동 측정
# 카드를 가이드박스에 맞춰 한 번만 촬영하면
# 자동으로 카드 검출 → mm/pixel 계산 → 균열 측정까지 완료
 
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO
import base64
import io
 
st.set_page_config(page_title="균열 자동 진단 V7", layout="wide")
st.title("🔍 콘크리트 균열 자동 진단 V7")
st.caption("💳 카드를 균열 위에 놓고 가이드박스에 맞춰 한 번만 촬영하세요.")
 
# ════════════════════════════════════════════════════════════════
# 상수 (표준 신용카드 ISO/IEC 7810 ID-1)
# ════════════════════════════════════════════════════════════════
CARD_W_MM = 85.60
CARD_H_MM = 53.98
CARD_ASPECT = CARD_W_MM / CARD_H_MM  # 약 1.586
 
# 가이드박스가 화면에서 차지하는 너비 비율 (40% 정도)
GUIDE_RATIO = 0.40
 
# ════════════════════════════════════════════════════════════════
# YOLO 모델
# ════════════════════════════════════════════════════════════════
@st.cache_resource
def load_yolo():
    return YOLO("bestcrack.pt")
 
# ════════════════════════════════════════════════════════════════
# 카드 자동 검출 (OpenCV 윤곽선 검출)
# ════════════════════════════════════════════════════════════════
def detect_card(img_np):
    """
    이미지에서 카드(직사각형)를 자동으로 검출.
    반환: (카드_가로_픽셀, 카드_세로_픽셀, 검출된_사각형_좌표) 또는 None
    """
    H, W = img_np.shape[:2]
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
 
    # 노이즈 제거
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
 
    # 적응형 이진화 + Canny edge
    edged = cv2.Canny(blurred, 50, 150)
 
    # 모폴로지 닫기 연산으로 선 연결
    kernel = np.ones((3, 3), np.uint8)
    edged = cv2.dilate(edged, kernel, iterations=2)
 
    # 윤곽선 검출
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
 
    if not contours:
        return None
 
    # 큰 윤곽선부터 정렬
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
 
    best_card = None
    best_score = 0
 
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < (W * H * 0.02):  # 너무 작은 건 제외 (이미지의 2% 미만)
            continue
        if area > (W * H * 0.8):  # 너무 큰 것도 제외
            continue
 
        # 사각형 근사
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
 
        if len(approx) != 4:
            continue
 
        # 최소외접사각형
        rect = cv2.minAreaRect(cnt)
        (cx, cy), (w, h), angle = rect
 
        if w == 0 or h == 0:
            continue
 
        # 카드 비율과 얼마나 비슷한지 확인
        long_side = max(w, h)
        short_side = min(w, h)
        aspect = long_side / short_side
 
        # 표준 카드 비율(1.586)과의 차이
        aspect_diff = abs(aspect - CARD_ASPECT)
        if aspect_diff > 0.25:  # 너무 다르면 제외
            continue
 
        # 점수: 면적이 크고 비율이 정확할수록 좋음
        score = area / (1 + aspect_diff * 10)
 
        if score > best_score:
            best_score = score
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            best_card = {
                "long_px": long_side,
                "short_px": short_side,
                "center": (cx, cy),
                "box": box,
                "angle": angle
            }
 
    return best_card
 
# ════════════════════════════════════════════════════════════════
# 라이브 카메라 + 가이드박스 컴포넌트
# ════════════════════════════════════════════════════════════════
def live_camera_with_guide(key="cam"):
    html_code = f"""
    <div style="font-family: -apple-system, sans-serif;">
        <div style="
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
            <div style="
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                width: {GUIDE_RATIO * 100}%;
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
                ">📐 카드를 여기에 맞추세요</div>
 
                <div style="position:absolute;top:50%;left:50%;width:30px;height:2px;background:red;transform:translate(-50%,-50%);"></div>
                <div style="position:absolute;top:50%;left:50%;width:2px;height:30px;background:red;transform:translate(-50%,-50%);"></div>
 
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
                padding: 14px 40px;
                font-size: 18px;
                font-weight: bold;
                border-radius: 8px;
                cursor: pointer;
                box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            ">📸 촬영 및 분석</button>
            <button id="switch-{key}" style="
                background: #555;
                color: white;
                border: none;
                padding: 14px 16px;
                font-size: 14px;
                border-radius: 8px;
                cursor: pointer;
                margin-left: 8px;
            ">🔄 전환</button>
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
            if (currentStream) currentStream.getTracks().forEach(t => t.stop());
            try {{
                currentStream = await navigator.mediaDevices.getUserMedia({{
                    video: {{
                        facingMode: useBackCamera ? {{ exact: 'environment' }} : 'user',
                        width: {{ ideal: 1920 }}, height: {{ ideal: 1080 }}
                    }}
                }});
                video.srcObject = currentStream;
            }} catch (e) {{
                try {{
                    currentStream = await navigator.mediaDevices.getUserMedia({{
                        video: {{ facingMode: useBackCamera ? 'environment' : 'user' }}
                    }});
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
            const w = video.videoWidth, h = video.videoHeight;
            canvas.width = w; canvas.height = h;
            canvas.getContext('2d').drawImage(video, 0, 0, w, h);
            const dataUrl = canvas.toDataURL('image/jpeg', 0.92);
            resultDiv.innerHTML = `
                <div style="margin-top:10px;">
                    <p style="color:#00aa00;font-weight:bold;">✅ 촬영 완료! 아래로 스크롤해서 데이터를 붙여넣으세요.</p>
                    <img src="${{dataUrl}}" style="max-width:200px;border:2px solid #00aa00;border-radius:8px;"/>
                    <br/><br/>
                    <textarea id="dataout-{key}" readonly style="
                        width:100%; height:80px; font-size:10px; font-family:monospace;
                    ">${{dataUrl}}</textarea>
                    <br/>
                    <button onclick="
                        navigator.clipboard.writeText(document.getElementById('dataout-{key}').value);
                        this.innerText='✅ 복사됨!';
                    " style="
                        background:#0088ff;color:white;border:none;
                        padding:10px 20px;border-radius:6px;cursor:pointer;
                        margin-top:8px;font-weight:bold;font-size:14px;
                    ">📋 데이터 복사하기</button>
                </div>
            `;
        }});
 
        startCamera();
    }})();
    </script>
    """
    components.html(html_code, height=720)
 
# ════════════════════════════════════════════════════════════════
# base64 → PIL
# ════════════════════════════════════════════════════════════════
def b64_to_pil(b64_str):
    try:
        if "," in b64_str:
            b64_str = b64_str.split(",", 1)[1]
        return Image.open(io.BytesIO(base64.b64decode(b64_str))).convert("RGB")
    except Exception as e:
        st.error(f"이미지 디코딩 실패: {e}")
        return None
 
# ════════════════════════════════════════════════════════════════
# 사이드바 (간단한 옵션만)
# ════════════════════════════════════════════════════════════════
st.sidebar.header("⚙️ 옵션")
conf_thres = st.sidebar.slider("YOLO 신뢰도 임계값", 0.05, 0.9, 0.25, 0.05)
st.sidebar.markdown("---")
st.sidebar.markdown("""
**📋 사용법**
1. 균열 위에 카드를 올림
2. 카메라가 카드를 가이드박스에 맞추도록 거리 조절
3. **📸 촬영 및 분석** 버튼
4. 데이터 복사해서 아래 박스에 붙여넣기
5. 자동으로 균열 측정 결과 표시
""")
 
# ════════════════════════════════════════════════════════════════
# 1. 라이브 카메라
# ════════════════════════════════════════════════════════════════
st.markdown("### 📷 카메라로 촬영")
st.markdown("""
**카드(신용카드/체크카드)를 균열 위에 올리고**, 녹색 가이드박스에 카드가 맞도록 거리를 조절한 뒤 **📸 촬영 및 분석** 버튼을 누르세요.
""")
 
live_camera_with_guide(key="main")
 
# ════════════════════════════════════════════════════════════════
# 2. 사진 데이터 입력
# ════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("### 📥 촬영 결과 분석")
 
tab1, tab2 = st.tabs(["📋 위 카메라에서 복사한 데이터 붙여넣기", "📁 파일 업로드"])
 
with tab1:
    img_b64 = st.text_area(
        "이미지 데이터 (base64)",
        height=80,
        placeholder="위의 '📋 데이터 복사하기' 버튼을 눌러서 복사한 후 여기에 붙여넣으세요...",
        label_visibility="collapsed"
    )
 
with tab2:
    img_upload = st.file_uploader(
        "카드가 포함된 균열 사진 업로드",
        type=["jpg", "jpeg", "png"]
    )
 
# 이미지 로드
pil_img = None
if img_b64.strip():
    pil_img = b64_to_pil(img_b64)
elif img_upload is not None:
    try:
        pil_img = Image.open(img_upload).convert("RGB")
    except Exception as e:
        st.error(f"❌ 이미지 로드 실패: {e}")
 
if pil_img is None:
    st.info("👆 위 카메라로 촬영하거나 파일을 업로드하면 자동 분석이 시작됩니다.")
    st.stop()
 
# ════════════════════════════════════════════════════════════════
# 3. 카드 자동 검출
# ════════════════════════════════════════════════════════════════
img_np = np.array(pil_img)
H, W = img_np.shape[:2]
 
with st.spinner("🔍 카드 검출 중..."):
    card_info = detect_card(img_np)
 
if card_info is None:
    st.error(
        "❌ **카드를 찾지 못했습니다.**\n\n"
        "다음을 확인해주세요:\n"
        "- 카드가 사진에 명확히 보이는지\n"
        "- 카드와 배경의 색 대비가 충분한지\n"
        "- 카드가 너무 기울어지지 않았는지\n"
        "- 조명이 균일한지"
    )
    st.image(pil_img, caption="입력 이미지", use_container_width=True)
    st.stop()
 
# mm/pixel 계산 (카드 긴 변 = 85.60mm)
scale = CARD_W_MM / card_info["long_px"]
 
# 카드 검출 결과 시각화
img_with_card = img_np.copy()
cv2.drawContours(img_with_card, [card_info["box"]], 0, (0, 255, 0), 5)
cv2.putText(
    img_with_card,
    f"Card detected ({card_info['long_px']:.0f}px x {card_info['short_px']:.0f}px)",
    (20, 50),
    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3
)
 
col1, col2 = st.columns(2)
with col1:
    st.markdown("**📷 카드 검출 결과**")
    st.image(img_with_card, use_container_width=True)
with col2:
    st.markdown("**📐 측정 기준**")
    st.write(f"📏 카드 픽셀 크기: **{card_info['long_px']:.0f} × {card_info['short_px']:.0f} px**")
    st.write(f"🔬 1 픽셀 = **{scale:.4f} mm**")
    st.write(f"🖼️ 이미지 크기: **{W} × {H} px**")
    st.success("✅ 카드 자동 검출 성공")
 
# ════════════════════════════════════════════════════════════════
# 4. 균열 검출 (YOLO)
# ════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("### 🎯 균열 측정 결과")
 
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
 
# 카드 영역은 균열 마스크에서 제외 (카드가 균열로 잡히는 것 방지)
card_mask = np.zeros((H, W), dtype=np.uint8)
cv2.fillPoly(card_mask, [card_info["box"]], 1)
full_mask = full_mask * (1 - card_mask)
 
if full_mask.sum() == 0:
    st.warning("⚠️ 균열 마스크가 비어있습니다. 카드 영역 밖에 균열이 보이는지 확인하세요.")
    st.stop()
 
# 면적·폭 계산
pixel_cnt = int(full_mask.sum())
area_cm2 = (pixel_cnt * scale * scale) / 100.0
dt = cv2.distanceTransform(full_mask, cv2.DIST_L2, 5)
max_width_mm = 2 * float(dt.max()) * scale
 
# 결과 메트릭
c1, c2, c3 = st.columns(3)
c1.metric("📏 mm/pixel", f"{scale:.4f}")
c2.metric("📐 균열 면적", f"{area_cm2:.2f} cm²")
c3.metric("📏 최대 균열 폭", f"{max_width_mm:.2f} mm")
 
# 시각화 (균열 빨강 + 카드 초록)
overlay = img_np.copy()
overlay[full_mask > 0] = [255, 50, 50]
blended = cv2.addWeighted(img_np, 0.55, overlay, 0.45, 0)
cv2.drawContours(blended, [card_info["box"]], 0, (0, 255, 0), 4)
 
st.image(blended, caption="🎯 검출 결과 (녹색: 카드 / 빨강: 균열)", use_container_width=True)
 
st.success(
    f"✅ 측정 완료 — 카드 기반 자동 보정\n\n"
    f"📊 총 균열 픽셀: {pixel_cnt:,}개 | 1px = {scale:.4f}mm"
)