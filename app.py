"""
ThresYm - Advanced Image Thresholding Tool
Enhanced Single File Application
"""
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io

# ==================== CONFIGURATION ====================
AVAILABLE_METHODS = [
    "Simple Threshold",
    "Otsu's Method",
    "Adaptive Mean",
    "Adaptive Gaussian",
    "Niblack",
    "Sauvola"
]

METHODS_INFO = {
    "Simple Threshold":   {"description": "Basic binary thresholding with a single global threshold value", "icon": "⚡", "badge": "GLOBAL"},
    "Otsu's Method":      {"description": "Automatic threshold selection using Otsu's bi-modal histogram algorithm", "icon": "🤖", "badge": "AUTO"},
    "Adaptive Mean":      {"description": "Local adaptive thresholding using mean of neighborhood pixels", "icon": "📐", "badge": "LOCAL"},
    "Adaptive Gaussian":  {"description": "Local adaptive thresholding using Gaussian-weighted neighborhood", "icon": "🌊", "badge": "LOCAL"},
    "Niblack":            {"description": "Local thresholding for images with varying illumination conditions", "icon": "🔭", "badge": "ADVANCED"},
    "Sauvola":            {"description": "Enhanced Niblack method, optimized for document and text images", "icon": "📄", "badge": "ADVANCED"},
}

DEFAULT_PARAMS = {
    'threshold': 127, 'block_size': 11, 'C': 2,
    'window_size': 15, 'k': -0.2, 'R': 128
}

BADGE_COLORS = {
    "GLOBAL":   "#ffc107",
    "AUTO":     "#39ff8a",
    "LOCAL":    "#00e5ff",
    "ADVANCED": "#b66dff",
}

# ==================== CUSTOM CSS ====================
def inject_custom_css():
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');
:root {
    --bg:       #0a0c10;
    --bg2:      #0f1117;
    --card:     #131720;
    --elevated: #1a2030;
    --border:   #1e2535;
    --borderB:  #2a3548;
    --cyan:     #00e5ff;
    --green:    #39ff8a;
    --amber:    #ffc107;
    --purple:   #b66dff;
    --red:      #ff4d6d;
    --t1:       #e8edf5;
    --t2:       #8892a4;
    --t3:       #4a5568;
    --mono:     'Space Mono', monospace;
    --display:  'Syne', sans-serif;
}
.stApp { background: var(--bg) !important; font-family: var(--mono) !important; }
.main .block-container { background: var(--bg) !important; padding: 1.5rem 2rem 3rem !important; max-width: 1400px !important; }
::-webkit-scrollbar { width:5px; height:5px; }
::-webkit-scrollbar-track { background: var(--bg2); }
::-webkit-scrollbar-thumb { background: var(--borderB); border-radius:99px; }
[data-testid="stSidebar"] { background: var(--bg2) !important; border-right: 1px solid var(--border) !important; }
[data-testid="stSidebarContent"] { background: var(--bg2) !important; }
h1,h2,h3,h4,h5,h6 { font-family: var(--display) !important; color: var(--t1) !important; }
p, span, div, label { color: var(--t2) !important; font-family: var(--mono) !important; }
[data-testid="stFileUploader"] {
    background: var(--card) !important;
    border: 1.5px dashed var(--borderB) !important;
    border-radius: 12px !important;
    padding: 1rem !important;
    transition: border-color 0.25s;
}
[data-testid="stFileUploader"]:hover { border-color: var(--cyan) !important; }
[data-testid="stFileUploadDropzone"] { background: transparent !important; }
[data-testid="stSlider"] [role="slider"] {
    background: var(--cyan) !important;
    border: 2px solid var(--bg) !important;
    box-shadow: 0 0 8px var(--cyan) !important;
}
[data-baseweb="select"] > div {
    background: var(--card) !important;
    border-color: var(--borderB) !important;
    border-radius: 8px !important;
}
[data-baseweb="select"] > div:focus-within {
    border-color: var(--cyan) !important;
    box-shadow: 0 0 0 2px rgba(0,229,255,0.15) !important;
}
[data-baseweb="tag"] {
    background: rgba(0,229,255,0.12) !important;
    border: 1px solid rgba(0,229,255,0.35) !important;
    color: var(--cyan) !important;
    border-radius: 4px !important;
    font-family: var(--mono) !important;
    font-size: 0.75rem !important;
}
.stDownloadButton > button, .stButton > button {
    background: transparent !important;
    color: var(--cyan) !important;
    border: 1px solid var(--cyan) !important;
    border-radius: 8px !important;
    font-family: var(--mono) !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    transition: all 0.2s !important;
    width: 100% !important;
}
.stDownloadButton > button:hover, .stButton > button:hover {
    background: rgba(0,229,255,0.1) !important;
    box-shadow: 0 0 16px rgba(0,229,255,0.3) !important;
    transform: translateY(-1px) !important;
}
[data-testid="stExpander"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}
[data-testid="stMetric"] {
    background: var(--elevated) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 0.6rem 0.8rem !important;
}
[data-testid="stMetricValue"] { color: var(--green) !important; font-family: var(--mono) !important; }
[data-testid="stMetricLabel"] p { font-size:0.65rem !important; letter-spacing:0.1em !important; text-transform:uppercase !important; color: var(--t3) !important; }
[data-testid="stImage"] img {
    border-radius: 8px !important;
    border: 1px solid var(--border) !important;
    transition: border-color 0.25s, box-shadow 0.25s;
}
[data-testid="stImage"] img:hover {
    border-color: var(--cyan) !important;
    box-shadow: 0 0 20px rgba(0,229,255,0.15) !important;
}
hr { border-color: var(--border) !important; margin: 1.2rem 0 !important; }
.sidebar-label {
    font-family: var(--mono) !important;
    font-size: 0.62rem !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    color: var(--t3) !important;
    padding-bottom: 0.4rem !important;
    border-bottom: 1px solid var(--border) !important;
    margin-bottom: 0.6rem !important;
    display: block;
}
</style>
""", unsafe_allow_html=True)


# ==================== IMAGE PROCESSOR ====================
class ImageProcessor:
    def __init__(self):
        self.original_image = None
        self.gray_image = None

    def load_image(self, uploaded_file):
        try:
            image = Image.open(uploaded_file)
            img_array = np.array(image)
            self.gray_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if len(img_array.shape) == 3 else img_array
            self.original_image = img_array
            return True
        except Exception as e:
            st.error(f"Error loading image: {e}")
            return False

    def get_grayscale(self):
        return self.gray_image

    def simple_threshold(self, threshold):
        _, binary = cv2.threshold(self.gray_image, threshold, 255, cv2.THRESH_BINARY)
        return binary

    def otsu_threshold(self):
        _, binary = cv2.threshold(self.gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    def adaptive_threshold(self, block_size, C, method='mean'):
        m = cv2.ADAPTIVE_THRESH_MEAN_C if method == 'mean' else cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        return cv2.adaptiveThreshold(self.gray_image, 255, m, cv2.THRESH_BINARY, block_size, C)

    def niblack_threshold(self, window_size, k):
        try:
            from skimage.filters import threshold_niblack
            thresh = threshold_niblack(self.gray_image, window_size=window_size, k=k)
            return (self.gray_image > thresh).astype(np.uint8) * 255
        except ImportError:
            st.error("scikit-image not installed. Run: pip install scikit-image")
            return None

    def sauvola_threshold(self, window_size, k, R):
        try:
            from skimage.filters import threshold_sauvola
            thresh = threshold_sauvola(self.gray_image, window_size=window_size, k=k, r=R)
            return (self.gray_image > thresh).astype(np.uint8) * 255
        except ImportError:
            st.error("scikit-image not installed. Run: pip install scikit-image")
            return None

    @staticmethod
    def image_to_bytes(image):
        buf = io.BytesIO()
        Image.fromarray(image).save(buf, format='PNG')
        return buf.getvalue()


# ==================== UI ====================

def render_header():
    st.markdown("""
<div style="padding:2.5rem 0 0.5rem; border-bottom:1px solid #1e2535; margin-bottom:1.5rem;">
  <span style="font-family:'Space Mono',monospace; font-size:0.62rem; letter-spacing:0.2em;
    color:#00e5ff; background:rgba(0,229,255,0.07); border:1px solid rgba(0,229,255,0.25);
    border-radius:3px; padding:0.2rem 0.7rem;">v2.0 · Computer Vision Lab</span>

  <div style="display:flex; align-items:baseline; gap:0.75rem; flex-wrap:wrap; margin-top:0.8rem;">
    <h1 style="font-family:'Syne',sans-serif !important; font-size:3.2rem !important;
      font-weight:800 !important; color:#e8edf5 !important; letter-spacing:-0.02em;
      margin:0; line-height:1;">ThresYm</h1>
    <span style="font-family:'Space Mono',monospace; font-size:0.82rem;
      color:#00e5ff; opacity:0.75;"></span>
  </div>

  <p style="font-family:'Space Mono',monospace !important; font-size:0.8rem !important;
    color:#8892a4 !important; margin-top:0.5rem; letter-spacing:0.04em;">
    Binary image segmentation and adaptive thresholding workbench
  </p>

  <div style="display:flex; gap:0.5rem; margin-top:1rem; flex-wrap:wrap;">
    <span style="font-family:'Space Mono',monospace; font-size:0.62rem; padding:0.2rem 0.6rem;
      background:rgba(57,255,138,0.08); border:1px solid rgba(57,255,138,0.22);
      color:#39ff8a; border-radius:3px; letter-spacing:0.08em;">6 METHODS</span>
    <span style="font-family:'Space Mono',monospace; font-size:0.62rem; padding:0.2rem 0.6rem;
      background:rgba(182,109,255,0.08); border:1px solid rgba(182,109,255,0.22);
      color:#b66dff; border-radius:3px; letter-spacing:0.08em;">REAL-TIME PARAMS</span>
    <span style="font-family:'Space Mono',monospace; font-size:0.62rem; padding:0.2rem 0.6rem;
      background:rgba(255,193,7,0.08); border:1px solid rgba(255,193,7,0.22);
      color:#ffc107; border-radius:3px; letter-spacing:0.08em;">SIDE-BY-SIDE COMPARE</span>
    <span style="font-family:'Space Mono',monospace; font-size:0.62rem; padding:0.2rem 0.6rem;
      background:rgba(0,229,255,0.08); border:1px solid rgba(0,229,255,0.22);
      color:#00e5ff; border-radius:3px; letter-spacing:0.08em;">PNG EXPORT</span>
  </div>
</div>
""", unsafe_allow_html=True)


def render_sidebar():
    with st.sidebar:
        st.markdown("""
<div style="display:flex; align-items:center; gap:0.6rem;
  padding:0.5rem 0 1.2rem; border-bottom:1px solid #1e2535; margin-bottom:1.2rem;">
  <div style="width:32px; height:32px; background:linear-gradient(135deg,#00e5ff22,#00e5ff44);
    border:1px solid #00e5ff66; border-radius:7px; display:flex; align-items:center;
    justify-content:center; font-size:1rem;">🔬</div>
  <div>
    <div style="font-family:'Syne',sans-serif; font-size:1rem; font-weight:800;
      color:#e8edf5; line-height:1;">ThresYm</div>
    <div style="font-family:'Space Mono',monospace; font-size:0.6rem;
      color:#4a5568; letter-spacing:0.1em;">CONTROL PANEL</div>
  </div>
</div>
""", unsafe_allow_html=True)

        st.markdown('<span class="sidebar-label">01 · Image Input</span>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Upload image",
            type=['png', 'jpg', 'jpeg'],
            help="PNG, JPG or JPEG — max 10 MB",
            label_visibility="collapsed"
        )

        if uploaded_file:
            try:
                img_preview = Image.open(uploaded_file)
                w, h = img_preview.size
                st.image(img_preview, use_column_width=True)
                st.markdown(f"""
<div style="display:flex; gap:0.4rem; flex-wrap:wrap; margin-top:0.4rem;">
  <span style="font-family:'Space Mono',monospace; font-size:0.62rem; padding:0.18rem 0.5rem;
    background:rgba(57,255,138,0.07); border:1px solid rgba(57,255,138,0.2);
    color:#39ff8a; border-radius:3px;">LOADED</span>
  <span style="font-family:'Space Mono',monospace; font-size:0.62rem; padding:0.18rem 0.5rem;
    background:#131720; border:1px solid #1e2535; color:#8892a4; border-radius:3px;">{w}x{h}px</span>
  <span style="font-family:'Space Mono',monospace; font-size:0.62rem; padding:0.18rem 0.5rem;
    background:#131720; border:1px solid #1e2535; color:#8892a4; border-radius:3px;">{uploaded_file.name.split('.')[-1].upper()}</span>
</div>
""", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Preview error: {e}")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<span class="sidebar-label">02 · Method Selection</span>', unsafe_allow_html=True)

        selected_methods = st.multiselect(
            "Methods",
            AVAILABLE_METHODS,
            default=["Simple Threshold"],
            max_selections=3,
            help="Select 1-3 thresholding methods",
            label_visibility="collapsed"
        )

        if selected_methods:
            with st.expander("Method details"):
                for method in selected_methods:
                    info = METHODS_INFO[method]
                    c = BADGE_COLORS.get(info["badge"], "#8892a4")
                    st.markdown(f"""
<div style="background:#131720; border:1px solid #1e2535; border-radius:8px;
  padding:0.8rem; margin-bottom:0.6rem; border-left:3px solid {c};">
  <div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.35rem;">
    <span style="font-size:0.9rem;">{info['icon']}</span>
    <span style="font-family:'Syne',sans-serif; font-size:0.85rem; font-weight:700; color:#e8edf5;">{method}</span>
    <span style="margin-left:auto; font-family:'Space Mono',monospace; font-size:0.58rem;
      padding:0.12rem 0.45rem; color:{c}; border:1px solid {c}44;
      background:{c}11; border-radius:3px; letter-spacing:0.1em;">{info['badge']}</span>
  </div>
  <div style="font-family:'Space Mono',monospace; font-size:0.68rem; color:#4a5568; line-height:1.5;">{info['description']}</div>
</div>
""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        all_params = {}

        if selected_methods:
            st.markdown('<span class="sidebar-label">03 · Parameters</span>', unsafe_allow_html=True)

            for method_idx, method in enumerate(selected_methods):
                info = METHODS_INFO[method]
                uid = str(method_idx)
                method_params = {}

                st.markdown(f"""
<div style="display:flex; align-items:center; gap:0.5rem; margin:0.9rem 0 0.4rem;">
  <span style="font-size:0.85rem;">{info['icon']}</span>
  <span style="font-family:'Syne',sans-serif; font-size:0.82rem; font-weight:700; color:#e8edf5;">{method}</span>
</div>
""", unsafe_allow_html=True)

                if method == "Simple Threshold":
                    method_params['threshold'] = st.slider(
                        "Threshold", 0, 255, DEFAULT_PARAMS['threshold'],
                        key=f"thresh_{uid}", help="Pixels above this value become white"
                    )

                elif method == "Otsu's Method":
                    st.markdown("""
<div style="font-family:'Space Mono',monospace; font-size:0.7rem; color:#39ff8a;
  padding:0.5rem 0.75rem; background:rgba(57,255,138,0.06);
  border:1px solid rgba(57,255,138,0.18); border-radius:6px; margin-bottom:0.5rem;">
  Auto-computes optimal threshold. No tuning needed.
</div>
""", unsafe_allow_html=True)
                    method_params['auto'] = True

                elif method == "Adaptive Mean":
                    method_params['block_size'] = st.slider("Block Size", 3, 51, DEFAULT_PARAMS['block_size'], step=2, key=f"am_b_{uid}")
                    method_params['C'] = st.slider("C Constant", -20, 20, DEFAULT_PARAMS['C'], key=f"am_c_{uid}")

                elif method == "Adaptive Gaussian":
                    method_params['block_size'] = st.slider("Block Size", 3, 51, DEFAULT_PARAMS['block_size'], step=2, key=f"ag_b_{uid}")
                    method_params['C'] = st.slider("C Constant", -20, 20, DEFAULT_PARAMS['C'], key=f"ag_c_{uid}")

                elif method == "Niblack":
                    method_params['window_size'] = st.slider("Window Size", 3, 51, DEFAULT_PARAMS['window_size'], step=2, key=f"nib_w_{uid}")
                    method_params['k'] = st.slider("k", -2.0, 2.0, -0.2, step=0.1, key=f"nib_k_{uid}")

                elif method == "Sauvola":
                    method_params['window_size'] = st.slider("Window Size", 3, 51, DEFAULT_PARAMS['window_size'], step=2, key=f"sau_w_{uid}")
                    method_params['k'] = st.slider("k", 0.1, 0.5, 0.2, step=0.05, key=f"sau_k_{uid}")
                    method_params['R'] = st.slider("R", 64, 256, DEFAULT_PARAMS['R'], key=f"sau_r_{uid}")

                all_params[method] = method_params

                if method_idx < len(selected_methods) - 1:
                    st.markdown('<hr style="border-color:#1e2535; margin:0.8rem 0;">', unsafe_allow_html=True)

        st.markdown("""
<div style="margin-top:2.5rem; padding-top:1rem; border-top:1px solid #1e2535;
  font-family:'Space Mono',monospace; font-size:0.6rem; color:#2a3548; text-align:center; line-height:1.8;">
  ThresYm v2.0 · Computer Vision Lab<br>opencv · scikit-image · streamlit
</div>
""", unsafe_allow_html=True)

    return uploaded_file, selected_methods, all_params


def render_results(original_gray, results, selected_methods):
    st.markdown("""
<div style="display:flex; align-items:center; gap:1rem; margin-bottom:1.5rem;">
  <div style="flex:1; height:1px; background:linear-gradient(90deg,#00e5ff44,transparent);"></div>
  <span style="font-family:'Space Mono',monospace; font-size:0.62rem; letter-spacing:0.2em;
    text-transform:uppercase; color:#00e5ff;">OUTPUT GRID</span>
  <div style="flex:1; height:1px; background:linear-gradient(90deg,transparent,#00e5ff44);"></div>
</div>
""", unsafe_allow_html=True)

    num_cols = len(selected_methods) + 1
    cols = st.columns(num_cols, gap="medium")
    accent_colors = ["#00e5ff", "#39ff8a", "#b66dff"]

    with cols[0]:
        st.markdown("""
<div style="background:#131720; border:1px solid #1e2535; border-radius:10px;
  padding:0.9rem; border-top:2px solid #4a5568; margin-bottom:0.6rem;">
  <div style="font-family:'Space Mono',monospace; font-size:0.62rem; letter-spacing:0.12em;
    text-transform:uppercase; color:#8892a4;">SOURCE IMAGE</div>
</div>
""", unsafe_allow_html=True)
        st.image(original_gray, use_column_width=True)
        h, w = original_gray.shape
        st.markdown(f"""
<div style="font-family:'Space Mono',monospace; font-size:0.62rem;
  color:#4a5568; text-align:center; margin-top:0.3rem;">{w} x {h} px · GRAYSCALE</div>
""", unsafe_allow_html=True)

    for idx, method in enumerate(selected_methods):
        info = METHODS_INFO[method]
        accent = accent_colors[idx % len(accent_colors)]

        with cols[idx + 1]:
            st.markdown(f"""
<div style="background:#131720; border:1px solid #1e2535; border-radius:10px;
  padding:0.9rem; border-top:2px solid {accent}; margin-bottom:0.6rem;">
  <div style="display:flex; align-items:center; gap:0.5rem;">
    <span style="font-size:0.9rem;">{info['icon']}</span>
    <span style="font-family:'Syne',sans-serif; font-size:0.82rem; font-weight:700; color:#e8edf5;">{method}</span>
    <span style="margin-left:auto; font-family:'Space Mono',monospace; font-size:0.58rem;
      padding:0.1rem 0.4rem; color:{accent}; border:1px solid {accent}44;
      background:{accent}11; border-radius:3px; letter-spacing:0.1em;">{info['badge']}</span>
  </div>
</div>
""", unsafe_allow_html=True)

            if results.get(method) is not None:
                result_img = results[method]
                st.image(result_img, use_column_width=True)

                white_px = int(np.sum(result_img == 255))
                black_px = int(np.sum(result_img == 0))
                total_px = int(result_img.size)
                white_pct = white_px / total_px * 100
                black_pct = 100 - white_pct

                c1, c2 = st.columns(2)
                with c1:
                    st.metric("White px", f"{white_px:,}")
                with c2:
                    st.metric("Black px", f"{black_px:,}")

                st.markdown(f"""
<div style="margin-top:0.6rem; background:#131720; border:1px solid #1e2535; border-radius:6px; padding:0.7rem;">
  <div style="display:flex; justify-content:space-between; margin-bottom:0.3rem;">
    <span style="font-family:'Space Mono',monospace; font-size:0.62rem; color:#4a5568;">WHITE</span>
    <span style="font-family:'Space Mono',monospace; font-size:0.62rem; color:#00e5ff;">{white_pct:.1f}%</span>
  </div>
  <div style="background:#0a0c10; border-radius:3px; height:5px; overflow:hidden;">
    <div style="width:{white_pct}%; height:100%;
      background:linear-gradient(90deg,#00e5ff,#39ff8a); border-radius:3px;"></div>
  </div>
  <div style="display:flex; justify-content:space-between; margin-top:0.3rem;">
    <span style="font-family:'Space Mono',monospace; font-size:0.62rem; color:#4a5568;">BLACK</span>
    <span style="font-family:'Space Mono',monospace; font-size:0.62rem; color:#b66dff;">{black_pct:.1f}%</span>
  </div>
</div>
""", unsafe_allow_html=True)

                safe_name = method.lower().replace(' ', '_').replace("'", '')
                img_bytes = ImageProcessor.image_to_bytes(result_img)
                st.download_button(
                    label="Download PNG",
                    data=img_bytes,
                    file_name=f"thresym_{safe_name}.png",
                    mime="image/png",
                    key=f"dl_{method}_{idx}"
                )
            else:
                st.markdown("""
<div style="padding:2rem 1rem; text-align:center; font-family:'Space Mono',monospace;
  font-size:0.7rem; color:#ff4d6d; border:1px dashed #ff4d6d44;
  border-radius:8px; margin-top:0.5rem;">Processing failed</div>
""", unsafe_allow_html=True)


def render_welcome():
    # Build all step cards HTML as one string
    step_cards = ""
    steps = [
        ("01", "#00e5ff", "Upload Image",       "PNG, JPG or JPEG — click the sidebar uploader to get started."),
        ("02", "#39ff8a", "Select Methods",     "Pick 1 to 3 thresholding algorithms to compare side-by-side."),
        ("03", "#b66dff", "Tune Parameters",    "Each method exposes independent controls for real-time tuning."),
        ("04", "#ffc107", "Export Results",     "Download any processed result as a full-resolution PNG file."),
    ]
    for num, color, title, desc in steps:
        step_cards += f"""
<div style="background:#131720; border:1px solid #1e2535; border-radius:10px;
  padding:1.1rem; border-left:3px solid {color};">
  <div style="font-family:'Space Mono',monospace; font-size:0.6rem; color:{color};
    letter-spacing:0.15em; margin-bottom:0.4rem;">STEP {num}</div>
  <div style="font-family:'Syne',sans-serif; font-size:0.92rem; font-weight:700;
    color:#e8edf5; margin-bottom:0.35rem;">{title}</div>
  <div style="font-family:'Space Mono',monospace; font-size:0.7rem; color:#4a5568; line-height:1.6;">{desc}</div>
</div>"""

    # Build method cards HTML as one string
    method_cards = ""
    for method in AVAILABLE_METHODS:
        info = METHODS_INFO[method]
        c = BADGE_COLORS.get(info["badge"], "#8892a4")
        method_cards += f"""
<div style="background:#0a0c10; border:1px solid #1e2535; border-radius:8px;
  padding:0.7rem 0.9rem; display:flex; align-items:flex-start; gap:0.6rem;">
  <span style="font-size:1rem; margin-top:2px;">{info['icon']}</span>
  <div>
    <div style="font-family:'Syne',sans-serif; font-size:0.82rem; font-weight:700;
      color:#e8edf5; margin-bottom:0.2rem;">{method}</div>
    <div style="font-family:'Space Mono',monospace; font-size:0.65rem; color:#2a3548;
      line-height:1.5; margin-bottom:0.35rem;">{info['description']}</div>
    <span style="font-family:'Space Mono',monospace; font-size:0.58rem; padding:0.1rem 0.45rem;
      color:{c}; border:1px solid {c}33; background:{c}11;
      border-radius:3px; letter-spacing:0.1em;">{info['badge']}</span>
  </div>
</div>"""

    # Render everything in a single st.markdown call
    st.markdown(f"""
<div style="background:#0f1117; border:1px solid #1e2535; border-radius:14px;
  padding:2.5rem; margin-top:0.5rem;">

  <div style="font-family:'Space Mono',monospace; font-size:0.62rem; letter-spacing:0.2em;
    text-transform:uppercase; color:#00e5ff; margin-bottom:1rem;">HOW TO USE</div>

  <div style="display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr));
    gap:1rem; margin-bottom:2rem;">
    {step_cards}
  </div>

  <div style="border-top:1px solid #1e2535; padding-top:1.5rem; margin-top:0.5rem;">
    <div style="font-family:'Space Mono',monospace; font-size:0.62rem; letter-spacing:0.2em;
      text-transform:uppercase; color:#8892a4; margin-bottom:1rem;">AVAILABLE ALGORITHMS</div>
    <div style="display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:0.7rem;">
      {method_cards}
    </div>
  </div>

  <div style="margin-top:1.8rem; background:#0a0c10; border:1px solid #1e2535;
    border-radius:8px; padding:1rem 1.2rem;">
    <div style="font-family:'Space Mono',monospace; font-size:0.6rem; letter-spacing:0.15em;
      color:#4a5568; margin-bottom:0.4rem;">INSTALL DEPENDENCIES</div>
    <code style="font-family:'Space Mono',monospace; font-size:0.75rem;
      color:#00e5ff; background:transparent;">pip install streamlit opencv-python numpy Pillow scikit-image</code>
  </div>

</div>
""", unsafe_allow_html=True)


# ==================== MAIN ====================
def main():
    st.set_page_config(
        page_title="ThresYm - Image Thresholding",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    inject_custom_css()

    if 'processor' not in st.session_state:
        st.session_state.processor = ImageProcessor()

    render_header()

    uploaded_file, selected_methods, all_params = render_sidebar()

    if uploaded_file and selected_methods:
        if st.session_state.processor.load_image(uploaded_file):
            original_gray = st.session_state.processor.get_grayscale()
            results = {}

            with st.spinner("Processing..."):
                for method in selected_methods:
                    p = all_params[method]
                    if method == "Simple Threshold":
                        result = st.session_state.processor.simple_threshold(p['threshold'])
                    elif method == "Otsu's Method":
                        result = st.session_state.processor.otsu_threshold()
                    elif method == "Adaptive Mean":
                        result = st.session_state.processor.adaptive_threshold(p['block_size'], p['C'], 'mean')
                    elif method == "Adaptive Gaussian":
                        result = st.session_state.processor.adaptive_threshold(p['block_size'], p['C'], 'gaussian')
                    elif method == "Niblack":
                        result = st.session_state.processor.niblack_threshold(p['window_size'], p['k'])
                    elif method == "Sauvola":
                        result = st.session_state.processor.sauvola_threshold(p['window_size'], p['k'], p['R'])
                    else:
                        result = None
                    results[method] = result

            render_results(original_gray, results, selected_methods)
        else:
            st.error("Failed to load image. Please try a different file.")

    elif uploaded_file and not selected_methods:
        st.markdown("""
<div style="text-align:center; padding:3rem; font-family:'Space Mono',monospace;
  font-size:0.8rem; color:#4a5568; border:1px dashed #1e2535;
  border-radius:12px; margin-top:1rem;">
  Select at least one thresholding method from the sidebar to continue.
</div>
""", unsafe_allow_html=True)

    else:
        render_welcome()


if __name__ == "__main__":
    main()