import streamlit as st
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageDraw
import sys
import os
import pandas as pd
import altair as alt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from streamlit_image_coordinates import streamlit_image_coordinates

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="CRC-Scan | Clinical AI Workspace",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Integrate STARC-9 repo modules safely
sys.path.append(os.path.abspath('STARC-9-Evaluation'))
try:
    from custom_models import get_custom_cnn
    from config import NUM_CLASSES, CLASS_NAMES
except ImportError:
    st.error("Failed to load STARC-9 internals. Make sure you're running this from the repository root.")
    st.stop()

# --- CLINICAL DESIGN SYSTEM ---
CLASS_COLORS = {
    "ADI": "#5DBCD2", "LYM": "#9470C4", "MUC": "#F5A623", "MUS": "#B8E986",
    "NCS": "#9B9B9B", "NOR": "#50E3C2", "BLD": "#D0021B", "FCT": "#F8E71C",
    "TUM": "#FF0055"
}
UNCERTAINTY_COLOR = "#FFFF00"

CLINICAL_GLOSSARY = {
    "ADI": "Adipose tissue (fat cells).",
    "LYM": "Lymphocytes (immune infiltration).",
    "MUC": "Mucin (secretory tissue).",
    "MUS": "Smooth muscle layers.",
    "NCS": "Necrotic debris (cell death).",
    "NOR": "Normal mucosal architecture.",
    "BLD": "Red blood cells (vascularity).",
    "FCT": "Loose connective tissue (stroma).",
    "TUM": "Malignant Adenocarcinoma cells."
}

# --- MODEL BACKEND (UNCHANGED LOGIC) ---
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- TISSUE VALIDATION LAYER ---
def validate_histology_image(image: Image.Image):
    """
    Heuristic validator that checks whether an uploaded image is likely
    an H&E-stained histopathology slide.

    H&E slides have a very specific color fingerprint:
      - Hematoxylin stains nuclei BLUE/PURPLE  (Hue ~ 200–300°)
      - Eosin stains cytoplasm PINK/RED         (Hue ~ 300–360° and 0–30°)
    
    Returns: (is_valid: bool, reason: str)
    """
    img_rgb = np.array(image.convert("RGB")).astype(np.float32)
    h, w = img_rgb.shape[:2]
    
    # --- Guard: minimum size ---
    if h < 64 or w < 64:
        return False, "Image is too small to be a diagnostic slide (minimum 64×64 px)."

    R, G, B = img_rgb[:,:,0], img_rgb[:,:,1], img_rgb[:,:,2]
    
    # --- Guard: near-uniform / blank image ---
    luminance = (0.299*R + 0.587*G + 0.114*B)
    if luminance.std() < 8.0:
        return False, "Image appears to be blank or near-uniform. Please upload a real histopathology scan."

    # --- Guard: mostly white background (scanner background) ---
    white_mask = (R > 230) & (G > 230) & (B > 230)
    white_fraction = white_mask.mean()
    if white_fraction > 0.95:
        return False, "Image is almost entirely white. Please ensure the slide has tissue content."

    # --- Core H&E Color Check ---
    # Convert to HSV for hue analysis (manual, fast)
    r_n, g_n, b_n = R/255.0, G/255.0, B/255.0
    cmax = np.maximum(np.maximum(r_n, g_n), b_n)
    cmin = np.minimum(np.minimum(r_n, g_n), b_n)
    delta = cmax - cmin + 1e-8

    # Saturation
    S = np.where(cmax > 0, delta / cmax, 0)
    
    # Hue (degrees 0–360)
    hue = np.zeros_like(cmax)
    mask_r = (cmax == r_n) & (delta > 0)
    mask_g = (cmax == g_n) & (delta > 0)
    mask_b = (cmax == b_n) & (delta > 0)
    hue[mask_r] = (60 * ((g_n[mask_r] - b_n[mask_r]) / delta[mask_r]) % 6)
    hue[mask_g] = (60 * ((b_n[mask_g] - r_n[mask_g]) / delta[mask_g]) + 2)
    hue[mask_b] = (60 * ((r_n[mask_b] - g_n[mask_b]) / delta[mask_b]) + 4)
    hue = hue * 60  # scale to 0-360 where cmax-based was 0-6
    # Re-normalize properly
    hue2 = np.zeros_like(R)
    valid = delta > 0.01
    hue2[mask_r & valid] = (((g_n - b_n) / delta * 60) % 360)[mask_r & valid]
    hue2[mask_g & valid] = (((b_n - r_n) / delta * 60) + 120)[mask_g & valid]
    hue2[mask_b & valid] = (((r_n - g_n) / delta * 60) + 240)[mask_b & valid]

    # Saturated pixels only (ignore white/gray background)
    sat_mask = (S > 0.10) & (cmax > 0.15)
    sat_pixels = sat_mask.sum()

    if sat_pixels < (h * w * 0.015):
        return False, "Insufficient color variation for H&E staining detection."

    # H&E hue bands:
    #   Eosin (pink):       hue2 ~  300–400 (extended)
    #   Hematoxylin (blue): hue2 ~ 200–300
    sat_hue = hue2[sat_mask]
    pink_mask  = (sat_hue >= 300) | (sat_hue <= 40)
    purple_mask = (sat_hue >= 200) & (sat_hue < 300)

    he_frac = (pink_mask.sum() + purple_mask.sum()) / (sat_pixels + 1e-8)

    # Threshold: 15% is extremely permissive to ensure NO legitimate tissue slides are rejected.
    if he_frac < 0.15:
        return (
            False,
            f"Image does not match H&E staining profiles (Detected {he_frac*100:.1f}% hues)."
        )

    return True, f"✅ Slide Verified ({he_frac*100:.1f}% H&E characteristic hues)."

@st.cache_resource
def load_clinical_model():
    """Loads inference model with the exact trained weights from STARC-9 repo."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_custom_cnn(num_classes=NUM_CLASSES)
    weights_path = os.path.abspath(os.path.join('weights', 'Model_weights', 'best_custom_cnn.pth'))
    load_status = "⚠️ No trained weights found."
    if os.path.exists(weights_path):
        try:
            checkpoint = torch.load(weights_path, map_location=device)
            state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
            clean_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(clean_state_dict, strict=True)
            load_status = f"✅ Weights loaded: {len(clean_state_dict)} tensors detected."
        except Exception as e:
            load_status = f"❌ Load Error: {str(e)}"
    model.to(device); model.eval()
    return model, device, load_status

if 'analysis_done' not in st.session_state: st.session_state.analysis_done = False
if 'final_composite' not in st.session_state: st.session_state.final_composite = None
if 'original_image' not in st.session_state: st.session_state.original_image = None
if 'df_counts' not in st.session_state: st.session_state.df_counts = None
if 'last_uploaded_name' not in st.session_state: st.session_state.last_uploaded_name = None

# --- UI HEADER ---
st.markdown("""
<div style='background-color: #1E1E1E; padding: 20px; border-radius: 10px; border-left: 5px solid #FF0055; margin-bottom: 25px;'>
    <h1 style='margin:0; color: white;'>🔬 CRC-Scan Clinical AI Workspace</h1>
    <p style='color: #888; margin: 5px 0 0 0;'>Diagnostic Histopathology Analysis & Tissue Decomposition</p>
</div>
""", unsafe_allow_html=True)

model, device, status = load_clinical_model()

# --- MAIN LAYOUT ---
col_sidebar, col_main = st.columns([1, 2.5], gap="large")

with col_sidebar:
    st.subheader("📁 Input Data")
    uploaded_file = st.file_uploader("Upload slide scan", type=['jpg', 'jpeg', 'png'])
    
    # Auto-reset if a new file is uploaded, and run tissue validation
    if uploaded_file and uploaded_file.name != st.session_state.last_uploaded_name:
        st.session_state.analysis_done = False
        st.session_state.last_uploaded_name = uploaded_file.name
        raw_img = Image.open(uploaded_file).convert("RGB")
        is_valid, reason = validate_histology_image(raw_img)
        st.session_state.original_image = raw_img if is_valid else None
        st.session_state.validation_ok = is_valid
        st.session_state.validation_msg = reason
    
    st.info(status)
    
    st.divider()
    st.subheader("⚙️ Analysis Params")
    confidence_threshold = st.slider("Uncertainty Threshold (%)", 0, 100, 70)
    st.caption("Lower threshold promotes AI decisiveness; higher threshold increases human review flags.")
    
    if uploaded_file:
        if st.session_state.get("validation_ok"):
            st.success(st.session_state.validation_msg)
            st.image(st.session_state.original_image, caption="Slide Preview", width=300)
        else:
            st.error(f"❌ **Invalid Image Rejected**\n\n{st.session_state.get('validation_msg', 'Unknown error.')}")

with col_main:
    if not st.session_state.analysis_done:
        if uploaded_file:
            if not st.session_state.get("validation_ok", False):
                st.markdown("""
                <div style='padding: 30px; border-radius: 12px; background: #2a1a1a; border: 2px solid #D0021B; text-align: center;'>
                    <h2 style='color: #FF4444;'>⛔ Non-Tissue Image Detected</h2>
                    <p style='color: #ccc; font-size: 1.1em;'>This image does not match the expected H&E histopathology profile.<br>
                    The clinical AI will not run analysis on non-tissue images to prevent false diagnostics.</p>
                    <p style='color: #888;'>Please upload a colorectal biopsy scan with pink/purple H&E staining.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("Analysis pending. Click **Execute AI Pipeline** below to begin.")
                if st.button("🚀 Execute AI Pipeline", type="primary", use_container_width=True):
                    # Analysis Logic
                    orig = st.session_state.original_image
                    w, h = orig.size
                    TILE_SIZE = 256
                    overlay = Image.new('RGBA', (w, h), (0, 0, 0, 0))
                    draw = ImageDraw.Draw(overlay)
                    counts = {cls: 0 for cls in CLASS_NAMES}
                    uncert = 0
                    
                    prog = st.progress(0)
                    y_steps = range(0, h, TILE_SIZE)
                    x_steps = range(0, w, TILE_SIZE)
                    total = len(y_steps) * len(x_steps)
                    curr = 0
                    
                    with torch.no_grad():
                        for y in y_steps:
                            for x in x_steps:
                                box = (x, y, min(x+TILE_SIZE, w), min(y+TILE_SIZE, h))
                                tile = orig.crop(box)
                                if tile.size[0] < 128 or tile.size[1] < 128: continue
                                
                                inp = transform(tile).unsqueeze(0).to(device)
                                probs = F.softmax(model(inp), dim=1)[0]
                                conf, pred = torch.max(probs, 0)
                                conf_pct = conf.item() * 100
                                
                                if conf_pct < confidence_threshold:
                                    uncert += 1
                                    draw.rectangle(box, fill=(255, 255, 0, 150), outline="#FFFF00", width=2)
                                else:
                                    lbl = CLASS_NAMES[pred.item()]
                                    counts[lbl] += 1
                                    r, g, b = int(CLASS_COLORS[lbl][1:3], 16), int(CLASS_COLORS[lbl][3:5], 16), int(CLASS_COLORS[lbl][5:7], 16)
                                    draw.rectangle(box, fill=(r, g, b, 150), outline="white", width=1)
                                
                                curr += 1
                                prog.progress(curr/total)
                    
                    st.session_state.final_composite = Image.alpha_composite(orig.convert('RGBA'), overlay)
                    
                    proc_total = sum(counts.values()) + uncert
                    df_data = [{"Tissue": k, "Count": v, "Percentage": (v/proc_total)*100} for k, v in counts.items()]
                    df_data.append({"Tissue": "UNCERTAIN", "Count": uncert, "Percentage": (uncert/proc_total)*100})
                    st.session_state.df_counts = pd.DataFrame(df_data)
                    st.session_state.analysis_done = True
                    st.rerun()
        else:
            st.info("Please upload a histopathology slide on the left to begin analysis.")

    else:
        # ANALYSIS OUTPUT VIEW
        tab1, tab2 = st.columns([2, 1])
        
        with tab1:
            st.markdown("### 🗺️ AI Heatmap Discovery")
            value = streamlit_image_coordinates(st.session_state.final_composite, key="heatmap_click", width=800)
            
            if value:
                x, y = value["x"], value["y"]
                TILE_SIZE = 256
                sx, sy = (x // TILE_SIZE) * TILE_SIZE, (y // TILE_SIZE) * TILE_SIZE
                raw_tile = st.session_state.original_image.crop((sx, sy, sx+TILE_SIZE, sy+TILE_SIZE))
                
                st.markdown("#### ⚡ Click-based Explainability (Grad-CAM)")
                tc1, tc2 = st.columns(2)
                cam = GradCAM(model=model, target_layers=[model.stage5[-1].conv2])
                grayscale = cam(input_tensor=transform(raw_tile).unsqueeze(0).to(device))[0, :]
                viz = show_cam_on_image(np.array(raw_tile).astype(np.float32)/255, grayscale, use_rgb=True)
                tc1.image(raw_tile, caption="Selected Tile", use_container_width=True)
                tc2.image(viz, caption="Activation Map", use_container_width=True)

        with tab2:
            st.markdown("### 📊 Distribution Analysis")
            df = st.session_state.df_counts
            
            # Donut Chart
            donut = alt.Chart(df).mark_arc(innerRadius=50).encode(
                theta="Percentage:Q",
                color=alt.Color("Tissue:N", scale=alt.Scale(domain=list(CLASS_COLORS.keys())+["UNCERTAIN"], 
                                                           range=list(CLASS_COLORS.values())+["#FFFF00"])),
                tooltip=["Tissue", "Percentage"]
            ).properties(height=300)
            st.altair_chart(donut, use_container_width=True)
            
        # Bar Chart
        bar = alt.Chart(df).mark_bar().encode(
            x=alt.X("Percentage:Q", title=None),
            y=alt.Y("Tissue:N", sort="-x", title=None),
            color=alt.Color("Tissue:N", legend=None),
            tooltip=["Tissue", "Percentage"]
        ).properties(height=280)
        st.altair_chart(bar, use_container_width=True)

        st.divider()
        st.markdown("### 🧠 Clinical Insights & Heuristics")
        
        # Calculate Heuristics
        df_idx = df.set_index("Tissue")
        tum_pct = df_idx.loc["TUM", "Percentage"]
        lym_pct = df_idx.loc["LYM", "Percentage"]
        ncs_pct = df_idx.loc["NCS", "Percentage"]
        uncert_pct = df_idx.loc["UNCERTAIN", "Percentage"]
        
        # TIL Score (Tumor Infiltrating Lymphocytes)
        til_score = (lym_pct / tum_pct * 100) if tum_pct > 0 else 0
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Tumor Content", f"{tum_pct:.1f}%", delta="Malignant" if tum_pct > 20 else "Low Risk", delta_color="inverse")
        c2.metric("TIL Ratio (Immune)", f"{til_score:.1f}%", help="Ratio of Lymphocytes found relative to Tumor mass.")
        c3.metric("Necrotic Load", f"{ncs_pct:.1f}%", help="Indicator of tumor aggression/starvation.")
        c4.metric("MD Assistance", f"{uncert_pct:.1f}%", delta="Human Review", delta_color="off")
            
        st.markdown("""
        **Summary Findings:**
        - **Tissue Dominance:** The slide is primarily composed of **{}**.
        - **Diagnostic Note:** {}
        """.format(
            df.sort_values("Percentage", ascending=False).iloc[0]["Tissue"],
            "Aggressive architectural patterns found with significant Necrosis." if ncs_pct > 5 else 
            "High immune infiltration detected (TIL), suggesting potential clinical response." if til_score > 30 else
            "Stable morphology detected with localized malignant clusters." if tum_pct > 0 else
            "Benign morphology verified across scanned area."
        ))

        if st.button("🗑️ Reset Workspace"):
            st.session_state.analysis_done = False
            st.rerun()
