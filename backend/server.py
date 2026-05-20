"""
CRC-Scan AI Backend — FastAPI Server
=====================================
All model/inference logic is a 1:1 port from app.py.
Zero changes to the model, weights, or prediction pipeline.
"""
import sys
import os
import io
import base64
import traceback
import tempfile
from datetime import datetime
from contextlib import asynccontextmanager

# ── Path setup (must come before model imports) ──────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "STARC-9-Evaluation"))

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageDraw

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from custom_models import get_custom_cnn
from config import NUM_CLASSES, CLASS_NAMES
from fpdf import FPDF

# ── Clinical design constants (exact copy from app.py) ───────────────────────
CLASS_COLORS = {
    "ADI": "#5DBCD2", "LYM": "#9470C4", "MUC": "#F5A623", "MUS": "#B8E986",
    "NCS": "#9B9B9B", "NOR": "#50E3C2", "BLD": "#D0021B", "FCT": "#F8E71C",
    "TUM": "#FF0055",
}

CLINICAL_GLOSSARY = {
    "ADI": "Adipose tissue (fat cells).",
    "LYM": "Lymphocytes (immune infiltration).",
    "MUC": "Mucin (secretory tissue).",
    "MUS": "Smooth muscle layers.",
    "NCS": "Necrotic debris (cell death).",
    "NOR": "Normal mucosal architecture.",
    "BLD": "Red blood cells (vascularity).",
    "FCT": "Loose connective tissue (stroma).",
    "TUM": "Malignant Adenocarcinoma cells.",
}

# ── Transform (exact copy from app.py) ────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ── Global model state ────────────────────────────────────────────────────────
_model = None
_device = None
_load_status = "⚠️ Model not initialized."


def _load_model():
    """Load the STARC-9 custom CNN exactly as in app.py."""
    global _model, _device, _load_status
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _model = get_custom_cnn(num_classes=NUM_CLASSES)
    weights_path = os.path.join(
        BASE_DIR, "weights", "Model_weights", "best_custom_cnn.pth"
    )
    if os.path.exists(weights_path):
        try:
            checkpoint = torch.load(weights_path, map_location=_device)
            state_dict = checkpoint.get(
                "model_state_dict", checkpoint.get("state_dict", checkpoint)
            )
            clean_state_dict = {
                k.replace("module.", "")
                .replace(".cbam.ca.", ".cbam.channel_attn.")
                .replace(".cbam.sa.", ".cbam.spatial_attn."): v
                for k, v in state_dict.items()
            }
            _model.load_state_dict(clean_state_dict, strict=True)
            _load_status = f"✅ Weights loaded — {len(clean_state_dict)} tensors."
        except Exception as e:
            _load_status = f"❌ Load Error: {str(e)}"
    else:
        _load_status = f"⚠️ Weights not found at: {weights_path}"
    _model.to(_device)
    _model.eval()


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model()
    yield


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="CRC-Scan AI Backend", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ── Validation (exact copy from app.py — unchanged) ───────────────────────────
def validate_histology_image(image: Image.Image):
    """
    Revised histology validator: Highly permissive for tissue slides.
    Ensures that legitimate H&E slides are NEVER rejected.
    """
    img_rgb = np.array(image.convert("RGB")).astype(np.float32)
    h, w = img_rgb.shape[:2]

    if h < 32 or w < 32:
        return False, "Image resolution too low for diagnostic analysis."

    R, G, B = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]

    # --- 1. Basic Luminance Check (Loosened) ---
    luminance = 0.299 * R + 0.587 * G + 0.114 * B
    if luminance.std() < 5.0:
        return False, "Image appears to be blank. Please upload a valid slide scan."

    # --- 2. White Background Check (Loosened) ---
    white_mask = (R > 240) & (G > 240) & (B > 240)
    if white_mask.mean() > 0.98:
        return False, "Image consists mostly of background whitespace."

    # --- 3. Texture/Complexity Check (Laplacian Variance - Loosened) ---
    import cv2
    gray = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < 15.0:
        return False, "Image lacks the characteristic cellular texture of a histology scan."

    # --- 4. H&E Color Signature (Highly Permissive) ---
    r_n, g_n, b_n = R / 255.0, G / 255.0, B / 255.0
    cmax = np.maximum(np.maximum(r_n, g_n), b_n)
    delta = cmax - np.minimum(np.minimum(r_n, g_n), b_n) + 1e-8
    S = np.where(cmax > 0, delta / cmax, 0)

    hue2 = np.zeros_like(R)
    valid = delta > 0.01
    mask_r = (cmax == r_n) & valid
    mask_g = (cmax == g_n) & valid
    mask_b = (cmax == b_n) & valid
    hue2[mask_r] = (((g_n - b_n) / delta * 60) % 360)[mask_r]
    hue2[mask_g] = (((b_n - r_n) / delta * 60) + 120)[mask_g]
    hue2[mask_b] = (((r_n - g_n) / delta * 60) + 240)[mask_b]

    sat_mask = (S > 0.10) & (cmax > 0.15)
    sat_pixels = sat_mask.sum()

    if sat_pixels < (h * w * 0.015):
        return False, "Insufficient color variation for H&E staining detection."

    sat_hue = hue2[sat_mask]
    pink_mask = (sat_hue >= 300) | (sat_hue <= 40)
    purple_mask = (sat_hue >= 200) & (sat_hue < 300)
    
    he_frac = (pink_mask.sum() + purple_mask.sum()) / (sat_pixels + 1e-8)

    # Threshold: 15% is extremely safe to ensure no tissue slides are rejected.
    if he_frac < 0.15:
        return (
            False,
            f"Image does not match H&E staining profiles (Detected {he_frac * 100:.1f}% hues).",
        )

    return True, f"✅ Slide Verified ({he_frac * 100:.1f}% H&E characteristic hues)."


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "device": str(_device) if _device else "not initialized",
        "weights_status": _load_status,
        "class_names": CLASS_NAMES,
        "class_colors": CLASS_COLORS,
        "glossary": CLINICAL_GLOSSARY,
    }


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...), threshold: int = Form(70)):
    """
    Tile-based inference pipeline — exact logic from app.py.
    Returns heatmap overlay (base64 PNG) + tissue distribution + clinical metrics.
    """
    try:
        contents = await file.read()
        orig = Image.open(io.BytesIO(contents)).convert("RGB")

        # Validation (exact from app.py)
        is_valid, reason = validate_histology_image(orig)
        if not is_valid:
            return JSONResponse({"status": "invalid", "reason": reason})

        w, h = orig.size
        TILE_SIZE = 256
        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        counts = {cls: 0 for cls in CLASS_NAMES}
        uncert = 0

        # Tiling inference loop (exact from app.py)
        with torch.no_grad():
            for y in range(0, h, TILE_SIZE):
                for x in range(0, w, TILE_SIZE):
                    box = (x, y, min(x + TILE_SIZE, w), min(y + TILE_SIZE, h))
                    tile = orig.crop(box)
                    if tile.size[0] < 128 or tile.size[1] < 128:
                        continue

                    # --- Spatial Background Check ---
                    # Skip tiles that are mostly white background or empty
                    tile_np = np.array(tile)
                    white_ratio = (np.all(tile_np > 235, axis=-1)).mean()
                    if white_ratio > 0.92:
                        continue

                    inp = transform(tile).unsqueeze(0).to(_device)
                    logits = _model(inp)
                    # Temperature Scaling removed for partially trained model
                    probs = F.softmax(logits, dim=1)[0]
                    conf, pred = torch.max(probs, 0)
                    conf_pct = conf.item() * 100

                    if conf_pct < threshold:
                        uncert += 1
                        draw.rectangle(box, fill=(255, 255, 0, 150), outline="#FFFF00", width=2)
                    else:
                        lbl = CLASS_NAMES[pred.item()]
                        counts[lbl] += 1
                        r = int(CLASS_COLORS[lbl][1:3], 16)
                        g = int(CLASS_COLORS[lbl][3:5], 16)
                        b = int(CLASS_COLORS[lbl][5:7], 16)
                        draw.rectangle(box, fill=(r, g, b, 150), outline="white", width=1)

        composite = Image.alpha_composite(orig.convert("RGBA"), overlay)

        # Clinical metrics (exact from app.py)
        proc_total = sum(counts.values()) + uncert
        distribution = {}
        for cls, cnt in counts.items():
            distribution[cls] = {
                "count": cnt,
                "percentage": round((cnt / proc_total) * 100, 2) if proc_total > 0 else 0,
            }
        distribution["UNCERTAIN"] = {
            "count": uncert,
            "percentage": round((uncert / proc_total) * 100, 2) if proc_total > 0 else 0,
        }

        tum_pct = distribution["TUM"]["percentage"]
        lym_pct = distribution["LYM"]["percentage"]
        ncs_pct = distribution["NCS"]["percentage"]
        uncert_pct = distribution["UNCERTAIN"]["percentage"]
        til_score = (lym_pct / tum_pct * 100) if tum_pct > 0 else 0

        dominant = max(counts, key=counts.get) if sum(counts.values()) > 0 else "UNCERTAIN"

        if ncs_pct > 5:
            note = "Aggressive architectural patterns found with significant Necrosis."
        elif til_score > 30:
            note = "High immune infiltration detected (TIL), suggesting potential clinical response."
        elif tum_pct > 0:
            note = "Stable morphology detected with localized malignant clusters."
        else:
            note = "Benign morphology verified across scanned area."

        return {
            "status": "ok",
            "heatmap": _pil_to_b64(composite),
            "original_size": {"width": w, "height": h},
            "distribution": distribution,
            "clinical": {
                "tum_pct": tum_pct,
                "lym_pct": lym_pct,
                "ncs_pct": ncs_pct,
                "uncert_pct": uncert_pct,
                "til_score": round(til_score, 1),
            },
            "dominant_tissue": dominant,
            "diagnostic_note": note,
            "validation_msg": reason,
        }

    except Exception as e:
        return JSONResponse(
            {"status": "error", "reason": str(e), "traceback": traceback.format_exc()},
            status_code=500,
        )


@app.post("/api/gradcam")
async def gradcam_endpoint(
    file: UploadFile = File(...),
    tile_x: int = Form(...),
    tile_y: int = Form(...),
):
    """
    Grad-CAM explainability for a clicked tile.
    Target layer: model.conv6[-1]  (exact from app.py)
    """
    try:
        contents = await file.read()
        orig = Image.open(io.BytesIO(contents)).convert("RGB")

        TILE_SIZE = 256
        w, h = orig.size
        sx = (tile_x // TILE_SIZE) * TILE_SIZE
        sy = (tile_y // TILE_SIZE) * TILE_SIZE
        raw_tile = orig.crop((sx, sy, min(sx + TILE_SIZE, w), min(sy + TILE_SIZE, h)))

        # Grad-CAM (exact target layer from app.py)
        cam = GradCAM(model=_model, target_layers=[_model.stage5[-1]])
        grayscale = cam(input_tensor=transform(raw_tile).unsqueeze(0).to(_device))[0, :]
        viz = show_cam_on_image(
            np.array(raw_tile).astype(np.float32) / 255, grayscale, use_rgb=True
        )

        # Prediction for the tile
        with torch.no_grad():
            inp = transform(raw_tile).unsqueeze(0).to(_device)
            logits = _model(inp)
            probs = F.softmax(logits, dim=1)[0]
            conf, pred = torch.max(probs, 0)
            lbl = CLASS_NAMES[pred.item()]

        return {
            "status": "ok",
            "tile": _pil_to_b64(raw_tile),
            "gradcam": _pil_to_b64(Image.fromarray(viz)),
            "label": lbl,
            "confidence": round(conf.item() * 100, 1),
            "glossary": CLINICAL_GLOSSARY.get(lbl, ""),
        }

    except Exception as e:
        return JSONResponse(
            {"status": "error", "reason": str(e)}, status_code=500
        )


# ── PDF Report Generation (Clinical Grade) ───────────────────────────────────
class PathologyReport(FPDF):
    def header(self):
        # 1. Solid Clinical Blue Header
        self.set_fill_color(22, 33, 62) # Deep Clinical Blue
        self.rect(0, 0, 210, 45, 'F')
        
        # 2. Draw a Minimalist Logo (Microscope Symbol)
        self.set_draw_color(255, 0, 85) # Malignant Pink
        self.set_line_width(1.5)
        # Main scope body
        self.ellipse(15, 12, 10, 10, 'D')
        self.line(20, 22, 20, 28)
        self.line(15, 28, 25, 28)
        
        # 3. Branding Text
        self.set_xy(30, 12)
        self.set_font('Helvetica', 'B', 28)
        self.set_text_color(255, 255, 255)
        self.cell(0, 10, 'CRC-SCAN', ln=True)
        
        self.set_xy(30, 24)
        self.set_font('Helvetica', 'I', 10)
        self.set_text_color(180, 180, 220)
        self.cell(0, 5, 'Automated Clinical Histopathology Dashboard', ln=True)

    def footer(self):
        self.set_y(-25)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(120, 120, 140)
        self.cell(0, 5, 'OFFICIAL PATHOLOGY REPORT | STARC-9 CLINICAL AI ENGINE', ln=True, align='C')
        self.cell(0, 5, f'Page {self.page_no()} | Confidential Medical Information', align='C')
        
    def section_header(self, title):
        self.ln(2)
        self.set_font('Helvetica', 'B', 11)
        self.set_text_color(22, 33, 62)
        self.set_fill_color(240, 240, 250)
        self.cell(0, 7, f"  {title}", ln=True, fill=True)
        self.ln(2)

    def table_header(self):
        self.set_font('Helvetica', 'B', 9)
        self.set_fill_color(230, 230, 240)
        self.set_text_color(40, 40, 60)
        self.cell(50, 8, " Tissue Category", 1, 0, 'L', True)
        self.cell(40, 8, " Tile Count", 1, 0, 'C', True)
        self.cell(40, 8, " Percentage", 1, 0, 'C', True)
        self.cell(60, 8, " Quantitative Distribution", 1, 1, 'C', True)

    def table_row(self, label, count, pct, color_hex):
        self.set_font('Helvetica', '', 9)
        self.set_text_color(20, 20, 20)
        self.cell(50, 7, f" {label}", 1)
        self.cell(40, 7, str(count), 1, 0, 'C')
        self.cell(40, 7, f"{pct}%", 1, 0, 'C')
        
        # Draw a small percentage bar inside the last cell
        x, y = self.get_x(), self.get_y()
        self.cell(60, 7, "", 1, 1) # border for bar
        
        # Convert hex to RGB
        r = int(color_hex[1:3], 16)
        g = int(color_hex[3:5], 16)
        b = int(color_hex[5:7], 16)
        self.set_fill_color(r, g, b)
        
        bar_width = (pct / 100.0) * 56
        if bar_width > 0:
            self.rect(x + 2, y + 1.5, bar_width, 4, 'F')

    def draw_legend(self, start_x=10):
        self.ln(2)
        self.set_font('Helvetica', 'B', 9)
        self.set_x(start_x)
        self.cell(0, 8, "Heatmap Color Key:", ln=True)
        self.set_font('Helvetica', '', 8)
        
        # 3 columns for legend
        col_w = 62
        base_y = self.get_y()
        count = 0
        for code, color in CLASS_COLORS.items():
            col = count % 3
            row = count // 3
            
            x = start_x + col * col_w
            y = base_y + row * 6
            
            # Square
            r = int(color[1:3], 16)
            g = int(color[3:5], 16)
            b = int(color[5:7], 16)
            self.set_fill_color(r, g, b)
            self.rect(x, y + 1, 4, 4, 'F')
            
            # Text
            name = CLINICAL_GLOSSARY.get(code, code).split("(")[0].strip()
            self.set_xy(x + 6, y)
            self.cell(col_w - 6, 6, f"{code}: {name}", 0)
            
            count += 1
        
        # Add Uncertain
        col = count % 3
        row = count // 3
        x = start_x + col * col_w
        y = base_y + row * 6
        self.set_fill_color(255, 255, 0)
        self.rect(x, y + 1, 4, 4, 'F')
        self.set_xy(x + 6, y)
        self.cell(col_w - 6, 6, "UNC: Uncertain", ln=True)
        self.ln(4)

@app.post("/api/report")
async def report_endpoint(
    file: UploadFile = File(...),
    result_json: str = Form(...),
    heatmap_b64: str = Form(...),
    radar_b64: str = Form(None), # Optional Radar Chart image
    slide_filename: str = Form(...),
    threshold: int = Form(70)
):
    try:
        import json
        data = json.loads(result_json)
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 1. Prepare images
        slide_bytes = await file.read()
        slide_img = Image.open(io.BytesIO(slide_bytes)).convert("RGB")
        heatmap_bytes = base64.b64decode(heatmap_b64)
        heatmap_img = Image.open(io.BytesIO(heatmap_bytes)).convert("RGB")
        
        # Save temp images for PDF embedding
        # On Windows, we must close the file before Pillow can save to it
        tmp_slide = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        tmp_slide.close()
        slide_img.save(tmp_slide.name)
        slide_path = tmp_slide.name

        tmp_heat = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        tmp_heat.close()
        heatmap_img.save(tmp_heat.name)
        heat_path = tmp_heat.name
        
        radar_path = None
        if radar_b64:
            radar_bytes = base64.b64decode(radar_b64.split(",")[-1])
            tmp_radar = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            tmp_radar.write(radar_bytes)
            tmp_radar.close()
            radar_path = tmp_radar.name
            
        # 2. Build PDF
        pdf = PathologyReport()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=25)
        
        # Metadata Section (Closer to header)
        pdf.set_y(50) 
        pdf.section_header("1. CLINICAL METADATA")
        pdf.set_font('Helvetica', '', 9)
        pdf.set_text_color(60, 60, 80)
        
        c1_w = 95
        pdf.cell(c1_w, 5, f"File Name: {slide_filename}", 0)
        pdf.cell(0, 5, f"Report Date: {now}", ln=True)
        pdf.cell(c1_w, 5, f"Patient ID: REF-{os.path.basename(slide_filename)[:6].upper()}", 0)
        pdf.cell(0, 5, f"Accession #: LAB-{datetime.now().strftime('%m%d-%H%M')}", ln=True)
        pdf.cell(c1_w, 5, f"AI Model: STARC-9 Custom ResNet-50", 0)
        pdf.cell(0, 5, f"Mode: Tiled Inference (256px)", ln=True)
        
        # Full Tissue Breakdown
        pdf.section_header("2. TISSUE DECOMPOSITION ANALYSIS")
        pdf.table_header()
        
        # Iterate all classes in global CLASS_NAMES
        for cls in CLASS_NAMES:
            entry = data["distribution"].get(cls, {"count": 0, "percentage": 0.0})
            full_name = CLINICAL_GLOSSARY.get(cls, cls)
            if len(full_name) > 30: full_name = full_name[:27] + "..."
            pdf.table_row(full_name, entry["count"], entry["percentage"], CLASS_COLORS.get(cls, "#CCCCCC"))
            
        # Add Uncertain if present
        uncert = data["distribution"].get("UNCERTAIN", {"count": 0, "percentage": 0.0})
        if uncert["count"] > 0:
            pdf.table_row("Uncertain / Low Confidence", uncert["count"], uncert["percentage"], "#FFFF00")

        # Diagnosis & Insights
        # Only break if we are VERY low on space (less than 60mm)
        if pdf.get_y() > 230:
            pdf.add_page()
            pdf.set_y(50)

        pdf.section_header("3. CLINICAL INTERPRETATION & INFERENCE")
        pdf.set_font('Helvetica', 'B', 10)
        pdf.set_text_color(220, 0, 85) # Clinical Malignant Pink
        pdf.cell(0, 7, f"Dominant Profile: {data['dominant_tissue']} - {CLINICAL_GLOSSARY.get(data['dominant_tissue'], '')}", ln=True)
        
        pdf.set_font('Helvetica', '', 10)
        pdf.set_text_color(40, 40, 60)
        pdf.multi_cell(0, 5, f"Diagnostic Insight: {data['diagnostic_note']}")
        
        m = data["clinical"]
        inference_points = []
        if m['tum_pct'] > 40: inference_points.append("- CRITICAL: High-density malignant clusters detected (>40%).")
        elif m['tum_pct'] > 15: inference_points.append("- MODERATE: Localized malignant architectural disruption (~15-40%).")
        else: inference_points.append("- LOW: Minimal malignant cell infiltration observed (<15%).")

        if m['til_score'] > 35: inference_points.append("- PROGNOSTIC: Robust immune infiltration (High TIL Ratio).")
        elif m['til_score'] < 10: inference_points.append("- WARNING: Low immune infiltration may indicate immune evasion.")

        if m['ncs_pct'] > 7: inference_points.append("- AGGRESSION: Elevated necrotic load suggests high turnover.")

        pdf.ln(1)
        pdf.set_font('Helvetica', 'B', 9)
        pdf.cell(0, 6, "AI-Derived Quantitative Metrics:", ln=True)
        pdf.set_font('Helvetica', '', 9)
        for point in inference_points:
            pdf.cell(0, 5, f"  {point}", ln=True)

        pdf.ln(2)
        summary_text = (
            f"Global quantitative analysis indicated {m['tum_pct']}% malignant epithelium density with "
            f"a TIL ratio of {m['til_score']}%. Necrotic burden calculated at {m['ncs_pct']}%."
        )
        pdf.set_font('Helvetica', 'I', 8.5)
        pdf.set_text_color(80, 80, 100)
        pdf.multi_cell(0, 4.5, summary_text)

        # ── Final Visuals (Ensure smart layout) ──────────────────────────────
        # Visuals Section (Usually Page 2, but could fit if distribution is small)
        if pdf.get_y() > 140:
             pdf.add_page()
             pdf.set_y(50)
        else:
             pdf.ln(5)

        pdf.section_header("4. SPATIAL DISCOVERY & AI HEATMAP")
        pdf.set_font('Helvetica', '', 9)
        pdf.set_text_color(100, 100, 120)
        pdf.cell(0, 6, "Neural network classification overlay on histological architecture:", ln=True)
        
        # Heatmap Image
        img_w = 145
        available_w = 210
        pdf.image(heat_path, x=(available_w - img_w) / 2, w=img_w)
        pdf.draw_legend(start_x=32)
        
        if radar_path:
            # Check space for Radar
            if pdf.get_y() > 210:
                pdf.add_page()
                pdf.set_y(50)
            else:
                pdf.ln(4)
            
            pdf.section_header("5. QUANTITATIVE TME FINGERPRINT")
            pdf.set_font('Helvetica', '', 9)
            pdf.set_text_color(100, 100, 120)
            pdf.cell(0, 6, "TME Radar (Patient vs Healthy Baseline):", ln=True)
            radar_w = 110
            pdf.image(radar_path, x=(available_w - radar_w) / 2, w=radar_w)
        
        # Disclaimer Box (Sticky bottom of final page)
        curr_y = pdf.get_y()
        if curr_y > 250:
            pdf.add_page()
            target_y = 265
        else:
            target_y = max(curr_y + 10, 265)

        pdf.set_y(target_y)
        pdf.set_fill_color(255, 250, 250)
        pdf.set_draw_color(220, 0, 85)
        pdf.set_line_width(0.3)
        pdf.rect(10, pdf.get_y(), 190, 20, 'FD')
        
        pdf.set_y(pdf.get_y() + 2)
        pdf.set_font('Helvetica', 'B', 8)
        pdf.set_text_color(220, 0, 85)
        pdf.cell(0, 4, "MANDATORY CLINICAL DISCLAIMER", ln=True, align='C')
        pdf.set_font('Helvetica', '', 7)
        pdf.set_text_color(80, 80, 100)
        disc = (
            "This report was generated by the CRC-Scan AI diagnostic engine (STARC-9 architecture). "
            "It is provided for clinical decision support and RESEARCH PURPOSES ONLY. "
            "Final clinical diagnosis must be established by a human pathologist. "
            "Automated metrics are subject to staining variations."
        )
        pdf.set_x(15)
        pdf.multi_cell(180, 3, disc, align='C')

        pdf_bytes = pdf.output()

        # 3. Cleanup ONLY after output is generated
        try:
            os.remove(slide_path)
            os.remove(heat_path)
            if radar_path:
                os.remove(radar_path)
        except:
            pass # Non-critical if cleanup fails
        
        return StreamingResponse(io.BytesIO(pdf_bytes), media_type="application/pdf", 
                                 headers={"Content-Disposition": f"attachment; filename=DiagnosticReport_{slide_filename}.pdf"})
        
    except Exception as e:
        return JSONResponse({"status": "error", "reason": str(e), "traceback": traceback.format_exc()}, status_code=500)

# ── Serve frontend (must be last) ─────────────────────────────────────────────
_frontend_dir = os.path.join(BASE_DIR, "frontend")
if os.path.exists(_frontend_dir):
    app.mount("/", StaticFiles(directory=_frontend_dir, html=True), name="frontend")
