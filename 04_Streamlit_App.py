from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIGURATION & AGRONOMIC DESIGN SYSTEM
# ============================================================

st.set_page_config(
    page_title="CitrusGro Pathology | Orchard Leaf Diagnostics",
    page_icon="🍊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    /* Global Typography & Orchard Canvas */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2.5rem;
        max-width: 1280px;
    }

    /* Sidebar Agronomic Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #143627 0%, #1b4332 100%);
        color: #f4f7f5;
    }
    [data-testid="stSidebar"] * {
        color: #f4f7f5 !important;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] section {
        background-color: rgba(255, 255, 255, 0.08) !important;
        border: 1.5px dashed rgba(255, 255, 255, 0.35) !important;
        border-radius: 12px !important;
    }
    [data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div {
        background-color: rgba(255, 255, 255, 0.1) !important;
        border: 1px solid rgba(255, 255, 255, 0.25) !important;
        border-radius: 10px !important;
    }

    /* Botanical Hero Header */
    .orchard-hero {
        position: relative;
        background: linear-gradient(125deg, #133325 0%, #1f513b 52%, #2d6a4f 82%, #c85a17 145%);
        padding: 2.1rem 2.5rem;
        border-radius: 20px;
        color: #ffffff;
        margin-bottom: 1.6rem;
        border: 1px solid rgba(255, 255, 255, 0.14);
        box-shadow: 0 14px 32px rgba(19, 51, 37, 0.18);
        overflow: hidden;
    }
    .orchard-kicker {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(255, 255, 255, 0.14);
        border: 1px solid rgba(255, 255, 255, 0.22);
        padding: 0.3rem 0.85rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.9px;
        text-transform: uppercase;
        margin-bottom: 0.75rem;
    }
    .orchard-title {
        font-family: 'Fraunces', Georgia, serif;
        font-size: 2.25rem;
        font-weight: 700;
        margin: 0;
        line-height: 1.15;
        letter-spacing: -0.3px;
    }
    .orchard-subtitle {
        font-size: 1rem;
        opacity: 0.92;
        margin-top: 0.55rem;
        margin-bottom: 1.25rem;
        max-width: 780px;
        line-height: 1.5;
    }
    .pipeline-strip {
        display: flex;
        flex-wrap: wrap;
        gap: 0.75rem;
        padding-top: 0.9rem;
        border-top: 1px solid rgba(255, 255, 255, 0.16);
        font-size: 0.82rem;
    }
    .pipeline-pill {
        background: rgba(0, 0, 0, 0.22);
        padding: 0.35rem 0.8rem;
        border-radius: 8px;
        font-weight: 600;
    }

    /* Botanical Field Cards */
    .agro-card {
        background-color: var(--secondary-background-color);
        border: 1px solid rgba(45, 106, 79, 0.22);
        border-radius: 16px;
        padding: 1.3rem 1.45rem;
        height: 100%;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.04);
    }
    .agro-card-accent {
        border-left: 6px solid #e85d04;
    }
    .agro-card-green {
        border-left: 6px solid #2d6a4f;
    }
    .agro-tag {
        display: inline-block;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.7px;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        margin-bottom: 0.5rem;
    }
    .tag-nutrient {
        background: rgba(232, 93, 4, 0.14);
        color: #d95300;
    }
    .tag-disease {
        background: rgba(185, 28, 28, 0.14);
        color: #c92a2a;
    }
    .tag-healthy {
        background: rgba(45, 106, 79, 0.16);
        color: #2b8a3e;
    }
    .agro-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.85px;
        opacity: 0.72;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    .agro-value {
        font-family: 'Fraunces', Georgia, serif;
        font-size: 1.55rem;
        font-weight: 700;
        margin: 0 0 0.45rem 0;
    }

    /* Prescription Box */
    .rx-box {
        background: linear-gradient(135deg, rgba(45, 106, 79, 0.10) 0%, rgba(232, 93, 4, 0.08) 100%);
        border: 1px solid rgba(45, 106, 79, 0.28);
        border-radius: 16px;
        padding: 1.35rem 1.6rem;
        margin-top: 1rem;
    }
    .rx-title {
        font-weight: 700;
        font-size: 1rem;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Custom Pigment Spectrum Bars */
    .spectrum-row {
        margin-bottom: 0.95rem;
    }
    .spectrum-header {
        display: flex;
        justify-content: space-between;
        font-size: 0.88rem;
        font-weight: 600;
        margin-bottom: 0.3rem;
    }
    .spectrum-track {
        width: 100%;
        height: 12px;
        background: rgba(128, 128, 128, 0.18);
        border-radius: 999px;
        overflow: hidden;
    }
    .spectrum-fill {
        height: 100%;
        border-radius: 999px;
    }

    /* Image Frame Styling */
    [data-testid="stImage"] img {
        border-radius: 14px;
        border: 2px solid rgba(45, 106, 79, 0.28);
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.08);
    }

    /* Primary Button Polish */
    .stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #1b4332 0%, #2d6a4f 100%) !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.7rem 1.2rem !important;
        font-weight: 700 !important;
        font-size: 1.02rem !important;
        letter-spacing: 0.3px;
        box-shadow: 0 6px 18px rgba(27, 67, 50, 0.28);
    }
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(90deg, #143627 0%, #1b4332 100%) !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# PROJECT PATHS, AGRONOMIC KNOWLEDGE BASE & MODEL
# ============================================================

MODEL_DIR = Path("model_files")
MODEL_PATH = MODEL_DIR / "svm_citrus_model.pkl"
FEATURE_COLUMNS_PATH = MODEL_DIR / "feature_columns.pkl"

CLASS_DISPLAY = {
    "Healthy": "Healthy Citrus Foliage",
    "Fe": "Iron Deficiency (Fe)",
    "Mg": "Magnesium Deficiency (Mg)",
    "Mn": "Manganese Deficiency (Mn)",
    "N": "Nitrogen Deficiency (N)",
    "Zn": "Zinc Deficiency (Zn)",
    "HLB": "HLB Disease (Citrus Greening)",
    "Greasy_spot": "Greasy Spot Fungal Disease"
}

AGRO_GUIDE = {
    "Healthy": {
        "icon": "🌿",
        "category": "Optimal Vigor",
        "tag_class": "tag-healthy",
        "foliage": "All Canopy Zones",
        "symptoms": "Rich, uniform chlorophyll pigmentation with intact cuticle and zero interveinal mottling.",
        "action": "Maintain balanced seasonal NPK fertigation and routine orchard scouting."
    },
    "Fe": {
        "icon": "🍋",
        "category": "Micronutrient Deficiency",
        "tag_class": "tag-nutrient",
        "foliage": "Young Terminal Leaves",
        "symptoms": "Distinctive green vein network against bright yellow interveinal tissue (lime-induced chlorosis).",
        "action": "Apply chelated iron (Fe-EDDHA) soil drench for alkaline soils or foliar iron sulfate spray."
    },
    "Mg": {
        "icon": "🍂",
        "category": "Macronutrient Deficiency",
        "tag_class": "tag-nutrient",
        "foliage": "Older Mature Leaves",
        "symptoms": "Characteristic inverted V-shaped green base with bronzed-yellow margins and leaf tip.",
        "action": "Apply foliar magnesium nitrate/sulfate or amend soil with dolomitic limestone."
    },
    "Mn": {
        "icon": "🌾",
        "category": "Micronutrient Deficiency",
        "tag_class": "tag-nutrient",
        "foliage": "Young & Mid-Shoot Leaves",
        "symptoms": "Diffuse pale-green interveinal bands bordered by blurred dark-green midrib and lateral veins.",
        "action": "Apply foliar manganese sulfate spray during spring or summer flush expansion."
    },
    "N": {
        "icon": "🌻",
        "category": "Macronutrient Deficiency",
        "tag_class": "tag-nutrient",
        "foliage": "Older Basal Leaves First",
        "symptoms": "Uniform pale yellow-green fading across both veins and blade without localized spots.",
        "action": "Split-apply nitrogen fertilizer (urea or ammonium nitrate) and verify root-zone irrigation."
    },
    "Zn": {
        "icon": "🌱",
        "category": "Micronutrient Deficiency",
        "tag_class": "tag-nutrient",
        "foliage": "New Growth Flushes",
        "symptoms": "Sharp creamy-yellow interveinal blotches with narrow, pointed leaves ('little-leaf' mottle).",
        "action": "Spray zinc sulfate or chelated zinc onto emerging spring flush leaves."
    },
    "HLB": {
        "icon": "🚨",
        "category": "Bacterial Vascular Disease",
        "tag_class": "tag-disease",
        "foliage": "Asymmetric Canopy Sectors",
        "symptoms": "Asymmetrical blotchy yellow mottling crossing leaf veins; corky, enlarged midribs.",
        "action": "Control Asian citrus psyllid vectors immediately, confirm via PCR assay, and isolate infected trees."
    },
    "Greasy_spot": {
        "icon": "🍄",
        "category": "Fungal Foliar Disease",
        "tag_class": "tag-disease",
        "foliage": "Mature Lower-Canopy Leaves",
        "symptoms": "Yellow-brown upper-surface blisters developing into dark, oily brown-black necrotic lesions underneath.",
        "action": "Apply copper fungicide or horticultural spray oil during summer flush and clear fallen leaf litter."
    }
}

CLASS_OPTIONS = list(CLASS_DISPLAY.keys())


@st.cache_resource
def load_model_bundle():
    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
    return model, feature_columns


if not MODEL_PATH.exists() or not FEATURE_COLUMNS_PATH.exists():
    st.error("The SVM model files were not found. Run the ML implementation notebook first.")
    st.stop()

try:
    model, FEATURE_COLUMNS = load_model_bundle()
except Exception as error:
    st.error("The SVM model could not be loaded.")
    st.code(str(error))
    st.stop()


# ============================================================
# IMAGE READING AND FEATURE EXTRACTION (100% UNCHANGED)
# ============================================================

def read_rgb_image(image_bytes):
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image_bgr is None:
        return None
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def read_leaf_region(region_bytes):
    region_array = np.frombuffer(region_bytes, dtype=np.uint8)
    return cv2.imdecode(region_array, cv2.IMREAD_GRAYSCALE)


def extract_features(image_rgb, leaf_region):
    """Extract the same 17 whole-leaf features used during training."""
    rgb_pixels = image_rgb[leaf_region].astype(np.float32)

    R = rgb_pixels[:, 0]
    G = rgb_pixels[:, 1]
    B = rgb_pixels[:, 2]

    hsv_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    hsv_pixels = hsv_image[leaf_region].astype(np.float32)

    H = hsv_pixels[:, 0]
    S = hsv_pixels[:, 1]
    V = hsv_pixels[:, 2]

    green_pixels = (H >= 35) & (H <= 90) & (S >= 40)
    yellow_pixels = (H >= 20) & (H < 35) & (S >= 40)
    dark_pixels = V < 100

    gray_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    gray_pixels = gray_image[leaf_region].astype(np.float32)

    edges = cv2.Canny(gray_image, 100, 200)

    feature_values = {
        "mean_R": np.mean(R),
        "mean_G": np.mean(G),
        "mean_B": np.mean(B),
        "std_R": np.std(R),
        "std_G": np.std(G),
        "std_B": np.std(B),
        "mean_H": np.mean(H),
        "mean_S": np.mean(S),
        "mean_V": np.mean(V),
        "std_H": np.std(H),
        "std_S": np.std(S),
        "std_V": np.std(V),
        "green_ratio": np.mean(green_pixels),
        "yellow_ratio": np.mean(yellow_pixels),
        "dark_ratio": np.mean(dark_pixels),
        "gray_std": np.std(gray_pixels),
        "edge_density": np.mean(edges[leaf_region] > 0)
    }

    return pd.DataFrame(
        [[feature_values[column] for column in FEATURE_COLUMNS]],
        columns=FEATURE_COLUMNS
    )


# ============================================================
# ORCHARD HERO HEADER
# ============================================================

st.markdown(
    """
    <div class="orchard-hero">
        <div class="orchard-kicker">🌱 Precision Citrus Agronomy • Foliar Spectral Vision</div>
        <h1 class="orchard-title">🍊 CitrusGro Leaf Pathology & Nutrient Analyzer</h1>
        <p class="orchard-subtitle">
            Non-destructive citrus orchard diagnostics powered by Support Vector Machine (SVM) classification
            across RGB color channels, HSV chlorophyll-chlorosis ratios, and Canny lesion texture descriptors.
        </p>
        <div class="pipeline-strip">
            <span class="pipeline-pill">📸 1. Leaf Specimen Capture</span>
            <span class="pipeline-pill">✂️ 2. Whole-Leaf Mask Isolation</span>
            <span class="pipeline-pill">🧬 3. 17-Feature Spectral Extraction</span>
            <span class="pipeline-pill">🩺 4. SVM Agronomic Diagnosis</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR FIELD STATION CONTROLS
# ============================================================

with st.sidebar:
    st.markdown("## 🌿 Orchard Field Station")
    st.caption("Upload a citrus leaf specimen to inspect pigment health and classify pathology.")

    st.markdown("### 1. Specimen Upload")
    image_file = st.file_uploader(
        "Leaf RGB Photograph *",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear photograph of an orange leaf blade."
    )

    region_file = st.file_uploader(
        "Foliar Segmentation Mask (Optional)",
        type=["png", "jpg", "jpeg"],
        help="Binary mask separating leaf lamina from soil or sky background."
    )

    st.markdown("---")
    st.markdown("### 2. Benchmark Verification")
    actual_class = st.selectbox(
        "Select Actual Class (Dataset Label)",
        CLASS_OPTIONS,
        format_func=lambda value: CLASS_DISPLAY[value],
        help="Select the known ground-truth label from your test dataset to validate SVM accuracy."
    )

    st.markdown("---")
    st.markdown("### ⚙️ Diagnostic Engine Specs")
    st.markdown(
        f"""
        - **Classifier:** Support Vector Machine (SVM)
        - **Spectral Space:** RGB + HSV Color Spaces
        - **Morphology:** Canny Edge & Gray Texture
        - **Input Vector:** `{len(FEATURE_COLUMNS)}` Extracted Features
        - **Pathology Scope:** `8` Citrus Classes
        """
    )


# ============================================================
# MAIN AGRONOMIC WORKSPACE
# ============================================================

if image_file is None:
    st.markdown("### 📖 Citrus Orchard Foliar Field Guide")
    st.info("👈 Upload a leaf specimen in the **Orchard Field Station** sidebar to run a live diagnosis, or browse the 8 supported citrus conditions below.")

    grid_cols = st.columns(4, gap="medium")
    for idx, key in enumerate(CLASS_OPTIONS):
        info = AGRO_GUIDE[key]
        with grid_cols[idx % 4]:
            st.markdown(
                f"""
                <div class="agro-card" style="margin-bottom: 1rem;">
                    <span class="agro-tag {info['tag_class']}">{info['category']}</span>
                    <div style="font-size: 1.5rem; margin-bottom: 0.25rem;">{info['icon']}</div>
                    <div style="font-weight: 700; font-size: 1.02rem; margin-bottom: 0.35rem;">{CLASS_DISPLAY[key]}</div>
                    <div style="font-size: 0.78rem; font-weight: 600; opacity: 0.7; margin-bottom: 0.45rem;">
                        📍 {info['foliage']}
                    </div>
                    <div style="font-size: 0.84rem; opacity: 0.88; line-height: 1.45;">
                        {info['symptoms']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
else:
    image_rgb = read_rgb_image(image_file.getvalue())
    if image_rgb is None:
        st.error("The leaf image could not be read.")
        st.stop()

    image_label = image_file.name

    if region_file is not None:
        leaf_region_image = read_leaf_region(region_file.getvalue())
        if leaf_region_image is None:
            st.error("The leaf-region file could not be read.")
            st.stop()

        if leaf_region_image.shape != image_rgb.shape[:2]:
            leaf_region_image = cv2.resize(
                leaf_region_image,
                (image_rgb.shape[1], image_rgb.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )

        leaf_region = leaf_region_image > 127
        if not np.any(leaf_region):
            st.error("The selected region file does not contain leaf pixels.")
            st.stop()
    else:
        leaf_region = np.ones(image_rgb.shape[:2], dtype=bool)

    # Specimen Microscopy / Visual Inspection
    st.markdown("### 🔬 Specimen Lamina Inspection")
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.image(
            image_rgb,
            caption=f"Raw Orchard Specimen • {image_label} ({image_rgb.shape[1]}×{image_rgb.shape[0]} px)",
            use_container_width=True
        )

    with col2:
        if region_file is not None:
            preview = image_rgb.copy()
            preview[~leaf_region] = 0
            coverage_pct = (np.sum(leaf_region) / leaf_region.size) * 100
            st.image(
                preview,
                caption=f"Isolated Leaf Lamina • {coverage_pct:.1f}% Active Foliar Surface",
                use_container_width=True
            )
        else:
            st.markdown(
                """
                <div class="agro-card agro-card-green" style="display:flex; flex-direction:column; justify-content:center; min-height:270px;">
                    <div style="font-size: 2.1rem; margin-bottom: 0.5rem;">🌿</div>
                    <div style="font-weight: 700; font-size: 1.15rem; margin-bottom: 0.45rem;">Full-Frame Spectral Scan Active</div>
                    <div style="opacity: 0.88; font-size: 0.94rem; line-height: 1.55;">
                        No binary leaf mask was supplied, so all pixels in the frame are sampled for pigment and edge analysis.
                        <br><br>
                        <strong>Agronomist Tip:</strong> Uploading a matching leaf-region mask removes soil/background interference and sharpens chlorosis ratios.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("<br>", unsafe_allow_html=True)
    predict_clicked = st.button("🩺 Analyze Leaf & Diagnose Condition", type="primary", use_container_width=True)

    if predict_clicked:
        with st.spinner("Sampling foliar chlorophyll, chlorosis ratios, and lesion edges..."):
            feature_row = extract_features(image_rgb, leaf_region)
            predicted_class = model.predict(feature_row)[0]

        st.divider()
        st.markdown("### 📋 Agronomic Diagnostic Report")

        # Ground Truth Verification Banner
        if predicted_class == actual_class:
            st.success("✅ **DIAGNOSIS VERIFIED** — The SVM prediction matches the selected ground-truth specimen label.")
        else:
            st.error("⚠️ **DIAGNOSTIC VARIANCE** — The SVM prediction differs from the selected ground-truth specimen label.")

        act_info = AGRO_GUIDE.get(actual_class, {"icon": "🏷️", "category": "Reference", "tag_class": "tag-healthy"})
        pred_info = AGRO_GUIDE.get(predicted_class, {"icon": "🤖", "category": "Diagnosis", "tag_class": "tag-nutrient", "foliage": "", "symptoms": "", "action": ""})

        c_actual, c_pred = st.columns(2, gap="large")

        with c_actual:
            st.markdown(
                f"""
                <div class="agro-card agro-card-green">
                    <span class="agro-tag {act_info['tag_class']}">{act_info['category']}</span>
                    <div class="agro-label">Reference Ground Truth (Actual Class)</div>
                    <div class="agro-value">{act_info['icon']} {CLASS_DISPLAY.get(actual_class, actual_class)}</div>
                    <div style="font-size: 0.86rem; opacity: 0.8;">
                        Typical Canopy Zone: <strong>{act_info.get('foliage', 'N/A')}</strong>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with c_pred:
            st.markdown(
                f"""
                <div class="agro-card agro-card-accent">
                    <span class="agro-tag {pred_info['tag_class']}">{pred_info['category']}</span>
                    <div class="agro-label">SVM Model Diagnosis (Predicted Class)</div>
                    <div class="agro-value">{pred_info['icon']} {CLASS_DISPLAY.get(predicted_class, predicted_class)}</div>
                    <div style="font-size: 0.88rem; opacity: 0.88; line-height: 1.45;">
                        {pred_info['symptoms']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Agronomist Field Recommendation Box
        st.markdown(
            f"""
            <div class="rx-box">
                <div class="rx-title">🚜 Recommended Orchard Management & Treatment</div>
                <div style="font-size: 0.95rem; line-height: 1.55;">
                    <strong>Target Zone:</strong> {pred_info['foliage']}  | 
                    <strong>Action Plan:</strong> {pred_info['action']}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # Detailed Tabs for Foliar Pigment Analysis & Raw Feature Vector
        tab_pigment, tab_features = st.tabs(["🌱 Foliar Pigment & Lesion Spectrum", "🧬 17-Feature Descriptor Vector"])

        with tab_pigment:
            green_val = float(feature_row["green_ratio"].iloc[0])
            yellow_val = float(feature_row["yellow_ratio"].iloc[0])
            dark_val = float(feature_row["dark_ratio"].iloc[0])
            edge_val = float(feature_row["edge_density"].iloc[0])

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Chlorophyll (Green)", f"{green_val * 100:.1f}%")
            m2.metric("Chlorosis (Yellow)", f"{yellow_val * 100:.1f}%")
            m3.metric("Necrotic / Dark Area", f"{dark_val * 100:.1f}%")
            m4.metric("Lesion Edge Density", f"{edge_val:.4f}")

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                f"""
                <div class="spectrum-row">
                    <div class="spectrum-header">
                        <span>🌿 Healthy Chlorophyll Pigmentation (H: 35–90)</span>
                        <span>{green_val * 100:.1f}%</span>
                    </div>
                    <div class="spectrum-track">
                        <div class="spectrum-fill" style="width: {min(green_val * 100, 100):.1f}%; background: #2d6a4f;"></div>
                    </div>
                </div>

                <div class="spectrum-row">
                    <div class="spectrum-header">
                        <span>🍋 Chlorotic Yellowing / Deficiency Indicator (H: 20–35)</span>
                        <span>{yellow_val * 100:.1f}%</span>
                    </div>
                    <div class="spectrum-track">
                        <div class="spectrum-fill" style="width: {min(yellow_val * 100, 100):.1f}%; background: #f59e0b;"></div>
                    </div>
                </div>

                <div class="spectrum-row">
                    <div class="spectrum-header">
                        <span>🍂 Dark Lesions / Necrotic Tissue (V < 100)</span>
                        <span>{dark_val * 100:.1f}%</span>
                    </div>
                    <div class="spectrum-track">
                        <div class="spectrum-fill" style="width: {min(dark_val * 100, 100):.1f}%; background: #7f1d1d;"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with tab_features:
            st.caption("Exact 17 numerical descriptors passed to the trained Support Vector Machine:")
            st.dataframe(
                feature_row.round(4),
                use_container_width=True,
                hide_index=True
            )

st.markdown("---")
st.caption("🍊 CitrusGro Leaf Pathology System • Precision Agriculture & Machine Learning (SVM)")