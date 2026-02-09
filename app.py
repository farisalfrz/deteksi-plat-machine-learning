import streamlit as st
import joblib
import numpy as np
from skimage import transform, feature, color, filters, morphology, exposure, util
from skimage.measure import label, regionprops
from PIL import Image, ImageOps

# ==========================================
# 1. KONFIGURASI & MODEL
# ==========================================
st.set_page_config(page_title="Sistem Deteksi Plat Nomor", layout="wide")

@st.cache_resource
def load_model():
    try:
        return joblib.load('model_svm_plat.pkl')
    except:
        return None

svm_model = load_model()

# ==========================================
# 2. FILTER WARNA (SATURATION CHECK)
# ==========================================
def is_sticker_color(roi_rgb):
    """
    Mendeteksi stiker berdasarkan saturasi warna.
    """
    if roi_rgb.ndim < 3: return False 
    
    hsv = color.rgb2hsv(roi_rgb)
    saturation = hsv[:, :, 1]
    avg_sat = np.mean(saturation)
    
    return avg_sat > 0.4

# ==========================================
# 3. DETEKSI (AUTO-ROTATE & POSITION LOGIC)
# ==========================================

def get_candidates(binary_img, source_name, img_w, img_h, original_rgb=None):
    label_img = label(binary_img)
    candidates = []
    
    for region in regionprops(label_img):
        # 1. Filter Area
        if region.area < 3000 or region.area > 300000: continue
        
        minr, minc, maxr, maxc = region.bbox
        width = maxc - minc
        height = maxr - minr
        if height == 0: continue
        
        ratio = width / height
        is_vertical = False

        # LOGIKA ORIENTASI
        # Horizontal: 2.0 - 8.0
        if 2.0 < ratio < 8.0:
            is_vertical = False
        # Vertical (Miring): 0.15 - 0.6
        elif 0.15 < ratio < 0.6:
            is_vertical = True
        else:
            continue 
            
        # 2. Filter Solidity
        if region.solidity > 0.35:
            score = region.area * region.solidity
            
            # Penalties Posisi
            center_y = (minr + maxr) / 2
            if center_y > img_h * 0.85: score *= 0.1 # Bawah (Lantai)
            if center_y < img_h * 0.10: score *= 0.5 # Atas (Langit)
            
            # Hukuman Warna
            if original_rgb is not None:
                roi_rgb = original_rgb[minr:maxr, minc:maxc]
                if is_sticker_color(roi_rgb):
                    score *= 0.2
            
            candidates.append({
                'region': region,
                'score': score,
                'source': source_name,
                'bbox': region.bbox,
                'is_vertical': is_vertical
            })
    return candidates

def robust_detect_plate(image_rgb):
    # Standardisasi
    h_orig, w_orig = image_rgb.shape[:2]
    new_w = 800
    new_h = int(h_orig * (new_w / w_orig))
    image_resized = transform.resize(image_rgb, (new_h, new_w), anti_aliasing=True)
    image_ubyte = util.img_as_ubyte(image_resized)
    
    gray = color.rgb2gray(image_ubyte)
    gray = exposure.equalize_adapthist(gray, clip_limit=0.03)
    
    candidates = []

    # STRATEGI 1: BRIGHT SPOT
    binary_bright = gray > 0.65 
    binary_bright = morphology.opening(binary_bright, morphology.square(3))
    binary_bright = morphology.closing(binary_bright, morphology.rectangle(5, 20))
    candidates += get_candidates(binary_bright, "Bright Spot", new_w, new_h, image_ubyte)

    # STRATEGI 2: TEXTURE EDGE
    sobel_v = filters.sobel(gray)
    thresh = filters.threshold_otsu(sobel_v)
    binary_edge = sobel_v > thresh
    binary_texture = morphology.closing(binary_edge, morphology.square(30))
    binary_texture = morphology.opening(binary_texture, morphology.square(5))
    candidates += get_candidates(binary_texture, "Texture Edge", new_w, new_h, image_ubyte)
    
    # STRATEGI 3: DARK BOX
    gray_inv = util.invert(gray)
    binary_dark = gray_inv > 0.6
    binary_dark = morphology.closing(binary_dark, morphology.square(30))
    candidates += get_candidates(binary_dark, "Dark Box", new_w, new_h, image_ubyte)

    if len(candidates) > 0:
        candidates = sorted(candidates, key=lambda x: x['score'], reverse=True)
        best = candidates[0]
        
        minr, minc, maxr, maxc = best['bbox']
        pad = 10
        minr = max(0, minr - pad); minc = max(0, minc - pad)
        maxr = min(new_h, maxr + pad); maxc = min(new_w, maxc + pad)
        
        plate_crop = image_ubyte[minr:maxr, minc:maxc]
        
        # --- FIX ERROR FLOAT VALUE ---
        if best['is_vertical']:
            # Rotate menghasilkan float. Kita harus kembalikan ke uint8 dengan aman.
            plate_crop = transform.rotate(plate_crop, angle=-90, resize=True, preserve_range=True)
            # Paksa jadi integer 0-255
            plate_crop = np.clip(plate_crop, 0, 255).astype(np.uint8)
            best['source'] += " (Rotated)"

        return plate_crop, best['source']

    return None, "Gagal"

# ==========================================
# 4. SEGMENTASI (NO SIDE CROP)
# ==========================================

def get_char_candidates(binary_img, target_h, target_w):
    cleaned = morphology.opening(binary_img, morphology.square(1))
    label_img = label(cleaned)
    regions = []
    
    for region in regionprops(label_img):
        minr, minc, maxr, maxc = region.bbox
        h_char, w_char = maxr-minr, maxc-minc
        
        if h_char > 0.25 * target_h and h_char < 0.95 * target_h:
            if w_char > 0.01 * target_w and w_char < 0.6 * target_w:
                if region.area > 20: 
                    regions.append(region)
    return len(regions), regions, cleaned

def segment_and_predict(plate_img, model):
    # Pastikan input sudah format integer 0-255
    if plate_img.dtype != np.uint8:
        plate_img = np.clip(plate_img, 0, 255).astype(np.uint8)

    # Potong frame atas/bawah (tetap), tapi JANGAN potong kiri kanan
    h_raw, w_raw = plate_img.shape[:2]
    crop_margin_y = int(h_raw * 0.05)
    
    if crop_margin_y > 0:
        plate_img = plate_img[crop_margin_y:-crop_margin_y, :]

    # Resize Standard
    h, w = plate_img.shape[:2]
    target_h = 75
    target_w = int(w * (target_h / h))
    plate_res = transform.resize(plate_img, (target_h, target_w), anti_aliasing=True)
    
    if plate_res.ndim == 3: gray = color.rgb2gray(plate_res)
    else: gray = plate_res
    
    p2, p98 = np.percentile(gray, (2, 98))
    gray = exposure.rescale_intensity(gray, in_range=(p2, p98))

    contestants = []

    # Sauvola
    t_sauvola = filters.threshold_sauvola(gray, window_size=45)
    b_sauvola = gray > t_sauvola
    contestants.append((b_sauvola, "Sauvola Norm"))
    contestants.append((~b_sauvola, "Sauvola Inv"))

    # Otsu
    t_otsu = filters.threshold_otsu(gray)
    b_otsu = gray > t_otsu
    contestants.append((b_otsu, "Otsu Norm"))
    contestants.append((~b_otsu, "Otsu Inv"))
    
    # Adaptive
    b_adapt = gray > filters.threshold_local(gray, 45, offset=0.05)
    contestants.append((b_adapt, "Adapt Norm"))
    contestants.append((~b_adapt, "Adapt Inv"))

    best_score = -1
    best_regions = []
    best_binary = b_otsu
    best_method = "None"

    for binary, name in contestants:
        count, regions, cleaned = get_char_candidates(binary, target_h, target_w)
        score = 0
        if 3 <= count <= 9: score = count * 10 
        else: score = count
        
        if score > best_score:
            best_score = score
            best_regions = regions
            best_binary = cleaned
            best_method = name

    best_regions = sorted(best_regions, key=lambda x: x.bbox[1])
    
    if len(best_regions) == 0:
        return best_binary, f"Gagal Segmentasi ({best_method})"

    text = ""
    for region in best_regions:
        minr, minc, maxr, maxc = region.bbox
        char_crop = best_binary[minr:maxr, minc:maxc]
        char_crop = char_crop.astype(float)
        
        char_resized = transform.resize(char_crop, (32, 32), anti_aliasing=True)
        feat = feature.hog(char_resized, orientations=9, pixels_per_cell=(8, 8), 
                           cells_per_block=(2, 2), visualize=False).reshape(1, -1)
        pred = model.predict(feat)[0]
        text += pred

    return best_binary, text

# ==========================================
# 5. UI STREAMLIT
# ==========================================
st.title("🛡️ ALPR System (Fixed & Stabilized)")
st.write("Versi Final: Perbaikan error float & rotasi gambar miring.")

uploaded_file = st.file_uploader("Upload Gambar", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Auto EXIF Rotate
    pil_image = Image.open(uploaded_file)
    pil_image = ImageOps.exif_transpose(pil_image) 
    pil_image = pil_image.convert('RGB')
    
    image_np = np.array(pil_image)
    
    st.image(image_np, caption="Input (Auto-Oriented)", use_container_width=True)
    
    if st.button("MULAI SCAN"):
        if svm_model is None: st.error("Model tidak ditemukan!")
        else:
            with st.spinner('Memproses...'):
                plate, method = robust_detect_plate(image_np)
                
                if plate is not None:
                    biner, text = segment_and_predict(plate, svm_model)
                    
                    st.divider()
                    st.success(f"✅ Lokasi: **{method}**")
                    
                    c1, c2, c3 = st.columns(3)
                    with c1: st.image(plate, caption="Crop Plat", use_container_width=True)
                    with c2: st.image(biner, caption="Segmentasi", use_container_width=True, clamp=True)
                    with c3: 
                        st.markdown("### Hasil Baca:")
                        st.title(f"`{text}`")
                else:
                    st.error("Gagal mendeteksi plat.")