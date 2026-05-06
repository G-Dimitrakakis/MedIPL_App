"""
processing.py
-------------
«Γέφυρα» μεταξύ Streamlit UI και των υπαρχόντων modules.

Κάθε συνάρτηση εδώ:
  1. Δέχεται παραμέτρους από τα Streamlit widgets
  2. Καλεί ΑΚΡΙΒΩΣ τις ίδιες συναρτήσεις που καλεί το MedIPL_Main.py
  3. Επιστρέφει numpy array για εμφάνιση
"""

import numpy as np

import modules.MedIPL_windowing as win
import modules.MedIPL_histograms as hist
import modules.MedIPL_spatial_filters as sf
import modules.MedIPL_frequency_filters as ff
import modules.MedIPL_reconstruction as rec
import streamlit as st


IMAGE_DEPTH = 255
TONES       = 256

def _hash_array(arr: np.ndarray) -> bytes:
    return arr.tobytes()

_cache_kwargs = {"hash_funcs": {np.ndarray: _hash_array}}
# ══════════════════════════════════════════════════════════════════════════════
# WINDOWING
# ══════════════════════════════════════════════════════════════════════════════

def apply_simple_window(im: np.ndarray, wc: int, ww: int) -> np.ndarray:
    return win.simpleWindow(im, wc, ww, IMAGE_DEPTH, TONES)


import numpy as np

def recommend_window(image, anatomy="🤖 Auto (Smart Detection)"):
    """
    Υπολογίζει το βέλτιστο παράθυρο (WC, WW) ανάλογα με τον τύπο ιστού (Anatomy).
    """
    # Δημιουργούμε μάσκα για να αγνοήσουμε τον εντελώς μαύρο "αέρα" εκτός σώματος
    threshold = np.max(image) * 0.02
    foreground = image[image > threshold]
    
    if foreground.size == 0:
        foreground = image.flatten()
        
    # Ορισμός των εκατοστημορίων ανάλογα με τον ιστό (Histogram Targeting)
    if anatomy == "🫁 Lungs (Dark details)":
        # Εστιάζουμε στα σκοτεινά (από 0% έως 80% της πληροφορίας)
        p_low = np.percentile(foreground, 0)
        p_high = np.percentile(foreground, 80)
        
    elif anatomy == "🧠 Brain (Narrow mid-tones)":
        # Στενό παράθυρο, εστιάζουμε αυστηρά στα μεσαία γκρι
        p_low = np.percentile(foreground, 40)
        p_high = np.percentile(foreground, 95)
        
    elif anatomy == "🦴 Bones (Highlights)":
        # Εστιάζουμε μόνο στα πολύ φωτεινά σημεία
        p_low = np.percentile(foreground, 70)
        p_high = np.percentile(foreground, 100)
        
    elif anatomy == "🫀 Soft Tissue (Abdomen/Chest)":
        # Ευρύ παράθυρο για γενική απεικόνιση κοιλίας
        p_low = np.percentile(foreground, 10)
        p_high = np.percentile(foreground, 90)
        
    else: 
        # "🤖 Auto (Smart Detection)" - Η γενική ασφαλής προσέγγιση
        p_low = np.percentile(foreground, 2)
        p_high = np.percentile(foreground, 98)

    # Υπολογισμός Width και Center
    ww = p_high - p_low
    wc = p_low + (ww / 2)
    
    # Failsafe
    if ww < 1:
        ww, wc = 255, 127
        
    return int(wc), int(ww)


def apply_advanced_window(im: np.ndarray, method: str, **kwargs) -> np.ndarray:
    """
    Εφαρμόζει Advanced Windowing technique.
    Αντιστοιχία με MedIPL_Main adv_choice 0-11.
    """
    if method == "Simple Display":
        return win.simpleDisplay(im, IMAGE_DEPTH, TONES)

    elif method == "Optimal Display":
        return win.optimalDisplay(im, TONES)

    elif method == "Broken Window":
        return win.brokenWindow(im, IMAGE_DEPTH, TONES,
                                kwargs.get("gray_val", 128),
                                kwargs.get("im_val", 128))

    elif method == "Double Window":
        return win.doubleWindow(im,
                                kwargs.get("ww1", 80),  kwargs.get("wl1", 60),
                                kwargs.get("ww2", 120), kwargs.get("wl2", 180),
                                IMAGE_DEPTH, TONES)

    elif method in ("Inverse", "Logarithmic", "Inv. Logarithmic",
                    "Power", "Sine", "Exponential"):
        choice_map = {
            "Inverse": 5, "Logarithmic": 6, "Inv. Logarithmic": 7,
            "Power": 8,   "Sine": 9,        "Exponential": 10,
        }
        lut = win.formPlotFunction(TONES, choice_map[method])
        mn, mx = np.min(im), np.max(im)
        if mx == mn:
            return im.copy()
        im_norm = np.round((TONES - 1) * (im - mn) / (mx - mn))
        N, M = im.shape
        im_out = np.zeros(im.shape, dtype=float)
        for i in range(N):
            for j in range(M):
                im_out[i, j] = lut[int(im_norm[i, j])]
        return im_out

    elif method == "Sigmoid":
        return win.sigmoid(im, TONES)
    elif method == "Cosine":
        return win.cosine(im, TONES)
    elif method == "Recip. Sqrt":
        return win.rSqrt(im, TONES)

    return im.copy()


def apply_preset_window(im: np.ndarray, preset: str) -> np.ndarray:
    return win.presetWindow(im, preset, IMAGE_DEPTH, TONES)


# ══════════════════════════════════════════════════════════════════════════════
# HISTOGRAM
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(**_cache_kwargs)
def apply_histogram_method(im: np.ndarray, method: str, **kwargs):
    """
    Returns: (im_out, h_original, h_processed)
    """
    # Clip στο 0-255 πριν δώσουμε την εικόνα στο histogram
    # (κάποια φίλτρα μπορεί να βγάλουν τιμές εκτός εύρους)
    im_safe = np.clip(im, 0, 255)

    if method == "Custom HE":
        im_out = hist.f_hequalization(im_safe, IMAGE_DEPTH, TONES)
    elif method == "Custom CDF":
        im_out = hist.CDF_equalization(im_safe, IMAGE_DEPTH, TONES)
    elif method == "CLAHE (skimage)":
        im_out = hist.CLAHE_skimage(im_safe)
    elif method == "OpenCV HE":
        im_out = hist.HE_opencv(im_safe).astype(float)
    elif method == "OpenCV CLAHE":
        im_out = hist.CLAHE_opencv(im_safe).astype(float)
    else:
        im_out = im_safe.copy()

    h_orig = hist.f_histogram(im_safe,  IMAGE_DEPTH, TONES)
    h_proc = hist.f_histogram(im_out,   IMAGE_DEPTH, TONES)
    return im_out, h_orig, h_proc


# ══════════════════════════════════════════════════════════════════════════════
# SPATIAL FILTERS
# ══════════════════════════════════════════════════════════════════════════════

_SPATIAL_MAP = {
    "SM1 - Cross Average":   ("smoothing",       0),
    "SM2 - Box Average":     ("smoothing",       1),
    "SM3 - Weighted Center": ("smoothing",       2),
    "SM4 - Gaussian":        ("smoothing",       3),
    "LM1 - 4-neighbor":      ("laplacian",       0),
    "LM2 - 8-neighbor":      ("laplacian",       1),
    "LM3 - Weighted":        ("laplacian",       2),
    "LM4 - Diagonal":        ("laplacian",       3),
    "LM5 - Edge Detection":  ("laplacian",       4),
    "HE1 - Cross Sharp":     ("highEnhancement", 0),
    "HE2 - Full Sharp":      ("highEnhancement", 1),
    "HE3 - Strong Sharp":    ("highEnhancement", 2),
    "HE4 - Strong Sharp v2": ("highEnhancement", 3),
    "Median Filter":         ("median",          0),
}

_KERNELS = {
    "smoothing": {
        0: np.array([[0,1,0],[1,1,1],[0,1,0]]),
        1: np.array([[1,1,1],[1,1,1],[1,1,1]]),
        2: np.array([[1,1,1],[1,2,1],[1,1,1]]),
        3: np.array([[1,2,1],[2,4,2],[1,2,1]]),
    },
    "laplacian": {
        0: np.array([[0,1,0],[1,-4,1],[0,1,0]]),
        1: np.array([[1,1,1],[1,-8,1],[1,1,1]]),
        2: np.array([[1,2,1],[2,-12,2],[1,2,1]]),
        3: np.array([[-1,2,-1],[2,-4,2],[-1,2,-1]]),
        4: np.array([[0,0,0],[0,-1,0],[0,1,0]]),
    },
    "highEnhancement": {
        0: np.array([[0,-1,0],[-1,5,-1],[0,-1,0]]),
        1: np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]]),
        2: np.array([[-1,-2,-1],[-2,13,-2],[-1,-2,-1]]),
        3: np.array([[-1,-2,-1],[-2,13,-2],[-1,-2,-1]]),
    },
}

SPATIAL_FILTER_NAMES = list(_SPATIAL_MAP.keys())

@st.cache_data(**_cache_kwargs)
def apply_spatial_filter(im: np.ndarray, filter_name: str):
    """
    Returns: (im_out, kernel_or_None, noisy_or_None)
    """
    category, idx = _SPATIAL_MAP[filter_name]
    kernel = None
    noisy  = None

    if category == "smoothing":
        im_out, _ = sf.smoothing(im, idx, IMAGE_DEPTH, TONES)
        kernel = _KERNELS["smoothing"][idx]

    elif category == "laplacian":
        im_out, _ = sf.laplacian(im, idx, IMAGE_DEPTH, TONES)
        kernel = _KERNELS["laplacian"][idx]

    elif category == "highEnhancement":
        im_out, _ = sf.highEnhancement(im, idx, IMAGE_DEPTH, TONES)
        kernel = _KERNELS["highEnhancement"][idx]

    elif category == "median":
        noisy, im_out = sf.medianFilter(im)

    else:
        im_out = im.copy()

    return im_out, kernel, noisy


# ══════════════════════════════════════════════════════════════════════════════
# FREQUENCY FILTERS
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(**_cache_kwargs)
def apply_frequency_filter(im: np.ndarray, filter_type: str, pass_type: str,
                            restoration: str, **kwargs):
    """
    Returns: (im_out, fh_1d, FH_2d, label)
    """
    M, N   = im.shape
    Flength = int(np.round(np.sqrt(M*M + N*N)))
    im_norm = np.round((TONES-1) * (im - np.min(im)) / (np.max(im) - np.min(im) + 1e-9))

    # ── Restoration ───────────────────────────────────────────────────────
    if restoration != "None":
        restore_map = {"Inverse Filter": 1, "Wiener Filter": 2, "Power Filter": 3}
        SIGMA   = 0.25
        fh      = ff.GaussianMTF(Flength)
        fhh, label = ff.generalizedWienerFilter(fh, restore_map[restoration], 2*SIGMA**2)
        FHH     = ff.from1dTo2dFilter(im_norm, fhh)
        im_out  = ff.DeconvolveImage(im_norm, FHH)
        im_out  = hist.imNormalize(im_out, TONES)
        return im_out, fhh, FHH, label

    # ── Standard filters ──────────────────────────────────────────────────
    pass_map = {"LP": 1, "HP": 2, "BR": 3, "BP": 4}
    TYPE     = pass_map[pass_type]
    cutoff   = kwargs.get("cutoff",    int(round(Flength * 0.3)))
    ndegree  = kwargs.get("order",     2)
    trans    = kwargs.get("bandwidth", int(round(Flength * 0.25)))

    if filter_type == "Butterworth":
        fh, label = ff.Butterworth(Flength, ndegree, cutoff, TYPE, trans)
    elif filter_type == "Exponential":
        fh, label = ff.Exponential(Flength, ndegree, cutoff, TYPE, trans)
    elif filter_type == "Gaussian":
        fh, label = ff.Gaussian(Flength, ndegree, cutoff, TYPE, trans)
    else:
        return im.copy(), None, None, "Unknown"

    FH     = ff.design2dFilter(im_norm, fh)
    im_out = ff.filterImage(im_norm, FH)
    im_out = hist.imNormalize(im_out, TONES)
    return im_out, fh, FH, label


# ══════════════════════════════════════════════════════════════════════════════
# RECONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════

_FBP_FILTERS = ["ramp", "shepp-logan", "cosine", "hamming", "hann", None]

@st.cache_data
def compute_sinogram(im: np.ndarray, n_angles: int = 180):
    return rec.imageToSinogram(im, N_proj=n_angles)

@st.cache_data
def apply_fbp(sinogram: np.ndarray, theta: np.ndarray, filter_name: str = "ramp"):
    filter_choice = _FBP_FILTERS.index(filter_name) if filter_name in _FBP_FILTERS else 0
    I_FBP, fname  = rec.FBP_reconstruction(sinogram, theta, filter_choice)
    # Normalize αποτέλεσμα στο 0-255 για σωστή εμφάνιση
    I_FBP = hist.imNormalize(I_FBP, 256)
    return I_FBP, fname

@st.cache_data
def apply_art(sinogram: np.ndarray, theta: np.ndarray, iterations: int = 5):
    I_ART = rec.ART_reconstruction(sinogram, theta, iterations)
    # Normalize αποτέλεσμα στο 0-255 για σωστή εμφάνιση
    I_ART = hist.imNormalize(I_ART, 256)
    return I_ART

