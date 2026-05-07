"""
app.py — MedIPL Streamlit Application v2
=========================================
Διορθώσεις:
  - Κάθε tab αποθηκεύει το δικό του processed image (win/hist/spat/freq_processed)
    ώστε να φαίνεται η διαφορά από το original σε κάθε re-run
  - Simple Window: εφαρμόζεται ΜΟΝΟ ως display layer, δεν «ψήνεται» στο αποτέλεσμα
  - Reconstruction: normalize output + layout χωρίς st.stop() μέσα σε with block
  - Presets → Recommended Window (mean/std based, εξαιρεί background)
  - Βελτιωμένα χρώματα UI
"""

import numpy as np
import streamlit as st
from streamlit_cropper import st_cropper
from PIL import Image
import pandas as pd
from processing import (
    apply_simple_window,
    apply_advanced_window,
    recommend_window,
    apply_histogram_method,
    apply_spatial_filter,
    apply_frequency_filter,
    compute_sinogram,
    apply_fbp,
    apply_art,
    SPATIAL_FILTER_NAMES,
    load_image
)
from ui_components import (
    render_header, render_footer, render_sidebar_windowing,
    render_image_pair, render_kernel_display, render_filter_plot,
)

st.set_page_config(
    page_title="MedIPL · Medical Image Processing Lab",
    page_icon="🔬", layout="wide", initial_sidebar_state="expanded",
)
with open("style.css", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ── Session State ─────────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "original_image": None,
        "wc": 128, "ww": 256,
        "last_kernel": None,
        "last_sinogram": None,
        # Κάθε tab έχει το δικό του αποτέλεσμα
        "win_processed":  None,
        "hist_processed": None,
        "spat_processed": None,
        "freq_processed": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-logo">🔬 MedIPL</div>', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-subtitle">Medical Image Processing Lab</p>', unsafe_allow_html=True)
    st.divider()

    st.markdown("### 📂 Image")
    uploaded = st.file_uploader(
        "BMP · PNG · JPG · DICOM",
        type=["bmp","png","jpg","jpeg","dcm"],
        label_visibility="collapsed",
    )
    if uploaded is not None:
    # Έλεγχος αν η εικόνα είναι όντως νέα
        if "last_uploaded_name" not in st.session_state or st.session_state.last_uploaded_name != uploaded.name:
            img = load_image(uploaded)
            if img is not None:
                st.session_state.original_image = img
                st.session_state.last_uploaded_name = uploaded.name
                # Reset μόνο όταν μπαίνει ΚΑΙΝΟΥΡΓΙΑ εικόνα
                for k in ("win_processed","hist_processed","spat_processed","freq_processed"):
                    st.session_state[k] = None
                st.rerun()

    st.divider()
    render_sidebar_windowing()

    # ── ROI Tool (μεταφέρεται εδώ) ──────────────────────
    st.divider()
    st.markdown("### 🔍 Analysis Tools")
    roi_enabled = st.checkbox("Enable ROI Tool", value=False)

    # ── Notes ────────────────────────────────────────────
    st.divider()
    st.markdown("### 📝 Notes")
    if "notes" not in st.session_state:
        st.session_state.notes = ""
    st.session_state.notes = st.text_area(
        "Clinical Notes",
        value=st.session_state.notes,
        height=180,
        placeholder="Write notes about the image, findings, processing steps..."
    )

    # ── Export ─────────────────────────────────────────────
    st.divider()
    st.markdown("### 💾 Export")

    import io
    from PIL import ImageDraw

    def _get_export_image():
        original = st.session_state.get("original_image")
        if original is None:
            return None
        
        # Χρησιμοποιούμε το export_source που ορίσαμε στα Tabs για να ξέρουμε ΠΟΙΑ εικόνα βλέπει ο χρήστης
        source_key = st.session_state.get("export_source")
        
        if source_key and st.session_state.get(source_key) is not None:
            raw_data = st.session_state[source_key]
        else:
            # Αν δεν έχει γίνει καμία επεξεργασία, πάρε την αρχική
            raw_data = original

        # Μετατροπή σε 8-bit για αποθήκευση (εφαρμογή κλίμακας 0-255)
        # Προσοχή: Αν η εικόνα είναι ήδη 0-255 από το ιστόγραμμα, το apply_simple_window ίσως την αλλάξει.
        # Καλύτερα να την κάνουμε clip και cast απευθείας:
        arr = np.clip(raw_data, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    export_img = _get_export_image()

    if export_img is None:
        st.caption("Upload an image to enable export.")
    else:
        # ROI overlay
        # Βεβαιώσου ότι το roi_enabled ορίζεται από το checkbox του ROI Tab
        if st.session_state.get("roi_enabled") and st.session_state.get("roi_box") is not None:
            box = st.session_state["roi_box"]
            export_img = export_img.copy()
            draw = ImageDraw.Draw(export_img)
            
            # Σχεδίαση με βάση τις συντεταγμένες που πλέον υπάρχουν!
            draw.rectangle(
                [box["left"], box["top"], box["left"] + box["width"], box["top"] + box["height"]], 
                outline='#00FF00', 
                width=5 # Κάνε το 5-10 για να φαίνεται σίγουρα
            )

        # Προετοιμασία αρχείου
        buf = io.BytesIO()
        export_img.save(buf, format="PNG")
        buf.seek(0)

        st.download_button(
            label="⬇️ Export as PNG",
            data=buf,
            file_name="medIPL_export.png",
            mime="image/png",
            use_container_width=True
        )

# ── Main ──────────────────────────────────────────────────────────────────────
render_header()

if st.session_state.original_image is None:
    st.markdown("""
        <div class="welcome-box">
            <h2>Welcome to MedIPL</h2>
            <p>Upload a medical image from the sidebar to begin processing.</p>
            <p class="supported">BMP &nbsp;·&nbsp; PNG &nbsp;·&nbsp; JPG &nbsp;·&nbsp; DICOM (.dcm)</p>
        </div>""", unsafe_allow_html=True)
    st.stop()

orig = st.session_state.original_image
wc   = st.session_state.wc
ww   = st.session_state.ww

def _wd(im):
    """Window Display: εφαρμόζει WC/WW sliders ΜΟΝΟ για εμφάνιση."""
    return apply_simple_window(im, wc, ww)


static_orig_display = apply_simple_window(orig, 128, 256).astype(np.uint8)

if roi_enabled:
    st.header("Region of Interest Analysis")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Το εργαλείο ROI. Επιστρέφει την κομμένη εικόνα ως Image object.
        # Χρησιμοποιούμε την static_orig_display για να είναι καθαρή η επιλογή.
        roi_coords = st_cropper(
            Image.fromarray(static_orig_display), 
            realtime_update=True, 
            box_color='#00FF00',
            aspect_ratio=None, # Επιτρέπει ελεύθερο σχήμα ορθογωνίου
            return_type='box')
        # Αποθηκεύουμε τις συντεταγμένες στο session_state για το sidebar
        st.session_state["roi_box"] = roi_coords
        st.session_state["roi_enabled"] = True
    
    with col2:
        st.markdown("##### ROI Preview")
        
        # Εφόσον πλέον παίρνουμε συντεταγμένες (coords), κόβουμε την εικόνα χειροκίνητα για το preview
        left, top, width, height = roi_coords['left'], roi_coords['top'], roi_coords['width'], roi_coords['height']
        
        # Κόψιμο (Crop) από τον πίνακα static_orig_display
        roi_array = static_orig_display[top:top+height, left:left+width]
        
        # Το υπόλοιπο display logic σου (Normalization κτλ)
        roi_min, roi_max = roi_array.min(), roi_array.max()
        if roi_max > roi_min:
            roi_rescaled = (roi_array - roi_min) / (roi_max - roi_min) * 255
        else:
            roi_rescaled = roi_array
            
        roi_final = roi_rescaled.astype(np.uint8)
        st.image(roi_final, use_container_width=True)
        
        st.markdown("##### Statistics")
        st.write(f"**Mean:** {np.mean(roi_array):.2f}")
        st.write(f"**Std Dev:** {np.std(roi_array):.2f}")

    st.divider()
    # Αν θέλεις να σταματάει εδώ η εκτέλεση όταν το ROI είναι ανοιχτό:
    # st.stop()
# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_win, tab_hist, tab_spat, tab_freq, tab_recon = st.tabs([
    "🪟  Windowing", "📊  Histogram", "🔲  Spatial Filters",
    "〰️  Frequency Filters", "🔄  Reconstruction",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 · WINDOWING
# ══════════════════════════════════════════════════════════════════════════════
with tab_win:
    st.markdown("### Windowing & Transformations")
    
    # ── 1. ΕΡΓΑΛΕΙΑ (ΠΑΝΩ ΜΕΡΟΣ) ──────────────────────────────────────────────
    
    # ── Intelligent Presets ──────────────────────────────────────────────
    st.markdown("**Intelligent Window Presets**")
    st.caption("Select the anatomy type to target specific image intensity ranges.")

    anatomy_options = [
        "🤖 Auto (Smart Detection)",
        "🫁 Lungs (Dark details)",
        "🧠 Brain (Narrow mid-tones)",
        "🦴 Bones (Highlights)",
        "🫀 Soft Tissue (Abdomen/Chest)"
    ]
    
    # Χωρίζουμε το χώρο για το dropdown, το κουμπί και τις πληροφορίες
    col_sel, col_btn, col_info = st.columns([2, 1.5, 1.5])
    
    with col_sel:
        # Dropdown χωρίς έξτρα τίτλο για εξοικονόμηση χώρου
        selected_anatomy = st.selectbox("Anatomy Preset", anatomy_options, label_visibility="collapsed")
        
    # Καλούμε τη συνάρτηση με την επιλογή του χρήστη
    rec_wc, rec_ww = recommend_window(orig, anatomy=selected_anatomy)
    
    with col_btn:
        if st.button("⚡ Apply Preset", type="primary", use_container_width=True):
            st.session_state.wc = rec_wc
            st.session_state.ww = rec_ww
            st.session_state.win_processed = apply_simple_window(orig, rec_wc, rec_ww)
            st.session_state["export_source"] = "win_processed"
            st.rerun()
            
    with col_info:
        # Δείχνουμε τις τιμές που θα εφαρμοστούν
        st.markdown(
            f'<div style="background-color: #1E1E1E; padding: 8px; border-radius: 5px; text-align: center; border: 1px solid #444;">'
            f'<span style="color: #4CAF50;">WC:</span> <b>{rec_wc}</b> &nbsp;|&nbsp; '
            f'<span style="color: #2196F3;">WW:</span> <b>{rec_ww}</b></div>',
            unsafe_allow_html=True,
        )

    # Advanced Window
    adv_method = "Simple Display"
    with st.expander("⚙️  Advanced Windowing Techniques", expanded=False):
        adv_methods = [
            "Simple Display", "Optimal Display", "Broken Window", "Double Window",
            "Inverse", "Logarithmic", "Inv. Logarithmic", "Power",
            "Sine", "Exponential", "Sigmoid", "Cosine", "Recip. Sqrt",
        ]
        adv_method = st.selectbox("Technique", adv_methods)
        adv_kwargs = {}
        
        if adv_method == "Broken Window":
            c1, c2 = st.columns(2)
            adv_kwargs["gray_val"] = c1.slider("Gray Value", 0, 255, 128)
            adv_kwargs["im_val"]   = c2.slider("Image Value", 0, 255, 128)
        elif adv_method == "Double Window":
            c1, c2 = st.columns(2)
            with c1:
                adv_kwargs["ww1"] = st.slider("Window 1 Width",  0, 255, 80)
                adv_kwargs["wl1"] = st.slider("Window 1 Center", 0, 255, 60)
            with c2:
                adv_kwargs["ww2"] = st.slider("Window 2 Width",  0, 255, 120)
                adv_kwargs["wl2"] = st.slider("Window 2 Center", 0, 255, 180)
                
        col_btn1, col_btn2 = st.columns([1, 1])
        with col_btn1:
            if st.button("Apply Advanced Window", type="primary", use_container_width=True):
                st.session_state.win_processed = apply_advanced_window(orig, adv_method, **adv_kwargs)
                st.session_state["export_source"] = "win_processed"
                st.rerun() # Προστέθηκε το rerun για να ανανεωθεί άμεσα η εικόνα!
        with col_btn2:
            if st.button("🔄 Reset to Original", use_container_width=True):
                st.session_state.win_processed = None
                st.rerun()

    st.divider()

    # ── 2. ΕΜΦΑΝΙΣΗ ΕΙΚΟΝΩΝ (ΚΑΤΩ ΜΕΡΟΣ) ──────────────────────────────────────
    
    # Καθορίζουμε τη βάση: αν έχει γίνει advanced επεξεργασία, δείχνουμε αυτή. Αλλιώς την original.
    base_for_right = st.session_state.win_processed if st.session_state.win_processed is not None else orig
    
    # Η αριστερή εικόνα μένει πάντα σταθερή
    left_display = apply_simple_window(orig, 128, 256) 
    
    # Στη δεξιά εφαρμόζουμε τα live sliders του sidebar (wc, ww)
    right_display = apply_simple_window(base_for_right, wc, ww)
    
    # Καθορισμός του label
    r_lbl = "Live Windowing" if st.session_state.win_processed is None else f"Advanced ({adv_method}) + Live Windowing"
    
    render_image_pair(
        left_display, 
        right_display,
        label_left="Original (Fixed View)",
        label_right=f"{r_lbl} [WC={wc}, WW={ww}]"
    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 · HISTOGRAM
# ══════════════════════════════════════════════════════════════════════════════
with tab_hist:
    st.markdown("### Histogram Enhancement")
    st.caption("Improve contrast using histogram-based methods.")

    mc, bc = st.columns([3, 1])
    with mc:
        hist_method = st.radio("Method",
            ["Custom HE","Custom CDF","CLAHE (skimage)","OpenCV HE","OpenCV CLAHE"],
            horizontal=True)
    with bc:
        st.markdown("&nbsp;")
        if st.button("Apply", type="primary", use_container_width=True):
            im_out, h_orig, h_proc = apply_histogram_method(orig, hist_method)
            st.session_state.hist_processed = im_out
            st.session_state["h_orig"] = h_orig
            st.session_state["h_proc"] = h_proc
            st.session_state["export_source"] = "hist_processed"

    st.divider()

    hist_result = st.session_state.hist_processed
    hist_result = st.session_state.get("hist_processed")
    if hist_result is None:
        render_image_pair(static_orig_display, static_orig_display, 
                         label_left="Original (Fixed)", label_right="—")
    else:
        render_image_pair(static_orig_display, _wd(hist_result), 
                         
            label_left="Original", label_right=f"{hist_method}",
            show_histogram=True,
            h_orig=st.session_state.get("h_orig"),
            h_proc=st.session_state.get("h_proc"),
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 · SPATIAL FILTERS
# ══════════════════════════════════════════════════════════════════════════════
with tab_spat:
    st.markdown("### Spatial Domain Filters")
    st.caption("Convolution-based filters applied directly on pixel values.")

    filter_groups = {
        "Smoothing":        [n for n in SPATIAL_FILTER_NAMES if n.startswith("SM")],
        "Laplacian":        [n for n in SPATIAL_FILTER_NAMES if n.startswith("LM")],
        "High Enhancement": [n for n in SPATIAL_FILTER_NAMES if n.startswith("HE")],
        "Median Filter":    ["Median Filter"],
    }

    gc, fc, bc = st.columns([1, 1, 1])
    with gc:
        group = st.radio("Filter Group", list(filter_groups.keys()))
    with fc:
        filter_name = st.radio("Filter", filter_groups[group])
    with bc:
        st.markdown("&nbsp;")
        if filter_name == "Median Filter":
            st.info("Adds Salt & Pepper noise, then removes it with 5×5 median.")
        if st.button("Apply Filter", type="primary"):
            im_out, kernel, noisy = apply_spatial_filter(orig, filter_name)
            st.session_state.spat_processed = im_out
            st.session_state.last_kernel    = kernel
            st.session_state["last_noisy"]  = noisy
            st.session_state["export_source"] = "spat_processed"
            

    if st.session_state.last_kernel is not None:
        st.divider()
        render_kernel_display(st.session_state.last_kernel, filter_name)

    st.divider()

    spat_result = st.session_state.spat_processed
    if spat_result is None:
        st.info("Select a filter and press **Apply Filter**.")
        # ΑΛΛΑΓΗ: Χρήση static_orig_display αντί για _wd(orig)
        render_image_pair(static_orig_display, static_orig_display,
                          label_left="Original (Fixed)", label_right="— awaiting processing —")
    else:
        noisy = st.session_state.get("last_noisy")
        if noisy is not None:
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown('<p class="img-label">Original (Fixed)</p>', unsafe_allow_html=True)
                # ΑΛΛΑΓΗ: Εμφάνιση της παγωμένης εικόνας χωρίς sliders
                st.image(static_orig_display, use_container_width=True)
            with c2:
                st.markdown('<p class="img-label">Salt & Pepper Noise</p>', unsafe_allow_html=True)
                # Εδώ συνήθως αφήνουμε το noisy ως έχει ή του βάζουμε ένα τυπικό windowing 
                # για να φαίνεται ο θόρυβος, π.χ. apply_simple_window(noisy, 128, 256)
                st.image(np.clip(noisy, 0, 255).astype(np.uint8), use_container_width=True)
            with c3:
                st.markdown('<p class="img-label">Filtered + Live Window</p>', unsafe_allow_html=True)
                # Εδώ σωστά εφαρμόζουμε το _wd στο αποτέλεσμα
                st.image(np.clip(_wd(spat_result), 0, 255).astype(np.uint8), use_container_width=True)
        else:
            # ΑΛΛΑΓΗ: static_orig_display αριστερά, _wd(spat_result) δεξιά
            render_image_pair(static_orig_display, _wd(spat_result),
                              label_left="Original (Fixed)", label_right=f"{filter_name} + Live Window")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 · FREQUENCY FILTERS
# ══════════════════════════════════════════════════════════════════════════════
with tab_freq:
    st.markdown("### Frequency Domain Filters")
    st.caption("Filters applied in the Fourier frequency domain.")

    lc, rc = st.columns([1, 2])
    with lc:
        filter_type = st.radio("Filter Type", ["Butterworth","Exponential","Gaussian"])
        pass_type   = st.radio("Pass Type", ["LP","HP","BR","BP"], horizontal=True)
        st.caption("LP · HP · BR (Band Reject) · BP (Band Pass)")
        st.divider()
        st.markdown("**Image Restoration**")
        restoration = st.selectbox("Restoration",
            ["None","Inverse Filter","Wiener Filter","Power Filter"],
            label_visibility="collapsed")

    with rc:
        M, N    = orig.shape
        Flength = int(np.round(np.sqrt(M*M + N*N)))
        freq_kwargs = {}
        if restoration == "None":
            freq_kwargs["cutoff"] = st.slider("Cutoff Frequency (D₀)", 1, Flength//2,
                                              value=int(round(Flength*0.3)))
            if filter_type in ("Butterworth","Exponential"):
                freq_kwargs["order"] = st.slider("Filter Order (n)", 1, 10, 2)
            if pass_type in ("BR","BP"):
                freq_kwargs["bandwidth"] = st.slider("Bandwidth (W)", 1, Flength//4,
                                                     value=int(round(Flength*0.25)))
        else:
            st.info(f"**{restoration}** uses Gaussian MTF as degradation model.")

        if st.button("Apply Frequency Filter", type="primary"):
            im_out, fh, FH, label = apply_frequency_filter(
                orig, filter_type=filter_type, pass_type=pass_type,
                restoration=restoration, **freq_kwargs)
            st.session_state.freq_processed = im_out
            st.session_state["freq_fh"]     = fh
            st.session_state["freq_FH"]     = FH
            st.session_state["freq_label"]  = label
            st.session_state["export_source"] = "freq_processed"

    st.divider()

    freq_result = st.session_state.freq_processed
    freq_label  = st.session_state.get("freq_label", "—")

    if freq_result is None:
        render_image_pair(static_orig_display, static_orig_display, # Και τα δύο παγωμένα αν δεν υπάρχει αποτέλεσμα
                        label_left="Original (Fixed)", 
                        label_right="Waiting for processing...")
    else:
        render_image_pair(static_orig_display, _wd(freq_result), # Αριστερά παγωμένο, δεξιά με sliders
                        label_left="Original (Fixed)", 
                        label_right=f"{freq_label} + Windowing",
                        show_fft=True)
        fh = st.session_state.get("freq_fh")
        FH = st.session_state.get("freq_FH")
        if fh is not None:
            st.divider()
            st.markdown("**Filter Visualization**")
            render_filter_plot(fh, FH, freq_label)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 · RECONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════
with tab_recon:
    st.markdown("### Image Reconstruction")
    st.caption("Compute sinograms and reconstruct using FBP or ART.")

    lc, rc = st.columns([1, 2])
    with lc:
        recon_method = st.radio("Method", ["FBP", "ART", "Both (FBP + ART)"])
        n_angles     = st.slider("Projection Angles", 10, 360, 180, 10)
        fbp_filter   = "ramp"
        if "FBP" in recon_method:
            fbp_sel    = st.selectbox("FBP Filter",
                ["ramp", "shepp-logan", "cosine", "hamming", "hann", "None"])
            fbp_filter = None if fbp_sel == "None" else fbp_sel
        art_iters = 5
        if "ART" in recon_method:
            art_iters = st.slider("ART Iterations", 1, 20, 5)

    with rc:
        
        st.markdown("&nbsp;")
        run_recon = st.button("▶  Run Reconstruction", type="primary")

    # --- Reconstruction logic ---
    if run_recon:
        sinogram = None
        with st.spinner("Computing..."):
            
            sinogram, theta = compute_sinogram(orig, n_angles=n_angles)

            if sinogram is not None:
                # ΑΠΟΘΗΚΕΥΣΗ ΣΤΟ STATE (Κρίσιμο!)
                st.session_state.last_sinogram = sinogram
                
                if "FBP" in recon_method:
                    I_FBP, fname = apply_fbp(sinogram, theta, filter_name=fbp_filter)
                    st.session_state["recon_FBP"] = I_FBP
                    st.session_state["recon_fname"] = fname
                
                if "ART" in recon_method:
                    st.session_state["recon_ART"] = apply_art(sinogram, theta, iterations=art_iters)
                
                st.session_state["recon_method"] = recon_method
                st.session_state["recon_iters"] = art_iters
                
                # Αναγκάζουμε το Streamlit να ξανατρέξει για να δει τα νέα αποτελέσματα
                st.rerun()
    st.divider()

    # --- Εμφάνιση Αποτελεσμάτων κάτω από το Divider ---
    sinogram = st.session_state.get("last_sinogram")
    I_FBP = st.session_state.get("recon_FBP")
    I_ART = st.session_state.get("recon_ART")
    stored_method = st.session_state.get("recon_method")

    if sinogram is None:
        st.info("Press the button to start reconstruction of the uploaded image")
    else:
        # Εσωτερική βοηθητική συνάρτηση για normalization
        def _prepare_for_display(img, target_shape=None):
            if img is None: return None
            
            # 1. Normalization (0-255)
            img_min, img_max = img.min(), img.max()
            if img_max > img_min:
                rescaled = (img - img_min) / (img_max - img_min) * 255
            else:
                rescaled = img
            out_img = rescaled.astype(np.uint8)
            
            # 2. Αφαίρεση περιττού πλαισίου (Center Crop)
            if target_shape is not None:
                h_target, w_target = target_shape[0], target_shape[1]
                h_current, w_current = out_img.shape[0], out_img.shape[1]
                
                # Υπολογισμός των ορίων για το κροπάρισμα στο κέντρο
                if h_current > h_target or w_current > w_target:
                    start_h = (h_current - h_target) // 2
                    start_w = (w_current - w_target) // 2
                    
                    # Κόψιμο ακριβώς στις διαστάσεις της αρχικής
                    out_img = out_img[start_h:start_h + h_target, start_w:start_w + w_target]
            
            return out_img

        # 1. ΣΥΝΟΓΡΑΜΜΑ (Οριζόντιο / Ξαπλωτό)
        st.markdown("#### 🔍 Sinogram Analysis")
        # Κάνουμε Transpose (.T) για να είναι οριζόντιο
        sino_horiz = _prepare_for_display(sinogram.T)
        st.image(sino_horiz, caption=f"Sinogram (Angles x Projections): {sinogram.shape}", use_container_width=True)
        
        st.divider()

        # 2. ΑΝΑΚΑΤΑΣΚΕΥΗ (Σύγκριση)
        st.markdown("####  Reconstruction Results")
        show_both = (stored_method == "Both (FBP + ART)")

        if show_both:
            # Τρεις στήλες: Original | FBP | ART
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**Original**")
                st.image(static_orig_display, use_container_width=True)
            with c2:
                st.markdown("**FBP Reconstruction**")
                st.image(_prepare_for_display(_wd(I_FBP),target_shape=static_orig_display.shape), use_container_width=True)
            with c3:
                st.markdown("**ART Reconstruction**")
                st.image(_prepare_for_display(_wd(I_ART),target_shape=static_orig_display.shape), use_container_width=True)
        else:
            # Δύο στήλες: Original | Result
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Original**")
                st.image(static_orig_display, use_container_width=True)
            with c2:
                if I_FBP is not None and "FBP" in str(stored_method):
                    st.markdown("**FBP Reconstruction**")
                    st.image(_prepare_for_display(_wd(I_FBP),target_shape=static_orig_display.shape), use_container_width=True)
                elif I_ART is not None:
                    st.markdown("**ART Reconstruction**")
                    st.image(_prepare_for_display(_wd(I_ART),target_shape=static_orig_display.shape), use_container_width=True)
st.info("⚠️ Research Use Only - This platform is intended for educational and research purposes. It is not a certified medical device and should not be used for primary clinical diagnosis or treatment decisions.")
render_footer()