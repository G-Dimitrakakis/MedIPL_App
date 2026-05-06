"""
ui_components.py
----------------
Επαναχρησιμοποιήσιμα UI components για το MedIPL Streamlit app.

Περιέχει:
  - render_header()           : Επικεφαλίδα σελίδας
  - render_sidebar_windowing(): WC/WW sliders (πάντα ορατά στο sidebar)
  - render_image_pair()       : Αρχική + επεξεργασμένη εικόνα δίπλα-δίπλα
  - render_kernel_display()   : Εμφάνιση convolution kernel ως πίνακας
  - render_histogram_chart()  : Ιστόγραμμα με matplotlib → Streamlit
  - render_fft_view()         : FFT magnitude spectrum
  - render_footer()           : Footer
"""

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — απαραίτητο για Streamlit


# ══════════════════════════════════════════════════════════════════════════════
# HEADER & FOOTER
# ══════════════════════════════════════════════════════════════════════════════

def render_header():
    """Κεντρική επικεφαλίδα της εφαρμογής."""
    st.markdown(
        """
        <div class="main-header">
            <h1>🔬 MedIPL</h1>
            <span>Medical Image Processing Lab · v1.0</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer():
    """Footer με πληροφορίες project."""
    st.markdown(
        """
        <div class="footer">
            MedIPL · Medical Image Processing Lab · Academic Project
        </div>
        """,
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR WINDOWING — πάντα ορατό
# ══════════════════════════════════════════════════════════════════════════════

def render_sidebar_windowing():
    """
    Εμφανίζει τους WC/WW sliders στο sidebar.
    Οι τιμές αποθηκεύονται στο st.session_state.wc / .ww
    ώστε να είναι διαθέσιμες παντού στην εφαρμογή.
    """
    st.markdown("### 🪟 Simple Window")
    st.caption("Always applied after any processing.")

    # Sliders — χρησιμοποιούμε key= για να συνδεθούν με session_state
    st.session_state.wc = st.slider(
        "Window Center (WC)",
        min_value=0,
        max_value=255,
        value=st.session_state.wc,
        key="wc_slider",
        help="Το κεντρικό σημείο του παραθύρου εμφάνισης (brightness).",
    )
    st.session_state.ww = st.slider(
        "Window Width (WW)",
        min_value=1,
        max_value=255,
        value=st.session_state.ww,
        key="ww_slider",
        help="Το εύρος του παραθύρου εμφάνισης (contrast).",
    )

    # Εμφάνιση των τιμών σε mono font
    st.markdown(
        f'<p style="font-family:\'IBM Plex Mono\', monospace; font-size:0.75rem; '
        f'color:#7d8590; margin-top:-0.5rem;">'
        f'Va = {max(0, int((2*st.session_state.wc - st.session_state.ww)/2))} · '
        f'Vb = {min(255, int((2*st.session_state.wc + st.session_state.ww)/2))}'
        f'</p>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE PAIR — αρχική + επεξεργασμένη δίπλα-δίπλα
# ══════════════════════════════════════════════════════════════════════════════

def render_image_pair(
    im_original: np.ndarray,
    im_processed: np.ndarray,
    label_left: str = "Original",
    label_right: str = "Processed",
    show_histogram: bool = False,
    show_fft: bool = False,
    h_orig: np.ndarray = None,
    h_proc: np.ndarray = None,
):
    """
    Εμφανίζει δύο εικόνες δίπλα-δίπλα με labels.
    Προαιρετικά εμφανίζει ιστογράμματα ή FFT spectrum από κάτω.

    Parameters
    ----------
    im_original  : numpy array αρχικής εικόνας
    im_processed : numpy array επεξεργασμένης εικόνας
    label_left   : τίτλος αριστερής εικόνας
    label_right  : τίτλος δεξιάς εικόνας
    show_histogram: αν True, εμφανίζει ιστογράμματα
    show_fft     : αν True, εμφανίζει FFT spectrum
    h_orig/h_proc: προ-υπολογισμένα ιστογράμματα (αν υπάρχουν)
    """
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown(f'<p class="img-label">{label_left}</p>', unsafe_allow_html=True)
        _show_image(im_original)
        _show_image_stats(im_original)

    with col_right:
        st.markdown(f'<p class="img-label">{label_right}</p>', unsafe_allow_html=True)
        _show_image(im_processed)
        _show_image_stats(im_processed)

    # ── Ιστογράμματα (για Histogram tab) ──────────────────────────────────
    if show_histogram and h_orig is not None and h_proc is not None:
        st.divider()
        st.markdown("**Histograms**")
        h_col1, h_col2 = st.columns(2)
        with h_col1:
            render_histogram_chart(h_orig, "Original Histogram", color="#7d8590")
        with h_col2:
            render_histogram_chart(h_proc, "Processed Histogram", color="#00d4ff")

    # ── FFT Spectrum (για Frequency tab) ──────────────────────────────────
    if show_fft:
        st.divider()
        st.markdown("**FFT Magnitude Spectrum**")
        f_col1, f_col2 = st.columns(2)
        with f_col1:
            render_fft_view(im_original, "Original FFT")
        with f_col2:
            render_fft_view(im_processed, "Processed FFT")


def _show_image(im: np.ndarray):
    """Helper: κανονικοποιεί και εμφανίζει εικόνα."""
    # Clip στο [0, 255] και μετατροπή σε uint8 για σωστή εμφάνιση
    im_display = np.clip(im, 0, 255).astype(np.uint8)
    st.image(im_display, use_container_width=True, clamp=True)


def _show_image_stats(im: np.ndarray):
    """Helper: εμφανίζει min/max/mean stats κάτω από εικόνα."""
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Min", f"{np.min(im):.0f}")
    col_b.metric("Max", f"{np.max(im):.0f}")
    col_c.metric("Mean", f"{np.mean(im):.1f}")


# ══════════════════════════════════════════════════════════════════════════════
# KERNEL DISPLAY
# ══════════════════════════════════════════════════════════════════════════════

def render_kernel_display(kernel: np.ndarray, filter_name: str):
    """
    Εμφανίζει το convolution kernel ως styled πίνακα.
    Θετικές τιμές: πράσινο, Αρνητικές: κόκκινο, Μηδέν: γκρι.
    """
    if kernel is None:
        return

    rows, cols = kernel.shape

    # Δημιουργία HTML πίνακα
    cells_html = ""
    for r in range(rows):
        cells_html += '<tr style="text-align:center;">'
        for c in range(cols):
            val = kernel[r, c]
            if val > 0:
                css_class = "positive"
            elif val < 0:
                css_class = "negative"
            else:
                css_class = "zero"
            # Format: χωρίς decimals αν integer
            fmt_val = f"{val:.0f}" if float(val) == int(val) else f"{val:.3f}"
            cells_html += (
                f'<td style="padding:0.3rem 0.6rem; font-family:\'IBM Plex Mono\', '
                f'monospace; font-size:0.9rem;" '
                f'class="{css_class}">{fmt_val}</td>'
            )
        cells_html += "</tr>"

    st.markdown(
        f"""
        <div class="kernel-box">
            <div class="kernel-title">Convolution Kernel — {filter_name}</div>
            <table style="border-collapse:collapse; margin:0 auto;">
                {cells_html}
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# HISTOGRAM CHART
# ══════════════════════════════════════════════════════════════════════════════

def render_histogram_chart(h: np.ndarray, title: str, color: str = "#00d4ff"):
    """
    Εμφανίζει ιστόγραμμα με matplotlib (dark theme).
    
    Parameters
    ----------
    h     : numpy array μεγέθους 256 (από hist.f_histogram)
    title : τίτλος γραφήματος
    color : χρώμα bars
    """
    fig, ax = plt.subplots(figsize=(4, 2.2))
    fig.patch.set_facecolor("#1c2333")
    ax.set_facecolor("#1c2333")

    ax.bar(np.arange(len(h)), h / (np.max(h) + 1e-9),
           color=color, alpha=0.85, width=1.0)
    ax.set_title(title, color="#e6edf3", fontsize=8, fontfamily="monospace")
    ax.set_xlabel("Gray Level", color="#7d8590", fontsize=7)
    ax.set_ylabel("Frequency (norm.)", color="#7d8590", fontsize=7)
    ax.tick_params(colors="#7d8590", labelsize=6)
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    ax.set_xlim(0, 255)

    plt.tight_layout(pad=0.5)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# FFT SPECTRUM VIEW
# ══════════════════════════════════════════════════════════════════════════════

def render_fft_view(im: np.ndarray, title: str):
    """
    Εμφανίζει το FFT magnitude spectrum μιας εικόνας (log scale).
    """
    fft = np.fft.fft2(im)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.log1p(np.abs(fft_shift))  # log για καλύτερη εμφάνιση

    fig, ax = plt.subplots(figsize=(4, 4))
    fig.patch.set_facecolor("#1c2333")
    ax.set_facecolor("#0d1117")
    ax.imshow(magnitude, cmap="inferno", aspect="auto")
    ax.set_title(title, color="#e6edf3", fontsize=8, fontfamily="monospace")
    ax.axis("off")
    plt.tight_layout(pad=0.3)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# 1D FILTER PLOT (για Frequency tab)
# ══════════════════════════════════════════════════════════════════════════════

def render_filter_plot(fh: np.ndarray, FH: np.ndarray, label: str):
    """
    Εμφανίζει:
      - Αριστερά: 1D filter profile
      - Δεξιά: 2D filter (fftshift για εμφάνιση)
    Ίδια λογική με το plot του MedIPL_Main (main_choice=4).
    """
    if fh is None or FH is None:
        return

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(4, 2.5))
        fig.patch.set_facecolor("#1c2333")
        ax.set_facecolor("#1c2333")

        ax.plot(fh, color="#00d4ff", lw=1.5, label="filter")
        N_fh = len(fh)
        ax.axvline(N_fh // 2, color="#3fb950", linestyle="--", lw=1, label="center")
        ax.plot(np.fft.fftshift(fh), color="#f85149", linestyle="--",
                lw=1, label="shifted")
        ax.legend(fontsize=6, facecolor="#161b22", labelcolor="#e6edf3")
        ax.set_title(f"1D Filter: {label}", color="#e6edf3",
                     fontsize=8, fontfamily="monospace")
        ax.tick_params(colors="#7d8590", labelsize=6)
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")
        ax.grid(True, color="#30363d", linewidth=0.5)

        plt.tight_layout(pad=0.5)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(4, 4))
        fig.patch.set_facecolor("#1c2333")
        ax.imshow(np.fft.fftshift(FH), cmap="viridis", aspect="auto")
        ax.set_title(f"2D Filter: {label}", color="#e6edf3",
                     fontsize=8, fontfamily="monospace")
        ax.axis("off")
        plt.tight_layout(pad=0.3)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
