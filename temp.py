# dsp_nir.py
# DSP-based NIR estimation WITHOUT wavelets
# Fully compatible with Raspberry Pi, OpenCV, NumPy 1.x

import cv2
import numpy as np


# ------------------------------ UTIL ------------------------------

def _to_float_gray(channel):
    """Ensure channel is grayscale float32."""
    if channel.ndim == 3:
        channel = cv2.cvtColor(channel, cv2.COLOR_BGR2GRAY)
    return channel.astype(np.float32)


# ------------------------------ 1. BASIC NIR MIX ------------------------------

def basic_nir_mix(b, g, r):
    """
    Weighted RGB combination to approximate NIR.
    These weights can be tuned later through calibration.
    """
    b = b.astype(np.float32)
    g = g.astype(np.float32)
    r = r.astype(np.float32)

    # Initial weights (tunable)
    aR, aG, aB = 0.7, 0.2, 0.1
    nir_basic = aR*r + aG*g + aB*b
    return nir_basic


# ------------------------------ 2. FFT LOW-PASS ------------------------------

def fft_lowpass(channel, radius_ratio=0.08):
    """
    Extract low-frequency components by removing high-frequency edges.
    NIR is naturally smooth -> low-frequency dominates.
    """
    ch = _to_float_gray(channel)

    f = np.fft.fft2(ch)
    fshift = np.fft.fftshift(f)

    h, w = ch.shape
    cy, cx = h // 2, w // 2
    radius = int(min(h, w) * radius_ratio)

    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((Y - cy)**2 + (X - cx)**2)
    mask = dist <= radius

    fshift_lp = fshift * mask
    f_ishift = np.fft.ifftshift(fshift_lp)
    img_back = np.fft.ifft2(f_ishift)

    return np.abs(img_back).astype(np.float32)


# ------------------------------ 3. LAPLACIAN BASE ------------------------------

def laplacian_base(channel, ksize=3):
    """
    Base layer = original - edges
    Smooth regions hold more NIR reflectance info.
    """
    ch = _to_float_gray(channel)
    lap = cv2.Laplacian(ch, cv2.CV_32F, ksize=ksize)
    base = ch - lap
    return base.astype(np.float32)


# ------------------------------ 4. MASTER NIR ESTIMATOR ------------------------------

def estimate_nir(frame,
                 w_basic=0.5, w_fft=0.3, w_lap=0.2,
                 radius_ratio=0.08):
    """
    Main NIR estimation method.
    Uses:
      - Basic RGB mix
      - FFT low-frequency extraction
      - Laplacian base layer
    Wavelet removed to ensure maximum compatibility.

    Returns:
        nir_est (float32)
        components (dict)
    """

    b, g, r = cv2.split(frame)

    # Individual components
    nir_basic = basic_nir_mix(b, g, r)
    nir_fft   = fft_lowpass(r, radius_ratio)
    nir_lap   = laplacian_base(r)

    # Weighted fusion
    w_sum = float(w_basic + w_fft + w_lap)
    nir_est = (w_basic*nir_basic +
               w_fft*nir_fft +
               w_lap*nir_lap) / (w_sum + 1e-6)

    components = {
        "nir_basic": nir_basic,
        "nir_fft":   nir_fft,
        "nir_lap":   nir_lap,
    }

    return nir_est.astype(np.float32), components


# ------------------------------ 5. SELF TEST ------------------------------

if __name__ == "__main__":
    test = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
    nir_est, comps = estimate_nir(test)
    print("NIR_est:", nir_est.shape)
    for k, v in comps.items():
        print(k, v.shape)
