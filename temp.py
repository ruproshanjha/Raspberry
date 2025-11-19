# dsp_nir.py
# DSP-based NIR estimation for Raspberry Pi multispectral pipeline

import cv2
import numpy as np
import pywt


def _to_float_gray(channel):
    """Ensure channel is float32 grayscale."""
    if channel.ndim == 3:
        channel = cv2.cvtColor(channel, cv2.COLOR_BGR2GRAY)
    return channel.astype(np.float32)


# ---------- 1. Basic RGB → NIR mix ----------

def basic_nir_mix(b, g, r):
    """
    Simple weighted mix of RGB to approximate NIR.
    Tune aR, aG, aB later with real data.
    """
    b = b.astype(np.float32)
    g = g.astype(np.float32)
    r = r.astype(np.float32)

    aR, aG, aB = 0.7, 0.2, 0.1
    nir_basic = aR * r + aG * g + aB * b
    return nir_basic


# ---------- 2. FFT low-pass component ----------

def fft_lowpass(channel, radius_ratio=0.08):
    """
    Take a channel, apply 2D FFT, keep only low frequencies
    inside a circular mask, and inverse FFT back.
    """
    ch = _to_float_gray(channel)

    f = np.fft.fft2(ch)
    fshift = np.fft.fftshift(f)

    h, w = ch.shape
    cy, cx = h // 2, w // 2
    radius = int(min(h, w) * radius_ratio)

    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
    mask = dist <= radius

    fshift_lp = fshift * mask
    f_ishift = np.fft.ifftshift(fshift_lp)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    return img_back.astype(np.float32)


# ---------- 3. Wavelet LL band using only pywt.dwt ----------

def wavelet_ll(channel):
    """
    Approximate 2D LL band using only 1D DWT (which exists in all PyWavelets versions).
    1) Haar DWT on each row -> keep approximation (cA)
    2) Haar DWT on each column of the result -> keep approximation
    3) Resize back to original size
    """
    ch = _to_float_gray(channel)

    # 1D DWT on rows
    LL_rows = []
    for row in ch:
        cA, cD = pywt.dwt(row, 'haar')
        LL_rows.append(cA)
    LL_rows = np.array(LL_rows, dtype=np.float32)

    # 1D DWT on columns of LL_rows
    LL_cols = []
    for col in LL_rows.T:
        cA, cD = pywt.dwt(col, 'haar')
        LL_cols.append(cA)
    LL_cols = np.array(LL_cols, dtype=np.float32).T  # back to (H', W')

    # Upsample to original size
    LL_up = cv2.resize(
        LL_cols,
        (ch.shape[1], ch.shape[0]),
        interpolation=cv2.INTER_LINEAR
    )

    return LL_up.astype(np.float32)


# ---------- 4. Laplacian base layer ----------

def laplacian_base(channel, ksize=3):
    """
    Edge vs base decomposition:
    base = channel - Laplacian(channel)
    Base is smoother, more NIR-dominant.
    """
    ch = _to_float_gray(channel)
    lap = cv2.Laplacian(ch, cv2.CV_32F, ksize=ksize)
    base = ch - lap
    return base.astype(np.float32)


# ---------- 5. Master NIR estimator ----------

def estimate_nir(frame, radius_ratio=0.08,
                 w_basic=0.4, w_fft=0.2, w_wave=0.2, w_lap=0.2):
    """
    Main function used by main.py

    Input:
        frame: BGR frame (uint8)
    Output:
        nir_est: fused NIR estimate (float32)
        components: dict of individual components for debugging / visualization
    """
    # Work at full resolution; if needed, resize before calling this
    b, g, r = cv2.split(frame)

    nir_basic = basic_nir_mix(b, g, r)
    nir_fft   = fft_lowpass(r, radius_ratio=radius_ratio)
    nir_wave  = wavelet_ll(r)
    nir_lap   = laplacian_base(r, ksize=3)

    # Weighted fusion
    w_sum = float(w_basic + w_fft + w_wave + w_lap)
    nir_est = (w_basic * nir_basic +
               w_fft   * nir_fft +
               w_wave  * nir_wave +
               w_lap   * nir_lap) / (w_sum + 1e-6)

    components = {
        "nir_basic": nir_basic,
        "nir_fft":   nir_fft,
        "nir_wave":  nir_wave,
        "nir_lap":   nir_lap,
    }

    return nir_est.astype(np.float32), components


# ---------- 6. Simple self-test ----------

if __name__ == "__main__":
    # Quick sanity check with a dummy image
    test = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
    nir_est, comps = estimate_nir(test)
    print("NIR_est shape:", nir_est.shape)
    for k, v in comps.items():
        print(k, v.shape)
