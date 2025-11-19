def estimate_nir(frame):
    b, g, r = cv2.split(frame)
    b = b.astype(np.float32)
    g = g.astype(np.float32)
    r = r.astype(np.float32)

    # Basic mix
    nir_basic = 0.7*r + 0.2*g + 0.1*b

    # FFT low pass
    nir_fft = fft_lowpass(r)

    # Laplacian base
    nir_lap = laplacian_base(r)

    # Fuse (no wavelet)
    nir_est = (0.5*nir_basic + 0.3*nir_fft + 0.2*nir_lap)

    components = {
        "nir_basic": nir_basic,
        "nir_fft": nir_fft,
        "nir_lap": nir_lap
    }
    return nir_est, components
