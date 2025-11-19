# indices.py
# All vegetation indices for multispectral Raspberry Pi project
# Fully NumPy + OpenCV compatible. No SciPy, no PyWavelets.

import cv2
import numpy as np


# ------------------------- Utility -------------------------

def norm8(x):
    """Normalize any float image to uint8 (0–255)."""
    x = x.astype(np.float32)
    x = cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX)
    return x.astype(np.uint8)


# ------------------------- NDVI -------------------------

def ndvi(nir, frame):
    b, g, r = cv2.split(frame)
    r = r.astype(np.float32)
    nir = nir.astype(np.float32)

    # Avoid division warnings
    bottom = (nir + r) + 1e-6

    ndvi_map = (nir - r) / bottom      # -1 to +1
    ndvi_map = (ndvi_map + 1) / 2.0    # 0 to 1
    ndvi_map = (ndvi_map * 255).astype(np.uint8)

    return ndvi_map


# ------------------------- VARI -------------------------

def vari(frame):
    b, g, r = cv2.split(frame)
    b = b.astype(np.float32)
    g = g.astype(np.float32)
    r = r.astype(np.float32)

    vari_map = (g - r) / (g + r - b + 1e-6)
    vari_map = (vari_map + 1) / 2.0
    vari_map = (vari_map * 255).astype(np.uint8)
    return vari_map


# ------------------------- EXG (Excess Green) -------------------------

def exg(frame):
    b, g, r = cv2.split(frame)
    b = b.astype(np.float32)
    g = g.astype(np.float32)
    r = r.astype(np.float32)

    exg_map = 2 * g - r - b
    return norm8(exg_map)


# ------------------------- WEED DETECTION -------------------------
# Simple method: weeds often have high EXG (very green), 
# and low NDVI (non-crop type vegetation).
# This is just a placeholder until your dataset calibration.

def weed(exg_map):
    exg_map = exg_map.astype(np.uint8)

    # High green means weeds or grass
    _, binary = cv2.threshold(exg_map, 150, 255, cv2.THRESH_BINARY)

    # Clean mask
    binary = cv2.medianBlur(binary, 5)

    return binary


# ------------------------- CROP HEALTH -------------------------

def crop_health(ndvi_map):
    ndvi_map = ndvi_map.astype(np.uint8)

    # Healthy crops = high NDVI
    healthy_mask = cv2.inRange(ndvi_map, 150, 255)

    # Unhealthy = low NDVI
    unhealthy_mask = cv2.inRange(ndvi_map, 0, 120)

    # Create color mask
    crop = np.zeros((ndvi_map.shape[0], ndvi_map.shape[1], 3), dtype=np.uint8)

    # Red = unhealthy
    crop[unhealthy_mask > 0] = (0, 0, 255)

    # Green = healthy
    crop[healthy_mask > 0] = (0, 255, 0)

    return crop
