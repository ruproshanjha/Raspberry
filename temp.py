# grid_display.py
import cv2
import numpy as np

# Normalize any float image to 8-bit
def norm8(x):
    x = x.astype(np.float32)
    x = cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX)
    return x.astype(np.uint8)

# Convert grayscale → BGR for stacking
def to_bgr(x):
    if len(x.shape) == 2:
        return cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)
    return x

def make_grid(input_frame, R, G, B, NIR, NDVI, CROP, WEED, VARI, out_w=480, out_h=320):
    # Normalize all channels
    R = norm8(R)
    G = norm8(G)
    B = norm8(B)
    NIR = norm8(NIR)
    NDVI = norm8(NDVI)
    CROP = norm8(CROP)
    WEED = norm8(WEED)
    VARI = norm8(VARI)

    # Colormaps for NDVI and VARI
    NDVI = cv2.applyColorMap(NDVI, cv2.COLORMAP_JET)
    VARI = cv2.applyColorMap(VARI, cv2.COLORMAP_JET)

    # Convert base channels to BGR
    R = to_bgr(R)
    G = to_bgr(G)
    B = to_bgr(B)
    NIR = to_bgr(NIR)
    CROP = to_bgr(CROP)
    WEED = to_bgr(WEED)

    # Resize everything to equal tile size
    # 3 columns, 3 rows → 9 tiles
    TW = out_w // 3
    TH = out_h // 3

    def resize_tile(t):
        return cv2.resize(t, (TW, TH))

    tiles = [
        [resize_tile(input_frame), resize_tile(R), resize_tile(G)],
        [resize_tile(B), resize_tile(NIR), resize_tile(NDVI)],
        [resize_tile(CROP), resize_tile(WEED), resize_tile(VARI)]
    ]

    # Build grid
    row1 = np.hstack(tiles[0])
    row2 = np.hstack(tiles[1])
    row3 = np.hstack(tiles[2])

    final = np.vstack([row1, row2, row3])

    return final
