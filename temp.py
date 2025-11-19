import cv2
import numpy as np

def norm8(x):
    x = x.astype(np.float32)
    x = cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX)
    return x.astype(np.uint8)

def to_bgr(x):
    if len(x.shape) == 2:
        return cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)
    return x

def make_grid(inp, R, G, B, NIR, NDVI, CROP, WEED, VARI, out_w=480, out_h=320):

    # Normalize everything
    R = to_bgr(norm8(R))
    G = to_bgr(norm8(G))
    B = to_bgr(norm8(B))
    NIR = to_bgr(norm8(NIR))
    CROP = to_bgr(norm8(CROP))
    WEED = to_bgr(norm8(WEED))
    NDVI = cv2.applyColorMap(norm8(NDVI), cv2.COLORMAP_JET)
    VARI = cv2.applyColorMap(norm8(VARI), cv2.COLORMAP_JET)
    inp = cv2.resize(inp, (out_w//3, out_h//3))

    # Resize tiles
    TW = out_w // 3
    TH = out_h // 3

    def RZ(x): return cv2.resize(x, (TW, TH))

    row1 = np.hstack([inp, RZ(R), RZ(G)])
    row2 = np.hstack([RZ(B), RZ(NIR), RZ(NDVI)])
    row3 = np.hstack([RZ(CROP), RZ(WEED), RZ(VARI)])

    final = np.vstack([row1, row2, row3])

    # Final enforcement (very important)
    final = cv2.resize(final, (out_w, out_h))
    return final
