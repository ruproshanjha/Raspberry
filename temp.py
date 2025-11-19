# fb_display.py
# Display an image directly on Raspberry Pi framebuffer (TFT screen)

import numpy as np

def fb_show(img, fb="/dev/fb0"):
    """
    img must be a 480x320 BGR image.
    Will write directly to framebuffer (SPI TFT).
    """
    h, w, c = img.shape
    if w != 480 or h != 320:
        raise ValueError("Image must be exactly 480x320")

    # OpenCV gives BGR → framebuffer needs RGB
    rgb = img[:, :, ::-1]

    # Send raw bytes directly to framebuffer
    with open(fb, "wb") as f:
        f.write(rgb.tobytes())
