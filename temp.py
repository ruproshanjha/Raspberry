# fb_display.py
# Correct framebuffer writer with RGB565 conversion

import fcntl
import struct
import numpy as np


# ioctl codes for framebuffer
FBIOGET_VSCREENINFO = 0x4600
FBIOGET_FSCREENINFO = 0x4602


def bgr2rgb565(img):
    """Convert 480x320 BGR888 → RGB565."""
    b = img[:,:,0].astype(np.uint16)
    g = img[:,:,1].astype(np.uint16)
    r = img[:,:,2].astype(np.uint16)

    rgb565 = ((r & 0xF8) << 8) | \
             ((g & 0xFC) << 3) | \
             (b >> 3)

    return rgb565.astype(np.uint16)


def fb_show(img, fb="/dev/fb0"):
    # Open framebuffer
    f = open(fb, "wb")

    # Get screen info
    vinfo = fcntl.ioctl(f, FBIOGET_VSCREENINFO, bytes(160))
    xres, yres, bpp = struct.unpack_from("I I I", vinfo, 0)

    # Verify resolution
    if img.shape[1] != xres or img.shape[0] != yres:
        raise ValueError(f"Image must be {xres}x{yres}, got {img.shape[1]}x{img.shape[0]}")

    # Convert to correct bit depth
    if bpp == 16:
        buf = bgr2rgb565(img).tobytes()
    elif bpp == 24:
        buf = img[:, :, ::-1].tobytes()  # BGR → RGB888
    else:
        raise ValueError(f"Unsupported framebuffer depth: {bpp}")

    f.write(buf)
    f.close()
