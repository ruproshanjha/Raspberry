import cv2
from dsp_nir import estimate_nir
from indices import ndvi, vari, exg, weed, crop_health
from grid_display import make_grid
from fb_display import fb_show   # For TFT output


def main():
    print("Starting Multispectral Camera...")

    # ---- Camera Init ----
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(3, 640)
    cap.set(4, 480)

    if not cap.isOpened():
        print("Camera FAILED to open")
        return
    print("Camera opened successfully")

    while True:
        # ---------------- READ FRAME ----------------
        ret, frame = cap.read()
        if not ret:
            print("Frame read failed")
            continue

        # ---------------- NIR ESTIMATION -------------
        nir, _ = estimate_nir(frame)

        # ---------------- INDICES ---------------------
        ndvi_map = ndvi(nir, frame)
        vari_map = vari(frame)
        exg_map  = exg(frame)
        weed_map = weed(exg_map)
        crop_map = crop_health(ndvi_map)

        # Split base channels
        b, g, r = cv2.split(frame)

        # ---------------- BUILD 3×3 GRID ---------------
        try:
            grid = make_grid(
                frame,
                r, g, b,
                nir,
                ndvi_map,
                crop_map,
                weed_map,
                vari_map,
                out_w=480,     # TFT width
                out_h=320      # TFT height
            )
        except Exception as e:
            print("GRID ERROR:", e)
            break

        # ---------------- DISPLAY ON HDMI ----------------
        cv2.imshow("MULTISPECTRAL GRID", grid)

        # ---------------- DISPLAY ON TFT -----------------
        # Try fb0 first (most SPI displays). If wrong, fb1 will be used.
        try:
            fb_show(grid, "/dev/fb0")
        except:
            fb_show(grid, "/dev/fb1")

        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


main()
