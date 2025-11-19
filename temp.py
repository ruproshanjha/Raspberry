# main.py
import cv2
from dsp_nir import estimate_nir
from indices import ndvi, vari, exg, weed, crop_health
from grid_display import make_grid

def main():
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(3, 640)
    cap.set(4, 480)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Estimate NIR
        nir, parts = estimate_nir(frame)

        # Vegetation indices
        ndvi_map = ndvi(nir, frame)
        vari_map = vari(frame)
        exg_map  = exg(frame)
        weed_map = weed(exg_map)
        crop_map = crop_health(ndvi_map)

        # Channels
        b, g, r = cv2.split(frame)

        # Build exact 480×320 grid
        grid = make_grid(
            frame,
            r, g, b,
            nir,
            ndvi_map,
            crop_map,
            weed_map,
            vari_map,
            out_w=480,
            out_h=320
        )

        cv2.imshow("MULTISPECTRAL GRID", grid)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

main()
