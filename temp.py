import cv2
from dsp_nir import estimate_nir
from indices import ndvi, vari, exg, weed, crop_health
from grid_display import make_grid

def main():
    print("Starting...")

    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    print("Camera opened:", cap.isOpened())

    while True:
        ret, frame = cap.read()
        print("frame read:", ret)
        if not ret:
            break

        print("frame shape:", frame.shape)

        # ----- NIR -----
        nir, parts = estimate_nir(frame)
        print("nir:", nir.shape)

        # ----- vegetation indices -----
        ndvi_map = ndvi(nir, frame)
        print("ndvi:", ndvi_map.shape)

        vari_map = vari(frame)
        print("vari:", vari_map.shape)

        exg_map = exg(frame)
        print("exg:", exg_map.shape)

        weed_map = weed(exg_map)
        print("weed:", weed_map.shape)

        crop_map = crop_health(ndvi_map)
        print("crop:", crop_map.shape)

        b, g, r = cv2.split(frame)

        # ----- GRID ------
        try:
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
            print("grid:", grid.shape)
        except Exception as e:
            print("GRID ERROR:", e)
            break

        cv2.imshow("GRID", grid)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

main()
