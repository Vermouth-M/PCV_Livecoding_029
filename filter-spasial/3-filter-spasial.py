import os
import cv2
import numpy as np

FOLDER_HASIL = r"filter-spasial\hasil_filter"
PATH_CITRA = r"filter-spasial\Image\image.jpeg"


# ======================================================================
# FUNGSI INTI
# ======================================================================

def konvolusi(f, w, tepi="replicate"):
    """Konvolusi manual, dipertahankan eksplisit agar mekanikanya terlihat."""
    m, n = w.shape
    a, b = m // 2, n // 2
    w_putar = np.flipud(np.fliplr(w))
    mode = {"zero": "constant", "replicate": "edge", "reflect": "reflect"}[tepi]
    f_pad = np.pad(f.astype(np.float64), ((a, a), (b, b)), mode=mode)
    g = np.zeros(f.shape, dtype=np.float64)
    for x in range(f.shape[0]):
        for y in range(f.shape[1]):
            jendela = f_pad[x:x + m, y:y + n]
            g[x, y] = np.sum(jendela * w_putar)
    return g


def ke_uint8(g):
    return np.clip(np.floor(g + 0.5), 0, 255).astype(np.uint8)


def unsharp(im, k=1.0, ukuran=3):
    imf = im.astype(np.float64)
    halus = cv2.blur(imf, (ukuran, ukuran))
    mask = imf - halus
    hasil = imf + k * mask
    return np.clip(hasil, 0, 255).astype(np.uint8)


def baca_citra():
    """Baca citra langsung dari PATH_CITRA (image/image.jpeg), tanpa tanya-tanya lagi."""
    if not os.path.isfile(PATH_CITRA):
        print(f"[!] File tidak ditemukan: {PATH_CITRA}")
        return None
    im = cv2.imread(PATH_CITRA, cv2.IMREAD_GRAYSCALE)
    if im is None:
        print(f"[!] Gagal membaca sebagai citra grayscale: {PATH_CITRA}")
        return None
    print(f"[i] Memindai citra: {PATH_CITRA} ({im.shape[1]}x{im.shape[0]})")
    return im


def simpan(nama, gambar):
    os.makedirs(FOLDER_HASIL, exist_ok=True)
    tujuan = os.path.join(FOLDER_HASIL, nama)
    cv2.imwrite(tujuan, gambar)
    print(f"    -> disimpan: {tujuan}")


# ======================================================================
# 1. DEMO KONVOLUSI MANUAL (tanpa file citra)
# ======================================================================

def menu_konvolusi():
    print("\n--- Demo Konvolusi Manual ---")
    f = np.array([[10, 10, 10, 10, 10],
                  [10, 50, 50, 50, 10],
                  [10, 50, 150, 50, 10],
                  [20, 40, 40, 40, 20],
                  [20, 20, 20, 20, 20]], dtype=np.float64)
    w = np.ones((3, 3), np.float64) / 9.0

    for tepi in ("zero", "replicate", "reflect"):
        hasil = konvolusi(f, w, tepi=tepi)
        print(f"\nHasil konvolusi (tepi={tepi}):")
        print(np.round(hasil, 2))

    # pembanding dengan OpenCV
    hasil_cv = cv2.filter2D(f, -1, cv2.flip(w, -1), borderType=cv2.BORDER_REPLICATE)
    print("\nHasil cv2.filter2D (tepi replicate, untuk verifikasi):")
    print(np.round(hasil_cv, 2))


# ======================================================================
# 2. SMOOTHING / LOWPASS
# ======================================================================

def menu_smoothing():
    print("\n--- Smoothing / Lowpass ---")
    im = baca_citra()
    if im is None:
        return

    hasil_box = cv2.blur(im, (3, 3))
    hasil_gauss = cv2.GaussianBlur(im, (5, 5), sigmaX=1.0)
    hasil_median = cv2.medianBlur(im, 3)

    print("\nPerbandingan simpangan baku (kontras):")
    for nama, hasil in [("asli", im), ("box", hasil_box),
                        ("gaussian", hasil_gauss), ("median", hasil_median)]:
        print("  %-8s std = %.2f" % (nama, hasil.std()))

    simpan("smoothing_box.png", hasil_box)
    simpan("smoothing_gaussian.png", hasil_gauss)
    simpan("smoothing_median.png", hasil_median)


# ======================================================================
# 3. SHARPENING / HIGHPASS
# ======================================================================

def menu_sharpening():
    print("\n--- Sharpening / Highpass ---")
    im = baca_citra()
    if im is None:
        return
    imf = im.astype(np.float64)

    kernel_lap = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], np.float64)
    lap = cv2.filter2D(imf, -1, kernel_lap, borderType=cv2.BORDER_REPLICATE)
    hasil_lap = np.clip(imf - lap, 0, 255).astype(np.uint8)

    hasil_unsharp = unsharp(im, k=1.0)
    hasil_highboost = unsharp(im, k=2.5)

    print("\nPerbandingan simpangan baku (kontras):")
    for nama, hasil in [("asli", im), ("laplacian", hasil_lap),
                        ("unsharp k=1.0", hasil_unsharp), ("highboost k=2.5", hasil_highboost)]:
        print("  %-16s std = %.2f" % (nama, hasil.std()))

    simpan("sharpening_laplacian.png", hasil_lap)
    simpan("sharpening_unsharp.png", hasil_unsharp)
    simpan("sharpening_highboost.png", hasil_highboost)


# ======================================================================
# 4. DETEKSI TEPI (SOBEL)
# ======================================================================

def menu_tepi():
    print("\n--- Deteksi Tepi (Sobel) ---")
    im = baca_citra()
    if im is None:
        return
    imf = im.astype(np.float64)

    gx = cv2.Sobel(imf, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(imf, cv2.CV_64F, 0, 1, ksize=3)
    besar_murah = np.abs(gx) + np.abs(gy)
    besar_tepat = np.sqrt(gx**2 + gy**2)
    arah = np.arctan2(gy, gx)

    hasil_murah = np.clip(besar_murah, 0, 255).astype(np.uint8)
    hasil_tepat = np.clip(besar_tepat, 0, 255).astype(np.uint8)

    print(f"  std gradien (|gx|+|gy|) = {hasil_murah.std():.2f}")
    print(f"  std gradien (sqrt)      = {hasil_tepat.std():.2f}")
    print(f"  rentang arah gradien    = {arah.min():.2f} s.d. {arah.max():.2f} radian")

    simpan("tepi_sobel_x.png", np.clip(np.abs(gx), 0, 255).astype(np.uint8))
    simpan("tepi_sobel_y.png", np.clip(np.abs(gy), 0, 255).astype(np.uint8))
    simpan("tepi_sobel_magnitude.png", hasil_tepat)


# ======================================================================
# MENU UTAMA
# ======================================================================

def tampilkan_menu():
    print("\n=========== MENU FILTER CITRA ===========")
    print("1. Demo konvolusi manual (tanpa file citra)")
    print("2. Smoothing / lowpass (box, gaussian, median)")
    print("3. Sharpening / highpass (laplacian, unsharp, highboost)")
    print("4. Deteksi tepi (sobel)")
    print("0. Keluar")
    print("===========================================")


def main():
    aksi = {
        "1": menu_konvolusi,
        "2": menu_smoothing,
        "3": menu_sharpening,
        "4": menu_tepi,
    }
    while True:
        tampilkan_menu()
        pilihan = input("Pilihan: ").strip()
        if pilihan == "0":
            print("Keluar.")
            break
        fungsi = aksi.get(pilihan)
        if fungsi is None:
            print("[!] Pilihan tidak dikenal, coba lagi.")
            continue
        fungsi()


if __name__ == "__main__":
    main()