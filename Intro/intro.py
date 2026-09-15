from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Folder output — ganti sesuai lokasi folder Anda (pastikan folder sudah ada)
OUTPUT_DIR = "output/"

def baca_gambar_grayscale(path):
    img = Image.open(path).convert("RGB")
    arr = np.array(img, dtype=np.float64)  # shape (H, W, 3)

    H, W, _ = arr.shape
    gray = np.zeros((H, W), dtype=np.float64)

    # Rumus luminansi manual: 0.299R + 0.587G + 0.114B
    for i in range(H):
        for j in range(W):
            r, g, b = arr[i, j]
            gray[i, j] = 0.299 * r + 0.587 * g + 0.114 * b

    return np.clip(gray, 0, 255).astype(np.uint8)


def transformasi_negatif(img):
    """s = (L-1) - r"""
    L = 256
    hasil = np.zeros_like(img)
    H, W = img.shape
    for i in range(H):
        for j in range(W):
            hasil[i, j] = (L - 1) - img[i, j]
    return hasil


def transformasi_log(img, c=None):
    """s = c * log(1 + r)"""
    H, W = img.shape
    r_max = img.max()
    if c is None:
        c = 255 / np.log(1 + r_max) if r_max > 0 else 1

    hasil = np.zeros((H, W), dtype=np.float64)
    for i in range(H):
        for j in range(W):
            hasil[i, j] = c * np.log(1 + img[i, j])

    return np.clip(hasil, 0, 255).astype(np.uint8)


def transformasi_gamma(img, gamma=1.0, c=1.0):
    """s = c * r^gamma  (power-law transform)"""
    H, W = img.shape
    hasil = np.zeros((H, W), dtype=np.float64)
    r_norm = img.astype(np.float64) / 255.0

    for i in range(H):
        for j in range(W):
            hasil[i, j] = c * (r_norm[i, j] ** gamma) * 255

    return np.clip(hasil, 0, 255).astype(np.uint8)


def peregangan_kontras(img, r1, s1, r2, s2):
    """
    Contrast stretching linear piecewise:
    - [0, r1]   -> [0, s1]
    - [r1, r2]  -> [s1, s2]
    - [r2, 255] -> [s2, 255]
    """
    H, W = img.shape
    hasil = np.zeros((H, W), dtype=np.float64)

    for i in range(H):
        for j in range(W):
            r = img[i, j]
            if r <= r1:
                s = (s1 / r1) * r if r1 != 0 else 0
            elif r <= r2:
                s = s1 + ((s2 - s1) / (r2 - r1)) * (r - r1)
            else:
                s = s2 + ((255 - s2) / (255 - r2)) * (r - r2) if r2 != 255 else s2
            hasil[i, j] = s

    return np.clip(hasil, 0, 255).astype(np.uint8)

def hitung_histogram(img, L=256):
    """Hitung frekuensi tiap level intensitas (0..L-1) secara manual."""
    hist = [0] * L
    H, W = img.shape
    for i in range(H):
        for j in range(W):
            hist[img[i, j]] += 1
    return hist


def hitung_cdf(hist):
    """Cumulative Distribution Function manual dari histogram."""
    cdf = [0] * len(hist)
    kumulatif = 0
    for k in range(len(hist)):
        kumulatif += hist[k]
        cdf[k] = kumulatif
    return cdf


def ekualisasi_histogram(img, L=256):
    """
    Ekualisasi histogram manual:
    1. Hitung histogram f(r_k)
    2. Hitung CDF
    3. Normalisasi CDF -> mapping s_k = round((L-1) * cdf(k) / (H*W))
    4. Petakan tiap piksel ke intensitas baru
    """
    H, W = img.shape
    total_piksel = H * W

    hist = hitung_histogram(img, L)
    cdf = hitung_cdf(hist)

    # Buat lookup table transformasi intensitas
    lut = [0] * L
    for k in range(L):
        lut[k] = round((L - 1) * cdf[k] / total_piksel)

    # Terapkan lookup table ke setiap piksel
    hasil = np.zeros((H, W), dtype=np.uint8)
    for i in range(H):
        for j in range(W):
            hasil[i, j] = lut[img[i, j]]

    return hasil, hist, lut


def tampilkan_hasil(img_asli, img_negatif, img_log, img_gamma, img_eq):
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))

    gambar = [img_asli, img_negatif, img_log, img_gamma, img_eq]
    judul = ["Asli", "Negatif", "Log", "Gamma", "Ekualisasi Histogram"]

    for k in range(5):
        axes[0, k].imshow(gambar[k], cmap="gray", vmin=0, vmax=255)
        axes[0, k].set_title(judul[k])
        axes[0, k].axis("off")

        axes[1, k].hist(gambar[k].ravel(), bins=256, range=(0, 255), color="black")
        axes[1, k].set_title(f"Histogram {judul[k]}")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + "hasil_transformasi.png", dpi=150)
    print("Hasil visualisasi disimpan di hasil_transformasi.png")


if __name__ == "__main__":
    path_gambar = "input.jpg"  # ganti dengan path gambar Anda

    img_gray = baca_gambar_grayscale(path_gambar)

    img_negatif = transformasi_negatif(img_gray)
    img_log = transformasi_log(img_gray)
    img_gamma = transformasi_gamma(img_gray, gamma=0.5)
    img_eq, hist_asli, lut = ekualisasi_histogram(img_gray)

    # Simpan hasil masing-masing sebagai file gambar
    Image.fromarray(img_gray).save(OUTPUT_DIR + "01_grayscale.png")
    Image.fromarray(img_negatif).save(OUTPUT_DIR + "02_negatif.png")
    Image.fromarray(img_log).save(OUTPUT_DIR + "03_log.png")
    Image.fromarray(img_gamma).save(OUTPUT_DIR + "04_gamma.png")
    Image.fromarray(img_eq).save(OUTPUT_DIR + "05_ekualisasi.png")

    tampilkan_hasil(img_gray, img_negatif, img_log, img_gamma, img_eq)

    print(f"Selesai. Semua hasil disimpan di folder '{OUTPUT_DIR}'.")