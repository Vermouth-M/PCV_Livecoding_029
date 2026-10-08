import os
import cv2
import numpy as np

folder = os.path.dirname(os.path.abspath(__file__))          # folder ti-eq
path_input = os.path.join(folder, "Image", "image.jpeg")
folder_output = os.path.join(folder, "output")
os.makedirs(folder_output, exist_ok=True)

img = cv2.imread(path_input)
if img is None:
    print("Gambar tidak ditemukan:", path_input)
    raise SystemExit

B = img[:, :, 0]
G = img[:, :, 1]
R = img[:, :, 2]
gray = (0.114 * B + 0.587 * G + 0.299 * R).astype(np.uint8)
r = gray.astype(np.float64)   # versi desimal untuk rumus

negatif = 255 - gray                                   # s = 255 - r

c = 255 / np.log(1 + 255)
log_img = (c * np.log(1 + r)).astype(np.uint8)         # s = c * log(1 + r)

gamma = 0.5
gamma_img = (255 * (r / 255) ** gamma).astype(np.uint8)  # s = 255 * (r/255)^gamma

r_min = gray.min()
r_max = gray.max()
stretch = ((r - r_min) / (r_max - r_min) * 255).astype(np.uint8)  # contrast stretching

def hitung_histogram(citra):
    hist = np.zeros(256, dtype=int)
    for k in range(256):                 # hitung berapa piksel yang bernilai k
        hist[k] = np.sum(citra == k)
    return hist


def ekualisasi(citra):
    hist = hitung_histogram(citra)
    total = citra.shape[0] * citra.shape[1]

    cdf = np.zeros(256)                  # CDF = jumlah kumulatif peluang
    jumlah = 0
    for k in range(256):
        jumlah = jumlah + hist[k]
        cdf[k] = jumlah / total

    tabel = (255 * cdf + 0.5).astype(np.uint8)   # s = round(255 * CDF)
    return tabel[citra]                           # ganti tiap piksel dengan nilai baru


def gambar_histogram(citra):
    hist = hitung_histogram(citra)
    kanvas = np.full((200, 512), 255, dtype=np.uint8)   # kanvas putih
    tertinggi = hist.max()
    for k in range(256):
        tinggi = int(hist[k] / tertinggi * 180)
        cv2.rectangle(kanvas, (k * 2, 199 - tinggi), (k * 2 + 1, 199), 0, -1)
    return kanvas


hasil_eq = ekualisasi(gray)
semua = {
    "1_original": gray,
    "2_negatif": negatif,
    "3_log": log_img,
    "4_gamma": gamma_img,
    "5_stretch": stretch,
    "6_ekualisasi": hasil_eq,
}

for nama, citra in semua.items():
    cv2.imwrite(os.path.join(folder_output, nama + ".png"), citra)
    cv2.imwrite(os.path.join(folder_output, nama + "_histogram.png"), gambar_histogram(citra))

print("Selesai! Hasil disimpan di:", folder_output)