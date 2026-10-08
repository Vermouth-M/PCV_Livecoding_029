# Konversi model warna: RGB -> CMYK, HSV, HSI (versi sederhana)
# OpenCV hanya untuk baca & simpan gambar.
# Semua rumus konversi ditulis manual (tanpa cv2.cvtColor).

import os
import cv2
import numpy as np

# ---------- 1. Lokasi file ----------
folder = os.path.dirname(os.path.abspath(__file__))          # folder model-warna
path_input = os.path.join(folder, "Image", "image.jpeg")
folder_output = os.path.join(folder, "output")
os.makedirs(folder_output, exist_ok=True)

# ---------- 2. Baca gambar & ambil kanal R, G, B (skala 0..1) ----------
img = cv2.imread(path_input)             # OpenCV membaca urutan B, G, R
if img is None:
    print("Gambar tidak ditemukan:", path_input)
    raise SystemExit

img = img.astype(np.float64) / 255
B = img[:, :, 0]
G = img[:, :, 1]
R = img[:, :, 2]

nilai_max = np.maximum(np.maximum(R, G), B)
nilai_min = np.minimum(np.minimum(R, G), B)
selisih = nilai_max - nilai_min

# ---------- 3. RGB -> CMYK ----------
K = 1 - nilai_max
pembagi = 1 - K                                   # = nilai_max
pembagi[pembagi == 0] = 1                         # hindari bagi nol (piksel hitam)
C = (1 - R - K) / pembagi
M = (1 - G - K) / pembagi
Y = (1 - B - K) / pembagi

# ---------- 4. RGB -> HSV ----------
V = nilai_max

S_hsv = np.zeros_like(V)
ada = nilai_max > 0                             
S_hsv[ada] = selisih[ada] / nilai_max[ada]

H_hsv = np.zeros_like(V)                         
tidak_abu = selisih > 0
d = np.where(tidak_abu, selisih, 1)               

merah = tidak_abu & (nilai_max == R)
hijau = tidak_abu & (nilai_max == G) & ~merah
biru = tidak_abu & ~merah & ~hijau

H_hsv[merah] = (60 * ((G - B) / d) % 360)[merah]
H_hsv[hijau] = (60 * ((B - R) / d + 2))[hijau]
H_hsv[biru] = (60 * ((R - G) / d + 4))[biru]


I = (R + G + B) / 3

S_hsi = np.zeros_like(I)
ada = I > 0
S_hsi[ada] = 1 - nilai_min[ada] / I[ada]

atas = 0.5 * ((R - G) + (R - B))
bawah = np.sqrt((R - G) ** 2 + (R - B) * (G - B))
bawah[bawah == 0] = 1                
theta = np.arccos(np.clip(atas / bawah, -1, 1)) * 180 / np.pi

H_hsi = theta.copy()
H_hsi[B > G] = 360 - theta[B > G]                
H_hsi[selisih == 0] = 0

def simpan(nama, kanal, nilai_maks=1.0):
    gambar = (kanal / nilai_maks * 255 + 0.5).astype(np.uint8)   
    cv2.imwrite(os.path.join(folder_output, nama + ".png"), gambar)

simpan("RGB_R", R); simpan("RGB_G", G); simpan("RGB_B", B)
simpan("CMYK_C", C); simpan("CMYK_M", M); simpan("CMYK_Y", Y); simpan("CMYK_K", K)
simpan("HSV_H", H_hsv, 360); simpan("HSV_S", S_hsv); simpan("HSV_V", V)
simpan("HSI_H", H_hsi, 360); simpan("HSI_S", S_hsi); simpan("HSI_I", I)

# Contoh nilai satu piksel (pojok kiri atas)
print("Piksel (0,0)  RGB  :", round(R[0, 0] * 255), round(G[0, 0] * 255), round(B[0, 0] * 255))
print("              CMYK :", round(C[0, 0], 3), round(M[0, 0], 3), round(Y[0, 0], 3), round(K[0, 0], 3))
print("              HSV  :", round(H_hsv[0, 0], 1), round(S_hsv[0, 0], 3), round(V[0, 0], 3))
print("              HSI  :", round(H_hsi[0, 0], 1), round(S_hsi[0, 0], 3), round(I[0, 0], 3))
print("Selesai! Hasil disimpan di:", folder_output)