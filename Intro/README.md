# PCV_Livecoding_029

# 1.intro.py
## Kontrol Keyboard
 
| Tombol      | Fungsi                                      |
|-------------|----------------------------------------------|
| **Spasi**   | Ganti mode warna (Original → Red → Green → Blue) |
| **1**       | Pindah ke mode webcam (webcam laptop/source lain)            |
| **2**       | Pindah ke mode gambar (gambar rektorat)               |
| **Esc**     | Keluar dari program                          |
 
## Cara Kerja
 
- Gambar/frame webcam dipecah per channel warna menggunakan `cv2.split()`.
- Channel yang tidak dipilih di-nol-kan (`zeros`), lalu digabung kembali dengan `cv2.merge()` sehingga hanya satu warna yang tampil.
- Mode warna berlaku sama baik untuk gambar statis maupun feed webcam.
- Saat pindah ke mode webcam, `cv2.VideoCapture(0)` dibuka; saat kembali ke gambar, webcam otomatis di-release agar tidak berjalan di background.
