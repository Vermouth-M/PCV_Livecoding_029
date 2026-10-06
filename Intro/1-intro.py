import cv2
import numpy as np

# ===== PENGATURAN (ubah di sini) =====
MODE = "video"        
IMAGE_PATH = "foto.jpg" 
VIDEO_SOURCE = 0        
# =====================================

NAMA_FILTER = ["Original", "Red", "Green", "Blue"]
nomor = 0 


def filter_warna(frame, nomor):
    b, g, r = cv2.split(frame)      
    nol = np.zeros_like(b)

    if nomor == 1:                  # Red
        return cv2.merge([nol, nol, r])
    if nomor == 2:                  # Green
        return cv2.merge([nol, g, nol])
    if nomor == 3:                  # Blue
        return cv2.merge([b, nol, nol])
    return frame                    # Original


if MODE == "image":
    img = cv2.imread(IMAGE_PATH)            # read image
    if img is None:
        print("Gambar tidak ditemukan:", IMAGE_PATH)
        exit()

    while True:
        hasil = filter_warna(img, nomor)
        cv2.putText(hasil, NAMA_FILTER[nomor], (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Filter Warna", hasil)   # show image

        key = cv2.waitKey(30) & 0xFF
        if key == 32:                       # spasi
            nomor = (nomor + 1) % 4
        elif key == ord("q"):
            break

else:
    cap = cv2.VideoCapture(VIDEO_SOURCE)

    while True:
        ret, frame = cap.read()
        if not ret:                         # video habis / webcam gagal
            break

        hasil = filter_warna(frame, nomor)
        cv2.putText(hasil, NAMA_FILTER[nomor], (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Filter Warna", hasil)

        key = cv2.waitKey(1) & 0xFF
        if key == 32:                       # spasi
            nomor = (nomor + 1) % 4
        elif key == ord("q"):
            break

    cap.release()

cv2.destroyAllWindows()