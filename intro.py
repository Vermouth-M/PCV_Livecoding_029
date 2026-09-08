import cv2
import numpy as np

img = cv2.imread('Image/image.jpeg')
mode = 0
list_mode = ["Original", "Red", "Green", "Blue"]

source = "image"
cap = None  

def apply_mode(image, mode):
    if mode == 0:
        return image
    b, g, r = cv2.split(image)
    zeros = np.zeros_like(b)
    if mode == 1: 
        return cv2.merge([zeros, zeros, r])
    elif mode == 2:
        return cv2.merge([zeros, g, zeros])
    elif mode == 3:  
        return cv2.merge([b, zeros, zeros])

cv2.namedWindow("Image")

while True:
    if source == "webcam":
        ret, frame = cap.read()
        if not ret:
            print("Webcam Error")
            break
        display_img = apply_mode(frame, mode)
        wait_time = 1
    else:
        display_img = apply_mode(img, mode)
        wait_time = 0  

    label = f"{list_mode[mode]}"
    cv2.putText(display_img, label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Image", display_img)

    key = cv2.waitKey(wait_time) & 0xFF

    if key == 32:  # Spacebar
        mode = (mode + 1) % 4
    elif key == ord('1'):  # Webcam
        if source != "webcam":
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("Webcam Error")
            else:
                source = "webcam"
    elif key == ord('2'):  #image
        if source != "image":
            if cap is not None:
                cap.release()
                cap = None
            source = "image"
    elif key == 27:  # Esc
        break

if cap is not None:
    cap.release()
cv2.destroyAllWindows()