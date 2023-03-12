# Import Library
import cv2
import numpy as np

# Memasukkan Gambar
img = cv2.imread('image.jpg', 0)

# Menentukan Ukuran Karnel
ksize = 3

# Menerapkan Min Filter
min_img = cv2.erode(img, np.ones((ksize, ksize), np.uint8))

# Menampilkan Hasil
cv2.imshow('Original Image', img)
cv2.imshow('Min Filter Result', min_img)
cv2.waitKey(0)
cv2.destroyAllWindows()