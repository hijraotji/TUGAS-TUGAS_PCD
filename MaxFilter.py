# Import Library
import cv2
import numpy as np

# Memasukkan Gambar
img = cv2.imread('image.jpg', 0)

# Menentukan Ukuran Kernel
ksize = 2

# Menerapkan Max Filter
max_img = cv2.dilate(img, np.ones((ksize, ksize), np.uint8))

# Menampilkan Hasil
cv2.imshow('Original Image', img)
cv2.imshow('Max Filter Result', max_img)
cv2.waitKey(0)
cv2.destroyAllWindows