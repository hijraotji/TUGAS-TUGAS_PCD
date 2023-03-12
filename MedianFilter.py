# Import Library
import cv2
import numpy as np

# Memasukkan Gambar
img = cv2.imread('image.jpg', 0)

# Menentukan Ukuran Karnel
ksize = 5

# Menerapkan Median Filter
median_img = cv2.medianBlur(img, ksize)

# Menampilkan Hasil
cv2.imshow('Original Image', img)
cv2.imshow('Median Filter Result', median_img)
cv2.waitKey(0)
cv2.destroyAllWindows()