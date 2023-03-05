# Hijra S. Otji
# F55121051

#import library
from PIL import Image, ImageChops
import numpy as np
import cv2

#memasukkan gambar
img1 = Image.open("Difference Image\gambar1.jpg")
img2 = Image.open("Difference Image\gambar2.png")

#membandingkan gambar
diif = ImageChops.difference(img1, img2)

#menampilkan perbandingan gambar 
diif.show()