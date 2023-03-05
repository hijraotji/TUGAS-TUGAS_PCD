# Hijra S. Otji
# F55121051

#import library
import numpy as np
import cv2

def avgblur(image, shift) :
    image=cv2.blur(image,(shift,shift))
    return image

gbr = cv2.imread("Averrage Filter\LENA.jpg")
x = avgblur(gbr, 20)
cv2.imshow("Gambar Lena",gbr)
cv2.imshow("Hasil",x)

cv2.waitKey()