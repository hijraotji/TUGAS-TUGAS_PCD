# Import Library
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Membaca Gambar Dengan Grayscale
img = cv2.imread('Lena.jpg', cv2.IMREAD_GRAYSCALE)

# Menghitung FFT (Fast Fourier Transform)
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20 * np.log(np.abs(fshift))

# Menampilkan Plot Gambar Asli dan FFT (Fast Fourier Transform)
fig, axs = plt.subplots(1, 2, figsize=(8, 6))
axs[0].imshow(img, cmap='gray')
axs[0].set_title('Input Image')
axs[1].imshow(magnitude_spectrum, cmap='gray')
axs[1].set_title('Magnitude Spectrum')
plt.show()