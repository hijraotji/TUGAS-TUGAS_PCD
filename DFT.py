# Import Library
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Membaca Gambar Dengan Grayscale
img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Menghitung DFT (Discrete Fourir Transform)
dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

# Menampilkan Plot Gambar Asli dan DFT (Discrete Fourir Transform)
fig, axs = plt.subplots(1, 2, figsize=(8, 6))
axs[0].imshow(img, cmap='gray')
axs[0].set_title('Input Image')
axs[1].imshow(magnitude_spectrum, cmap='gray')
axs[1].set_title('Magnitude Spectrum')
plt.show()