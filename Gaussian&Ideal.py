import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load image
img = cv2.imread('lena.png', 0)

# DFT
dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

# Gaussian Lowpass Filter
rows, cols = img.shape
crow, ccol = rows // 2, cols // 2
D = 30  # cutoff radius
H = np.zeros((rows, cols, 2), np.float32)
for i in range(rows):
    for j in range(cols):
        distance = np.sqrt((i - crow) ** 2 + (j - ccol) ** 2)
        H[i, j] = np.exp(-0.5 * ((distance / D) ** 2))
Gaussian_lowpass = dft_shift * H
Gaussian_lowpass_shift = np.fft.ifftshift(Gaussian_lowpass)
img_back_Gaussian = cv2.idft(Gaussian_lowpass_shift)
img_back_Gaussian = cv2.magnitude(img_back_Gaussian[:, :, 0], img_back_Gaussian[:, :, 1])

# Ideal Highpass Filter
D_0 = 50  # cutoff radius
H_ideal = np.zeros((rows, cols, 2), np.float32)
for i in range(rows):
    for j in range(cols):
        distance = np.sqrt((i - crow) ** 2 + (j - ccol) ** 2)
        if distance > D_0:
            H_ideal[i, j] = 1
Ideal_highpass = dft_shift * H_ideal
Ideal_highpass_shift = np.fft.ifftshift(Ideal_highpass)
img_back_Ideal = cv2.idft(Ideal_highpass_shift)
img_back_Ideal = cv2.magnitude(img_back_Ideal[:, :, 0], img_back_Ideal[:, :, 1])

# Plot
plt.subplot(221), plt.imshow(img, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(222), plt.imshow(img_back_Gaussian, cmap='gray')
plt.title('Gaussian Lowpass Filter'), plt.xticks([]), plt.yticks([])
plt.subplot(223), plt.imshow(img, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(224), plt.imshow(img_back_Ideal, cmap='gray')
plt.title('Ideal Highpass Filter'), plt.xticks([]), plt.yticks([])
plt.show()