import cv2
import numpy as np
from matplotlib import pyplot as plt

# Read image in grayscale
img = cv2.imread('Lena.jpg', cv2.IMREAD_GRAYSCALE)

# Compute DFT
dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

# Ideal Lowpass Filter
rows, cols = img.shape
crow, ccol = rows // 2, cols // 2
mask = np.zeros((rows, cols, 2), np.float32)
mask[crow - 30 : crow + 30, ccol - 30 : ccol + 30] = 1
fshift = dft_shift * mask
ideal_lp = cv2.idft(np.fft.ifftshift(fshift))
ideal_lp = cv2.magnitude(ideal_lp[:, :, 0], ideal_lp[:, :, 1])

# Butterworth Lowpass Filter
D0 = 30
n = 4
u, v = np.meshgrid(np.arange(-cols/2, cols/2), np.arange(-rows/2, rows/2))
d = np.sqrt(u**2 + v**2)
H = 1 / (1 + (d/D0)**(2*n))
H = cv2.resize(H, (cols, rows))
bw_lp = cv2.idft(np.fft.ifftshift(fshift))
bw_lp = cv2.magnitude(bw_lp[:, :, 0], bw_lp[:, :, 1])

# Plot original image and filtered images
fig, axs = plt.subplots(1, 3, figsize=(12, 6))
axs[0].imshow(img, cmap='gray')
axs[0].set_title('Input Image')
axs[1].imshow(ideal_lp, cmap='gray')
axs[1].set_title('Ideal Lowpass Filter')
axs[2].imshow(bw_lp, cmap='gray')
axs[2].set_title('Butterworth Lowpass Filter')
plt.show()
