import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
img = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)

# Calculate FFT
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)

# Set filter parameters
D0 = 30  # cutoff frequency
n = 4  # order of the filter

# Butterworth Highpass Filter
rows, cols = img.shape
crow, ccol = rows // 2, cols // 2
butterworth_highpass = np.zeros((rows, cols), dtype=np.float32)
for i in range(rows):
    for j in range(cols):
        D = np.sqrt((i - crow) ** 2 + (j - ccol) ** 2)
        if D == 0:
            D = 0.00000001
        butterworth_highpass[i, j] = 1 / (1 + (D0 / D) ** (2 * n))

# Apply Butterworth Highpass Filter
butterworth_filtered = np.multiply(fshift, butterworth_highpass)
butterworth_img = np.fft.ifft2(np.fft.ifftshift(butterworth_filtered)).real

# Gaussian Highpass Filter
gaussian_highpass = np.zeros((rows, cols), dtype=np.float32)
for i in range(rows):
    for j in range(cols):
        D = np.sqrt((i - crow) ** 2 + (j - ccol) ** 2)
        gaussian_highpass[i, j] = 1 - np.exp(-(D ** 2) / (2 * (D0 ** 2)))

# Apply Gaussian Highpass Filter
gaussian_filtered = np.multiply(fshift, gaussian_highpass)
gaussian_img = np.fft.ifft2(np.fft.ifftshift(gaussian_filtered)).real

# Display images
plt.figure(figsize=(10, 10))
plt.subplot(221), plt.imshow(img, cmap='gray'), plt.title('Original Image')
plt.axis('off')
plt.subplot(222), plt.imshow(butterworth_img, cmap='gray'), plt.title('Butterworth Highpass Filter')
plt.axis('off')
plt.subplot(223), plt.imshow(gaussian_img, cmap='gray'), plt.title('Gaussian Highpass Filter')
plt.axis('off')
plt.show()