import cv2
import matplotlib.pyplot as plt

# Abre a imagem
original_image = cv2.imread('images/low_contrast.jpg', 0)
#original_image = cv2.imread('images/low.jpg', 0)
#original_image = cv2.imread('images/pout.jpg', 0)

# Equaliza o histograma da imagem
equalized_image = cv2.equalizeHist(original_image)

# Mostra as imagens
plt.figure()
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Original")
plt.imshow(original_image, cmap='gray')

plt.subplot(1,2,2)
plt.axis("off")
plt.title("After Histogram equalization")
plt.imshow(equalized_image, cmap='gray')

plt.show()
