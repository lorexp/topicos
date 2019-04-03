import cv2
import matplotlib.pyplot as plt


# Abre a imagem
original_image = cv2.imread("images/estadio.jpg")
#original_image = cv2.imread("images/coins.jpg")
#original_image = cv2.imread("images/goalkeeper.jpg")
hsv_original = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)

# Abre a imagem que gera o histograma
roi_image = cv2.imread("images/grama.jpg")
#roi_image = cv2.imread("images/bronze.jpg")
#roi_image = cv2.imread("images/pitch_ground.jpg")

hsv_roi = cv2.cvtColor(roi_image, cv2.COLOR_BGR2HSV)

# ROI histograma
roi_hist = cv2.calcHist([hsv_roi], [0, 1], None, [180, 256], [0, 180, 0, 256])

# Máscara
mask = cv2.calcBackProject([hsv_original], [0, 1], roi_hist, [0, 180, 0, 256], 1)

# Filtra a imagem para remover ruídos
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
mask = cv2.filter2D(mask, -1, kernel)
_, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
mask = cv2.merge((mask, mask, mask))

# Gera a imagem de saída
result = cv2.bitwise_and(original_image, mask)


# Mostra as imagens
plt.figure()

plt.subplot(1,4,1)
plt.axis("off")
plt.title("Original")
plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))

plt.subplot(1,4,2)
plt.axis("off")
plt.title("Roi")
plt.imshow(cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB))

plt.subplot(1,4,3)
plt.axis("off")
plt.title("Mask")
plt.imshow(mask)

plt.subplot(1,4,4)
plt.axis("off")
plt.title("Result")
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

plt.show()
