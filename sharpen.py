import cv2
import numpy as np

image = cv2.imread('images/moon.jpg')

kernel = np.array([[-1,-1,-1], 
                   [-1, 8,-1],
                   [-1,-1,-1]])
                   

sharpened = cv2.filter2D(image, -1, kernel)

cv2.imshow('Image Sharpening', sharpened)



cv2.waitKey(0)
cv2.destroyAllWindows()
