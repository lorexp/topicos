import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2

#Function for plotting abs:
pic_n = 1

def show_abs(I, plot_title):
    plt.title(plot_title)
    plt.tight_layout()
    plt.axis('off')
    plt.imshow(abs(I), cm.gray)

#Reading of the image into numpy array:
A0 = cv2.imread('images/moon1.jpg', 0)
A0 = np.float64(A0)
A0 -= np.amin(A0)#map values to the (0, 255) range
A0 *= 255.0/np.amax(A0)

#Kernel for negative Laplacian
kernel = np.array([[-1,-1,-1], 
                   [-1, 8,-1],
                   [-1,-1,-1]])

#Convolution of the image with the kernel:
Lap = cv2.filter2D(A0, -1, kernel)

#Map Laplacian to some new range:
Laps = Lap*100.0/np.amax(Lap) #Sharpening factor!

A = A0 + Laps #Add negative Laplacian to the original image

A = abs(A) #Get rid of negative values

A *= 255.0/np.amax(A)

plt.figure(pic_n)
pic_n += 1
plt.subplot(1,3,1)
show_abs(A0, 'Original image')
plt.subplot(1,3,2)
show_abs(Laps, 'Scaled Laplacian')
plt.subplot(1,3,3)
show_abs(A, 'Laplacian filtered img')
plt.show()