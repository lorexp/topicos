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
print('Img         ', np.amax(A0), np.amin(A0))
#>>> Img          255.0 0.0

#Kernel for negative Laplacian
kernel = np.array([[-1,-1,-1], 
                   [-1, 8,-1],
                   [-1,-1,-1]])

#Convolution of the image with the kernel:
Lap = cv2.filter2D(A0, -1, kernel)

#Laplacian now has negative values
print('Original Lap', np.amax(Lap), np.amin(Lap))
#>>> Original Lap 1151.0 -1166.0

#Map Laplacian to some new range:
Laps = Lap*100.0/np.amax(Lap) #Sharpening factor!
print('Scaled Lap  ', np.amax(Laps), np.amin(Laps))
#>>> Scaled Lap   100.0 -101.303

A = A0 + Laps #Add negative Laplacian to the original image

print('SharpImg    ', np.amax(A), np.amin(A))
#>>> SharpImg     350.917 -81.06

A = abs(A) #Get rid of negative values
print('SharpImg abs', np.amax(A), np.amin(A))

A *= 255.0/np.amax(A)
print('SharpImg after scaling', np.amax(A), np.amin(A))
#>>> SharpImg abs 350.917 0.0

# Local Histogram Equalization with OpenCV:
A_cv2 = cv2.equalizeHist(A.astype(np.uint8))

plt.figure(pic_n)
pic_n += 1
plt.subplot(1,2,1)
show_abs(Lap, 'Laplacian')
plt.subplot(1,2,2)
show_abs(Laps, 'Scaled Laplacian')

plt.figure(pic_n)
pic_n += 1
plt.subplot(1,3,1)
show_abs(A0, 'Original image')
plt.subplot(1,3,2)
show_abs(A, 'Laplacian filtered img')
plt.subplot(1,3,3)
show_abs(A_cv2, 'Local Hist equalized img')
plt.show()