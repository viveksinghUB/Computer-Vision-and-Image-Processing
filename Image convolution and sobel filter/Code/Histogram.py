import numpy as np
import cv2
import matplotlib.pyplot as plt

#Step 1
img = cv2.imread("lena_gray.jpg",0)#loading the file to display
M,N= img.shape[:2]
newimg=np.copy(img)
intensities=np.arange(256)
H=np.zeros(256,dtype=int)
Hc=np.zeros(256)
T=np.zeros(256)

#Step 2
for i in range(256):
    H[i]=np.count_nonzero(img==i)

plt.figure('Image Histogram')  
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency of intensity level')
plt.title('Image Histogram')
plt.plot(intensities,H)
plt.show()  

#Step 3
Hp=np.cumsum(H)
plt.figure('Cumulative Histogram')  
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency of intensity level')
plt.title('Cumulative Histogram')
plt.plot(intensities,Hp)
plt.show() 

#Step 4
factor=(256.0-1.0)/(M*N)
T=factor*Hp
plt.figure('Transformation Function')  
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency of intensity level')
plt.title('Transformation Function')
plt.plot(intensities,T)

#Step 5

for i in range(0, M):
    for j in range(0, N):
        newimg[i][j] = T[img[i][j]]

Hnew=np.zeros(256,dtype=int)
for i in range(255):
    Hnew[i]=np.count_nonzero(newimg==i)

plt.figure('New Image Histogram')  
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency of intensity level')
plt.title('New Image Histogram')
plt.plot(intensities,Hnew)
plt.show()

#Display both image side by side
vis = np.concatenate((img, newimg), axis=1)
cv2.imshow('out.png', vis)