import numpy as np # import NumPy module
import cv2         # import OpenCV module
from matplotlib import pyplot as plt # import matplotlib module

# Reading an image in default mode  
frame1 = cv2.imread('Figures\SoccerPracticeFrame1.png')
frame2 = cv2.imread('Figures\SoccerPracticeFrame2.png')

# "absdiff": computing a per-element absolute difference of 2 frames
diff = cv2.absdiff(frame1, frame2)
gray_img = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY) # convert color- to gray-image
blur_img = cv2.GaussianBlur(gray_img, (5,5) , 0) # smooths image with "GaussianBlur": Kernel Size = 5x5
_, thresh = cv2.threshold(blur_img, 10, 255, cv2.THRESH_BINARY)

plt.subplot(221), plt.imshow(blur_img,cmap = 'gray'), plt.title("Blur Grayscale Image")
plt.subplot(222), plt.imshow(thresh,cmap = 'gray'), plt.title("Thresholded Image")

# "erode" method: thining foreground objects
ErodeImg = cv2.erode(thresh, None, iterations = 3)  
# "dilate" method: enlarging foreground objects
DilateImg = cv2.dilate(ErodeImg, None, iterations = 12) 

plt.subplot(223), plt.imshow(ErodeImg,cmap = 'gray'), plt.title("Eroded Image")
plt.subplot(224), plt.imshow(DilateImg,cmap = 'gray'), plt.title("Dilated Image")
plt.show()

