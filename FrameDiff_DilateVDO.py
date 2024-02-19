import numpy as np # import NumPy module
import cv2         # import OpenCV module.

# Initiate video capture for video file
cap = cv2.VideoCapture('Figures\SoccerPractice.mp4')
# Reading two consecutive frames from video 
ret, frame1 = cap.read()
ret, frame2 = cap.read()

while cap.isOpened():
    # "absdiff": computing a per-element absolute difference of 2 frames
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY) # convert color- to gray-image
    blur = cv2.GaussianBlur(gray, (5,5) , 0) # smooths image with "GaussianBlur": Kernel Size = 5x5
    _, thresh = cv2.threshold(blur, 10, 255, cv2.THRESH_BINARY)
    # "erode" method: thining foreground objects
    ErodeImg = cv2.erode(thresh, None, iterations = 3)
    # "dilate" method: enlarging foreground objects
    DilateImg = cv2.dilate(ErodeImg, None, iterations = 15)
    # "findContours" computes the object contour from binary image
    # from external horizontal, vertical, and diagonal line segments 
    contours, _ = cv2.findContours(DilateImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame1, contours, -1, (0,255,0), 2)  # -1 option for draw all contours
    
    # plot the moving contour
    cv2.imshow('Moving Contour',frame1)
    frame1 = frame2 # reset frame1 using frame2
    ret, frame2 = cap.read() # read next frame (frame2) from video
    if cv2.waitKey(10) == 27:
        break

cap.release()
cv2.destroyAllWindows()
