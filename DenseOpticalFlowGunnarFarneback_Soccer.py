import numpy as np # import NumPy module
import cv2         # import OpenCV module.

# Initiate video capture for video file
cap = cv2.VideoCapture('Figures\SoccerTraining.mp4')
#cap = cv2.VideoCapture('Figures\SoccerTraining.mp4') 
ret, frame1 = cap.read() # Reading the first frames from video 
PrevGrayframe = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY) # convert to gray-scale image
hsv_mask = np.zeros_like(frame1) # create zero mask with same dimension as frame1
hsv_mask[...,1] = 255 # Set image saturation value to its max value

while cap.isOpened():
    ret, frame2 = cap.read() # Reading next frames from video 
    NextGrayframe = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY) # convert to gray-scale image

    # Calculates dense optical flow by Gunnar-Farneback method
    flow = cv2.calcOpticalFlowFarneback(PrevGrayframe, NextGrayframe, None, pyr_scale = 0.5,\
           levels = 3, winsize = 15, iterations = 3, poly_n = 5, poly_sigma = 1.2, flags = 0)
    # Compute magnitude and phase angle of 2D velocity vector
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    # Set hue channel according to the angle of velocity vector
    hsv_mask[...,0] = ang*180/np.pi/2
    # Set intensity channle according to normalized magnitude of velocity vector
    hsv_mask[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    # Convert from HSV to BGR color-space
    rgb_mask = cv2.cvtColor(hsv_mask,cv2.COLOR_HSV2BGR)
    
    cv2.imshow('Original Frame',frame2)
    cv2.imshow('Dense Optical Flow',rgb_mask)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite('opticalfb.png',frame2)
        cv2.imwrite('opticalhsv.png',rgb_mask)
    PrevGrayframe = NextGrayframe

cap.release()
cv2.destroyAllWindows()

