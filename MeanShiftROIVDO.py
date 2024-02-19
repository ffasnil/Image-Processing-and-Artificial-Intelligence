import numpy as np # import NumPy module
import cv2         # import OpenCV module.

# Initiate video capture for video file
cap = cv2.VideoCapture('Figures\slow.mp4')

frame = None      # each frame from VDO file
roiPts = []       # 4 corner point of ROI
inputMode = False # Check for mouse-click input mode

# Callback function to create ROI from 4 points, clicked by the user 
def click_and_crop(event, x, y, flags, param):
    # Declare global variables: 1)current frame, 2) list of ROI points, 3) input mode
    global frame, roiPts, inputMode
 
    # Checking conditions: ROI selection/input mode, left-mouse click, if 4 points are selected or not?
    if inputMode and event == cv2.EVENT_LBUTTONDOWN and len(roiPts) < 4:
        roiPts.append((x, y)) # update a list of ROI points with (x, y) location of mouse click
        cv2.circle(frame, (x, y), 4, (0, 255, 0), 2) # draw green circle at mouse click (x,y) location
        cv2.imshow("image", frame)

# Attaching the callback into the video window
cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_crop)

# Setup the termination criteria, either 10 iteration or move by at least 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
roiBox = None # intialize ROI region as none/empty

while (True): # Main loop
    ret, frame = cap.read() # Reading next frames from video 
    if roiBox is None and len(roiPts) < 4:
        inputMode = True # indicate that we are in input mode 
        orig = frame.copy() # copy the first frame of VDO file 
        # keep looping until 4 reference-corner points are selected for ROI;
        # press any key to exit ROI selction mode once 4 points have been selected
        while len(roiPts) < 4:
            cv2.imshow("image", frame)
            cv2.waitKey(0)
        # determine the top-left and bottom-right points
        roiPts = np.array(roiPts)
        s = roiPts.sum(axis = 1)
        tl = roiPts[np.argmin(s)] # tl denotes top-left corner point
        br = roiPts[np.argmax(s)] # br denotes bottom-right corner point
         
        roi = orig[tl[1]:br[1], tl[0]:br[0]] # crop ROI-bounding box from original frame
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV) # convert ROI box to HSV-color space
        # compute 2D histogram for ROI box from Hue-&Value-channels
        roi_hist = cv2.calcHist([roi],[0,2], None, [180,256], [0,180,0,255])
        # setup an initial search window for meanshift method
        roiBox = (tl[0], tl[1], br[0]-tl[0], br[1]-tl[1])
    elif roiBox is not None:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # convert current frame to HSV-color space
        
        backProj = cv2.calcBackProject([hsv],[0,2],roi_hist,[0,180,0,255],scale=2) # compute backproject from roi_hist
        
        ret, roiBox = cv2.meanShift(backProj, roiBox, term_crit) # "meanshift": compute the new window to track object

        x,y,w,h = roiBox
        print("width roi = ",w," height roi = ",h)
        frame = cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0) ,2) # Draw new windown on current frame
        cv2.circle(frame, (int(x+(w/2.0)), int(y+(h/2.0))), 4, (0, 255, 0), 2) # Draw a circle center of new window 

    cv2.imshow("Object Tracking", frame)
    keyboard = cv2.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break
        
cv2.destroyAllWindows()
cap.release()