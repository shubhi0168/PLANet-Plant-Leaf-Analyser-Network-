import numpy as np
import cv2

def nothing(useless=None):
    pass

cv2.namedWindow("Mask")
cap = cv2.VideoCapture(0)

#Defining the trackbars
cv2.createTrackbar('R_l','Mask',26,255,nothing)
cv2.createTrackbar('G_l','Mask',46,255,nothing)
cv2.createTrackbar('B_l','Mask',68,255,nothing)

cv2.createTrackbar('R_h','Mask',108,255,nothing)
cv2.createTrackbar('G_h','Mask',138,255,nothing)
cv2.createTrackbar('B_h','Mask',155,255,nothing)

while True:
    
    #Getting the position of the trackbads
    R_l = cv2.getTrackbarPos('R_l', 'Mask')
    G_l = cv2.getTrackbarPos('G_l', 'Mask')
    B_l = cv2.getTrackbarPos('B_l', 'Mask')
    
    R_h = cv2.getTrackbarPos('R_h', 'Mask')
    G_h = cv2.getTrackbarPos('G_h', 'Mask')
    B_h = cv2.getTrackbarPos('B_h', 'Mask')
    
    #Getting frame, blurring it and converting rgb to hsv
    _,frame = cap.read()
    blurred_frame = cv2.blur(frame,(5,5),0)    
    hsv_frame = cv2.cvtColor(blurred_frame,cv2.COLOR_BGR2HSV)
    
    #Defining color theshold
    low_green = np.array([R_l, G_l, B_l])
    high_green = np.array([R_h, G_h, B_h])
    green_mask = cv2.inRange(hsv_frame, low_green, high_green)

    #Morphological adjestments
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    opening = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    close = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    #Getting the largest contour
    contours,_ = cv2.findContours(green_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)    
    
    try:
        
        biggest = sorted(contours,key=cv2.contourArea,reverse=True)[0]
        cv2.drawContours(frame,biggest,-1,(255,0,0),1)
        
        #Creating blank mask and filling in the contour
        blank_mask = np.zeros(frame.shape, dtype=np.uint8)
        cv2.fillPoly(blank_mask, [biggest], (255,255,255))
        blank_mask = cv2.cvtColor(blank_mask, cv2.COLOR_BGR2GRAY)
        result = cv2.bitwise_and(frame,frame,mask=blank_mask)

        x,y,w,h = cv2.boundingRect(blank_mask)
        ROI = result[y:y+h, x:x+w]

        cv2.imshow('Mask', ROI)
        cv2.imshow('frame', frame)
        
    except IndexError:
        continue
        
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
