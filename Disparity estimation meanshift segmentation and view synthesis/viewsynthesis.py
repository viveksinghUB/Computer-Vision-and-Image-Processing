from __future__ import print_function
import cv2
import numpy as np
import sys

#THE PROGRAMS DISPLAYS THE MIDVIEW GENERATED FORM LEFT DISPARITY
#RIGHT DISPARITY AND BY THE COMBINATION OF BOTH.


left_img = cv2.imread('./PA2Data/view1.png')  #read it as a grayscale image
right_img = cv2.imread('./PA2Data/view5.png')
disp1=cv2.imread('./PA2Data/disp1.png',0) 
disp5=cv2.imread('./PA2Data/disp5.png',0) 
rowcol=left_img.shape
midview=np.zeros((left_img.shape),np.uint8)
leftview=np.zeros((left_img.shape),np.uint8)
rightview=np.zeros((left_img.shape),np.uint8)
avgmidview=np.zeros((left_img.shape),np.uint8)
holevalue=np.zeros((left_img.shape),np.uint8)
#COMPUTING SYNTHESZED VIEW
for row in range(0,rowcol[0]):
    for col in range(0,rowcol[1]):
        shiftedcolumn=col-(disp1[row][col]/2)
        midview[row][shiftedcolumn]=left_img[row][col]

#FILLING IN THE HOLES
leftview=midview
for row in range(0,rowcol[0]):
    for col in range(0,rowcol[1]):
        shiftedcolumn=col+(disp5[row][col]/2)
        if(shiftedcolumn<463):
            rightview[row][shiftedcolumn]=right_img[row][col]
            if(midview[row][shiftedcolumn].all()==0):
                holevalue[row][shiftedcolumn]=right_img[row][col]
                midview[row][shiftedcolumn]=right_img[row][col]

#view images

cv2.namedWindow('LeftView', cv2.WINDOW_NORMAL)#create a window for display
cv2.imshow('LeftView',leftview)#show our image inside it.
cv2.namedWindow('Right View', cv2.WINDOW_NORMAL)#create a window for display
cv2.imshow('Right View',rightview)#show our image inside it.                
cv2.namedWindow('MidView', cv2.WINDOW_NORMAL)#create a window for display
cv2.imshow('MidView',midview)#show our image inside it.

#write to directory
#cv2.imwrite( "./PA2 output/View Synthesis/LeftView.jpg", leftview ); 
#cv2.imwrite( "./PA2 output/View Synthesis/Right View.jpg", rightview ); 
#cv2.imwrite( "./PA2 output/View Synthesis/MidView.jpg", midview );  
