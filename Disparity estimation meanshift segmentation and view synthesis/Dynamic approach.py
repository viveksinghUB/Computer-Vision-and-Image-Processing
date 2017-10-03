import cv2
import numpy as np

#DISPARITY IMAGE OF LEFT IMAGE IS DisparityImage
#DISPARITY IMAGE OF RIGHT IMAGE IS DisparityImage2

left_img = cv2.imread('./PA2Data/view1.png', 0) #read it as a grayscale image
right_img = cv2.imread('./PA2Data/view5.png', 0)

rowcol= left_img.shape
DisparityMatrix=np.zeros(rowcol,np.uint8)
DisparityMatrix2=np.zeros(rowcol,np.uint8)

OcclusionCost =5
#(You can adjust this, depending on how much threshold you want to give for noise)

#For Dynamic Programming you have build a cost matrix. Its dimension will be numcols x numcols
numcols=int(left_img.shape[1])
CostMatrix = np.zeros((numcols,numcols),dtype=float)
DirectionMatrix = np.zeros((numcols,numcols),dtype=float)
SolnMatrix = np.zeros((numcols,numcols),dtype=float)
#(This is important in Dynamic Programming. You need to know which direction you need traverse)

#We first populate the first row and column values of Cost Matrix

for i in range(numcols):
    CostMatrix[i,0] = i*OcclusionCost
    CostMatrix[0,i] = i*OcclusionCost
 
# Use the pseudocode from "A Maximum likelihood Stereo Algorithm" paper given as reference
for rownum in range(rowcol[0]):    
    for i in range(numcols):
        for j in range(numcols):
            min1=CostMatrix[i-1,j-1]+np.abs((left_img[rownum,i]-right_img[rownum,j]),dtype=float)
            min2=CostMatrix[i-1,j]+OcclusionCost
            min3=CostMatrix[i,j-1]+OcclusionCost
            CostMatrix[i,j]=cmin=np.min((min1,min2,min3))
            if(min1==cmin):
                DirectionMatrix[i,j]=1
            if(min2==cmin):
                DirectionMatrix[i,j]=2
            if(min3==cmin):
                DirectionMatrix[i,j]=3
                    
    p=q=numcols-1
    
    while(p != 0 and q!=0) :
        if(DirectionMatrix[p,q]==1):
            DisparityMatrix[rownum,p]=p-q
            DisparityMatrix2[rownum,q]=q-p
            p=p-1
            q=q-1
        elif(DirectionMatrix[p,q]==2):
            p=p-1
        elif(DirectionMatrix[p,q]==3):
            q=q-1
               
          
    
cv2.namedWindow('DisparityImage', cv2.WINDOW_NORMAL)#create a window for display
cv2.imshow('DisparityImage',DisparityMatrix)#show our image inside it.  
cv2.namedWindow('DisparityImage2', cv2.WINDOW_NORMAL)#create a window for display
cv2.imshow('DisparityImage2',DisparityMatrix2)#show our image inside it.
 
#WRITE IMAGES TO DIRECTORY
#cv2.imwrite( "./PA2 output/dynamic programing/DisparityMatrix1_dp.jpg", DisparityMatrix );  
#cv2.imwrite( "./PA2 output/dynamic programing/DisparityMatrix2_dp.jpg", DisparityMatrix2 );