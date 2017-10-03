import cv2
import numpy as np

left_img = cv2.imread('./PA2Data/view1.png', 0)  #read it as a grayscale image
right_img = cv2.imread('./PA2Data/view5.png', 0)
groundtruth1=cv2.imread('./PA2Data/disp1.png', 0) 
groundtruth2=cv2.imread('./PA2Data/disp5.png', 0) 
DisparityMatrix1_3=cv2.imread('./PA2 output/disparity images/DisparityMatrix1_3.jpg', 0) 
DisparityMatrix2_3=cv2.imread('./PA2 output/disparity images/DisparityMatrix2_3.jpg', 0) 
DisparityMatrix1_7=cv2.imread('./PA2 output/disparity images/DisparityMatrix1_7.jpg', 0) 
DisparityMatrix2_7=cv2.imread('./PA2 output/disparity images/DisparityMatrix2_7.jpg', 0) 
DisparityMatrix1_9=cv2.imread('./PA2 output/disparity images/DisparityMatrix1_9.jpg', 0) 
DisparityMatrix2_9=cv2.imread('./PA2 output/disparity images/DisparityMatrix2_9.jpg', 0) 


#back propogation consistency check
 
def consistencyCheck(disp1,disp2,blocksize):
    
    bordersize=blocksize/2;
    rowcol= disp1.shape
    totalrow=rowcol[0]
    totalcol=rowcol[1]
    totalsize=totalrow*totalcol;
    
    newdisparity1=np.zeros(rowcol)
    newdisparity2=np.zeros(rowcol)
    #FORM NEW DISPARITY IMAGE
    for x in range(0,totalrow):
        for y in range(0,totalcol):
            disparity1=disp1[x][y];disparity2=disp2[x][y]
            index1=np.abs(y-disparity1);index2=y+disparity2 #DIVES THE INDEX AT WHICH PIXEL SHOULD LIE
            #calculates new disparity based on left disparity
            if(index1>=0 and index1<totalcol):
                if(disp2[x][index1]==disparity1):
                    newdisparity1[x][y]=disparity1
                else:
                    newdisparity1[x][y]=0
            #calculates new disparity based on right disparity
            if(index2>=0 and index2<totalcol):
                if(disp1[x][index2]==disparity2):
                    newdisparity2[x][y]=disparity2
                else:
                    newdisparity2[x][y]=0
    #CHECK MSE
    mse1=computeMse(newdisparity1[bordersize:-bordersize,bordersize:-bordersize],groundtruth1)
    mse2=computeMse(newdisparity2[bordersize:-bordersize,bordersize:-bordersize],groundtruth2) 

    return newdisparity1,newdisparity2,mse1,mse2

def computeMse(m,n):
    #EXCLUDING ZEROS WHILE COMPUTING MSE
    mse=0;count=1;
    for x in range(0,m.shape[0]):
        for y in range(0,m.shape[0]):
            if(m[x][y]!=0):
                count=count+1
                mse=mse+(m[x][y]-n[x][y])**2
    return mse/count            
    
#BLOCK SIZE=3
blocksize=3
newdisparity1_3,newdisparity2_3,mse1_3,mse2_3= consistencyCheck(DisparityMatrix1_3,DisparityMatrix2_3,blocksize)  
cv2.namedWindow('newdisparity1_3', cv2.WINDOW_NORMAL)#create a window for display
cv2.imshow('newdisparity1_3',newdisparity1_3/newdisparity1_3.max())#show our image inside it.
cv2.namedWindow('newdisparity2_3', cv2.WINDOW_NORMAL)#create a window for display
cv2.imshow('newdisparity2_3',newdisparity1_3/newdisparity1_3.max())#show our image inside it.
print("MSE for consistency check using left disparity image, and block size = " ,blocksize," is ",mse1_3)
print("MSE for for consistency check using right disparity image, and block size = " ,blocksize," is ",mse2_3)

#BLOCK SIZE=7
blocksize=7
newdisparity1_7,newdisparity2_7,mse1_7,mse2_7= consistencyCheck(DisparityMatrix1_7,DisparityMatrix2_7,blocksize)  
cv2.namedWindow('newdisparity1_7', cv2.WINDOW_NORMAL)#create a window for display
cv2.imshow('newdisparity1_7',newdisparity1_7/newdisparity1_7.max())#show our image inside it.
cv2.namedWindow('newdisparity2_7', cv2.WINDOW_NORMAL)#create a window for display
cv2.imshow('newdisparity2_7',newdisparity1_7/newdisparity1_7.max())#show our image inside it.
print("MSE for consistency check using left disparity image, and block size = " ,blocksize," is ",mse1_7)
print("MSE for for consistency check using right disparity image, and block size = " ,blocksize," is ",mse2_7)


#BLOCK SIZE=9
blocksize=9
newdisparity1_9,newdisparity2_9,mse1_9,mse2_9= consistencyCheck(DisparityMatrix1_9,DisparityMatrix2_9,blocksize)  
cv2.namedWindow('newdisparity1_9', cv2.WINDOW_NORMAL)#create a window for display
cv2.imshow('newdisparity1_9',newdisparity1_3/newdisparity1_9.max())#show our image inside it.
cv2.namedWindow('newdisparity2_9', cv2.WINDOW_NORMAL)#create a window for display
cv2.imshow('newdisparity2_9',newdisparity1_9/newdisparity1_9.max())#show our image inside it.
print("MSE for consistency check using left disparity image, and block size = " ,blocksize," is ",mse1_9)
print("MSE for for consistency check using right disparity image, and block size = " ,blocksize," is ",mse2_9)


#WRITE BACK TRACKED DISPARITY IMAGES TO FILE
#uncomment and create sub folders to save output
#cv2.imwrite( "./PA2 output/consistency check/newdisparity1_3.jpg", newdisparity1_3 );
#cv2.imwrite( "./PA2 output/consistency check/newdisparity2_3.jpg", newdisparity2_3 );
#cv2.imwrite( "./PA2 output/consistency check/newdisparity1_7.jpg", newdisparity1_7 );
#cv2.imwrite( "./PA2 output/consistency check/newdisparity2_7.jpg", newdisparity2_7 );
#cv2.imwrite( "./PA2 output/consistency check/newdisparity1_9.jpg", newdisparity1_9 );
#cv2.imwrite( "./PA2 output/consistency check/newdisparity2_9.jpg", newdisparity2_9 );
#
