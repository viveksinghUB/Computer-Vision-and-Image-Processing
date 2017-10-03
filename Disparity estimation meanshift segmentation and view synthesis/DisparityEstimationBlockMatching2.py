
#1. save the data in ./PA2Data/
import cv2
import numpy as np
import sys

left_img = cv2.imread('./PA2Data/view1.png', 0)  #read it as a grayscale image
right_img = cv2.imread('./PA2Data/view5.png', 0)
groundtruth1=cv2.imread('./PA2Data/disp1.png', 0) 
groundtruth2=cv2.imread('./PA2Data/disp5.png', 0) 


#METHOD TO RETURN DISPARITY BETWEEN LEFT AND RIGHT IMAGE
def calculatedisparity(left_img,right_img,blocksize,search_range,lr):
    print("Calculating disparity map for block size ",blocksize," left to right = ",lr )    
    bordersize=blocksize/2
    rowcol= left_img.shape
    DisparityMatrix=np.zeros(rowcol)
    left=cv2.copyMakeBorder(left_img, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize, borderType= cv2.BORDER_CONSTANT, value=[0,0,0] )
    right=cv2.copyMakeBorder(right_img, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize, borderType= cv2.BORDER_CONSTANT, value=[0,0,0] )
    DisparityMatrix=cv2.copyMakeBorder(DisparityMatrix, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize, borderType= cv2.BORDER_CONSTANT, value=[0,0,0] )

    for centrepx in range(bordersize,left.shape[0]-bordersize):
        for centrepy in range(bordersize,left.shape[1]-bordersize):
            #select the left block to match
            startrow=centrepx-bordersize;endrow=centrepx+bordersize
            startcol=centrepy-bordersize;endcol=centrepy+bordersize
            left_block=np.array(left[startrow:endrow+1,startcol: endcol+1])
            #get the search range
            minpy,maxpy=calculatesearchrange(centrepy,search_range,bordersize,left.shape[1],lr)
            index=centrepy 
            minssd=sys.maxint   
            for colselect in range(minpy,maxpy+1):
                startcol=colselect-bordersize;endcol=colselect+bordersize
                right_block=np.array(right[startrow:endrow+1,startcol: endcol+1])
                ssd=np.sum(np.square(left_block-right_block))
                if(ssd<minssd):
                    minssd=ssd
                    index=colselect
            
            
            d2=np.abs(centrepy-index)
            DisparityMatrix[centrepx,centrepy] = d2#plain disparity
    return DisparityMatrix

#METHOD TO COMPUTE MIN AND MAX SEARCH RANGE
def calculatesearchrange(centrepy,search_range,bordersize,totalcol,lr):
    if(search_range==0):
        #then it searches 78 to left or right depending on whether left image is actually left
        if(lr==1):
            #left is actual left
            minpy=centrepy-78
            maxpy=centrepy
            if(minpy<bordersize):
                minpy=bordersize
        else:
            minpy=centrepy
            maxpy=centrepy+78
            if(maxpy>=(totalcol-bordersize)):
                #last pixel
                maxpy=totalcol-bordersize-1       
    else:
        #+- search_range px from current px
        #if +-search_rangepx  crosses boundary then min +30 or max-30
        minpy=centrepy-search_range
        maxpy=centrepy+search_range
        if(minpy<bordersize):
            minpy=bordersize
            maxpy=minpy+2*search_range;
        elif(maxpy>(totalcol-bordersize)):#last pixel
            maxpy=totalcol-bordersize
            minpy=maxpy-2*search_range
    return minpy,maxpy
            

# COMPUTING DISPARITY OF LEFT TO RIGHT AND RIGHT TO LEFT IMAGE 

rowcol= left_img.shape
lastrow=rowcol[0];
lastcol=rowcol[1];
totlsize=lastrow*lastcol;

#DEFINE THE PARAMETERS. IF SEARCH RANGE IS 0 THEN IT SEARCHES WITHIN 78 PIXELS
#IF SEARCH RANGE IS X THEN IT SEARCHES WITHIN +- X RANGE
search_range=0 #

#BLOCKSIZE=3
blocksize=3
bordersize=blocksize/2
DisparityMatrix1_3=calculatedisparity(left_img,right_img,blocksize,search_range,1)
DisparityMatrix2_3=calculatedisparity(right_img,left_img,blocksize,search_range,0)
mse1_3=np.sum(np.square(DisparityMatrix1_3[bordersize:-bordersize,bordersize:-bordersize]-groundtruth1)) / totlsize
mse2_3=np.sum(np.square(DisparityMatrix1_3[bordersize:-bordersize,bordersize:-bordersize]-groundtruth2)) / totlsize
print("MSE for disparity map left to right, and block size =3 is ",mse1_3)
print("MSE for disparity map right to left, and block size =3 is ",mse2_3)

#BLOCKSIZE=7
blocksize=7
bordersize=blocksize/2
DisparityMatrix1_7=calculatedisparity(left_img,right_img,blocksize,search_range,1)
DisparityMatrix2_7=calculatedisparity(right_img,left_img,blocksize,search_range,0)
mse1_7=np.sum(np.square(DisparityMatrix1_7[bordersize:-bordersize,bordersize:-bordersize]-groundtruth1)) / totlsize
mse2_7=np.sum(np.square(DisparityMatrix2_7[bordersize:-bordersize,bordersize:-bordersize]-groundtruth2)) / totlsize
print("MSE for disparity map left to right, and block size =7 is ",mse1_7)
print("MSE for disparity map right to left, and block size =7 is ",mse2_7)

#BLOCKSIZE=9
blocksize=9
bordersize=blocksize/2
DisparityMatrix1_9=calculatedisparity(left_img,right_img,blocksize,search_range,1)
DisparityMatrix2_9=calculatedisparity(right_img,left_img,blocksize,search_range,0)
mse1_9=np.sum(np.square(DisparityMatrix1_9[bordersize:-bordersize,bordersize:-bordersize]-groundtruth1)) / totlsize
mse2_9=np.sum(np.square(DisparityMatrix2_9[bordersize:-bordersize,bordersize:-bordersize]-groundtruth2)) / totlsize
print("MSE for disparity map left to right, and block size =9 is ",mse1_9)
print("MSE for disparity map right to left, and block size =9 is ",mse2_9)

#Display Disparity Maps

#BLOCKSIZE=3
cv2.namedWindow('DisparityMatrix1_3', cv2.WINDOW_NORMAL)#create a window for display
cv2.imshow('DisparityMatrix1_3',DisparityMatrix1_3/DisparityMatrix1_3.max())#show our image inside it. 
cv2.namedWindow('DisparityMatrix2_3', cv2.WINDOW_NORMAL)#create a window for display
cv2.imshow('DisparityMatrix2_3',DisparityMatrix2_3/DisparityMatrix2_3.max())#show our image inside it.
#BLOCKSIZE=7 
cv2.namedWindow('DisparityMatrix1_7', cv2.WINDOW_NORMAL)#create a window for display
cv2.imshow('DisparityMatrix1_7',DisparityMatrix1_7/DisparityMatrix1_7.max())#show our image inside it. 
cv2.namedWindow('DisparityMatrix2_7', cv2.WINDOW_NORMAL)#create a window for display
cv2.imshow('DisparityMatrix2_7',DisparityMatrix2_7/DisparityMatrix2_7.max())#show our image inside it.
#BLOCKSIZE=9
cv2.namedWindow('DisparityMatrix1_9', cv2.WINDOW_NORMAL)#create a window for display
cv2.imshow('DisparityMatrix1_9',DisparityMatrix1_9/DisparityMatrix1_9.max())#show our image inside it. 
cv2.namedWindow('DisparityMatrix2_9', cv2.WINDOW_NORMAL)#create a window for display
cv2.imshow('DisparityMatrix2_9',DisparityMatrix2_9/DisparityMatrix2_9.max())#show our image inside it.



#WRITE DISPARITY IMAGES TO FILE
#uncomment and create sub folders to save output
cv2.imwrite( "./PA2 output/disparity images/DisparityMatrix1_3.jpg", DisparityMatrix1_3 );
cv2.imwrite( "./PA2 output/disparity images/DisparityMatrix2_3.jpg", DisparityMatrix2_3 );
cv2.imwrite( "./PA2 output/disparity images/DisparityMatrix1_7.jpg", DisparityMatrix1_7 );
cv2.imwrite( "./PA2 output/disparity images/DisparityMatrix2_7.jpg", DisparityMatrix2_7 );
cv2.imwrite( "./PA2 output/disparity images/DisparityMatrix1_9.jpg", DisparityMatrix1_9 );
cv2.imwrite( "./PA2 output/disparity images/DisparityMatrix2_9.jpg", DisparityMatrix2_9 );
