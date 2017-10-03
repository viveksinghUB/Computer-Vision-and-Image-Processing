import cv2
import numpy as np
import random
import logging



#INITIALIZE VARIABLES AND LOAD DATA
img =np.array( cv2.imread('./PA2Data/Butterfly.jpg'))
rc=img.shape
newimg=np.zeros(rc)
f=np.zeros((rc[0]*rc[1],5))
fnew=np.zeros((rc[0]*rc[1],5))
row=0;col=0;

#SET THE PARAMETERS h AND iter i.e. shift_threshold
h=150
shift_threshold=10.0

def enableLogging(check,logger,hdlr):
    if(check):
        
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        hdlr.setFormatter(formatter)
        logger.addHandler(hdlr) 
        logger.setLevel(logging.DEBUG)

#SET TO  ENABLE LOGGING
logger = logging.getLogger('Mean Shift')
hdlr = logging.FileHandler('./Mean Shift.log')

enableLogging(True,logger,hdlr);


#DEFINE DISTANCE
def eucledianDistance(a,b):
    return np.sqrt(np.sum((a - b) ** 2))
    
#FORMING THE MATRIX F 166500 X 5 WITH RGBXY VALUES
for x in range(0,rc[0]):
    for y in range(0,rc[1]):
        for z in range(0,rc[2]):
            f[row][col]=img[x][y][z]
            col=col+1
        f[row][col]=x;
        col=col+1;
        f[row][col]=y;
        col=0;row=row+1;

initialmean=np.array(f[random.randint(0,f.shape[0]-1)]).reshape(1,5)
fnew=f

while(fnew.size>0 ):
    cluster=np.empty((0,5))
    delindex=[]
    nrow=fnew.shape[0]
    for i in range(0,nrow):  #iterate over rows to find value close to mean
        dist=eucledianDistance(fnew[i],initialmean)
        if(dist<=h): #FORM THE CLUSTER OF POINTS WITH DISTANCE <H
            cluster = np.append(cluster, fnew[i].reshape(1,5), axis=0)
            delindex.append(i)#FORM LIST OF INDICES CAPTURED IN CLUSTER
    #logger.info("1 iteration Cluster shape = %i %i",cluster.shape[0],cluster.shape[1])        
    if(cluster.size > 0):
        newmean=np.mean(cluster,axis=0).reshape(1,5)
        #logger.info("cluster shape = %i %i",cluster.shape[0],cluster.shape[1])
        dist2=eucledianDistance(newmean,initialmean)
        if(dist2<shift_threshold):
            fnew=np.delete(fnew,delindex, axis=0)#delete picked rows
            if(fnew.size>0):#initialize a random mean
                initialmean=np.array(fnew[random.randint(0,fnew.shape[0]-1)]).reshape(1,5)
            for j in range(0,cluster.shape[0]):#fill in the mean value to make new image
                row=int(cluster[j][3]);
                col=int(cluster[j][4]);
                newimg[row][col][0]=int(newmean[0][0]);#red
                newimg[row][col][1]=int(newmean[0][1]);#green
                newimg[row][col][2]=int(newmean[0][2]);#blue
        else:
            #IF MEAN NOT NEAR shift_threshold distance INITIAL MEAN, shift to new mean
            initialmean=newmean    
    else:
        #IF NO ITEAM H DISTANCE NEAR INITIAL MEAN
        initialmean=np.array(f[random.randint(0,f.shape[0])]).reshape(1,5)
        
cv2.namedWindow('Segmented Image', cv2.WINDOW_NORMAL)#create a window for display
cv2.imshow('Segmented Image',newimg/newimg.max())#show our image inside it.

#close the file handler for logging
hdlr.close