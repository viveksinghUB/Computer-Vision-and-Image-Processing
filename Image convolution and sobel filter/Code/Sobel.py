# -*- coding: utf-8 -*-
#Ref : PythonReview2

import numpy as np
import cv2

img = cv2.imread("lena_gray.jpg",0)#loading the file to display
#img=Gx = np.array([[0,0,0,0,0,0,0],[0,0,1,1,1,1,0],[0,0,0,0,1,1,0],[0,0,0,1,1,1,0],[0,0,0,0,0,0,0]])
row, col= img.shape[:2]
#create a border of unit length across the image
bordersize=1
border=cv2.copyMakeBorder(img, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize, borderType= cv2.BORDER_CONSTANT, value=[0,0,0] )
rowcol=img.shape
#the two sobel kernels / high pass filters
def twodconvolution():
    Gx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    Gy = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    #initialize variables
    xconvolved=np.zeros(rowcol)
    yconvolved=np.zeros(rowcol)
    grad=np.zeros(rowcol)
    for i in range(1,border.shape[0]-1):
        for j in range(1,border.shape[1]-1):
            a=np.array(border[i-1:i+2,j-1:j+2],dtype=int)
            xconvolved[i-1,j-1]=np.sum(a*Gx)
            yconvolved[i-1,j-1]=np.sum(a*Gy)
    
    #normalize between 0 to 1
    xconvolved_n=xconvolved/xconvolved.max()
    yconvolved_n=yconvolved/yconvolved.max()
    grad=((xconvolved_n**2)+(yconvolved_n**2)**(1/2.0))
    
    cv2.namedWindow('Gx', cv2.WINDOW_NORMAL)#create a window for display
    cv2.imshow('Gx',xconvolved_n)#show our image inside it.
    
    cv2.namedWindow('Gy', cv2.WINDOW_NORMAL)#create a window for display
    cv2.imshow('Gy',yconvolved_n)#show our image inside it.
    
    cv2.namedWindow('G', cv2.WINDOW_NORMAL)#create a window for display
    cv2.imshow('G',grad)#show our image inside it.

def oneDconvolution():
    Gx_v = np.array([1,2,1]).reshape(3,1)
    Gx_h = np.array([-1,0,1])
    Gy_v = np.array([-1,0,1]).reshape(3,1)
    Gy_h = np.array([1,2,1])
    xconvolved_h=np.zeros(border.shape)
    yconvolved_h=np.zeros(border.shape)
    xconvolved=np.zeros(rowcol)
    yconvolved=np.zeros(rowcol)
    
    for i in range(1,border.shape[0]-1):
        for j in range(1,border.shape[1]-1):
            v=np.array(border[i,j-1:j+2])
            xconvolved_h[i,j]=np.sum(v*Gx_h)
            yconvolved_h[i,j]=np.sum(v*Gy_h)
            
    for i in range(1,border.shape[0]-1):
        for j in range(1,border.shape[1]-1):
            h1=np.array(xconvolved_h[i-1:i+2,j]).reshape(3,1)
            h2=np.array(yconvolved_h[i-1:i+2,j]).reshape(3,1)
            xconvolved[i-1,j-1]=np.sum(h1*Gx_v)
            yconvolved[i-1,j-1]=np.sum(h2*Gy_v)        
            
    xconvolved_n=xconvolved/xconvolved.max()
    yconvolved_n=yconvolved/yconvolved.max()       
    grad=((xconvolved_n**2)+(yconvolved_n**2)**(1/2.0))       
    cv2.namedWindow('Gx', cv2.WINDOW_NORMAL)#create a window for display
    cv2.imshow('Gx',xconvolved_n)#show our image inside it.
    
    cv2.namedWindow('Gy', cv2.WINDOW_NORMAL)#create a window for display
    cv2.imshow('Gy',yconvolved_n)#show our image inside it.
    
    cv2.namedWindow('G', cv2.WINDOW_NORMAL)#create a window for display
    cv2.imshow('G',grad)#show our image inside it.
        
#twodconvolution()  
oneDconvolution()    
        