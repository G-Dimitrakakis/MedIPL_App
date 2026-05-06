# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 16:37:59 2021

@author: medisp-2
"""

#moduleFreEnhancementFunctions 

import numpy as np
import matplotlib.pyplot as plt
# import moduleUtils as U
# import cv2
# import warnings



print('')
    #------------------------------------------------------     
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])
#---------------------------------------------------    
def imPlot(im,title):
    plt.imshow(im,cmap=plt.cm.gray,vmin=0,vmax=255);
    plt.title (title);plt.axis("off");
#----------------------------------------------------
def imNormalize(w,tones):
    mx=np.max(w);mn=np.min(w);
    w=(tones-1)*(w-mn)/(mx-mn);    
    w=np.round(w)
    return w


#--------------------------------------------------------
def ampl_fft2( im):
    im1=np.fft.fft2(im)
    im1=np.fft.fftshift(im1)
    im1=np.round(10.0*np.log(np.abs(im1)+1))
    return(im1) 
##-----------------------------------------------------
#-------------------------------------------------------
def design2dFilter(im,fh):
    y=np.size(im,0);x=np.size(im,1);
    FH=np.zeros(np.shape(im),dtype=float);
    
    for k in range (y):
        for m in range(x):
            K=y/2-k+1;
            M=x/2-m+1;
            ir = np.int32(np.sqrt( ( K*K +M*M ) )+0.5)  ;
            if(ir>len(fh)):
                ir=len(fh);
            FH[k][m]=fh[ir];
    
    FH=np.fft.fftshift(FH)
    return(FH)
#---------------------------------------------------    
def filterImage(im,FH,tones):
    # print(np.shape(im))
    Fim=np.fft.fft2(im)
    Fim1=Fim*FH
    im1=np.real(np.fft.ifft2(Fim1))
    return (im1)
#----------------------------------------------------
def loadImage(imName):
    
    print(imName[-3:])
    
    if (imName[-3:]=='dcm'):
        import pydicom as dicom
        #anaconda prompt -->   pip install pydicom
        print('in dcm')
        print(imName)
        dcHeader=dicom.dcmread(imName)
        im=dcHeader.pixel_array

    else: #elif (imName[-3:]=='bmp'):
        # import cv2    
        # im=cv2.imread(imName)
        im=plt.imread(imName)
        print('in bmp')
    
    L=np.shape(im)
    if(len(L)==3):
        im=rgb2gray(im)
        print('------- rgb image ----------');
    
    return(im)    
#-------------------------------------------------------------
#---------------------------------------------------------------
# @jit('void(double[:,:],double,double,double,double)')
def simpleWindow(im,wc,ww,image_depth, tones):
    im1=np.asarray(im, dtype=float)
    Vb=(2.0*wc+ww)/2.0;
    Va=Vb-ww; 
    if(Vb>image_depth):
        Vb=image_depth;
    if(Va<0): 
        Va=0;
    im1=(((tones-1)*((im1-Va)/(Vb-Va))))
    M=np.size(im1,0)
    N=np.size(im1,1)
    for i in range (0,M):
        for j in range(N):
            if(  (im1[i][j]>=Va) and (im1[i][j]<=Vb)  ):    
                im1[i][j]=(((tones-1)*(im1[i][j]-Va)/(Vb-Va)))
            elif (im1[i][j]<Va):
                im1[i][j]=0;
            elif (im1[i][j]>Vb):
                im1[i][j]=tones-1;
    return im1

