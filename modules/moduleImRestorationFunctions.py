# -*- coding: utf-8 -*-
"""
Created on Tue May  4 13:33:43 2021

@author: Cavouras
"""
import numpy as np
import matplotlib.pyplot as plt
import time
# moduleImRestarationFunctions
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])
#----------------------------------------------------

def imPlot(im,title,fz):
    plt.figure(figsize=(fz,fz));
    plt.imshow(im,cmap=plt.cm.gray,vmin=0,vmax=255);
    plt.title (title);plt.axis("off");
#----------------------------------------------------
def imNormalize(w,tones):
    mx=np.max(w);mn=np.min(w);
    w=(tones-1)*(w-mn)/(mx-mn);    
    w=np.round(w)
    return w
#--------------------------------------------------------
def ampl_fft2(im):
    t1=float(time.time())
    im1=np.fft.fft2(im)
    im1=np.fft.fftshift(im1)
    im1=np.round(10.0*np.log(np.abs(im1)+1))
    t2=float(time.time());dt=t2-t1;
    print("time for amplitute calculation  =%10.7f " % (dt));
    return(im1)
#--------------------------------------------------------
    
def BlurImage(im,FH):
     Fim=np.fft.fft2(im)
     Fim1=Fim*np.fft.fftshift(FH)
     im1=np.real(np.fft.ifft2(Fim1))
     return im1
    
#---------------------------------------------------------
def GaussianMTF(N):
    
    fh=np.zeros(np.int32(N),dtype=float)
    
    
    if((N % 2)==0):
        L=np.round(N/2+1)
        M=np.round(N/2+2)
    else:
        L=np.round(N/2+0.5)
        M=np.round(N/2+1+0.5)

    sigma=L/2-1
    for k in range(np.int32(L)):
        fh[k]=np.exp(-k**2/(2*sigma**2))
    
    for k in range (np.int32(M-1),np.int32(N)):
        fh[k]=fh[np.int32(N-k)]
    return(fh)
#---------------------------------------------------------
def from1dTo2dFilter(im,fh):
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
    FH=FH/np.amax(FH)
    
    return FH
#-----------------------------------------------------------------
def addNoise(im1,percentNoise):
    M=np.size(im1,0);N=np.size(im1,1);
    for i in range(np.uint32(M)):
        for j in range(np.uint32(N)):
            im1[i][j]=im1[i][j] + percentNoise*im1[i][j]*np.random.rand()
    
    im1=255*im1/(np.amax(im1))
    return im1

#--------------------------------------------------------------- 

def generalizedWienerFilter(fh,filtType,SIGMA):
    N=len(fh)

    C=2*SIGMA**2
    
    if (filtType==1):
        a=1;b=0#Inverse Filter
        sText=" Inverse Filter";
    elif (filtType==2):
        a=0;b=1;
        sText=" Wiener Filter";
    elif(filtType==3):
        a=0.5;b=1;
        sText=" Power Filter";
    fhh=np.zeros(np.int32(N),dtype=float)
    
    
    for k in range(np.int32(N)):
        fhh[k]=(((fh[k]**2)/ (fh[k]**2+b*C))**(1-a)) *(1/fh[k])
        if(fhh[k]<C):
            fhh[k]=(((fh[k]**2)/ (fh[k]**2+b*C))**(1-a)) *(1/C)

    fhh=fhh/np.max(fhh)
    return (fhh, sText)  
#--------------------------------------------------------------- 
      
def DeconvolveImage(im,FH):
    Fim=np.fft.fft2(im)
    Fim1=Fim*np.fft.fftshift(FH)
    im1=np.real(np.fft.ifft2(Fim1))
    return im1
#---------------------------------------------------------------
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
#--------------------------------------------------------------
def ST_ERROR(diag_orig,diag_im):
    SE1=np.sqrt(np.sum((diag_orig-diag_im)**2)/len(diag_orig))
    return (SE1)
