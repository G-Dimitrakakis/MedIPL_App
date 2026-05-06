# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 16:02:40 2021

@author: medisp-2
"""

#module FreqDom_1d_Filters.py
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
import warnings
from scipy import signal
warnings.simplefilter('ignore')#default


#-----------------------------------------------------------------
def Ideal(N,fco,TYPE,enh,trans,w):
    fh=np.zeros(np.int32(N),dtype=float)
    if((N % 2)==0):
        L=np.round(N/2+1)
        M=np.round(N/2+2)
    else:
        L=np.round(N/2+0.5)
        M=np.round(N/2+1+0.5)

    if(TYPE==1):#LP
        fh=np.ones(np.int32(N),dtype=float)
        sText=" Ideal LP filter";
        for k in range(np.int32(fco),np.int32(L)):
            fh[k]=0+enh;#LP
 
    elif(TYPE==2):     
        fh=np.zeros(np.int32(N),dtype=float)+enh
        for k in range(np.int32(fco),np.int32(L)):
            fh[k]=1;#Ideal HP
            sText=" Ideal HP filter"
    elif(TYPE==3):     
        d=trans;
        fh=np.ones(np.int32(N),dtype=float)
        for k in range(np.int32(d-w/2+0.5),np.int32(d+w/2+0.5)):
            fh[k]=0+enh;#Ideal BR
        sText=" Ideal BR filter"

    elif(TYPE==4):     
        d=trans;
        fh=np.zeros(np.int32(N),dtype=float)+enh
        
        for k in range(np.int32(d-(w/2) + 0.5),np.int32(d+(w/2)+0.5)):
            fh[k]=1;#Ideal BP
        sText=" Ideal BP filter"    
    else: 
         print("------NO SUCH FILTER, FILTERS BETWEEN 1-4" )

    
    for k in range (np.int32(M-1),np.int32(N)):
        fh[k]=fh[np.int32(N-k)]
    return(fh/np.max(fh),sText)    
    #--------------------------------------------------
def Butterworth(N,ndegree,fco,TYPE,trans):
    fh=np.zeros(np.int32(N),dtype=float)
    print("--BUTTERWORTH FILTERS----")
    
    if((N % 2)==0):
        L=np.round(N/2+1)
        M=np.round(N/2+2)
    else:
        L=np.round(N/2+0.5)
        M=np.round(N/2+1+0.5)

    if (TYPE==1):#BLP
#        fh=np.zeros(np.int32(N),dtype=float)
        for k in range(np.int32(L)):
            fh[k]=1.0/(1.0+0.414* np.power( (k/fco), (2*ndegree)) );#LP
        sText='Butterworth LP'
    elif(TYPE==2):     
#        fh=np.zeros(np.int32(N),dtype=float)+enh
        for k in range(np.int32(L)):
            fh[k]=1.0/(1.0+0.414* np.power( (fco/(k+0.001)), (2*ndegree)) );#HP
        for k in range(np.int32(L)):    
            if ( k<int(N/2-trans) ):
              fh[k] = fh[k+int(trans)];
            else:
              fh[k] = fh[int(N/2)];  
        sText='Butterworth HP'     
              
    elif(TYPE==3):     
        d=trans;
        for k in range(np.int32(L)):
            fh[k]=1.0/(1.0+0.414* np.power( (fco/(k-d+0.001)), (2*ndegree)) );#BR
        sText='Butterworth BR'
    elif(TYPE==4):     
        d=trans;enh=0.001
        fh=np.zeros(np.int32(N),dtype=float)+enh
        
        for k in range(np.int32(L)):
            fh[k]=1.0/(1.0+0.414* np.power( ((k-d)/fco), (2*ndegree)) );#BP
        sText='Butterworth BP'    
    else: 
        print("------NO SUCH FILTER, FILTERS BETWEEN 1-4" )
    
    for k in range (np.int32(M-1),np.int32(N)):
        fh[k]=fh[np.int32(N-k)]
        
    return(fh/np.max(fh),sText)
#--------------------------------------------------------------
def Exponential( N,ndegree,fco,TYPE,trans):
    fh=np.zeros(np.int32(N),dtype=float)
    print("--EXPONENTIAL FILTERS----")
    
    if((N % 2)==0):
        L=np.round(N/2+1)
        M=np.round(N/2+2)
    else:
        L=np.round(N/2+0.5)
        M=np.round(N/2+1+0.5)

    if (TYPE==1):#ELP
#        fh=np.zeros(np.int32(N),dtype=float)
        for k in range(np.int32(L)):
            fh[k]=np.exp( (-np.log(2))*(k/fco)**ndegree)
        sText='Exponential LP'
    elif(TYPE==2): #EHP    
#        fh=np.zeros(np.int32(N),dtype=float)+enh
        for k in range(np.int32(L)):
            fh[k]=np.exp((-np.log(2))*(fco/(k+0.0001))**ndegree)

        for k in range(np.int32(L)):    
            if ( k<int(N/2-trans) ):
              fh[k] = fh[k+int(trans)];
            else:
              fh[k] = fh[int(N/2)];  
        sText='Exponential HP'     
              
    elif(TYPE==3):     
        d=trans;
        for k in range(np.int32(L)):
            fh[k]=np.exp((-np.log(2))*(fco/(k-d+0.00001))**ndegree)
        sText='Exponential BR'

    elif(TYPE==4):     
        d=trans;enh=0.001
        fh=np.zeros(np.int32(N),dtype=float)+enh
        
        for k in range(np.int32(L)):
            fh[k]=np.exp((-np.log(2))*((k-d)/fco)**ndegree)
            
        sText='Exponential BP'
    else: 
        print("------NO SUCH FILTER, FILTERS BETWEEN 1-4" )
    
    for k in range (np.int32(M-1),np.int32(N)):
        fh[k]=fh[np.int32(N-k)]
        
    return(fh/np.max(fh),sText)


#--------------------------------------------------------------
def Gaussian( N,ndegree,fco,TYPE,trans):
    fh=np.zeros(np.int32(N),dtype=float)
    print("--GAUSSIAN FILTERS----")
    
    if((N % 2)==0):
        L=np.round(N/2+1)
        M=np.round(N/2+2)
    else:
        L=np.round(N/2+0.5)
        M=np.round(N/2+1+0.5)

    if (TYPE==1):#gLP
#        fh=np.zeros(np.int32(N),dtype=float)
        for k in range(np.int32(L)):
            fh[k]=np.exp( -(k**2/(2*fco**2))**ndegree)
        sText='Gaussian LP'
    elif(TYPE==2): #GHP    
#        fh=np.zeros(np.int32(N),dtype=float)+enh
        for k in range(np.int32(L)):
            fh[k]=np.exp(-(2*fco**2/(k+0.0001)**2)**ndegree)

        for k in range(np.int32(L)):    
            if ( k<int(N/2-trans) ):
              fh[k] = fh[k+int(trans)];
            else:
              fh[k] = fh[int(N/2)];  
        sText='Gaussian HP'     
              
    elif(TYPE==3):     
        d=trans;
        for k in range(np.int32(L)):
            fh[k]=np.exp(-(2*fco**2/(k-d+0.00001)**2)**ndegree)
        sText='Gaussian BR'

    elif(TYPE==4):     
        d=trans;enh=0.001
        fh=np.zeros(np.int32(N),dtype=float)+enh
        
        for k in range(np.int32(L)):
            fh[k]=np.exp(-((k-d)**2/(2*fco**2))**ndegree)
            
        sText='Gaussian BP'
    else: 
        print("------NO SUCH FILTER, FILTERS BETWEEN 1-4" )
    
    for k in range (np.int32(M-1),np.int32(N)):
        fh[k]=fh[np.int32(N-k)]
        
    return(fh/np.max(fh),sText)



