# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 14:14:37 2020

@author: medisp-2
"""
import time
import sys
import numpy as np
import matplotlib.pyplot as plt

def loadImage(imageFile):
    imtype = imageFile.split('.')[-1].lower()
    
    if imtype != 'dcm':
        im = plt.imread(imageFile)
        im = np.asarray(im, dtype=float)
        L = np.shape(im)
        if len(L) == 3:
            im = rgb2gray(im)
        image_depth = 255
    
    elif imtype == 'dcm':
        import pydicom as dicom
        ds = dicom.dcmread(imageFile)
        im = np.array(ds.pixel_array, dtype=float)
        mn = np.min(im); mx = np.max(im)
        im = np.round(255 * (im - mn) / (mx - mn))
        image_depth = 255
    
    return im, image_depth
def cls():
    print(chr(27) + "[2J") 
def pause():
    input("PRESS ENTER TO CONTINUE.")
    #    ------------------------------------------------------------
def tic():
    t1=float(time.time());
    return (t1)

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])    

#------------------------------------------------------------
def toc(t1,s):
    t2=float(time.time());dt=t2-t1;
    s1='time taken '+s 
    print("%s %e" % (s1,dt) )     
#---------------------------------------------------------
def RETURN():
    sys.exit()
     