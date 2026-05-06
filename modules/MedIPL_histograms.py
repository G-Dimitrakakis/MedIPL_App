import numpy as np
from modules.MedIPL_visualization import imPlot
import matplotlib.pyplot as plt
from skimage import exposure
import cv2

def imNormalize(w, tones):
    mx = np.max(w); mn = np.min(w)
    w = (tones-1) * (w-mn) / (mx-mn)
    w = np.round(w)
    return w

def f_histogram(A, image_depth, tones):
    minA = 0; maxA = image_depth
    if(np.max(A) > (tones-1)):
        B = np.round((tones-1) * ((A-minA) / (maxA-minA)))
    else:
        B = A
    M = np.size(B, 0)
    N = np.size(B, 1)
    Bval = np.reshape(B, M*N)
    h = np.zeros(tones, dtype=float)
    for i in range(np.size(Bval)):
        val = np.int16(Bval[i])
        h[val] = h[val] + 1
    return h

def f_hequalization(A, image_depth, tones):
    minA = 0; maxA = image_depth
    B = np.round((tones-1) * ((A-minA) / (maxA-minA)))
    M = np.size(B, 0); N = np.size(B, 1)
    Bval = np.reshape(B, M*N)
    p = np.argsort(Bval)
    neq = np.int32((M*N) / tones + 0.5)
    BL = len(Bval)
    az = np.int32(np.fix((N*M) / neq))
    zRem = np.int32(np.remainder(BL, neq))
    D = np.zeros(M*N)
    k = -1
    for i in range(0, (neq*az), neq):
        k = k+1
        for j in range(0, neq):
            D[i+j] = k
    if(zRem > 0):
        for i in range((neq*az), ((az*neq)+zRem)):
            D[i] = tones-1
    L = np.zeros(M*N)
    k = -1
    for i in range(M):
        for j in range(N):
            k = k+1
            L[p[k]] = D[k]
    Z = np.reshape(L, B.shape)
    Z = imNormalize(Z, tones)
    return Z

def CDF_equalization(im, image_depth, tones):
    B = np.round((tones-1) * ((im) / (np.max(im))))
    M = np.size(im, 0); N = np.size(im, 1)
    CDFh = np.zeros(tones, dtype=float)
    CDFq = np.zeros(tones, dtype=float)
    h = f_histogram(im, image_depth, tones)
    tone_values = ((M*N) / tones)
    q = (tone_values * np.ones(tones, dtype=float))
    for i in range(tones):
        for j in range(i+1):
            CDFh[i] = CDFh[i] + h[j]
            CDFq[i] = CDFq[i] + q[j]
    B = CDFh[np.int32(B)] / tone_values - 1
    B = np.round(B)
    return B

def CLAHE_skimage(im):
    im_CLAHE = exposure.equalize_adapthist(np.uint8(im), clip_limit=0.03)
    im_CLAHE = 255 * im_CLAHE / np.max(im_CLAHE)
    return im_CLAHE

def HE_opencv(im):
    im_uint8 = np.uint8(np.clip(im, 0, 255))
    im_eq_cv = cv2.equalizeHist(im_uint8)
    return im_eq_cv

def CLAHE_opencv(im):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    im_uint8 = np.uint8(np.clip(im, 0, 255))
    im_clahe_cv = clahe.apply(im_uint8)
    return im_clahe_cv