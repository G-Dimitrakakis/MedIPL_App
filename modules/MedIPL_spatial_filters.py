import numpy as np
from scipy import signal
import time

def tic():
    t1 = float(time.time())
    return t1

def toc(t1, s):
    t2 = float(time.time())
    dt = t2 - t1
    s = 'Time for processing using %s: ' % s
    print("%s %f" % (s, dt))

def conv2(im, mask):
    im1 = signal.convolve2d(im, mask, mode='same')
    return im1

def SaltPepperNoise(im):
    r = np.random.ranf(im.shape) < 0.01
    im1 = np.asarray(im, dtype=float)
    im1 = (1-r)*im + r*200
    r = np.random.ranf(im.shape) < 0.1
    im1 = (1-r)*im1 - r*200
    return im1

def clipImage(im, image_depth):
    im1 = np.asarray(im, dtype=float)
    M = np.size(im1, 0)
    N = np.size(im1, 1)
    for i in range(M):
        for j in range(N):
            if im1[i][j] > image_depth:
                im1[i][j] = image_depth
            elif im1[i][j] < 0:
                im1[i][j] = 0
    return im1

def getKernelTitle(kernel):
    k = np.asarray(kernel, dtype=float)
    k_flat = np.reshape(k, np.size(k, 0) * np.size(k, 1))
    return 'Kernel: ' + str(k_flat)

# ── SMOOTHING FILTERS ─────────────────────────────────────────────────────────

def smoothing(im, idMask, image_depth, tones):
    kernels = {
        0: np.array([[0,1,0],[1,1,1],[0,1,0]]),   # SM1 - Cross average
        1: np.array([[1,1,1],[1,1,1],[1,1,1]]),   # SM2 - Box average
        2: np.array([[1,1,1],[1,2,1],[1,1,1]]),   # SM3 - Weighted center
        3: np.array([[1,2,1],[2,4,2],[1,2,1]])    # SM4 - Gaussian
    }
    kernel = kernels[idMask]
    title = getKernelTitle(kernel)
    sK = np.sum(kernel)
    if sK > 0:
        kernel = kernel / sK
    im1 = np.asarray(im, dtype=float)
    t1 = tic()
    im_out = conv2(im1, kernel)
    toc(t1, 'Smoothing conv2d')
    return im_out, title

# ── LAPLACIAN FILTERS ─────────────────────────────────────────────────────────

def laplacian(im, idMask, image_depth, tones):
    kernels = {
        0: np.array([[0,1,0],[1,-4,1],[0,1,0]]),      # LM1 - 4-neighbor
        1: np.array([[1,1,1],[1,-8,1],[1,1,1]]),      # LM2 - 8-neighbor
        2: np.array([[1,2,1],[2,-12,2],[1,2,1]]),     # LM3 - Weighted
        3: np.array([[-1,2,-1],[2,-4,2],[-1,2,-1]]),  # LM4 - Diagonal
        4: np.array([[0,0,0],[0,-1,0],[0,1,0]])       # Edge detection
    }

    kernel = kernels[idMask]
    title = getKernelTitle(kernel)
    sK = np.sum(kernel)
    if sK > 0:
        kernel = kernel / sK
    im1 = np.asarray(im, dtype=float)
    im_out = conv2(im1, kernel)
    im_out = clipImage(im_out, image_depth)
    from modules.MedIPL_histograms import imNormalize
    im_out = imNormalize(im_out, tones)
    return im_out, title

# ── HIGH ENHANCEMENT FILTERS ──────────────────────────────────────────────────

def highEnhancement(im, idMask, image_depth, tones):
    kernels = {
        0: np.array([[0,-1,0],[-1,5,-1],[0,-1,0]]),       # HE1 - Cross
        1: np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]]),   # HE2 - Full
        2: np.array([[-1,-2,-1],[-2,13,-2],[-1,-2,-1]]),  # HE3 - Strong
        3: np.array([[-1,-2,-1],[-2,13,-2],[-1,-2,-1]])   # HE4 - Strong v2
    }
    kernel = kernels[idMask]
    title = getKernelTitle(kernel)
    sK = np.sum(kernel)
    if sK > 0:
        kernel = kernel / sK
    im1 = np.asarray(im, dtype=float)
    im_out = conv2(im1, kernel)
    im_out = clipImage(im_out, image_depth)
    from modules.MedIPL_histograms import imNormalize
    im_out = imNormalize(im_out, tones)
    return im_out, title

# ── MEDIAN FILTER ─────────────────────────────────────────────────────────────

def medianFilter(im):
    im1 = np.asarray(im, dtype=float)
    im1 = SaltPepperNoise(im1)
    t1 = tic()
    im_out = signal.medfilt2d(im1, (5, 5))
    toc(t1, 'Median Filter')
    return im1, im_out  # επιστρέφει και την εικόνα με θόρυβο για σύγκριση