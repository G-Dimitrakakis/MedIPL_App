import numpy as np
from skimage.transform import radon, iradon, iradon_sart

def sigNorm(x):
    xmax = np.max(x); xmin = np.min(x)
    x = (x - xmin) / (xmax - xmin)
    return x

def imageToSinogram(im, N_proj=180):
    theta = np.arange(0, N_proj)
    mn = np.min(im); mx = np.max(im)
    im_norm = (im - mn) * (255 / (mx - mn))
    sinogram = radon(im_norm, theta=theta, circle=False)
    return sinogram, theta

def FBP_reconstruction(sinogram, theta, filterChoice=0):
    filters = ['ramp', 'shepp-logan', 'cosine', 'hamming', 'hann', None]
    filter_name = filters[filterChoice]
    if filter_name is None:
        I_FBP = iradon(sinogram, theta=theta, filter_name=None)
    else:
        I_FBP = iradon(sinogram, theta=theta, filter_name=filter_name,
                       output_size=sinogram.shape[0])
    return I_FBP, filter_name if filter_name else 'None'

def ART_reconstruction(sinogram, theta, iterations=5):
    I_ART = iradon_sart(sinogram, theta=theta)
    for i in range(iterations):
        I_ART = iradon_sart(sinogram, theta=theta, image=I_ART)
    return I_ART