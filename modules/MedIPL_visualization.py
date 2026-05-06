import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def histCumsum(im):
    hist,bins = np.histogram(im.flatten(),256,[0,256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max()/ cdf.max()
    return(cdf_normalized)

def imPlot(im,title):
    
    im = Image.fromarray( np.asarray( im, dtype="uint8"), "L" )
    plt.imshow(im,cmap=plt.cm.gray,vmin=0,vmax=255);
    plt.title (title);plt.axis("off");