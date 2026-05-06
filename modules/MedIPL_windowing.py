import numpy as np
import math

def simpleWindow(im, wc, ww, image_depth, tones):
    im1 = np.zeros(im.shape, dtype=float)
    Vb = (2.0 * wc + ww) / 2.0
    if Vb > image_depth:
        Vb = image_depth
    Va = Vb - ww
    if Va < 0:
        Va = 0
    M = np.size(im, 0)
    N = np.size(im, 1)
    for i in range(M):
        for j in range(N):
            Vm = im[i][j]
            if Vm < Va:
                t = 0
            elif Vm > Vb:
                t = tones - 1
            else:
                t = (((tones-1) * (Vm-Va) / (Vb-Va)))
            im1[i][j] = np.round(t)
    return im1

def simpleDisplay(im, image_depth, tones):
    im1 = np.round(((tones-1) / (image_depth-1)) * (im - 0))
    return im1

def optimalDisplay(im, tones):
    vmn = int(np.min(im))
    vmx = int(np.max(im))
    im1 = np.around((tones-1) * (im-vmn) / (vmx-vmn))
    return im1

def brokenWindow(im, image_depth, tones, gray_val, im_val):
    im = np.asarray(im, dtype=float)
    im1 = np.zeros(im.shape, dtype=float)
    N = np.size(im, 0)
    M = np.size(im, 1)
    for i in range(N):
        for j in range(M):
            if im[i][j] <= im_val:
                im1[i][j] = (gray_val / im_val) * im[i][j]
            else:
                im1[i][j] = (((tones-1) - (gray_val+1)) / (image_depth - (im_val+1))) * (im[i][j] - (im_val+1)) + (gray_val+1)
    im1 = np.round(im1)
    return im1

def doubleWindow(im, ww1, wl1, ww2, wl2, image_depth, tones):
    im = np.asarray(im, dtype=float)
    im1 = np.zeros(im.shape, dtype=float)
    N = np.size(im, 0)
    M = np.size(im, 1)
    half = (tones / 2) - 1
    ve1 = round((2.0 * wl1 + ww1) / 2.0)
    vs1 = ve1 - ww1
    ve2 = round((2.0 * wl2 + ww2) / 2.0)
    vs2 = ve2 - ww2
    if vs2 < ve1:
        new_point = round((vs2 + ve1) / 2.0)
        ve1 = new_point
        vs2 = ve1
    if vs1 < 0:
        vs1 = 0
    if ve2 > image_depth:
        ve2 = image_depth
    for i in range(N):
        for j in range(M):
            if im[i][j] < vs1:
                im1[i][j] = 0
            if im[i][j] >= vs1 and im[i][j] <= ve1:
                im1[i][j] = round(((half-0) / (ve1-vs1)) * (im[i][j]-vs1) + 0.0)
            if im[i][j] > ve1 and im[i][j] < vs2:
                im1[i][j] = half + 1
            if im[i][j] >= vs2 and im[i][j] <= ve2:
                im1[i][j] = round((((tones-1) - (half+1)) / (ve2-vs2)) * (im[i][j]-vs2) + (half+1))
            if im[i][j] > ve2:
                im1[i][j] = tones - 1
    return im1

def formPlotFunction(tones, choice):
    w = np.zeros(tones)
    for i in range(0, tones):
        if choice == 5:
            w[i] = tones - i - 1
        elif choice == 6:
            r = 0.05
            w[i] = math.log(1 + r * i)
        elif choice == 7:
            c = tones - 1
            w[i] = math.exp(i) ** (1/c) - 1
        elif choice == 8:
            gamma = 2
            w[i] = i ** gamma
        elif choice == 9:
            w[i] = np.sin(2 * np.pi * i / (4 * (tones-1)))
        elif choice == 10:
            w[i] = np.exp(i / 20)
    w = (tones-1) * ((w - np.min(w)) / (np.max(w) - np.min(w)))
    return w

def sigmoid(im, tones):
    im = np.asarray(im, dtype=float)
    w = np.zeros(tones)
    for i in range(tones):
        w[i] = (1 / (1 + math.exp(-i / 70))) * 255
    mn = np.min(im); mx = np.max(im)
    im1 = np.round((tones-1) * (im-mn) / (mx-mn))
    N = np.size(im, 0); M = np.size(im, 1)
    for i in range(N):
        for j in range(M):
            v = int(im1[i, j])
            im1[i, j] = np.int32(w[v])
    return im1

def cosine(im, tones):
    im = np.asarray(im, dtype=float)
    w = np.zeros(tones)
    for i in range(tones):
        w[i] = (1 - np.cos(2 * np.pi * i / (4 * (tones-1)))) * 255
    mn = np.min(im); mx = np.max(im)
    im1 = np.round((tones-1) * (im-mn) / (mx-mn))
    N = np.size(im, 0); M = np.size(im, 1)
    for i in range(N):
        for j in range(M):
            v = int(im1[i, j])
            im1[i, j] = np.int32(w[v])
    return im1

def rSqrt(im, tones):
    im = np.asarray(im, dtype=float)
    w = np.zeros(tones)
    for i in range(1, tones):
        w[i] = 1 / np.sqrt(i) * 255
    w = (tones-1) * ((w - np.min(w)) / (np.max(w) - np.min(w)))
    mn = np.min(im); mx = np.max(im)
    im1 = np.round((tones-1) * (im-mn) / (mx-mn))
    N = np.size(im, 0); M = np.size(im, 1)
    for i in range(N):
        for j in range(M):
            v = int(im1[i, j])
            im1[i, j] = np.int32(w[v])
    return im1

def presetWindow(im, preset, image_depth, tones):
    presets = {
    'brain':       {'wc': 128, 'ww': 60},   # στενό παράθυρο κεντρικά
    'bone':        {'wc': 200, 'ww': 100},  # φωτεινές δομές
    'lung':        {'wc': 80,  'ww': 200},  # πλατύ παράθυρο
    'abdomen':     {'wc': 128, 'ww': 150},  # μεσαίο παράθυρο
    'soft_tissue': {'wc': 110, 'ww': 120},  # ελαφρά στενό
}
    if preset not in presets:
        print(f'Unknown preset: {preset}')
        return im
    wc = presets[preset]['wc']
    ww = presets[preset]['ww']
    im1 = simpleWindow(im, wc, ww,image_depth, tones)
    return im1