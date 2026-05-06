import numpy as np

# ── 1D FILTER DESIGN ─────────────────────────────────────────────────────────

def Butterworth(N, ndegree, fco, TYPE, trans):
    fh = np.zeros(np.int32(N), dtype=float)
    if (N % 2) == 0:
        L = np.round(N/2+1)
        M = np.round(N/2+2)
    else:
        L = np.round(N/2+0.5)
        M = np.round(N/2+1+0.5)
    if TYPE == 1:
        for k in range(np.int32(L)):
            fh[k] = 1.0/(1.0+0.414*np.power((k/fco), (2*ndegree)))
        sText = 'Butterworth LP'
    elif TYPE == 2:
        for k in range(np.int32(L)):
            fh[k] = 1.0/(1.0+0.414*np.power((fco/(k+0.001)), (2*ndegree)))
        for k in range(np.int32(L)):
            if k < int(N/2-trans):
                fh[k] = fh[k+int(trans)]
            else:
                fh[k] = fh[int(N/2)]
        sText = 'Butterworth HP'
    elif TYPE == 3:
        d = trans
        for k in range(np.int32(L)):
            fh[k] = 1.0/(1.0+0.414*np.power((fco/(k-d+0.001)), (2*ndegree)))
        sText = 'Butterworth BR'
    elif TYPE == 4:
        d = trans; enh = 0.001
        fh = np.zeros(np.int32(N), dtype=float)+enh
        for k in range(np.int32(L)):
            fh[k] = 1.0/(1.0+0.414*np.power(((k-d)/fco), (2*ndegree)))
        sText = 'Butterworth BP'
    for k in range(np.int32(M-1), np.int32(N)):
        fh[k] = fh[np.int32(N-k)]
    return fh/np.max(fh), sText

def Exponential(N, ndegree, fco, TYPE, trans):
    fh = np.zeros(np.int32(N), dtype=float)
    if (N % 2) == 0:
        L = np.round(N/2+1)
        M = np.round(N/2+2)
    else:
        L = np.round(N/2+0.5)
        M = np.round(N/2+1+0.5)
    if TYPE == 1:
        for k in range(np.int32(L)):
            fh[k] = np.exp((-np.log(2))*(k/fco)**ndegree)
        sText = 'Exponential LP'
    elif TYPE == 2:
        for k in range(np.int32(L)):
            fh[k] = np.exp((-np.log(2))*(fco/(k+0.0001))**ndegree)
        for k in range(np.int32(L)):
            if k < int(N/2-trans):
                fh[k] = fh[k+int(trans)]
            else:
                fh[k] = fh[int(N/2)]
        sText = 'Exponential HP'
    elif TYPE == 3:
        d = trans
        for k in range(np.int32(L)):
            fh[k] = np.exp((-np.log(2))*(fco/(k-d+0.00001))**ndegree)
        sText = 'Exponential BR'
    elif TYPE == 4:
        d = trans; enh = 0.001
        fh = np.zeros(np.int32(N), dtype=float)+enh
        for k in range(np.int32(L)):
            fh[k] = np.exp((-np.log(2))*((k-d)/fco)**ndegree)
        sText = 'Exponential BP'
    for k in range(np.int32(M-1), np.int32(N)):
        fh[k] = fh[np.int32(N-k)]
    return fh/np.max(fh), sText

def Gaussian(N, ndegree, fco, TYPE, trans):
    fh = np.zeros(np.int32(N), dtype=float)
    if (N % 2) == 0:
        L = np.round(N/2+1)
        M = np.round(N/2+2)
    else:
        L = np.round(N/2+0.5)
        M = np.round(N/2+1+0.5)
    if TYPE == 1:
        for k in range(np.int32(L)):
            fh[k] = np.exp(-(k**2/(2*fco**2))**ndegree)
        sText = 'Gaussian LP'
    elif TYPE == 2:
        for k in range(np.int32(L)):
            fh[k] = np.exp(-(2*fco**2/(k+0.0001)**2)**ndegree)
        for k in range(np.int32(L)):
            if k < int(N/2-trans):
                fh[k] = fh[k+int(trans)]
            else:
                fh[k] = fh[int(N/2)]
        sText = 'Gaussian HP'
    elif TYPE == 3:
        d = trans
        for k in range(np.int32(L)):
            fh[k] = np.exp(-(2*fco**2/(k-d+0.00001)**2)**ndegree)
        sText = 'Gaussian BR'
    elif TYPE == 4:
        d = trans; enh = 0.001
        fh = np.zeros(np.int32(N), dtype=float)+enh
        for k in range(np.int32(L)):
            fh[k] = np.exp(-((k-d)**2/(2*fco**2))**ndegree)
        sText = 'Gaussian BP'
    for k in range(np.int32(M-1), np.int32(N)):
        fh[k] = fh[np.int32(N-k)]
    return fh/np.max(fh), sText

# ── 2D FILTER APPLICATION ─────────────────────────────────────────────────────

def design2dFilter(im, fh):
    y = np.size(im, 0); x = np.size(im, 1)
    FH = np.zeros(np.shape(im), dtype=float)
    for k in range(y):
        for m in range(x):
            K = y/2-k+1
            M = x/2-m+1
            ir = np.int32(np.sqrt((K*K+M*M))+0.5)
            if ir > len(fh):
                ir = len(fh)
            FH[k][m] = fh[ir]
    FH = np.fft.fftshift(FH)
    return FH

def filterImage(im, FH):
    Fim = np.fft.fft2(im)
    Fim1 = Fim * FH
    im1 = np.real(np.fft.ifft2(Fim1))
    return im1

# ── IMAGE RESTORATION ─────────────────────────────────────────────────────────

def GaussianMTF(N):
    fh = np.zeros(np.int32(N), dtype=float)
    if (N % 2) == 0:
        L = np.round(N/2+1)
        M = np.round(N/2+2)
    else:
        L = np.round(N/2+0.5)
        M = np.round(N/2+1+0.5)
    sigma = L/2-1
    for k in range(np.int32(L)):
        fh[k] = np.exp(-k**2/(2*sigma**2))
    for k in range(np.int32(M-1), np.int32(N)):
        fh[k] = fh[np.int32(N-k)]
    return fh

def from1dTo2dFilter(im, fh):
    y = np.size(im, 0); x = np.size(im, 1)
    FH = np.zeros(np.shape(im), dtype=float)
    for k in range(y):
        for m in range(x):
            K = y/2-k+1
            M = x/2-m+1
            ir = np.int32(np.sqrt((K*K+M*M))+0.5)
            if ir > len(fh):
                ir = len(fh)
            FH[k][m] = fh[ir]
    FH = FH/np.amax(FH)
    return FH

def generalizedWienerFilter(fh, filtType, SIGMA):
    N = len(fh)
    C = 2*SIGMA**2
    if filtType == 1:
        a = 1; b = 0
        sText = 'Inverse Filter'
    elif filtType == 2:
        a = 0; b = 1
        sText = 'Wiener Filter'
    elif filtType == 3:
        a = 0.5; b = 1
        sText = 'Power Filter'
    fhh = np.zeros(np.int32(N), dtype=float)
    for k in range(np.int32(N)):
        fhh[k] = (((fh[k]**2)/(fh[k]**2+b*C))**(1-a))*(1/fh[k])
        if fhh[k] < C:
            fhh[k] = (((fh[k]**2)/(fh[k]**2+b*C))**(1-a))*(1/C)
    fhh = fhh/np.max(fhh)
    return fhh, sText

def DeconvolveImage(im, FH):
    Fim = np.fft.fft2(im)
    Fim1 = Fim * np.fft.fftshift(FH)
    im1 = np.real(np.fft.ifft2(Fim1))
    return im1