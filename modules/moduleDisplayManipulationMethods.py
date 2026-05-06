import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')# Qt5Agg

#----------------FUNCTIONS-----------------------------------
    
#---------------------------------------------------
def simpleDisplay(im,image_depth,tones):
    im1=np.zeros(im.shape, dtype=float)
    im1=np.round(( (tones-1)/(image_depth-1) ) * (im-0))
    return im1
#---------------------------------------------------
def imPlot(im,title,tones,fz):
    # plt.figure(figsize=(fz,fz));
    plt.imshow(im,cmap=plt.cm.gray,vmin=0,vmax=tones);
    plt.title (title);plt.axis("off");
#---------------------------------------------------
def optimalDisplay(im,tones):
    im1=np.zeros(im.shape, dtype=float) 
    vmn=int(np.min(im))
    vmx=int(np.max(im))
    im1=(((tones-1)*(im-vmn)/(vmx-vmn)))
    im1=np.around(im1)
    return im1
#--------------------------------------------------------

def simpleWindow(im,wc,ww,image_depth, tones):
    im1=np.zeros(im.shape, dtype=float)

    Vb=(2.0*wc+ww)/2.0;
    if(Vb>image_depth):
        Vb=image_depth;
    Va=Vb-ww;
    if(Va<0): 
        Va=0;
    M=np.size(im,0)#image rows
    N=np.size(im,1)#image columns        
    
    for i in range(M):
        for j in range(N):
            Vm=im[i][j]
            if(Vm<Va): 
                t=0
            elif(Vm>Vb):
                t=tones-1
            else: 
                t=(((tones-1)*(Vm-Va)/(Vb-Va)))
            im1[i][j]=np.round(t)
    return im1


#--------------------------------------------------------

def brokenWindow(im,image_depth,tones,gray_val,im_val):
    im=np.asarray(im,dtype=float)
    N=np.size(im,0)#image rows
    M=np.size(im,1)#image columns
    im1=np.zeros(im.shape, dtype=float)
    for i in range(N):
        for j in range(M):
            if(im[i][j]<=im_val):
                im1[i][j]=(gray_val/(im_val))*im[i][j]
            else:
                im1[i][j]= (  ((tones-1)-(gray_val+1))/(image_depth-(im_val+1)) )*(im[i][j]-(im_val+1)) + (gray_val+1);
    im1=np.round(im1)
    return im1


#-------------------------------------------------------

def doubleWindow(im,ww1,wl1,ww2,wl2,image_depth,tones):
    im=np.asarray(im,dtype=float)
    im1=np.zeros(im.shape, dtype=float)
    N=np.size(im,0)#image rows
    M=np.size(im,1)#image columns
    
    half= (tones/2)-1;
    ve1=round ( (2.0*wl1+ww1)/2.0 );
    vs1=ve1-ww1;
    ve2=round ( (2.0*wl2+ww2)/2.0 );
    vs2=ve2-ww2;
    if(vs2<ve1):
        new_point=round ( ((vs2+ve1))/2.0);
        ve1=new_point;
        vs2=ve1;
    
    if(vs1<0):
        vs1=0;
    if(ve2>image_depth):
        ve2=image_depth;
    
    for i in range (N):
        for j in range (M):
            if (im[i][j]<vs1) :
                im1[i][j]=0; 
            if ( im[i][j] >= vs1 and im[i][j]<=ve1 ):
                t= round ( (  (half-0)/(ve1-vs1) )*(im[i][j]-vs1) + 0.0);
                im1[i][j]=round(t);
            
            if ( im[i][j]>ve1 and im[i][j]<vs2 ):
                im1[i][j]=half+1;
            
            if ( im[i][j]>=vs2 and im[i][j]<=ve2 ):
                t= round ( (  ( (tones-1)-(half+1) )/(ve2-vs2) )*(im[i][j]-vs2) + (half+1));
                im1[i][j]=round(t);
            
            if (im[i][j]>ve2):
                im1[i][j]=tones-1;
    return im1
#----------------------------------------------------------
def formPlotFunction(tones,Choice):
    import math
    w=np.zeros(tones)
    # Choice=4
    for i in range (0,tones):
        if (Choice==0):#inverse
            w[i]=tones-i-1
            text='inverse'
        elif(Choice==1):#logarithmic
            r=0.05
            w[i]=math.log(1+r*i)
            text='logarithmic'
        elif(Choice==2):#inverse logarithmic
            c=128
            w[i]=np.exp(i)**(1/c)-1
            text='inverse logarithmic'
        elif(Choice==3):#power
            gamma=0.55    
            w[i]=i**gamma
            text='power'
        elif(Choice==4):
            Val=4
            w[i]=np.sin(2*np.pi*i/(Val*(tones-1)))
            text='sine-window'
        elif(Choice==5):#exponential
            Val=90
            w[i]=1-np.exp(-i/Val)
            text='exp-window'
                
    w=(tones-1)*((w-np.min(w))/(np.max(w)-np.min(w)))        
    # print(w)        
    # print('wMin: %d, wMax: %d: ' %(np.min(w),np.max(w)))    
    # print('display function '+text+': \n',w)
    
    return(w,text)
#---------------------------------------------------------
# def getLinearWindowingFunction(im,Choice):
#     if (idFunction==0):#Optimal
#        im1= optimalDisplay(im,tones)

#     elif(idFunction==1):#Simple Window
#         wc=50;ww=250;
#         im1=simpleWindow(im,wc,ww,image_depth,tones)
       
#     elif(idFunction==2):#Broken Window
#        gray_val=128;im_val=70
#        im1=brokenWindow(im,image_depth,tones,gray_val,im_val)
       
#     elif(idFunction==3):#Double Window
#         ww1=100;wl1=50
#         ww2=100;wl2=150
#         im1=doubleWindow(im,ww1,wl1,ww2,wl2,image_depth,tones)
#     return(im1)
#-----------------------------------------------------------
def getNonLinearFunction_FormProcessedImage(im,fChoice):

    im=np.asarray(im,float)
    tones=256;image_depth=256
    x = np.arange(0, image_depth)
    y = np.zeros(image_depth)
    y_initial = (tones-1) / (image_depth-1) * x

    (w,sText)=formPlotFunction(tones,fChoice)

    fz=8#size of display 
    image_depth=256;tones=256;
    
    N=np.size(im,0)#image rows
    M=np.size(im,1)#image columns
    #normalize to 0:tones-1 the original image 
    mn=np.min(im);mx=np.max(im);
    im1=np.round((tones-1)*(im-mn)/(mx-mn))
    #trasform image to w function
    for i in range(0,N):#N
        for j in range(M):
            v=int(im1[i,j])
            im1[i,j]=np.int32(w[v])  
    return (im1,y_initial,w)
#------------------------------------------------------     
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])    
#-----------------------------------------------------------
def loadImage(imageFile):
    im=plt.imread(imageFile);
    im=np.asarray(im, dtype=float)
    L=np.shape(im)
    if(len(L)==3):
        im=rgb2gray(im)
    return(im)   
#----------------------------------------------------------------
def WindowingFunction(Choice, tones, image_depth, **kwargs):
    x = np.arange(0, image_depth)
    y = np.zeros(image_depth)
    y_initial = (tones-1) / (image_depth-1) * x
    if Choice == 0: #Optimal Window
        
        wc = kwargs['wc']; ww = kwargs['ww']
        Vb = (2.0*wc + ww) / 2.0
        if Vb > image_depth: Vb = image_depth
        Va = Vb - ww
        if Va < 0: Va = 0
        for i, v in enumerate(x):
            if v < Va: y[i] = 0
            elif v > Vb: y[i] = tones - 1
            else: y[i] = (tones-1) * (v-Va) / (Vb-Va)

    elif Choice == 1: #Simple Window
        wc = kwargs['wc']; ww = kwargs['ww']
        Vb = (2.0*wc + ww) / 2.0
        if Vb > image_depth: Vb = image_depth
        Va = Vb - ww
        if Va < 0: Va = 0
        for i, v in enumerate(x):
            if v < Va: y[i] = 0
            elif v > Vb: y[i] = tones - 1
            else: y[i] = (tones-1) * (v-Va) / (Vb-Va)

    elif Choice == 2: # Broken window
        gray_val = kwargs['gray_val']; im_val = kwargs['im_val']
        for i, v in enumerate(x):
            if v <= im_val:
                y[i] = (gray_val / im_val) * v
            else:
                y[i] = (((tones-1)-(gray_val+1)) / (image_depth-(im_val+1))) * (v-(im_val+1)) + (gray_val+1)

    elif Choice == 3: #double window
        ww1=kwargs['ww1']; wl1=kwargs['wl1']
        ww2=kwargs['ww2']; wl2=kwargs['wl2']
        half = (tones/2) - 1
        ve1 = round((2.0*wl1+ww1)/2.0)
        vs1 = ve1 - ww1
        ve2 = round((2.0*wl2+ww2)/2.0)
        vs2 = ve2 - ww2
        if vs2 < ve1:
            new_point = round((vs2+ve1)/2.0)
            ve1 = new_point; vs2 = ve1
        if vs1 < 0: vs1 = 0
        if ve2 > image_depth: ve2 = image_depth
        for i, v in enumerate(x):
            if v < vs1: y[i] = 0
            elif v >= vs1 and v <= ve1: y[i] = ((half)/(ve1-vs1))*(v-vs1)
            elif v > ve1 and v < vs2: y[i] = half + 1
            elif v >= vs2 and v <= ve2: y[i] = (((tones-1)-(half+1))/(ve2-vs2))*(v-vs2)+(half+1)
            elif v > ve2: y[i] = tones - 1
   
    y=(tones-1)*(y-min(y))/(max(y)-min(y)) 
    return(y_initial,y)
#----------------------------------------------------    

def imNormalize(w,tones):
    mx=np.max(w);mn=np.min(w);
    w=(tones-1)*(w-mn)/(mx-mn);    
    w=np.round(w)
    return w
def f_histogram(A,image_depth,tones):
    minA=0;maxA=image_depth;
    if(np.max(A)>(tones-1)):
        B=np.round((tones-1)*((A-minA)/(maxA-minA)))
    else:
        B=A
    #calculate histogram
    M=np.size(B,0)
    N=np.size(B,1)
    Bval=np.reshape(B,M*N)
    h=np.zeros(tones,dtype=float)
    for i in range(np.size(Bval)):
        val=np.int16(Bval[i])
        h[val]=h[val]+1;
    return (h)
#----------------------------------------------------

def f_hequalization (A,image_depth,tones):
    #implement HE method
    minA=0;maxA=image_depth;
    B=np.round((tones-1)*((A-minA)/(maxA-minA) ))
    M=np.size(B,0);N=np.size(B,1)
    Bval=np.reshape(B,M*N)#from 2d to 1d dimension
    p=np.argsort(Bval)#keep initial positions of sorted values 
    neq=np.int32((M*N)/tones+0.5) #calculate number of pixels per gray-tone
    BL=len(Bval)
    az=np.int32(np.fix((N*M)/neq))#get integral number of pixels per gray level
    zRem=np.int32(np.remainder(BL,neq))#get  remainder of pixels if not intergal part of neq
    D=np.zeros(M*N)#define 1d array to hold the transformed image matrix
    k=-1;
    for i in range(0,(neq*az),neq):
        k=k+1
        for j in range(0,neq):
            D[i+j]=k
    if(zRem>0):
        for i in range( (neq*az),( (az*neq)+zRem) ):
            D[i]=tones-1
    #reassign equalized values into their proper position
    L=np.zeros(M*N)
    k=-1;
    for i in range (M):
        for j in range (N):
            k=k+1;
            L[p[k]]=D[k];
    #finalize equalized image
    Z=np.reshape(L,B.shape)
    Z=imNormalize(Z,tones)
    return (Z)    
#--------------------------------------------------
# @jit('void(double[:,:],double,double)')
def CDF_equalization (im,image_depth,tones):
    B=np.round((tones-1)*((im)/(np.max(im)) ))
    M=np.size(im,0);N=np.size(im,1)
    CDFh=np.zeros(tones,dtype=float)
    CDFq=np.zeros(tones,dtype=float)
    h=f_histogram(im,image_depth,tones)
    tone_values=((M*N)/tones)
    q=(tone_values*np.ones(tones,dtype=float));
    for i in range (tones):
        for j in range (i+1):
            CDFh[i]=CDFh[i]+h[j];
            CDFq[i]=CDFq[i]+q[j];
    B=CDFh[np.int32(B)]/tone_values-1
    B=np.round(B)
    return B

def histCumsum(im):
    hist,bins = np.histogram(im.flatten(),256,[0,256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max()/ cdf.max()
    return(cdf_normalized)
