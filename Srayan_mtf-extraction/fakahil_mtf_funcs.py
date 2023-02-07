import numpy as np
from scipy import special
import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import scipy
import math
from numpy import fft
from numpy.fft import fft, ifft, fftshift, ifftshift
from scipy.optimize import curve_fit
import statsmodels.api as sm
lowess = sm.nonparametric.lowess
import cv2
from skimage import io

########
# fakahil functions
########


## Define Erfc function
def Erfc(x,sigma,a):

 y = a*special.erfc(-x/(sigma*np.sqrt(2)))
 return y

# Compute the first derivative

def process_limb(x,y):
    frac = 1/30.
    d1 = np.diff(y)/np.diff(x)
    xd1 = 0.5*(x[:-1]+x[1:])
    xd2 = 0.5*(xd1[:-1]+xd1[1:])
    d2 = np.diff(d1)/np.diff(xd1)
    w = lowess(d1, xd1, frac)
    s_x, s_y = w[:,0], w[:,1]
    #ind = np.where(d1==d1.max())
    #print  x[ind]
    ind = np.where(s_y==s_y.max())
    #temp =  x[ind] 
    temp = s_x[ind] 
    x = (x-  temp)#x[ind]
    # print(temp)
    # plt.plot(xd1,d1)
    # plt.plot(s_x,s_y)
    # plt.axvline(x=temp)
    # plt.show()
    return temp,x#, y


def detect_edge(x,y):
    frac = 1/30.
    d1 = np.diff(y)/np.diff(x)
    xd1 = 0.5*(x[:-1]+x[1:])
    w = lowess(d1, xd1, frac)
    s_x, s_y = w[:,0], w[:,1]
    ind = np.where(s_y==s_y.max())
    # plt.plot(x,y)
    # plt.plot(s_x,s_y)
    # plt.axvline(x[ind])
    # plt.show()
    #plt.clf()
    #print ind
    return s_x[ind],y[ind]
    

def merge(x,y):
    x = np.concatenate(x)
    y = np.concatenate(y)
    return x,y

def sort(x,y):
    points = zip(x,y)

    points.sort()
    xx = np.array([p[0] for p in points])
    yy = np.array([p[1] for p in points])
    return xx,yy

def compute_lsf(x,y):
    d1 = np.abs(np.diff(y)/np.diff(x))
    xd1 = 0.5*(x[:-1]+x[1:])
    return xd1, d1

def compute_mtf(lsf_x,lsf_y):
    lsf = fftshift(lsf_y)
    mtf = fft(lsf)
    f  = (lsf_x.size) * np.arange(0, lsf_x.size, step = 1)
    
    #freq = np.linspace(0,N*0.5,size/2)
    #freq = np.arange(0,N*0.5,1./(size))
    mtf= np.abs(mtf)
    mtf = mtf/mtf.max()
    # plt.plot(f[f <= 1 / 2], mtf[f <= 1 / 2])
    # plt.show()
    #return freq, mtf[0:size/2]


def Lowess(y,x,frac):
    w = sm.nonparametric.lowess(y, x, frac)
    s_x, s_y = w[:,0], w[:,1]
    return s_x,s_y

theta = 3

########
# our functions
########

def load_stack(path, x_min=400, x_max=700, y_min=300, y_max=600, show=False):
    imgs = io.imread(path+'/MMStack_Pos0.ome.tif')[:, x_min:x_max, y_min:y_max]
    if show:
        plt.imshow(imgs[0], cmap="gray")
        plt.title(f'Image 1 of {len(imgs)}')
        plt.show()
    print(f"Loaded stack of {imgs.shape[0]} images, dimensions {imgs.shape[1:]}")
    return imgs

def average_stack(imgs, im_per_pos=5, show=False):
    # adapted from: https://stackoverflow.com/a/69721142
    image_sets = imgs.reshape((len(imgs)//im_per_pos, im_per_pos, imgs.shape[1], imgs.shape[2]))
    avg_stack = np.mean(image_sets, axis=1)
    if show:
        plt.imshow(avg_stack[0], cmap="gray")
        plt.title(f'Image 1 of {len(avg_stack)}')
        plt.show()
    print(f"Averaged original stack of {len(imgs)} down to {len(avg_stack)}.")
    return avg_stack


def remove_background(imgs, bg_light, bg_dark, show=False):
    # adapted from: https://stackoverflow.com/a/73082666
    imgs_minus_bg = np.clip(imgs - bg_dark, 0, imgs.max())
    light_minus_bg = np.clip(bg_light - bg_dark, 0, bg_light.max())
    divided = np.clip(imgs_minus_bg / light_minus_bg, 0, imgs_minus_bg.max())
    if show:
        f, axes = plt.subplots(2, 2, figsize=(10, 10))
        axes[0][0].imshow(imgs[0], cmap="gray")
        axes[0][0].set_title(f"imgs, 1 of {len(imgs)}")
        axes[0][1].imshow(imgs_minus_bg[0], cmap="gray")
        axes[0][1].set_title(f"imgs_minus_bg, 1 of {len(imgs_minus_bg)}")
        axes[1][0].imshow(light_minus_bg[0], cmap="gray")
        axes[1][0].set_title(f"light_minus_bg, 1 of {len(light_minus_bg)}")
        axes[1][1].imshow(divided[0], cmap="gray")
        axes[1][1].set_title(f"divided, 1 of {len(divided)}")
        plt.tight_layout()
        plt.show()
    print(f"Removed background and divided illumination from {len(divided)} images.")
    return divided


def allComputeMTF(edge):
    rows = np.arange(edge.shape[0]) 
    n_rows = len(rows) 
    i = 0
    profile = np.zeros((n_rows,edge.shape[1]))
    for row in rows:
        profile[i] = edge[row,:]
        i = i+1

    coords_x = []
    coords_y = []
    x = np.arange(edge.shape[1])

    for row in range(n_rows):
        x_e,y_e = detect_edge(x,profile[row])
        coords_x.append(x_e)
        coords_y.append(row)

    x = sorted(coords_x)
    y = sorted(coords_y)

    from scipy.optimize import curve_fit
    x = np.asarray(x)
    y = np.asarray(y)
    def f(x, A, B): 
        return A*x + B
    p0 = [1,1]
    popt, pcov = scipy.optimize.curve_fit(f,x[:,0],y,p0)

    angle =np.rad2deg(np.arctan(popt[0]))

    rows = np.arange(30,30+19,1)
    n_rows = len(rows)
    i = 0
    profile = np.zeros((n_rows,edge.shape[1]))
    for row in rows:
        profile[i] = edge[row,:]
        i = i+1

    x = np.arange(edge.shape[1])
    shifted_x =[]
    Temp = []
    temp =  process_limb(x,profile[0])[0]
    shifted_x.append(process_limb(x,profile[0])[1])
    Temp.append(temp)

    for row in range(1,n_rows):
        x = np.arange(edge.shape[1])
        temp =temp + np.tan(math.radians(theta))
        x = x - temp
        shifted_x.append(x)
        Temp.append(temp)

    shifted_x = np.asarray(shifted_x)

    shifted_x = shifted_x*np.cos(math.radians(theta))

    all_x, all_y = merge(shifted_x,profile)

    all_x = sorted(all_x)
    all_y = sorted(all_y)

    s_x,s_y = Lowess(all_y,all_x,1/40.)

    esfx =s_x
    esfy =s_y

    lsfx,lsfy = compute_lsf(esfx,esfy)
    lsfy2 = lsfy*np.hanning(len(lsfy))

    # plt.figure(figsize=(10,10))
    size = lsfx.shape[0]
    size2 = edge.shape[1]
    lsf = fftshift(lsfy)
    mtf = fft(lsf)
    f_s = 1./19
    f  = (1./(f_s*size) * np.arange(0, size, step = 1))
    f2 = (1./(size2) * np.arange(0, size2, step = 1))
    mtf = np.abs(mtf)
    mtf = mtf/mtf.max()
    mtf2 = mtf[:edge.shape[1]]
    plt.plot(f[f <= n_rows / 2], mtf[f <= n_rows / 2])
    plt.scatter(f2, mtf2,label='Slanted edge')
    plt.xlim(0,0.5)
    plt.legend()
    plt.xlabel('cycles / pixel')
    plt.ylabel('MTF')
    # plt.show()
