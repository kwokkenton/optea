import numpy as np
from skimage import io
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from celluloid import Camera
from IPython.display import HTML
import scipy as sp

def add_position_labels(axes, slices, positions):
    """
    Given 2x2 subplot axes intended to display each quartile
    image from a stack, apply the appropriate titles
    from a list of positions of stack images in mm. 
    """
    
    axes[0][0].set_title(f'{positions[0]} mm')
    axes[0][1].set_title(f'{positions[slices//4]} mm')
    axes[1][0].set_title(f'{positions[slices//2]} mm')
    axes[1][1].set_title(f'{positions[(slices*3)//4]} mm')
    
    return axes


def load_disp_img(path, positions, x_min=300, x_max=800, y_min=300, y_max=600, disp=False):
    """
    Load and crop stack, optionally display quartile images
    """
    
    im = io.imread(path)[:, x_min:x_max, y_min:y_max]
    
    if disp:
        f, axes = plt.subplots(2,2)
        slices = len(im)

        axes[0][0].imshow(im[0], cmap='gray')
        axes[0][1].imshow(im[slices//4], cmap='gray')
        axes[1][0].imshow(im[slices//2], cmap='gray')
        axes[1][1].imshow(im[(slices*3)//4], cmap='gray')

        axes = add_position_labels(axes, slices, positions)

        plt.tight_layout()
        plt.show()

    return im

def line_prof_splash(im, positions):
    """
    Plot vertical line profiles through centre,
    for quartile images in stack
    """
    
    f, axes = plt.subplots(2,2,figsize=(15, 15))
    slices = len(im)
    centre = im.shape[2]//2
    
    axes[0][0].plot(im[0,:,centre])
    axes[0][1].plot(im[slices//4,:,centre])
    axes[1][0].plot(im[slices//2,:,centre])
    axes[1][1].plot(im[(slices*3)//4,:,centre])
    
    axes = add_position_labels(axes, slices, positions)

    plt.show()


def fourier(frame, plot=False):
    """
    Return Fourier transform array and frequency axes, optionally plot log(FFT) in 2D
    
    Adapted from:
    - https://thepythoncodingbook.com/2021/08/30/2d-fourier-transform-in-python-and-fourier-synthesis-of-images/
    - https://stackoverflow.com/a/39201385
    
    TODO: understand how to adapt to real lengths
    """
    
    ft = np.fft.ifftshift(frame)
    ft = np.fft.fft2(ft)
    ft = np.fft.fftshift(ft)
    
    freq_x = np.fft.fftfreq(ft.shape[0])
    freq_y = np.fft.fftfreq(ft.shape[1])
    
    if plot:
        plt.imshow(
            np.log10(np.abs(ft)),
            extent=(freq_x.min(),freq_x.max(),freq_y.min(),freq_y.max()),
            aspect=1
        )
        plt.colorbar()
        plt.xlabel('f_x (/pixel)')
        plt.ylabel('f_y (/pixel)')
        plt.show()
               
    return ft, (freq_x, freq_y)


def fft_plot_centre_y(frame):
    ft, freqs = fourier(frame)
    plt.plot(np.fft.fftshift(freqs[0]), (np.abs(ft)[:,ft.shape[1]//2]))
    plt.xlabel('f_y (/pixel)')
    plt.show()
    

def fft_plot_quartiles(im, positions):
    # TODO: are there better ways to represent this? or to quantify it?
    slices = len(im)
    qrts = [0, slices//4, slices//2, (slices*3)//4]
    
    ax = plt.figure().add_subplot(projection='3d')

    for q in qrts:
        ft, freqs = fourier(im[q])
        ax.plot(np.fft.fftshift(freqs[0]), (np.abs(ft)[:,ft.shape[1]//2]), positions[q])
        
    ax.set_xlabel('f_y (/pixel)')
    ax.set_zlabel('distance (mm)')
    ax.view_init(azim=20, elev=20, vertical_axis='y')
    plt.show()

    
def speed_run(path, positions):
    # quick and dirty comparison - run everything for each stack
    # this will vomit plots all over the notebook - be warned!
    im = load_disp_img(path, positions)
    line_prof_splash(im, positions)
    ft, freqs = fourier(im[len(im)//2], plot=True)
    fft_plot_centre_y(im[len(im)//2])
    fft_plot_quartiles(im, positions)

    
def fft_plot_quartiles_compare(paths, positions_lists, imgnames):  
    fig = plt.figure()
    num = len(paths)
    
    for i in range(num):
        ax = fig.add_subplot(1, num, i+1, projection='3d')
        
        im = load_disp_img(paths[i], positions_lists[i])
        slices = len(im)
        qrts = [0, slices//4, slices//2, (slices*3)//4]
        # qrts = range(slices)

        for q in qrts:
            ft, freqs = fourier(im[q])
            ax.plot(np.fft.fftshift(freqs[0]), (np.abs(ft)[:,ft.shape[1]//2]), positions_lists[i][q])

        ax.set_xlabel('f_y (/pixel)')
        ax.set_zlabel('distance (mm)')
        ax.set_title(imgnames[i])
        ax.view_init(azim=20, elev=20, vertical_axis='y')

    plt.tight_layout()
    plt.show()

    
def fft_find_peaks(frame, thresh=0.02e9, disp=False):
    ft, freqs = fourier(frame)
    peak_indices, properties = find_peaks(np.abs(ft)[:,ft.shape[1]//2], height=thresh)
    
    if disp:
        plt.plot(np.fft.fftshift(freqs[0]), (np.abs(ft)[:,ft.shape[1]//2]))
        plt.plot(np.fft.fftshift(freqs[0])[peak_indices], properties['peak_heights'],'x')
        plt.xlabel('f_y (/pixel)')
        plt.show()
        
    return peak_indices, properties


def fft_find_peaks_anim(im, positions, thresh=0.02e9):
    fig = plt.figure(figsize=(15, 10))
    plt.xlabel('f_y (/pixel)')
    camera = Camera(fig)
    
    for i in range(len(im)):
        ft, freqs = fourier(im[i])
        peak_indices, properties = find_peaks(np.abs(ft)[:,ft.shape[1]//2], height=thresh)
        line = plt.plot(np.fft.fftshift(freqs[0]), (np.abs(ft)[:,ft.shape[1]//2]))
        plt.plot(np.fft.fftshift(freqs[0])[peak_indices], properties['peak_heights'],'x')
        plt.legend(line, [f'{positions[i-1]} mm'])
        camera.snap()
    
    animation = camera.animate()
    return animation



def peak_distance_plots(paths, positions_lists, imgnames):
    fig = plt.figure(figsize=(15, 5))
    num = len(paths)

    for i in range(num):
        ax = fig.add_subplot(1, num, i+1)
        im = load_disp_img(paths[i], positions_lists[i])
        peaks_lists, _ = map(list, zip(*[fft_find_peaks(im[i], thresh=0.05e9) for i in range(len(im))]))
        ax.plot(positions_lists[i], [max(peaks) for peaks in peaks_lists])
        ax.set_ylabel('distance between furthest peaks')
        ax.set_xlabel('depth (mm)')
        ax.set_title(imgnames[i])
        
    plt.tight_layout()
    plt.show()


def peak_height_plots(paths, positions_lists, imgnames):
    fig = plt.figure(figsize=(15, 5))
    num = len(paths)

    for i in range(num):
        ax = fig.add_subplot(1, num, i+1)
        im = load_disp_img(paths[i], positions_lists[i])
        # need to set a very low threshold, else it breaks (as there is no second-highest peak)
        # TODO: use a more robust method to get the second-highest peak
        peaks_lists, properties_lists = map(list, zip(*[fft_find_peaks(im[i], thresh=0.001e9) for i in range(len(im))]))
        ax.plot(positions_lists[i], [sorted(properties['peak_heights'])[-2] for properties in properties_lists])
        ax.set_ylabel('height of second-max peak')
        ax.set_xlabel('depth (mm)')
        ax.set_title(imgnames[i])
        
    plt.tight_layout()
    plt.show()


def fit_square_to_lineprof(im, frame, positions, disp=False):
    def rect(t, ampl, pxperline, offset):
        return ampl*sp.signal.square(t*(2*np.pi)/pxperline*2)/2+offset

    def sine(t, ampl, pxperline, offset):
        # fitting a sine wave works
        # (fitting square wave directly doesn't...)
        return ampl*np.sin(t*(2*np.pi)/pxperline*2)/2+offset

    centre = im.shape[2]//2
    lineprof = im[frame,:,centre]
    length = len(lineprof)
    t = np.arange(0, length, 1)

    if disp:
        plt.plot(lineprof, '--')
        plt.title(f'{positions[frame]} mm')
        plt.xlabel('pixels')
        plt.ylabel('amplitude')

    popt, pcov = sp.optimize.curve_fit(sine, t, lineprof, p0=[40000,31,25000])
    ampl, pxperline, offset = popt

    idealsquare = rect(t, ampl, pxperline, offset)

    if disp:
        plt.plot(t, idealsquare)
        print(f"Fitted: {pxperline:.3f} pixels per line")
        plt.show()
    
    idealgrating = np.tile(rect(t, ampl, pxperline, offset),(im.shape[2],1)).T

    if disp:
        f, axes = plt.subplots(1,2)
        axes[0].imshow(im[frame], cmap='gray')
        axes[0].set_title('True image')
        axes[1].imshow(idealgrating, cmap='gray')
        axes[1].set_title('Ideal grating')
        plt.show()
    
    return idealsquare, idealgrating


def get_mtf(im, frame, positions, disp=False):
    idealsquare, idealgrating = fit_square_to_lineprof(im, frame, positions)

    ft_im, freqs_im = fourier(im[13])
    ft_ideal, freqs_ideal = fourier(idealgrating)

    mtf = 1/((np.abs(ft_ideal)[:,ft_ideal.shape[1]//2])/(np.abs(ft_im)[:,ft_im.shape[1]//2]))[1:]

    if disp:
        plt.plot(np.fft.fftshift(freqs_ideal[0]), (np.abs(ft_ideal)[:,ft_ideal.shape[1]//2]), label='Ideal grating')
        plt.plot(np.fft.fftshift(freqs_im[0]), (np.abs(ft_im)[:,ft_im.shape[1]//2]), '--', label='True image')
        plt.xlabel('f_y (/pixel)')
        plt.legend()
        plt.show()
        
        plt.plot(np.fft.fftshift(freqs_im[0])[1:], mtf)
        plt.title('Modulation transfer function')
        plt.show()

    return mtf