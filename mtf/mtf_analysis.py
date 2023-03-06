import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import cv2
from scipy.optimize import curve_fit

# gaussian function and initial guess borrowed from Kenton's code
def gaussian(x, sigma):
    return np.exp(-x**2/(2*sigma**2))

def exact_otf(f, alpha, wvln=562e-9, n=1.33):
    # taken from Stokseth 1969 paper
    s = (wvln*f)/(n*np.sin(alpha))
    beta = np.arccos(s/2)
    return (1/np.pi)*((2*beta) - np.sin(2*beta))

# TODO: add in jinc function for defocus mtf

def mtf_ft_plot(freqs, MTF, in_focus, f_number):
    t_freqs = freqs[: len(freqs) // 2]
    t_freqs_2 = np.linspace(0, t_freqs[-1], 1000)
    t_MTF = MTF[in_focus][: len(MTF[in_focus]) // 2]

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].plot(t_freqs, t_MTF, label='data')
    ax[0].set_title(f"in-focus MTF at f/{f_number}")
    ax[0].set_xlabel("line pairs per millimetre")
    
    popt_g, pcov_g = curve_fit(gaussian, t_freqs, t_MTF, p0=[23])
    ax[0].plot(t_freqs, gaussian(t_freqs, *popt_g), 'r-', label=f'Gaussian, sigma={popt_g[0]:.2e}')
    popt_e, pcov_e = curve_fit(exact_otf, t_freqs, t_MTF, p0=[0.2])
    ax[0].plot(t_freqs, exact_otf(t_freqs, *popt_e), 'g-', label=f'"Exact OTF", alpha={popt_e[0]:.2e}')
    ax[0].legend()

    N = len(t_freqs)
    T = np.mean(np.diff(t_freqs))

    yf_data = fft(t_MTF)
    yf_gauss = fft(gaussian(t_freqs_2, *popt_g))
    yf_exact = fft(exact_otf(t_freqs_2, *popt_e))
    xf = fftfreq(N, T)
    xf_2 = fftfreq(len(t_freqs_2))
    ax[1].plot(xf * 1000, 2.0 / N * np.abs(yf_data), "bx", label='FT{data}')
    ax[1].plot(xf_2 * 1000, 2.0 / N * np.abs(yf_gauss), "r.", label='FT{Gaussian}')
    ax[1].plot(xf_2 * 1000, 2.0 / N * np.abs(yf_exact), "g.", label='FT{"exact"}')
    ax[1].legend()
    ax[1].set_title("FT(MTF)")
    ax[1].set_xlim(-50, 50)
    ax[1].set_xlabel("distance (microns)")
    plt.show()


def mtf_contour(freqs, pos, MTF, f_number, fixticks=True):
    plt.contourf(freqs, pos, MTF, 100, cmap=plt.cm.get_cmap("Greens", 10))
    plt.xlim(0, 0.5 / 6.45e-3)
    plt.xlabel("line pairs per mm")
    plt.ylabel("Defocus (mm)")
    plt.title(f"MTF versus defocus at f/{f_number}")
    plt.colorbar(ticks=list(np.linspace(0, 1, 11)))
    plt.show()


def focal_scan_mtf(freqs, pos, MTF, f_number, shifts):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    mappable = ax[0].contourf(freqs, pos, MTF, 100, cmap="Greens")
    ax[0].set_xlim(0, 0.5 / 6.45e-3)
    ax[0].set_xlabel("line pairs per mm")
    ax[0].set_ylabel("Defocus (mm)")
    ax[0].set_title(f"MTF at f/{f_number}")
    plt.colorbar(mappable, ax=ax[0])

    mtf_sum = np.zeros_like(MTF)
    for shift in shifts:
        M = np.float32([[1, 0, 0], [0, 1, shift]])
        mtf_sum += cv2.warpAffine(MTF, M, (MTF.shape[1], MTF.shape[0]))
    mtf_sum *= 1.0 / mtf_sum.max()

    mappable = ax[1].contourf(freqs, pos, mtf_sum, 100, cmap="Greens")
    ax[1].set_xlim(0, 0.5 / 6.45e-3)
    ax[1].set_xlabel("line pairs per mm")
    ax[1].set_ylabel("Defocus (mm)")
    ax[1].set_title(f"Summed MTF at f/{f_number}")
    plt.colorbar(mappable, ax=ax[1])

    plt.show()


def all_analysis(f_number, in_focus, shifts):
    with open(f"mtfs/f{f_number}_mtf.pickle", "rb") as f:
        freqs, pos, MTF = pickle.load(f)
        mtf_contour(freqs, pos, MTF, f_number)
        mtf_ft_plot(freqs, MTF, in_focus, f_number)
        focal_scan_mtf(freqs, pos, MTF, f_number, shifts)
        return freqs, pos, MTF
