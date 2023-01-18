# imports
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd
import gc  # garbage collection to manage ram usage
import cv2
from skimage import io
from skimage.transform import iradon
from scipy.signal import find_peaks
from mpl_toolkits.mplot3d import Axes3D
from celluloid import Camera
from IPython.display import HTML
from tqdm import tqdm

# general image handling
def add_position_labels(axes, slices, positions):
    """
    Given 2x2 subplot axes intended to display each quartile
    image from a stack, apply the appropriate titles
    from a list of positions of stack images in mm.
    """

    axes[0][0].set_title(f"{positions[0]} mm")
    axes[0][1].set_title(f"{positions[slices//4]} mm")
    axes[1][0].set_title(f"{positions[slices//2]} mm")
    axes[1][1].set_title(f"{positions[(slices*3)//4]} mm")

    return axes


def load_disp_img(
    path, positions, x_min=300, x_max=800, y_min=300, y_max=600, disp=False
):
    """
    Load and crop stack, optionally display quartile images
    """

    im = io.imread(path)[:, x_min:x_max, y_min:y_max]

    if disp:
        f, axes = plt.subplots(2, 2)
        slices = len(im)

        axes[0][0].imshow(im[0], cmap="gray")
        axes[0][1].imshow(im[slices // 4], cmap="gray")
        axes[1][0].imshow(im[slices // 2], cmap="gray")
        axes[1][1].imshow(im[(slices * 3) // 4], cmap="gray")

        axes = add_position_labels(axes, slices, positions)

        plt.tight_layout()
        plt.show()

    return im


def line_prof_splash(im, positions):
    """
    Plot vertical line profiles through centre,
    for quartile images in stack
    """

    f, axes = plt.subplots(2, 2, figsize=(15, 15))
    slices = len(im)
    centre = im.shape[2] // 2

    axes[0][0].plot(im[0, :, centre])
    axes[0][1].plot(im[slices // 4, :, centre])
    axes[1][0].plot(im[slices // 2, :, centre])
    axes[1][1].plot(im[(slices * 3) // 4, :, centre])

    axes = add_position_labels(axes, slices, positions)

    plt.show()


###
# dof
###


def fourier(frame, plot=False):
    """
    Return Fourier transform array and frequency axes, optionally plot 
    log(FFT) in 2D

    Adapted from:
    - https://thepythoncodingbook.com/2021/08/30/2d-fourier-transform-in\
        -python-and-fourier-synthesis-of-images/
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
            extent=(freq_x.min(), freq_x.max(), freq_y.min(), freq_y.max()),
            aspect=1,
        )
        plt.colorbar()
        plt.xlabel("f_x (/pixel)")
        plt.ylabel("f_y (/pixel)")
        plt.show()

    return ft, (freq_x, freq_y)


def fft_plot_centre_y(frame):
    ft, freqs = fourier(frame)
    plt.plot(np.fft.fftshift(freqs[0]), (np.abs(ft)[:, ft.shape[1] // 2]))
    plt.xlabel("f_y (/pixel)")
    plt.show()


def fft_plot_quartiles(im, positions):
    # TODO: are there better ways to represent this? or to quantify it?
    slices = len(im)
    qrts = [0, slices // 4, slices // 2, (slices * 3) // 4]

    ax = plt.figure().add_subplot(projection="3d")

    for q in qrts:
        ft, freqs = fourier(im[q])
        ax.plot(
            np.fft.fftshift(freqs[0]), (np.abs(ft)[:, ft.shape[1] // 2]), positions[q]
        )

    ax.set_xlabel("f_y (/pixel)")
    ax.set_zlabel("distance (mm)")
    ax.view_init(azim=20, elev=20, vertical_axis="y")
    plt.show()


def speed_run(path, positions):
    # quick and dirty comparison - run everything for each stack
    # this will vomit plots all over the notebook - be warned!
    im = load_disp_img(path, positions)
    line_prof_splash(im, positions)
    ft, freqs = fourier(im[len(im) // 2], plot=True)
    fft_plot_centre_y(im[len(im) // 2])
    fft_plot_quartiles(im, positions)


def fft_plot_quartiles_compare(paths, positions_lists, imgnames):
    fig = plt.figure()
    num = len(paths)

    for i in range(num):
        ax = fig.add_subplot(1, num, i + 1, projection="3d")

        im = load_disp_img(paths[i], positions_lists[i])
        slices = len(im)
        qrts = [0, slices // 4, slices // 2, (slices * 3) // 4]
        # qrts = range(slices)

        for q in qrts:
            ft, freqs = fourier(im[q])
            ax.plot(
                np.fft.fftshift(freqs[0]),
                (np.abs(ft)[:, ft.shape[1] // 2]),
                positions_lists[i][q],
            )

        ax.set_xlabel("f_y (/pixel)")
        ax.set_zlabel("distance (mm)")
        ax.set_title(imgnames[i])
        ax.view_init(azim=20, elev=20, vertical_axis="y")

    plt.tight_layout()
    plt.show()


def fft_find_peaks(frame, thresh=0.02e9, disp=False):
    ft, freqs = fourier(frame)
    peak_indices, properties = find_peaks(
        np.abs(ft)[:, ft.shape[1] // 2], height=thresh
    )

    if disp:
        plt.plot(np.fft.fftshift(freqs[0]), (np.abs(ft)[:, ft.shape[1] // 2]))
        plt.plot(
            np.fft.fftshift(freqs[0])[peak_indices], properties["peak_heights"], "x"
        )
        plt.xlabel("f_y (/pixel)")
        plt.show()

    return peak_indices, properties


def fft_find_peaks_anim(im, positions, thresh=0.02e9):
    fig = plt.figure(figsize=(15, 10))
    plt.xlabel("f_y (/pixel)")
    camera = Camera(fig)

    for i in range(len(im)):
        ft, freqs = fourier(im[i])
        peak_indices, properties = find_peaks(
            np.abs(ft)[:, ft.shape[1] // 2], height=thresh
        )
        line = plt.plot(np.fft.fftshift(freqs[0]), (np.abs(ft)[:, ft.shape[1] // 2]))
        plt.plot(
            np.fft.fftshift(freqs[0])[peak_indices], properties["peak_heights"], "x"
        )
        plt.legend(line, [f"{positions[i-1]} mm"])
        camera.snap()

    animation = camera.animate()
    return animation


def peak_distance_plots(paths, positions_lists, imgnames):
    fig = plt.figure(figsize=(15, 5))
    num = len(paths)

    for i in range(num):
        ax = fig.add_subplot(1, num, i + 1)
        im = load_disp_img(paths[i], positions_lists[i])
        peaks_lists, _ = map(
            list, zip(*[fft_find_peaks(im[i], thresh=0.05e9) for i in range(len(im))])
        )
        ax.plot(positions_lists[i], [max(peaks) for peaks in peaks_lists])
        ax.set_ylabel("distance between furthest peaks")
        ax.set_xlabel("depth (mm)")
        ax.set_title(imgnames[i])

    plt.tight_layout()
    plt.show()


def peak_height_plots(paths, positions_lists, imgnames):
    fig = plt.figure(figsize=(15, 5))
    num = len(paths)

    for i in range(num):
        ax = fig.add_subplot(1, num, i + 1)
        im = load_disp_img(paths[i], positions_lists[i])
        # need to set a very low threshold, else it breaks (as there is no second-highest peak)
        # TODO: use a more robust method to get the second-highest peak
        peaks_lists, properties_lists = map(
            list, zip(*[fft_find_peaks(im[i], thresh=0.001e9) for i in range(len(im))])
        )
        ax.plot(
            positions_lists[i],
            [sorted(properties["peak_heights"])[-2] for properties in properties_lists],
        )
        ax.set_ylabel("height of second-max peak")
        ax.set_xlabel("depth (mm)")
        ax.set_title(imgnames[i])

    plt.tight_layout()
    plt.show()


def fit_square_to_lineprof(im, frame, positions, init_f, disp=False):
    def rect(t, ampl, pxperline, offset):
        return ampl * sp.signal.square(t * (2 * np.pi) / pxperline * 2) / 2 + offset

    def sine(t, ampl, pxperline, offset):
        # fitting a sine wave works
        # (fitting square wave directly doesn't...)
        return ampl * np.sin(t * (2 * np.pi) / pxperline * 2) / 2 + offset

    centre = im.shape[2] // 2
    lineprof = im[frame, :, centre]
    length = len(lineprof)
    t = np.arange(0, length, 1)

    if disp:
        plt.plot(lineprof, "--")
        plt.title(f"{positions[frame]} mm")
        plt.xlabel("pixels")
        plt.ylabel("amplitude")

    popt, pcov = sp.optimize.curve_fit(sine, t, lineprof, p0=[40000, init_f, 25000])
    ampl, pxperline, offset = popt

    idealsquare = rect(t, ampl, pxperline, offset)

    if disp:
        plt.plot(t, idealsquare)
        print(f"Fitted: {pxperline:.3f} pixels per line")
        plt.show()

    idealgrating = np.tile(rect(t, ampl, pxperline, offset), (im.shape[2], 1)).T

    if disp:
        f, axes = plt.subplots(1, 2)
        axes[0].imshow(im[frame], cmap="gray")
        axes[0].set_title("True image")
        axes[1].imshow(idealgrating, cmap="gray")
        axes[1].set_title("Ideal grating")
        plt.show()

    return idealsquare, idealgrating


def get_mtf(im, frame, positions, disp=False):
    idealsquare, idealgrating = fit_square_to_lineprof(im, frame, positions)

    ft_im, freqs_im = fourier(im[frame])
    ft_ideal, freqs_ideal = fourier(idealgrating)

    mtf = (
        1
        / (
            (np.abs(ft_ideal)[:, ft_ideal.shape[1] // 2])
            / (np.abs(ft_im)[:, ft_im.shape[1] // 2])
        )[1:]
    )

    if disp:
        plt.plot(
            np.fft.fftshift(freqs_ideal[0]),
            (np.abs(ft_ideal)[:, ft_ideal.shape[1] // 2]),
            label="Ideal grating",
        )
        plt.plot(
            np.fft.fftshift(freqs_im[0]),
            (np.abs(ft_im)[:, ft_im.shape[1] // 2]),
            "--",
            label="True image",
        )
        plt.xlabel("f_y (/pixel)")
        plt.legend()
        plt.show()

        plt.plot(np.fft.fftshift(freqs_im[0])[1:], mtf)
        plt.title("Modulation transfer function")
        plt.show()

    return mtf


def get_mtf_peakdivide(
    im, frame, positions, init_f, thresh_true=0, thresh_ideal=0.25e9
):

    idealsquare, idealgrating = fit_square_to_lineprof(
        im, frame, positions, init_f, disp=True
    )  # fit in-focus

    ft_im, freqs_im = fourier(im[frame])
    ft_ideal, freqs_ideal = fourier(idealgrating)

    plt.figure(figsize=(4, 3))
    peaks_im, props_im = fft_find_peaks(im[frame], thresh=thresh_true, disp=True)
    plt.figure(figsize=(4, 3))
    peaks_ideal, props_ideal = fft_find_peaks(
        idealgrating, thresh=thresh_ideal, disp=True
    )

    d = {
        "image_indices": peaks_im,
        "image_heights": props_im["peak_heights"],
        "ideal_indices": peaks_ideal,
        "ideal_heights": props_ideal["peak_heights"],
    }

    df = pd.DataFrame(data=d)

    image_indices = df["image_indices"].tolist()
    df["ideal_heights_at_image_indices"] = np.abs(ft_ideal)[:, ft_im.shape[1] // 2][
        image_indices
    ]
    ideal_indices = df["ideal_indices"].tolist()
    df["image_heights_at_ideal_indices"] = np.abs(ft_im)[:, ft_im.shape[1] // 2][
        ideal_indices
    ]

    plt.plot(
        np.fft.fftshift(freqs_im[0])[image_indices],
        df["image_heights"] / df["ideal_heights"],
        "-x",
    )
    plt.xlabel("f_y (/pixel)")
    plt.title("MTF from dividing peak heights")
    plt.show()

    return


###
# autoalign
###


def recolour_image(before, bgr=[1.5, 0.75, 1.25]):
    """
    Recolour greyscale image `before` to new colour `bgr`
    in blue-green-red format

    Adapted from: https://stackoverflow.com/a/58142700
    """

    before = cv2.cvtColor(before, cv2.COLOR_GRAY2BGR)
    b, g, r = cv2.split(before)

    np.multiply(b, bgr[0], out=b, casting="unsafe")
    np.multiply(g, bgr[1], out=g, casting="unsafe")
    np.multiply(r, bgr[2], out=r, casting="unsafe")

    after = cv2.merge([b, g, r])

    return after


def split_images(im):
    """
    Return first and mid-point images from stack
    """

    return im[0], im[int(len(im) / 2)]


def overlay_images(img1, img2, translation):
    """
    Recolour two greyscale images to red and green,
    flip second image,
    overlay with 50% transparency
    """

    M = np.float32([[1, 0, translation], [0, 1, 0]])

    img_0 = recolour_image(img1, bgr=[0, 0, 255])
    img_180 = np.fliplr(recolour_image(img2, bgr=[0, 255, 0]))
    img_180_shift = cv2.warpAffine(img_180, M, (img_180.shape[1], img_180.shape[0]))
    blended = cv2.addWeighted(img_0, 0.5, img_180_shift, 0.5, 0.0)
    blended = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)

    return blended


def subtract_images(img1, img2, translation):
    """
    Translate second image relative to first, and subtract
    """

    M = np.float32([[1, 0, translation], [0, 1, 0]])

    img2 = np.fliplr(img2)
    img_180_shift = cv2.warpAffine(img2, M, (img2.shape[1], img2.shape[0]))
    subtracted = cv2.subtract(img1, img_180_shift)

    return subtracted


def misalignment(img1, img2, offset):
    """
    Return sum of intensity in central half of difference
    between two images, as measure of misalignment
    """

    x = subtract_images(img1, img2, offset)
    width = np.shape(x)[1]
    intensity = np.concatenate(x[:, width // 4 : width * 3 // 4]).sum()

    return intensity


def align_images(img1, img2):
    """
    Slide two images past each other and find offset which minimises misalignment
    """

    width = img1.shape[1]
    frac = 4  # decreasing factor tries larger range of offsets
    offsets = np.arange(
        -width // frac, width // frac
    )  # working with integer offsets only
    misalignments = [misalignment(img1, img2, i) for i in tqdm(offsets)]
    stack = np.column_stack((offsets, misalignments))
    optimal = stack[np.argmin(stack[:, 1]), 0]

    return optimal


def plot_alignment(im):
    """
    Split image, align front and back projections, and plot
    """

    img1, img2 = split_images(im)
    optimal = align_images(img1, img2)
    f, axes = plt.subplots(1, 2)
    axes[0].imshow(overlay_images(img1, img2, 0).astype(np.uint8))  # rescale to [0,255]
    axes[0].set_title("Original")
    axes[1].imshow(overlay_images(img1, img2, optimal).astype(np.uint8))
    axes[1].set_title(f"Aligned (offset {int(optimal)} px)")
    plt.show()


def apply_offset(im):
    """
    Cut `offset` pixels from side of all images in z-stack
    to recentre axis of rotation
    """

    img1, img2 = split_images(im)
    optimal = align_images(img1, img2)

    M = np.float32([[1, 0, -optimal / 2], [0, 1, 0]])

    corrected = np.array(
        [cv2.warpAffine(slce, M, (slce.shape[1], slce.shape[0])) for slce in tqdm(im)]
    )

    return corrected


def recon(im):
    """
    Reconstruct tomographic image using inverse radon transform
    """
    theta = np.linspace(0, 360, np.shape(im)[0], endpoint=False)
    sinogram = np.moveaxis(im, 0, -1)

    reconstruction_fbp = np.zeros((im.shape[1], im.shape[2], im.shape[2]))
    for i in tqdm(range(im.shape[1])):
        reconstruction_fbp[i] = iradon(sinogram[i], theta=theta, filter_name="ramp")

    return reconstruction_fbp


def recon_part(im, frac=0.5):
    """
    Reconstruct part of tomographic image using inverse radon transform
    """
    theta = np.linspace(0, 360, np.shape(im)[0], endpoint=False)
    sinogram = np.moveaxis(im, 0, -1)

    reconstruction_fbp = np.zeros((int(im.shape[1] * frac), im.shape[2], im.shape[2]))
    for i in tqdm(range(int(im.shape[1] * frac))):
        reconstruction_fbp[i] = iradon(sinogram[i], theta=theta, filter_name="ramp")

    return reconstruction_fbp


def disp_slices(reconstructed):
    """
    Display 4 slices through reconstructed image
    """

    f, axes = plt.subplots(2, 2)
    slices = len(reconstructed)
    axes[0][0].imshow(reconstructed[0], cmap="gray")
    axes[0][1].imshow(reconstructed[slices // 4], cmap="gray")
    axes[1][0].imshow(reconstructed[slices // 2], cmap="gray")
    axes[1][1].imshow(reconstructed[(slices * 3) // 4], cmap="gray")
    plt.show()


###
# averaging
###


def load_stack(path, x_min=400, x_max=700, y_min=300, y_max=600, show=False):
    imgs = io.imread(path + "/MMStack_Pos0.ome.tif")[:, x_min:x_max, y_min:y_max]
    if show:
        plt.imshow(imgs[0])
        plt.title("First image")
        plt.show()
    print(f"Loaded stack of {imgs.shape[0]} images, dimensions {imgs.shape[1:]}")
    return imgs


def average_stack(imgs, im_per_pos=5, show=False):
    # adapted from: https://stackoverflow.com/a/69721142
    image_sets = imgs.reshape(
        (len(imgs) // im_per_pos, im_per_pos, imgs.shape[1], imgs.shape[2])
    )
    avg_stack = np.mean(image_sets, axis=1)
    if show:
        plt.imshow(avg_stack[0])
        plt.title("First image")
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
        axes[0][0].imshow(imgs[0])
        axes[0][0].set_title("First image, imgs")
        axes[0][1].imshow(imgs_minus_bg[0])
        axes[0][1].set_title("First image, imgs_minus_bg")
        axes[1][0].imshow(light_minus_bg[0])
        axes[1][0].set_title("First image, light_minus_bg")
        axes[1][1].imshow(divided[0])
        axes[1][1].set_title("First image, divided")
        plt.tight_layout()
        plt.show()
    print(f"Removed background and divided illumination from {len(divided)} images.")
    return divided
