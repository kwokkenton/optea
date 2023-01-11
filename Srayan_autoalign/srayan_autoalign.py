import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from skimage.transform import iradon
import gc  # garbage collection to manage ram usage


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
