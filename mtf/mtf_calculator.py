# Copyright (c) 2012 The Chromium OS Authors. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.

# Adapted by Srayan Gangopadhyay, 2023-02-22


import cv2
import math
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import itertools
import pickle


def _ExtractPatch(sample, edge_start, edge_end, desired_width, crop_ratio):
    """Crops a patch from the test target."""
    # Identify the edge direction.
    vec = edge_end - edge_start
    # Discard both ends so that we MIGHT not include other edges
    # in the resulting patch.
    # TODO: Auto-detect if the patch covers more than one edge!
    safe_start = (1 - crop_ratio) * edge_start + crop_ratio * edge_end
    safe_end = crop_ratio * edge_start + (1 - crop_ratio) * edge_end
    minx = int(round(min(safe_start[0], safe_end[0])))
    miny = int(round(min(safe_start[1], safe_end[1])))
    maxx = int(round(max(safe_start[0], safe_end[0])))
    maxy = int(round(max(safe_start[1], safe_end[1])))
    if abs(vec[0]) > abs(vec[1]):  # near-horizontal edge
        ylb = max(0, miny - desired_width)
        yub = min(sample.shape[0], maxy + desired_width + 1)
        patch = np.transpose(sample[ylb:yub, minx : (maxx + 1)])
    else:  # near-vertical edge
        xlb = max(0, minx - desired_width)
        xub = min(sample.shape[1], maxx + desired_width + 1)
        patch = sample[miny : (maxy + 1), xlb:xub]
    # Make sure white is on the left.
    if patch[0, 0] < patch[-1, -1]:
        patch = np.fliplr(patch)
    # Make a floating point copy.
    patch = np.asfarray(patch)
    # TODO(sheckylin) Correct for vignetting.
    return patch


def _FindEdgeSubPix(patch, desired_width):
    """Locates the edge position for each scanline with subpixel precision."""
    ph = patch.shape[0]
    pw = patch.shape[1]
    # Get the gradient magnitude along the x direction.
    k_gauss = np.transpose(cv2.getGaussianKernel(7, 1.0))
    temp = cv2.filter2D(patch, -1, k_gauss, borderType=cv2.BORDER_REFLECT)
    k_diff = np.array([[-1, 1.0]])
    grad = abs(cv2.filter2D(temp, -1, k_diff, borderType=cv2.BORDER_REPLICATE))
    # Estimate subpixel edge position for each scanline.
    ys = np.arange(ph, dtype=np.float64)
    xs = np.empty(ph, dtype=np.float64)
    x_dummy = np.arange(pw, dtype=np.float64)
    for y in range(ph):
        # 1st iteration.
        b = np.sum(x_dummy * grad[y])
        a = np.sum(grad[y])
        c = int(round(b / a))
        # 2nd iteration due to bias of different num of black and white pixels.
        dw = min(min(c, desired_width), pw - c - 1)
        b = np.sum(x_dummy[(c - dw) : (c + dw + 1)] * grad[y, (c - dw) : (c + dw + 1)])
        a = np.sum(grad[y, (c - dw) : (c + dw + 1)])
        xs[y] = int(round(b / a))
    # Fit a second-order polyline for subpixel accuracy.
    fitted_line = np.polyfit(ys, xs, 1)
    fitted_parabola = np.polyfit(ys, xs, 2)
    angle = math.atan(fitted_line[0])
    pb = np.poly1d(fitted_parabola)
    centers = [pb(y) for y in range(ph)]
    return angle, centers


def _AccumulateLine(patch, centers):
    """Adds up the scanlines along the edge direction."""
    ph = patch.shape[0]
    pw = patch.shape[1]
    # Determine the final line length.
    w = min(int(round(np.min(centers))), pw - int(round(np.max(centers))) - 1)
    w4 = 2 * w + 1
    # Accumulate a 4x-oversampled line.
    psf4x = np.zeros((4, w4), dtype=np.float64)
    counts = np.zeros(4, dtype=np.float64)
    for y in range(ph):
        ci = int(round(centers[y]))
        idx = 3 + 4 * ci - int(4 * centers[y] + 2)
        psf4x[idx] += patch[y, (ci - w) : (ci + w + 1)]
        counts[idx] += 1
    counts = np.expand_dims(counts, axis=1)
    psf4x /= counts
    psf = np.diff(psf4x.transpose().flatten())
    return psf


def _GetResponse(psf, angle):
    """Composes the MTF curve."""
    w = psf.shape[0]
    # Compute FFT.
    magnitude = abs(np.fft.fft(psf))
    # Smooth the result a little bit.
    # This is equivalent to applying window in the spatial domain
    # as enforced in the Imatest's algorithm.
    k_gauss = np.transpose(cv2.getGaussianKernel(7, 1.0))
    cv2.filter2D(magnitude, -1, k_gauss, magnitude, borderType=cv2.BORDER_REFLECT)
    # Slant correction factor.
    slant_correction = math.cos(angle)
    # Compose MTF curve.
    # Normalize the low frequency response to 1 and compensate for the
    # finite difference.
    rw = int(w / 4 + 1)
    magnitude = magnitude[0:rw] / magnitude[0]
    freqs = np.arange(rw, dtype=np.float64) * 4 / w / slant_correction
    attns = magnitude / (np.sinc(np.arange(rw, dtype=np.float64) / w) ** 2)
    return freqs, attns


def _FindMTF50P(freqs, attns, use_50p):
    """Locates the MTF50P given the MTF curve."""
    peak50 = (attns.max() if use_50p else 1.0) / 2.0
    idx = np.nonzero(attns < peak50)[0]
    if idx.shape[0] == 0:
        return freqs[-1]
    idx = idx[0]
    # Linear interpolation.
    ratio = (peak50 - attns[idx - 1]) / (attns[idx] - attns[idx - 1])
    return freqs[idx - 1] + (freqs[idx] - freqs[idx - 1]) * ratio


def Compute(sample, edge_start, edge_end, desired_width, crop_ratio, use_50p=True):
    """Computes the MTF50P value of an edge.
    This function implements the slanted-edge MTF calculation method as used in
    the Imatest software. For more information, please visit
    http://www.imatest.com/docs/sharpness/ . The function in default setting
    computes the so-called MTF50P value instead of the traditional MTF50 in
    order to better cope with the in-camera post-sharpening. The result quality
    improves with longer edges and wider margin widths.
    Args:
      sample: The test target image. It needs to be single-channel.
      edge_start: The (rough) start point of the edge.
      edge_end: The (rough) end point of the edge.
      desired_width: Desired margin width on both sides of the edge.
      crop_ratio: The truncation ratio at the both ends of the edge.
      use_50p: Compute whether the MTF50P value or the MTF50 value.
    Returns:
      1: The MTF50P value.
      2, 3: The MTF curve, represented by lists of frequencies and
          attenuation values.
    """
    patch = _ExtractPatch(sample, edge_start, edge_end, desired_width, crop_ratio)
    angle, centers = _FindEdgeSubPix(patch, desired_width)
    # print(f'Detected edge angle: {np.rad2deg(angle)}')
    psf = _AccumulateLine(patch, centers)
    freqs, attns = _GetResponse(psf, angle)
    return _FindMTF50P(freqs, attns, use_50p), freqs, attns


#################
# OUR FUNCTIONS #
#################


def load_stack(path, x_min=400, x_max=700, y_min=300, y_max=600, show=False):
    imgs = io.imread(path + "/MMStack_Pos0.ome.tif")[:, x_min:x_max, y_min:y_max]
    if show:
        plt.imshow(imgs[0], cmap="gray")
        plt.title(f"Image 1 of {len(imgs)}")
        plt.show()
    print(f"Loaded stack of {imgs.shape[0]} images, dimensions {imgs.shape[1:]}")
    return imgs


def average_stack(imgs, im_per_pos=5, show=False):
    # adapted from: https://stackoverflow.com/a/6972selection[142
    image_sets = imgs.reshape(
        (len(imgs) // im_per_pos, im_per_pos, imgs.shape[1], imgs.shape[2])
    )
    avg_stack = np.mean(image_sets, axis=1)
    if show:
        plt.imshow(avg_stack[0], cmap="gray")
        plt.title(f"Image 1 of {len(avg_stack)}")
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


def load_and_divide(img_dir, bg_dir, img_name, bg_light_name, bg_dark_name, crop):
    img_path = str(img_dir + "/" + img_name)
    imgs = load_stack(
        img_path, x_min=crop[0], x_max=crop[1], y_min=crop[2], y_max=crop[3], show=False
    )

    avg_stack = average_stack(imgs)

    bg_light_path = bg_dir + "/" + bg_light_name
    bg_dark_path = bg_dir + "/" + bg_dark_name
    bg_light = average_stack(
        load_stack(
            bg_light_path, x_min=crop[0], x_max=crop[1], y_min=crop[2], y_max=crop[3]
        )
    )
    bg_dark = average_stack(
        load_stack(
            bg_dark_path, x_min=crop[0], x_max=crop[1], y_min=crop[2], y_max=crop[3]
        )
    )

    divided = remove_background(avg_stack, bg_light, bg_dark)
    return divided


def load_no_divide(img_dir, img_name, crop):
    img_path = str(img_dir + "/" + img_name)
    imgs = load_stack(
        img_path, x_min=crop[0], x_max=crop[1], y_min=crop[2], y_max=crop[3], show=False
    )

    avg_stack = average_stack(imgs)

    return avg_stack


def allComputeMTF(
    edge, position, edge_start, edge_end, desired_width, crop_ratio, plot=False
):
    _, freqs, attns = Compute(
        edge, np.array(edge_start), np.array(edge_end), desired_width, crop_ratio
    )

    if plot:
        plt.plot(freqs / 6.45e-3, attns, "-.", label=f"{position} mm")
        plt.xlim(0, 0.5 / 6.45e-3)
        plt.legend()
        plt.xlabel("line pairs / mm")
        plt.ylabel("MTF")

    return freqs / 6.45e-3, attns


def gen_MTF_plots(
    edge,
    f_number,
    depths,
    selection,
    edge_start,
    edge_end,
    desired_width,
    crop_ratio,
    save_plots=False,
    save_mtfs = False
):
    # plot the extracted patch as visual check
    plt.figure(figsize=(1, 1))
    plt.imshow(
        _ExtractPatch(
            edge[0], np.array(edge_start), np.array(edge_end), desired_width, crop_ratio
        ), cmap='gray'
    )
    plt.title("Extracted patch")
    plt.show()

    # generate the plots
    positions = []
    frequencies = []
    mtfs = []

    focus = depths[len(depths) // 2]

    # COMPUTING MTFS
    for i in np.arange(0, len(edge), 1):
        freqs, mtf = allComputeMTF(
            edge[i],
            depths[i],
            edge_start,
            edge_end,
            desired_width,
            crop_ratio,
            plot=True,
        )
        positions.append(depths[i])
        frequencies.append(freqs)
        mtfs.append(mtf)

    # ALL MTF PLOT

    plt.title(f"all mtfs, f/{f_number}")
    if save_plots:
        plt.savefig(f"plots/all-mtfs-f{f_number}.png")
    plt.show()

    # SELECTION OF MTFS
    plt.plot(
        frequencies[selection[0]],
        mtfs[selection[0]],
        "g-",
        label=f"{np.abs(positions[selection[0]]-focus)}",
    )
    plt.plot(
        frequencies[selection[1]],
        mtfs[selection[1]],
        "g--",
        label=f"{np.abs(positions[selection[1]]-focus)}",
    )
    plt.plot(
        frequencies[selection[2]],
        mtfs[selection[2]],
        "g:",
        label=f"{np.abs(positions[selection[2]]-focus)}",
    )

    plt.xlim(0, 0.5 / 6.45e-3)
    plt.legend(title="Defocus (mm)")
    plt.xlabel("line pairs per mm")
    plt.ylabel("MTF")
    plt.title(f"MTF at f/{f_number}")
    if save_plots:
        plt.tight_layout()
        plt.savefig(f"plots/selection-mtfs-f{f_number}.png", dpi=1000)
    plt.show()

    # JAMES-STYLE PLOT
    mtf_len = min([len(mtf) for mtf in mtfs])
    mtfs_cropped = [mtf[:mtf_len] for mtf in mtfs]
    all_mtfs = list(itertools.chain.from_iterable(mtfs_cropped))
    MTF = np.array(all_mtfs).reshape(len(depths), mtf_len)

    index_20lp = 61
    mtf_20lp = MTF[:, index_20lp]
    min_pos = np.min((np.array(positions) - focus)[mtf_20lp >= 0.2])
    max_pos = np.max((np.array(positions) - focus)[mtf_20lp >= 0.2])

    plt.contourf(
        frequencies[0][:mtf_len],
        np.array(positions) - focus,
        MTF,
        100,
        cmap=plt.cm.get_cmap("Greens", 10),
    )
    plt.plot(
        [frequencies[0][index_20lp], frequencies[0][index_20lp]],
        [min_pos, max_pos],
        "-w",
        linewidth="5",
    )

    plt.xlim(0, 0.5 / 6.45e-3)
    plt.xlabel("line pairs per mm")
    plt.ylabel("Defocus (mm)")
    plt.title(f"MTF versus defocus at f/{f_number}")
    plt.colorbar(ticks=list(np.linspace(0, 1, 11)))
    if save_plots:
        plt.tight_layout()
        plt.savefig(f"plots/contour-f{f_number}.png", dpi=1000)
    plt.show()

    print(f"Depth of field = {max_pos - min_pos}")

    mtf_tuple = (frequencies[0][:mtf_len], np.array(positions) - focus, MTF)

    if save_mtfs:
        with open(f'mtfs/f{f_number}_mtf.pickle', 'wb') as f:
            pickle.dump(mtf_tuple, f)

    # return mtf_tuple