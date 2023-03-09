import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_mask(arr, mask, titles):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    arr_copy = arr.copy()
    arr_copy[mask] = 1

    sns.heatmap(arr_copy, cmap=['black', 'r'], ax=axs[0])
    axs[0].set_aspect('equal')
    axs[0].set_title(f'{titles[0]} LEDs ({np.count_nonzero(arr_copy)})')
    posns_1 = np.transpose(np.where(arr_copy==1))

    arr_copy = arr.copy()
    arr_copy[~mask] = 1

    sns.heatmap(arr_copy, cmap=['black', 'r'], ax=axs[1])
    axs[1].set_aspect('equal')
    axs[1].set_title(f'{titles[1]} LEDs ({np.count_nonzero(arr_copy)})')
    posns_2 = np.transpose(np.where(arr_copy==1))

    plt.tight_layout()
    plt.show()

    return posns_1, posns_2


def plot_array(arr, titles):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    arr_copy = arr.copy()

    sns.heatmap(arr_copy, cmap=['black', 'r'], ax=axs[0])
    axs[0].set_aspect('equal')
    axs[0].set_title(f'{titles[0]} LEDs ({np.count_nonzero(arr_copy)})')
    posns_1 = np.transpose(np.where(arr_copy==1))

    arr_copy = np.invert(arr.astype(bool)).astype(int)

    sns.heatmap(arr_copy, cmap=['black', 'r'], ax=axs[1])
    axs[1].set_aspect('equal')
    axs[1].set_title(f'{titles[1]} LEDs ({np.count_nonzero(arr_copy)})')
    posns_2 = np.transpose(np.where(arr_copy==1))

    plt.tight_layout()
    plt.show()

    return posns_1, posns_2


def get_led_posns(array_width, circle_radius):
    """
    Given side length of LED array (`array_width`) and radius of circle
    required for darkfield/brightfield split (`circle_radius`), plot
    LED positions, and return positions in array of LEDs for brightfield, 
    darkfield, and top/bottom phase contrast imaging (`XXX_posns`).
    """

    x = np.arange(0, array_width)
    y = np.arange(0, array_width)
    arr = np.zeros((y.size, x.size))

    cx = array_width//2
    cy = array_width//2
    r = circle_radius

    circle_mask = (x[np.newaxis,:]-cx)**2 + (y[:,np.newaxis]-cy)**2 < r**2
    bright_posns, dark_posns = plot_mask(arr, circle_mask, ['Brightfield', 'Darkfield'])

    tb_phase_mask = x < x.size//2
    top_posns, bottom_posns = plot_mask(arr, tb_phase_mask, ['Top', 'Bottom'])
    
    lr_phase_array = np.zeros_like(arr)
    lr_phase_array[:, :x.size//2] = 1
    left_posns, right_posns = plot_array(lr_phase_array, ['Left', 'Right'])

    return bright_posns, dark_posns, top_posns, bottom_posns, left_posns, right_posns