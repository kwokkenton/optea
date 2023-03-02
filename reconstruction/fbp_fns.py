# IMPORTS
from __future__ import division
from os import mkdir
from os.path import join, isdir
from imageio import get_writer
import astra
import numpy as np
from tifffile import imsave
from skimage import io
import matplotlib.pyplot as plt
from tqdm import tqdm, trange


# FUNCTION DEFINITIONS

def run_fbp(im, output_dir, max_val, save_recon=True, progbar=False):    
    # Turn into sinogram
    im = np.moveaxis(im, 0, -2).astype('float64')

    # Preallocate Memory
    reconstruction_fbp = np.zeros((im.shape[0], im.shape[2], im.shape[2]))
    # Infer parameters from inputs
    num_of_projections = im.shape[1]
    detector_rows = im.shape[0]  # Vertical size of detector [pixels].
    detector_cols = im.shape[-1]  # Horizontal size of detector [pixels].
    angles = np.linspace(0, 2 * np.pi, num=num_of_projections, endpoint=False)
    
    
    if progbar:
        itrbl = trange(im.shape[0], desc='Reconstructing slices: ')
    else:
        itrbl = range(im.shape[0])
    for i in itrbl:
        # Copy projection images into ASTRA Toolbox.
        proj_geom = astra.create_proj_geom('parallel', 1.0, detector_cols, angles)
        projections_id = astra.data2d.create('-sino', proj_geom, im[i])

        # Create reconstruction.
        vol_geom = astra.creators.create_vol_geom(detector_cols, detector_cols,
                                                detector_rows)
        reconstruction_id = astra.data2d.create('-vol', vol_geom, data=0)
        alg_cfg = astra.astra_dict('FBP_CUDA')
        alg_cfg['ProjectionDataId'] = projections_id
        alg_cfg['ReconstructionDataId'] = reconstruction_id
        algorithm_id = astra.algorithm.create(alg_cfg)
        astra.algorithm.run(algorithm_id)
        reconstruction = astra.data2d.get(reconstruction_id)

        # Limit and scale reconstruction.
        reconstruction[reconstruction < 0] = 0
        reconstruction /= max_val
        reconstruction = np.round(reconstruction * 65535).astype(np.uint16)

        # Save reconstruction.
        if save_recon: 
            if not isdir(output_dir):
                mkdir(output_dir)
            imsave(join(output_dir, 'reco%04d.tif' % i), reconstruction)
        
        # Save reconstruction to preallocated memory
        reconstruction_fbp[i] = reconstruction
        
        # Cleanup.
        astra.algorithm.delete(algorithm_id)
        astra.data2d.delete(reconstruction_id)
        astra.data2d.delete(projections_id)

    return reconstruction_fbp


def reconstruct(im, output_dir):
    reconstruction_fbp = run_fbp(im, output_dir, 65535, save_recon=False, progbar=True)
    max_val = np.max(reconstruction_fbp)
    print(f'Reconstructed, max={max_val}. Repeating to rescale.')
    reconstruction_fbp = run_fbp(im, output_dir, max_val, progbar=True)
    print(f'Saved reconstruction to {output_dir}.')
    return reconstruction_fbp


def align(im, bead_row):
    bead_im = im[:, bead_row-2:bead_row+3 , :] # get bead row +/- 2
    offsets = np.arange(1, 16)
    offset_list = []
    maxes = []
    for offset in offsets[::-1]:
        # negative crops
        offset_list.append(-offset)
        crop_im = bead_im[:,:,:-offset]
        reconstruction_fbp = run_fbp(crop_im, None, 65535, save_recon=False)
        maxes.append(np.max(reconstruction_fbp))
    # zero crop
    offset_list.append(0)
    reconstruction_fbp = run_fbp(crop_im, None, 65535, save_recon=False)
    maxes.append(np.max(reconstruction_fbp))
    for offset in offsets:
        # positive crops
        offset_list.append(offset)
        crop_im = bead_im[:,:,offset:]
        reconstruction_fbp = run_fbp(crop_im, None, 65535, save_recon=False)
        maxes.append(np.max(reconstruction_fbp))
    plt.plot(offset_list, maxes, '.')
    plt.xlabel('offset')
    plt.ylabel('maximum value')
    plt.show()
    optimal = offset_list[np.argmax(maxes)]
    print(f'Best offset: {optimal} pixels')
    if optimal==0:
        reconstruction_fbp = run_fbp(bead_im, None, 65535, save_recon=False)
        plt.imshow(reconstruction_fbp[len(reconstruction_fbp)//2])
        plt.title('Reconstructed bead')
        plt.show()
        return im
    elif optimal<0:
        crop_im = bead_im[:,:,:-optimal]
        recon_misaligned = run_fbp(bead_im, None, 65535, save_recon=False)
        recon_aligned = run_fbp(crop_im, None, 65535, save_recon=False)
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(recon_misaligned[len(recon_misaligned)//2])
        axs[0].set_title('Original bead recon')
        axs[1].imshow(recon_misaligned[len(recon_misaligned)//2])
        axs[1].set_title('Aligned bead recon')
        plt.show()
        return im[:,:,:-optimal]
    elif optimal>0:
        crop_im = bead_im[:,:,optimal:]
        recon_misaligned = run_fbp(bead_im, None, 65535, save_recon=False)
        recon_aligned = run_fbp(crop_im, None, 65535, save_recon=False)
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(np.sum(recon_misaligned, axis=0))
        axs[0].set_title('Original bead recon')
        axs[1].imshow(np.sum(recon_aligned, axis=0))
        axs[1].set_title('Aligned bead recon')
        plt.show()
        return im[:,:,optimal:]