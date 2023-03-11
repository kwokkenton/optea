import numpy as np
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, rescale, iradon
from skimage import io
import astra
import scipy.io


def compare_recons(phantom):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    mpbl = axs[0].imshow(phantom, cmap='gray')
    plt.colorbar(mpbl)
    axs[0].set_title('Input image')

    # generate sinogram with ASTRA
    vol_geom = astra.create_vol_geom(256, 256)
    proj_geom = astra.create_proj_geom('parallel', 1.0, 256, np.linspace(0,np.pi,180,False))
    proj_id = astra.create_projector('cuda', proj_geom, vol_geom)
    sinogram_id, sinogram_astra = astra.create_sino(phantom, proj_id)

    mpbl = axs[1].imshow(sinogram_astra.T, cmap='gray')
    plt.colorbar(mpbl)
    axs[1].set_title('ASTRA sinogram')

    # generate sinogram with skimage
    theta = np.linspace(0., 180., max(phantom.shape), endpoint=False)
    sinogram_skimage = radon(phantom, theta=theta)
    dx, dy = 0.5 * 180.0 / max(phantom.shape), 0.5 / sinogram_skimage.shape[0]

    mpbl = axs[2].imshow(sinogram_skimage, cmap='gray',
               extent=(-dx, 180.0 + dx, -dy, sinogram_skimage.shape[0] + dy),
               aspect='auto')
    plt.colorbar(mpbl)
    axs[2].set_title('skimage sinogram')

    plt.show()

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    rec_id = astra.data2d.create('-vol', vol_geom)
    cfg = astra.astra_dict('FBP_CUDA')
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = sinogram_id
    cfg['option'] = { 'FilterType': 'Ram-Lak' }
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)
    recon_astra = astra.data2d.get(rec_id)

    mpbl = axs[0].imshow(recon_astra, cmap='gray')
    plt.colorbar(mpbl)
    axs[0].set_title('ASTRA recon')

    recon_skimage = iradon(sinogram_skimage, theta=theta, filter_name='ramp')

    mpbl = axs[1].imshow(recon_skimage, cmap='gray')
    plt.colorbar(mpbl)
    axs[1].set_title('skimage recon')

    plt.show()


def compare_recons_2(stack, row):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    mpbl = axs[0].imshow(stack[0], cmap='gray')
    plt.colorbar(mpbl)
    axs[0].axhline(y=row, color = 'red', linestyle = '--')
    axs[0].set_title('Input stack (first projection)')
    
    # reshape into sinogram (need to transpose for ASTRA)
    sinogram = np.moveaxis(stack, 0, -1)[row]
    mpbl = axs[1].imshow(sinogram, cmap='gray')
    plt.colorbar(mpbl)
    axs[1].set_title(f'Sinogram (row {row})')
    
    # ASTRA setup
    im_len = stack.shape[2]
    vol_geom = astra.create_vol_geom(im_len, im_len)
    proj_geom = astra.create_proj_geom('parallel', 1.0, im_len, np.linspace(0,2*np.pi,stack.shape[0],False))
    proj_id = astra.create_projector('cuda', proj_geom, vol_geom)
    projections_id = astra.data2d.create('-sino', proj_geom, sinogram.T)
    rec_id = astra.data2d.create('-vol', vol_geom)
    cfg = astra.astra_dict('FBP_CUDA')
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = projections_id
    cfg['option'] = { 'FilterType': 'Ram-Lak' }
    alg_id = astra.algorithm.create(cfg)

    # skimage setup
    theta = np.linspace(0., 360., 400, endpoint=False)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    
    # astra reconstruction
    astra.algorithm.run(alg_id)
    recon_astra = astra.data2d.get(rec_id)

    mpbl = axs[0].imshow(recon_astra, cmap='gray')
    plt.colorbar(mpbl)
    axs[0].set_title('ASTRA recon')

    # skimage reconstruction
    recon_skimage = iradon(sinogram, theta=theta, filter_name='ramp')

    mpbl = axs[1].imshow(recon_skimage, cmap='gray')
    plt.colorbar(mpbl)
    axs[1].set_title('skimage recon')

    plt.show()
    
    # rescaling (original method)
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    
    astra_max = np.max(recon_astra)
    recon_astra[recon_astra < 0] = 0
    recon_astra /= astra_max
    recon_astra = np.round(recon_astra * 65535).astype(np.uint16)
    
    mpbl = axs[0].imshow(recon_astra, cmap='gray')
    plt.colorbar(mpbl)
    axs[0].set_title('ASTRA recon (rescaled)')
    
    skimage_max = np.max(recon_skimage)
    recon_skimage[recon_skimage < 0] = 0
    recon_skimage /= skimage_max
    recon_skimage = np.round(recon_skimage * 65535).astype(np.uint16)
    
    mpbl = axs[1].imshow(recon_skimage, cmap='gray')
    plt.colorbar(mpbl)
    axs[1].set_title('skimage recon (rescaled)')
    
    plt.show()
    
    # rescaling (new method)
    astra.algorithm.run(alg_id)
    recon_astra = astra.data2d.get(rec_id)
    recon_skimage = iradon(sinogram, theta=theta, filter_name='ramp')
    
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    
    astra_max = np.max(recon_astra)
    astra_min = np.min(recon_astra)
    recon_astra = (recon_astra - astra_min) / (astra_max - astra_min) * 65535
    
    skimage_max = np.max(recon_skimage)
    skimage_min = np.min(recon_skimage)
    recon_skimage = (recon_skimage - skimage_min) / (skimage_max - skimage_min) * 65535
    
    mpbl = axs[0].imshow(recon_astra, cmap='gray')
    plt.colorbar(mpbl)
    axs[0].set_title('ASTRA recon (rescaled - fixed)')
    
    
    mpbl = axs[1].imshow(recon_skimage, cmap='gray')
    plt.colorbar(mpbl)
    axs[1].set_title('skimage recon (rescaled - fixed)')
    
    plt.show()
    