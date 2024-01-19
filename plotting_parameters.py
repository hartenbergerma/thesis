import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import spectral as sp
import os
import sys
from preprocessing import get_array, smooth_spectral


tum_blue_brand ='#3070b3'
tum_blue_dark = '#072140' #
tum_blue_dark_2 = '#0e396e' #
tum_blue_light = '#5e94d4' #
tum_blue_light_2 = '#c2d7ef' #
tum_orange = '#f7b11e' #
tum_grey_1 = '#20252a'
tum_grey_3 = '#475058'
tum_grey_5 = '#abb5be'
tum_grey_7 = '#dde2e6'
tum_red = '#ea7237' #
tum_red_dark = '#d95117'
tum_green = '#9fba36'
tum_green_dark = '#7d922a'
tum_pink = '#b55ca5' #
tum_yellow = '#fed702' #

tum_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'tum', [(0,tum_blue_dark), (0.2,tum_blue_dark_2), (0.4,tum_pink), (0.7,tum_red), (0.8,tum_orange), (1,tum_yellow)])


def set_plotting_style(style):
    if style == "latex":
        mpl.rcParams.update(
            {
                "font.family": "serif",
                "font.size": 8,
                "text.usetex": True,
                "axes.unicode_minus": False,
            }
        )
    if style =="default":
        plt.rcdefaults()


def plot_ref_full():
    '''
    Creates a figure with the size of a full column in a latex document.
    '''
    fig, ax = plt.subplots(figsize=(7.2,1))
    ax.axis('off')
    plt.show()

def plot_ref_half():
    '''
    Creates a figure with the size of a half column in a latex document.
    '''
    fig, ax = plt.subplots(figsize=(3.4,1))
    ax.axis('off')
    plt.show()


def plot_spectrum(spectr, wavelengths, fig, ax, legend=False, nspectr=None):
    '''
    Plot the spectra of the pixels in spectr. The number of spectra to plot can be specified with nspectr.
    If nspectr is None, all spectra are plotted, otherwise the spectra are evenly spaced.
    input:
        spectr: np.array of shape (N,k) where k is the number of bands and N is the number of spectra
        wavelengths: np.array of shape (k,) containing the wavelengths of the bands
        fig: figure to plot on
        ax: axis to plot on
        legend: if True, plot a legend/ colorbar
        nspectr: number of spectra to plot
    output:
        ax: axis with plot
    '''
    if nspectr is None:
        nspectr = spectr.shape[1]

    N = spectr.shape[0]
       
    colors = tum_cmap(np.linspace(0, 1, nspectr))
    idxs = np.linspace(0, N-1, nspectr, dtype=int)

    for i, idx in enumerate(idxs):
        ax.plot(wavelengths, spectr[idx,:], color=colors[i])
    ax.set_xlabel('Wavelength (nm)')
    # ax.set_ylabel('Intensity')

    if legend:
        sm = plt.cm.ScalarMappable(cmap=tum_cmap, norm=plt.Normalize(0, nspectr - 1))
        sm.set_array(np.arange(0, nspectr))  # Specify the array directly
        fig.tight_layout()
        cbar_ax = ax.inset_axes([0.65, 0.9, 0.3, 0.05])
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
        cbar.ax.set_xticks([])
        cbar.set_label("Pixel \n Spectra")

def plot_class_spectra(img, gt_map, nspectr=None, bands=None, figsize=(18,5)):
    '''
    Plot the spectra of the pixels in the classes Normal, Tumor, and Hypervasculized of the ground truth map.
    input:
        img: spectral.ImageArray or np.array containing the image
        gt_map: spectral.ImageArray or np.array containing the ground truth map
        nspectr: number of spectra to plot for each class
        bands: np.array containing the wavelengths of the bands
        figsize: size of the figure
    output:
        fig: figure with plot
        axs: axes with plots
    '''
    if bands is None:
        if isinstance(img, sp.io.bilfile.BilFile):
            bands = img.bands.centers
        else: 
            raise ValueError("bands must be specified if img is not a spectral.ImageArray")

    img = get_array(img)
    gt_map = gt_map.asarray()

    class_labels = ["Not labled", "Normal", "Tumor", "Hypervasculized", "Background"]
    class_ids = [1, 2, 3]
    fig, axs = plt.subplots(1, len(class_ids), figsize=figsize)
    # plot pixel spectra for each class
    for i, class_id in enumerate(class_ids):
        mask = np.where(gt_map[:,:,0] == class_id)
        class_pixels = img[mask]
        if len(mask[0]) == 0: 
            continue
        plot_spectrum(class_pixels, bands, fig=fig, ax=axs[i], legend=True, nspectr=nspectr)
        axs[i].set_title(class_labels[class_id])

    return fig, axs

def plot_bands(img, bands=[200, 300, 400, 500, 600, 700], figsize=(18,3)):
    '''
    Plot selected bands of the image as gray-scale images.
    input:
        img: spectral.ImageArray or np.array containing the image
        bands: list of bands to plot
        figsize: size of the figure
    output:
        fig: figure with plot
        axs: axes with plots
    '''
    img = get_array(img)
    fig, axs = plt.subplots(1, len(bands), figsize=figsize)
    for i, band in enumerate(bands):
        axs[i].imshow(img[:,:,band], cmap='gray', aspect='equal')
        axs[i].set_title(f'Band {band}')
        axs[i].axis('off')
    return fig, axs

def get_rgb(img, bands=[109,192,425]):
    '''
    Convert the image to RGB using the specified bands
    input:
        img: image to convert, SpyFile or array-like
        bands: bands to use for RGB, list of ints
    output:
        image as numpy array
    '''
    if isinstance(img, sp.io.bilfile.BilFile):
        if "default bands" in img.metadata:
            bands = [int(band) for band in img.metadata["default bands"]]
    img = get_array(img)
    img_rgb = np.stack([img[:,:,bands[2]], img[:,:,bands[1]], img[:,:,bands[0]]], axis=-1).squeeze()
    img_rgb = (img_rgb - img_rgb.min(axis=(0,1)))/ (img_rgb.max(axis=(0,1)) - img_rgb.min(axis=(0,1)))
    return img_rgb

def plot_img(img, gt_map=None, class_labels=None, class_colors=None, bands=[109,192,425], figsize=(5,5), legend=True):
    '''
    Plot the image. If gt_map is provided, the labeled pixels are overlayed on the image.
    input:
        img: image to plot, SpyFile or array-like
        bands: bands to use for RGB, list of ints
        gt_map: ground truth map, np.array or SpyFile
        class_labels: class labels for gt_map, list of strings
        filename: optional, filename to save the plot, str
    output:
        figure handle to the plot
    '''
    img_rgb = get_rgb(img, bands=bands)

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img_rgb)
    if gt_map is not None:
        gt_map = gt_map.asarray()
        class_ids = np.unique(gt_map)
        if class_labels is None:
            raise ValueError("class_labels must be provided when gt_map is provided")
        for class_id in class_ids[class_ids != 0]:
            mask = np.where(gt_map[:,:,0] == class_id)
            if class_colors is None:
                ax.scatter(mask[1], mask[0], marker='.', s=1, label=class_labels[class_id])
            else:
                ax.scatter(mask[1], mask[0], marker='.', s=1, label=class_labels[class_id], color=class_colors[class_id])
    ax.axis('off')
    if legend:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., markerscale=10)
    return fig, ax

def plot_class_dist(img, gt_map, bands, class_ids, class_labels, class_colors, figsize=(5,4), legend_loc='upper left'):
    '''
    Plot the spectral mean and std of the specified classes.
    input:
        img: image to plot, SpyFile or array-like
        gt_map: ground truth map, np.array or SpyFile
        bands: band centers, list of floats
        class_ids: class ids to plot, list of ints
        class_labels: class names, list of strings
        class_colors: colors to use for plotting, list of strings
        figsize: figure size, tuple of floats
        legend_loc: location of the legend, string or None if no legend should be plotted
    output:
        fig: figure handle to the plot
        ax: axis handle to the plot
    '''
    img = get_array(img)
    gt_map = gt_map.asarray()

    fig, ax = plt.subplots(figsize=figsize)
    for class_id in class_ids:
        mask = np.where(gt_map[:, :, 0] == class_id)
        class_std = np.std(img[mask], axis=0)
        class_mean = np.mean(img[mask], axis=0)
        ax.plot(bands, class_mean, label=class_labels[class_id], color=class_colors[class_id])
        ax.fill_between(bands, class_mean - class_std, class_mean + class_std, alpha=0.25, color=class_colors[class_id])
    ax.set_xlim([400, 1000])
    ax.set_ylabel('Intensity')
    ax.set_xlabel('Wavelength (nm)')
    if legend_loc is not None:
        # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        ax.legend(loc=legend_loc, handlelength=0.8)
    return fig, ax