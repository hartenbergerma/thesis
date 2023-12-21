import sys
import os
import spectral as sp
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
preprocessing_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(preprocessing_path)
from preprocessing import *
from plotting_parameters import *



gradeIVpatients = ["008-01"]#, "008-02", "010-03", "012-01", "012-02", "014-01", "015-01", "016-04", "016-05", "017-01", "020-01", "025-02"]


for patient in gradeIVpatients:
    data_folder = f"C:/Users/User/OneDrive/Dokumente/Uni/Master/Masterarbeit/datasets/helicoid/{patient}"

    # Load the sp data
    img = sp.open_image(data_folder + "/raw.hdr")
    white_ref = sp.open_image(data_folder + "/whiteReference.hdr")
    dark_ref = sp.open_image(data_folder + "/darkReference.hdr")
    gt_map = sp.open_image(data_folder + "/gtMap.hdr")

    # get illumination spectrum E
    E = np.subtract(white_ref.asarray(), dark_ref.asarray(), dtype=np.float32)
    E = E.squeeze()
    E = smooth_spectral(E, 5)
    E_mean = np.mean(E, axis=0, dtype=np.float32)

    # get mapping to subspace orthogonal to E
    P_E = np.eye(E_mean.shape[0]) - np.outer(E_mean, E_mean)/np.dot(E_mean, E_mean)

    # project white reference onto subspace orthogonal to E
    E_proj = np.einsum('ij,kj->ki', P_E, E)

    # get reflected spectrum R
    R = np.subtract(img.asarray(), dark_ref.asarray(), dtype=np.float32)

    # apply mapping to data
    R_E = np.einsum('ij,klj->kli', P_E, R)
    R_E = np.load(f"{data_folder}/R_E.npy")
    os.makedirs(f"{data_folder}/results", exist_ok=True)
    print("Calculated R_E")

    # normalize R
    R_E_norm = R_E / alpha[:,:,np.newaxis]
    np.savez_compressed(f"{data_folder}/results/R_E.npz", R_E=R_E, R_E_norm=R_E_norm, alpha=alpha, allow_pickle=True)




    # get plots of the data
    band_centers = img.bands.centers
    class_labels = ["Not labled", "Normal", "Tumor", "Hypervasculized", "Background"]

    # Plot white reference before and after projection
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    im0 = axs[0].imshow(E, cmap='viridis', aspect='auto') 
    cb0 = plt.colorbar(im0)
    cb0.set_label('Intensity')
    axs[0].set_xticks(np.arange(0,E.shape[1],100))
    axs[0].set_xticklabels([f'{wavelength:.0f}' for wavelength in band_centers[::100]])
    axs[0].set_xlabel('Wavelength (nm)')
    axs[0].set_ylabel('Pixel')
    axs[0].set_title('Before projection')
    im1 = axs[1].imshow(E_proj, cmap='viridis', aspect='auto')
    cb1 = plt.colorbar(im1, ax=axs[1])
    cb1.set_label('Intensity')
    axs[1].set_xticks(np.arange(0,E_proj.shape[1],100))
    axs[1].set_xticklabels([f'{wavelength:.0f}' for wavelength in band_centers[::100]])
    axs[1].set_xlabel('Projected wavelength (nm)')
    axs[1].set_ylabel('Pixel')
    axs[1].set_title('After projection')
    fig.suptitle('White Reference')
    plt.savefig(f"{data_folder}/results/white_reference_proj.svg", bbox_inches='tight')
    plt.close(fig)

    # plot images and spectra
    bands = [100, 200, 300, 400, 500, 600, 700, 800]

    fig, axs = plt.subplots(1, len(bands), figsize=(25,3))
    for i, band in enumerate(bands):
        # fixed aspect ratio
        axs[i].imshow(R[:,:,band], cmap='gray', aspect='equal') 
        axs[i].set_title(f"Band {band}")
    fig.suptitle('Raw image for different bands')
    plt.savefig(f"{data_folder}/results/raw_images.svg", bbox_inches='tight')
    plt.close(fig)

    fig = plot_class_spectra(R, gt_map, nspectr=25, bands=img.bands.centers)
    plt.savefig(f"{data_folder}/results/raw_spectra.svg", bbox_inches='tight')
    plt.close(fig)

    fig, axs = plt.subplots(1, len(bands), figsize=(25,3))
    for i, band in enumerate(bands):
        # fixed aspect ratio
        axs[i].imshow(R_E[:,:,band], cmap='gray', aspect='equal') 
        axs[i].set_title(f"Band {band}")
    fig.suptitle('Mapped image for different bands')
    plt.savefig(f"{data_folder}/results/mapped_images.svg", bbox_inches='tight')
    plt.close(fig)

    fig = plot_class_spectra(R_E, gt_map, nspectr=25, bands=img.bands.centers)
    plt.savefig(f"{data_folder}/results/mapped_spectra.svg", bbox_inches='tight')
    plt.close(fig)

    fig, axs = plt.subplots(1, len(bands), figsize=(25,3))
    for i, band in enumerate(bands):
        # fixed aspect ratio
        axs[i].imshow(R_E_norm[:,:,band], cmap='gray', aspect='equal') 
        axs[i].set_title(f"Band {band}")
    fig.suptitle('Mapped and normalized image for different bands')
    plt.savefig(f"{data_folder}/results/mapped_norm_images.svg", bbox_inches='tight')
    plt.close(fig)

    fig = plot_class_spectra(R_E_norm, gt_map, nspectr=25, bands=img.bands.centers)
    plt.savefig(f"{data_folder}/results/mapped_norm_spectra.svg", bbox_inches='tight')
    plt.close(fig)

    # PCA on the data
    max_samples = 1000 # maximum number of samples per class
    configs = ["raw", "calibrated", "calibrated_norm" "mapped", "mapped_norm"]

    img_raw = img.asarray()
    img_calibrated = calibrate_img(img, white_ref, dark_ref)

    img_mapped = R_E
    img_mapped_norm = R_E_norm

    for i, pca_img in enumerate([img_raw, img_calibrated, img_mapped, img_mapped_norm]):
        N = pca_img[np.where(gt_map.asarray()[:,:,0] == 1)]
        T = pca_img[np.where(gt_map.asarray()[:,:,0] == 2)]
        B = pca_img[np.where(gt_map.asarray()[:,:,0] == 3)]
        print(f"{N.shape=}, {T.shape=}, {B.shape=}")

        # select random samples from each class such that the number of samples is equal to the smallest class or max_samples
        nsamples = min(N.shape[0], T.shape[0], B.shape[0])
        if nsamples == 0:
            break
        if nsamples > max_samples:
            nsamples = max_samples

        idx_N = np.random.choice(N.shape[0], nsamples, replace=False)
        N = N[idx_N]
        idx_T = np.random.choice(T.shape[0], nsamples, replace=False)
        T = T[idx_T]
        idx_B = np.random.choice(B.shape[0], nsamples, replace=False)
        B = B[idx_B]

        N_and_T_and_B = np.concatenate((N, T, B))
        N_and_T = np.concatenate((N, T))
        N_and_B = np.concatenate((N, B))
        B_and_T = np.concatenate((T, B))

        pca = PCA(2)
        plt.figure()
        pca.fit(N_and_T_and_B)
        plt.plot(band_centers, pca.components_[0], label="ntb")
        pca.fit(N_and_T)
        plt.plot(band_centers, pca.components_[0], label="nt")
        pca.fit(N_and_B)
        plt.plot(band_centers, pca.components_[0], label="nb")
        pca.fit(B_and_T)
        plt.plot(band_centers, pca.components_[0], label="bt")
        plt.legend()
        plt.title("Inter-class PCA components")
        plt.savefig(f"{data_folder}/results/{configs[i]}_pca_components.svg", bbox_inches='tight')
        plt.close()

        ntb = pca.fit_transform(N_and_T_and_B)
        plt.figure()
        Y = np.repeat([1, 2, 3], nsamples)
        for j in range(1,4):
            plt.scatter(ntb[Y==j, 0], ntb[Y==j, 1], label=class_labels[j])
        plt.xlabel('component 1')
        plt.ylabel('component 2')
        plt.title('Projected data after inter-class PCA')
        plt.legend()
        plt.savefig(f"{data_folder}/results/{configs[i]}_pca_projected.svg", bbox_inches='tight')
        plt.close()