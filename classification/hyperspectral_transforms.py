import os
import sys
import torch
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from extinctions import *
from preprocessing import *

def get_endmembers():
    bands_range = [520,900]
    band_centers = np.arange(bands_range[0], bands_range[1]+1)
    extinction_dict = get_extinctions(bands_range)
    scatter_simple = (band_centers/500)**(-1.2)
    endmembers = np.vstack((extinction_dict["cyt_c_ox"],
                            extinction_dict["cyt_c_red"], 
                            extinction_dict["cyt_b_ox"], 
                            extinction_dict["cyt_b_red"], 
                            extinction_dict["cyt_oxi_ox"], 
                            extinction_dict["cyt_oxi_red"], 
                            extinction_dict["hb"], 
                            extinction_dict["hbo2"], 
                            extinction_dict["water"], 
                            extinction_dict["fat"], 
                            scatter_simple))
    endmembers = smooth_spectral(endmembers,5)
    endmembers = torch.from_numpy(endmembers)
    return endmembers

def get_endmembers_mc():
    spectra_folder = "./mc_sim/spectra_mc/"
    hb_mc = np.loadtxt(spectra_folder + "m_hhb_50.txt")
    hbo2_mc = np.loadtxt(spectra_folder + "m_hbo2_50.txt")
    cyt_c_ox_mc = np.loadtxt(spectra_folder + "m_cyt_c_ox_20.txt")
    cyt_c_red_mc = np.loadtxt(spectra_folder + "m_cyt_c_red_20.txt")
    cyt_b_ox_mc = np.loadtxt(spectra_folder + "m_cyt_b_ox_20.txt")
    cyt_b_red_mc = np.loadtxt(spectra_folder + "m_cyt_b_red_20.txt")
    cyt_oxi_ox_mc = np.loadtxt(spectra_folder + "m_cyt_oxi_ox_20.txt")
    cyt_oxi_red_mc = np.loadtxt(spectra_folder + "m_cyt_oxi_red_20.txt")
    scatter_mc = np.loadtxt(spectra_folder + "m_scatter_40.txt")
    water_mc = np.loadtxt(spectra_folder + "m_water_200.txt")
    fat_mc = np.loadtxt(spectra_folder + "m_fat_200.txt")

    endmembers = np.vstack((cyt_c_ox_mc,
                            cyt_c_red_mc, 
                            cyt_b_ox_mc, 
                            cyt_b_red_mc, 
                            cyt_oxi_ox_mc, 
                            cyt_oxi_red_mc, 
                            hb_mc, 
                            hbo2_mc, 
                            water_mc, 
                            fat_mc, 
                            scatter_mc))
    endmembers = smooth_spectral(endmembers,5)
    endmembers = torch.from_numpy(endmembers)
    return endmembers


class ToTensor(object):
    def __call__(self, sample):
        x, y = sample
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
    
class OSP(object):
    def __init__(self, ref_pixel=None):
        self.endmembers = get_endmembers()
        self.n = self.endmembers.shape[0]
        self.ref_pixel = ref_pixel

        # precompute projection matrices
        self.projection_matrices = []
        for i in range(self.n):
            endmember_remove = torch.cat((self.endmembers[:i], self.endmembers[i+1:]), dim=0)
            P = torch.eye(endmember_remove.shape[1]) - endmember_remove.T @ torch.pinverse(endmember_remove).T
            self.projection_matrices.append(P)

    def __call__(self, sample):
        x, y = sample
        if self.ref_pixel is not None:
            x = x - self.ref_pixel
        x_osp = torch.empty((self.n))
        for i in range(self.n):
            endmember_target = self.endmembers[i]
            P = self.projection_matrices[i]
            x_osp[i] = endmember_target[None,:] @ P @ x
        return x_osp, y
    

class CEM(object):
    def __init__(self):
        self.endmembers = get_endmembers()
        self.n = self.endmembers.shape[0]

    def __call__(self, sample):
        x, y = sample
        x_cem = torch.empty((self.n))
        for i in range(self.n):
            t = self.endmembers[i,:]
            R_hat = (t @ t[:,None]) / x.shape[-1] + torch.eye(t.shape[0])
            Rinv = torch.inverse(R_hat)
            x_cem[i] = torch.dot(t, Rinv @ x) / torch.dot(t, Rinv @ t)
        return x_cem, y

class CEM_mc(object):
    def __init__(self, ref_pixel):
        self.endmembers = get_endmembers_mc()
        self.n = self.endmembers.shape[0]
        self.ref_pixel = ref_pixel

    def __call__(self, sample):
        x, y = sample
        x = x - self.ref_pixel
        x_cem = torch.empty((self.n))
        for i in range(self.n):
            t = self.endmembers[i,:]
            R_hat = (t @ t[:,None]) / x.shape[-1] + torch.eye(t.shape[0])
            Rinv = torch.inverse(R_hat)
            x_cem[i] = (t[:,None] @ Rinv @ x) / (t[:,None] @ Rinv @ t)
        return x_cem, y
    
class ConcatTransform(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        _, y = sample
        transformed_samples = [transform(sample) for transform in self.transforms]
        x, _ = zip(*transformed_samples)
        x = torch.cat(x, dim=0)
        return x, y