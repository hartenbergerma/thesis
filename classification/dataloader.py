import sys
import os
import torch
import numpy as np
from torch.utils.data import Dataset
import spectral as sp

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from preprocessing import *

class helicoid_Dataset(Dataset):
    def __init__(self, data_folders, files, transform=None):
        self.data = []
        self.labels = []
        self.transform = transform
        for data_folder in data_folders:
            img_data = []
            img_labels = []
            for file in files:
                img_data.append(np.load(os.path.join(data_folder, file)))
            img_data = np.concatenate(self.data, axis=1)
            print(img_data.shape)
            img_labels = np.load(os.path.join(data_folder, 'labels_data.npy'))

            self.data.append(img_data)
            self.labels.append(img_labels)
        
        self.data = np.concatenate(self.data, axis=0)
        print(self.data.shape)
        self.labels = np.concatenate(self.labels, axis=0)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx] - 1
        if self.transform:
            x, y = self.transform((x, y))
        return x, y
    
def get_dataloaders(data_folders, files, batch_size=32, shuffle=False, transform=None):
    dataset = helicoid_Dataset(data_folders, files, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=128)
    return dataloader

# def img_test_dataloader(data_folder, files, batch_size=32, transform=None):
#     dataset = helicoid_Dataset(data_folder, files, transform=transform)
#     print(len(dataset))
#     test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=128)
#     return test_loader


# class helicoid_Dataset(Dataset):
#     def __init__(self, data_folder, files, transform=None):
#         self.data = []
#         for file in files:
#             self.data.append(np.load(os.path.join(data_folder, file)))
#         self.data = np.concatenate(self.data, axis=1)
#         self.labels = np.load(os.path.join(data_folder, 'labels_data.npy'))
#         self.transform = transform
        
#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, idx):
#         x = self.data[idx]
#         y = self.labels[idx] - 1
#         if self.transform:
#             x, y = self.transform((x, y))
#         return x, y
    
# def get_dataloaders(data_folder, files, batch_size=32, train_test_split=0.8, val_test_split=0.5, transform=None):
#     dataset = helicoid_Dataset(data_folder, files, transform=transform)
#     train_size = int(train_test_split * len(dataset))
#     val_size = int((1 - train_test_split) * val_test_split * len(dataset))
#     test_size = len(dataset) - train_size - val_size
#     train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
#     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=128)
#     val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=128)
#     test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=128)
#     return train_loader, val_loader, test_loader

# def img_test_dataloader(data_folder, files, batch_size=32, transform=None):
#     dataset = helicoid_Dataset(data_folder, files, transform=transform)
#     print(len(dataset))
#     test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=128)
#     return test_loader



# class helicoid_Dataset(Dataset):
#     def __init__(self, transform=None):
#         self.data = np.load('own_labels/normal_tumor_blood/preprocessed_data.npy')
#         self.labels = np.load('own_labels/normal_tumor_blood/labels_data.npy')
#         self.transform = transform
        
#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, idx):
#         x = self.data[idx]
#         y = self.labels[idx] - 1
#         if self.transform:
#             x, y = self.transform((x, y))
#         return x, y


# class helicoid_Dataset_test(Dataset):
#     def __init__(self, img_folder, transform=None):
#         self.transform = transform

#         img = sp.open_image(img_folder + "/raw.hdr")
#         white_ref = sp.open_image(img_folder + "/whiteReference.hdr")
#         dark_ref = sp.open_image(img_folder + "/darkReference.hdr")
#         # preprocessing
#         bands_range = [520,900]
#         img_interp, band_centers = bands_lin_interpolation(img, img.bands.centers, bands_range)
#         white_ref_interp, _ = bands_lin_interpolation(white_ref, img.bands.centers, bands_range)
#         dark_ref_interp, _ = bands_lin_interpolation(dark_ref, img.bands.centers, bands_range)
#         img_calib = calibrate_img(img_interp, white_ref_interp, dark_ref_interp)
#         img_calib_norm = img_calib / np.linalg.norm(img_calib, axis=(0,1), ord=1, keepdims=True)
#         img_smooth = smooth_spectral(img_calib_norm, 5)

#         self.data = img_smooth.reshape(-1, img_smooth.shape[-1])
#         self.labels = np.zeros_like(self.data)

#     def __len__(self):
#         return self.data.shape[0]
    
#     def __getitem__(self, idx):
#         x = self.data[idx]
#         y = self.labels[idx] - 1
#         if self.transform:
#             x, y = self.transform((x, y))
#         return x, y


# def get_img_test_dataloader(img_folder, transform=None):
#     dataset = helicoid_Dataset_test(img_folder, transform=transform)
#     test_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=128)
#     return test_loader