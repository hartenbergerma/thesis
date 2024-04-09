import os
import json
import torch
import numpy as np
import lightning.pytorch as pl
from torch.utils.data import TensorDataset, DataLoader

from sklearn.decomposition import PCA

class Helicoid_Dataset_Loader():
    def __init__(self, files, n_dim=None):
        self.files = files
        self.n_dim = n_dim
        if n_dim is not None:
            self.pca = PCA(n_components=n_dim)

    def load_data(self, patient_folders, mode='labeled', return_img_shape=False):
        data = []
        labels = []
        for patient_folder in patient_folders:
            print(f"loading image {patient_folder}")
            patient_folder = os.path.join('/home/martin_ivan/code/own_labels/npj_database/', patient_folder)
            img_data = []
            img_labels = np.load(os.path.join(patient_folder, 'gtMap.npy')).astype(int)
            img_shape = img_labels.squeeze().shape
            for file in self.files:
                img_data_all = np.load(os.path.join(patient_folder, file))
                if mode == 'labeled':
                    img_data.append(img_data_all[(img_labels !=0)])
                elif mode == 'all':
                    img_data.append(img_data_all.reshape(-1, img_data_all.shape[-1]))
                else:
                    raise ValueError("Unknown mode")
            img_data = np.concatenate(img_data, axis=1)
            data.append(img_data)
            if mode == 'labeled':
                labels.append(img_labels[(img_labels !=0)])
            elif mode == 'all':
                labels.append(img_labels.reshape(-1))
            else:
                raise ValueError("Unknown mode")
        data = np.concatenate(data, axis=0)
        labels = np.concatenate(labels, axis=0) - 1
        if return_img_shape:
            return data, labels, img_shape
        return data, labels
    
    def balance_dataset(self, data, labels, per_class_count=3000):
        class_counts = np.unique(labels, return_counts=True)[1]
        data_balanced = []
        labels_balanced = []
        for i in range(len(class_counts)):
            idx_class = np.where(labels == i)[0]
            np.random.seed(0)
            random_idx = np.random.choice(idx_class, per_class_count, replace=False)
            data_balanced.append(data[random_idx])
            labels_balanced.append(labels[random_idx])
        data_balanced = np.concatenate(data_balanced, axis=0)
        labels_balanced = np.concatenate(labels_balanced, axis=0)
        return data_balanced, labels_balanced
    
    def to_device(self, data, labels):
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        data = torch.tensor(data, dtype=torch.float32).to(device)
        labels = torch.tensor(labels, dtype=torch.long).to(device)
        return data, labels

    def get_train_dataset(self, train_patient_folders, balance_dataset=False):
        data, labels = self.load_data(train_patient_folders, mode='labeled')
        if balance_dataset:
            data, labels = self.balance_dataset(data, labels)
        if self.n_dim is not None:
            data, labels = self.pca.fit_transform(data)
        data, labels = self.to_device(data, labels)
        return TensorDataset(data, labels)
    
    def get_val_dataset(self, val_patient_folders):
        data, labels = self.load_data(val_patient_folders, mode='labeled')
        if self.n_dim is not None:
            data, labels = self.pca.transform(data)
        data, labels = self.to_device(data, labels)
        return TensorDataset(data, labels)
    
    def get_test_dataset(self, test_patient_folders):
        data, labels = self.load_data(test_patient_folders, mode='labeled')
        if self.n_dim is not None:
            data, labels = self.pca.transform(data)
        data, labels = self.to_device(data, labels)
        return TensorDataset(data, labels)
    
    def get_predict_datasets(self, test_patient_folders):
        img_datasets = []
        img_shapes = []
        for test_patient_folder in test_patient_folders:
            data, labels, img_shape = self.load_data([test_patient_folder], mode='all', return_img_shape=True)
            if self.n_dim is not None:
                data, labels = self.pca.transform(data)
            data, labels = self.to_device(data, labels)
            img_datasets.append(TensorDataset(data, labels))
            img_shapes.append(img_shape)
        return img_datasets, img_shapes



# def get_helicoid_Dataset(patient_folders, files, mode='labeled'):
#         data = []
#         labels = []
#         for patient_folder in patient_folders:
#             print(f"loading image {patient_folder}")
#             patient_folder = os.path.join('/home/martin_ivan/code/own_labels/npj_database/', patient_folder)
#             img_data = []
#             img_labels = np.load(os.path.join(patient_folder, 'gtMap.npy')).astype(int)
#             for file in files:
#                 if file == 'preprocessed_reduced':
#                     img_data_all = np.load(os.path.join(patient_folder, "preprocessed.npy"))[:,:,0::4]
#                 else:
#                     img_data_all = np.load(os.path.join(patient_folder, file))
#                 if (mode == 'labeled') or (mode == 'labeled_balanced'):
#                     # img_data.append(img_data_all[(img_labels !=0) & (img_labels != 4)])
#                     img_data.append(img_data_all[(img_labels !=0)])
#                 elif mode == 'all':
#                     img_data.append(img_data_all.reshape(-1, img_data_all.shape[-1]))
#                 else:
#                     raise ValueError("Unknown mode")
#             img_data = np.concatenate(img_data, axis=1)

#             data.append(img_data)
#             if (mode == 'labeled') or (mode == 'labeled_balanced'):
#                 # self.labels.append(img_labels[(img_labels !=0) & (img_labels != 4)])
#                 labels.append(img_labels[(img_labels !=0)])
#             elif mode == 'all':
#                 labels.append(img_labels.reshape(-1))
#             else:
#                 raise ValueError("Unknown mode")
                    
#         data = np.concatenate(data, axis=0)
#         labels = np.concatenate(labels, axis=0) - 1

#         if mode == 'labeled_balanced':
#             print("Balancing the dataset")
#             ## select 3000 samples from each class
#             # per_class_count = 3000
#             per_class_count = np.min(np.unique(labels, return_counts=True)[1])
#             class_counts = np.unique(labels, return_counts=True)[1]
#             data_balanced = []
#             labels_balanced = []
#             for i in range(len(class_counts)):
#                 idx_class = np.where(labels == i)[0]
#                 np.random.seed(0)
#                 random_idx = np.random.choice(idx_class, per_class_count, replace=False)
#                 data_balanced.append(data[random_idx])
#                 labels_balanced.append(labels[random_idx])

#             data = np.concatenate(data_balanced, axis=0)
#             labels = np.concatenate(labels_balanced, axis=0)

#         data_size_GB = data.nbytes / (1024 ** 3)
#         print(f"The size of the data array is: {data_size_GB} GB")

#         if torch.cuda.is_available():
#             device = "cuda"
#         else:
#             device = "cpu"
#         data = torch.tensor(data, dtype=torch.float32).to(device)
#         labels = torch.tensor(labels, dtype=torch.long).to(device)

#         return TensorDataset(data, labels)

class HelicoidDataModule(pl.LightningDataModule):
    def __init__(self, files, fold="fold1"):
        super().__init__()
        self.fold = fold
        self.files = files
        self.setup()
        self.dataset_loader = Helicoid_Dataset_Loader(files)

    def setup(self, stage=None):
        with open('/home/martin_ivan/code/own_labels/folds.json') as f:
            folds = json.load(f)

        if stage=="fit":
            # self.dataset_train = get_helicoid_Dataset(folds[self.fold]["train"], self.files, mode='labeled_balanced')
            # self.dataset_val = get_helicoid_Dataset(folds[self.fold]["val"], self.files)
            self.dataset_train = self.dataset_loader.get_train_dataset(folds[self.fold]["train"])
            self.dataset_val = self.dataset_loader.get_val_dataset(folds[self.fold]["val"])
        if stage=="val":
            self.dataset_val = self.dataset_loader.get_val_dataset(folds[self.fold]["val"])
        if stage=="test":
            # self.dataset_test = get_helicoid_Dataset(folds[self.fold]["test"], self.files)
            self.dataset_test = self.dataset_loader.get_test_dataset(folds[self.fold]["test"])
        if stage=="predict":
            # self.dataset_predict = get_helicoid_Dataset(folds[self.fold]["test"], self.files, mode='all')
            print(folds[self.fold]["test"])
            self.datasets_predict, self.test_img_shapes = self.dataset_loader.get_predict_datasets(folds[self.fold]["test"])
            self.image_ids = folds[self.fold]["test"]

    def train_dataloader(self, batch_size=64):
        return DataLoader(self.dataset_train, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    
    def val_dataloader(self, batch_size=1024):
        return DataLoader(self.dataset_val, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)
    
    def test_dataloader(self, batch_size=1024): 
        return DataLoader(self.dataset_test, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)
    
    def predict_dataloader(self, batch_size=1024):
        dataloaders = []
        for dataset in self.datasets_predict:
            dataloaders.append(DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False))
        return dataloaders, self.test_img_shapes, self.image_ids

    def sample_size(self):
        size = self.dataset_train.tensors[0].shape[1]
        print(f"------------- sample size: {size} -------------")
        return size
    
    def class_distribution(self):
        dist =  torch.unique(self.dataset_train.tensors[1], return_counts=True)[1].float()
        print(f"------------- class distribution: {dist} -------------")
        return dist
    
    def num_classes(self):
        num = len(torch.unique(self.dataset_train.tensors[1]))
        print(f"------------- number of classes: {num} -------------")
        return num
    
    def get_fold(self):
        return self.fold



# class helicoid_Dataset(Dataset):
#     def __init__(self, patient_folders, files, transform=None, mode='labeled'):
#         self.data = []
#         self.labels = []
#         self.transform = transform
#         for patient_folder in patient_folders:
#             print(f"loading image {patient_folder}")
#             patient_folder = os.path.join('/home/martin_ivan/code/own_labels/npj_database/', patient_folder)
#             img_data = []
#             img_labels = np.load(os.path.join(patient_folder, 'gtMap.npy')).astype(int)
#             for file in files:
#                 if file == 'preprocessed_reduced':
#                     img_data_all = np.load(os.path.join(patient_folder, "preprocessed.npy"))[:,:,0::4]
#                 else:
#                     img_data_all = np.load(os.path.join(patient_folder, file))
#                 if mode == 'labeled':
#                     # img_data.append(img_data_all[(img_labels !=0) & (img_labels != 4)])
#                     img_data.append(img_data_all[(img_labels !=0)])
#                 elif mode == 'all':
#                     img_data.append(img_data_all.reshape(-1, img_data_all.shape[-1]))
#                 else:
#                     raise ValueError("Unknown mode")
#             img_data = np.concatenate(img_data, axis=1)

#             self.data.append(img_data)
#             if mode == 'labeled':
#                 # self.labels.append(img_labels[(img_labels !=0) & (img_labels != 4)])
#                 self.labels.append(img_labels[(img_labels !=0)])
#             elif mode == 'all':
#                 self.labels.append(img_labels.reshape(-1))
#             else:
#                 raise ValueError("Unknown mode")
                    
#         self.data = np.concatenate(self.data, axis=0)
#         self.labels = np.concatenate(self.labels, axis=0)
#         print(f"------------- label counts: {np.unique(self.labels, return_counts=True)} -------------")
#         data_size_GB = self.data.nbytes / (1024 ** 3)
#         print(f"The size of the data array is: {data_size_GB} GB")

#         if torch.cuda.is_available():
#             device = "cuda"
#         else:
#             device = "cpu"
#         self.data = torch.tensor(self.data, dtype=torch.float32).to(device)
#         self.labels = torch.tensor(self.labels, dtype=torch.long).to(device)

#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, idx):
#         x = self.data[idx]
#         y = self.labels[idx] - 1
#         # if self.transform:
#         #     x, y = self.transform((x, y))
#         return x, y
    
# class ToTensor(object):
#     def __call__(self, sample):
#         x, y = sample
#         return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

# class HelicoidDataModule(pl.LightningDataModule):
#     def __init__(self, files, batch_size=64, fold="fold1"):
#         super().__init__()
#         self.batch_size = batch_size
#         self.fold = fold
#         self.files = files
#         self.transform = ToTensor()

#         self.setup()

#     def setup(self, stage=None):
#         with open('/home/martin_ivan/code/own_labels/folds.json') as f:
#             folds = json.load(f)

#         if stage=="fit":
#             self.dataset_train = helicoid_Dataset(folds[self.fold]["train"], self.files, transform=self.transform)
#             self.dataset_val = helicoid_Dataset(folds[self.fold]["val"], self.files, transform=self.transform)
#         if stage=="test":
#             self.dataset_test = helicoid_Dataset(folds[self.fold]["test"], self.files, transform=self.transform)
#         if stage=="predict":
#             self.dataset_predict = helicoid_Dataset(folds[self.fold]["test"], self.files, transform=self.transform, mode='all')

#     def train_dataloader(self):
#         return DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=True, num_workers=32)
    
#     def val_dataloader(self):
#         return DataLoader(self.dataset_val, batch_size=self.batch_size, shuffle=False, num_workers=32)
    
#     def test_dataloader(self): 
#         return DataLoader(self.dataset_test, batch_size=self.batch_size, shuffle=False, num_workers=32)
    
#     def predict_dataloader(self):
#         return DataLoader(self.dataset_predict, batch_size=self.batch_size, shuffle=False, num_workers=32)
    
#     def sample_size(self):
#         return self.dataset_train.data.shape[1]
    
#     def class_distribution(self):
#         dist =  torch.unique(self.dataset_train.labels, return_counts=True)[1].float()
#         print(f"------------- class distribution: {dist} -------------")
#         return dist
#         # return np.unique(self.dataset_train.labels, return_counts=True)[1]
    
#     def num_classes(self):
#         num = len(torch.unique(self.dataset_train.labels))
#         print(f"------------- number of classes: {num} -------------")
#         return num
#         # return len(np.unique(self.dataset_train.labels))
    

    
# def get_dataloader(fold, files, batch_size=32, transform=None):
#     # load folds.json
#     with open('./own_labels/folds.json') as f:
#         folds = json.load(f)

#     dataset_train = helicoid_Dataset(folds[fold]["train"], files, transform=transform, mode='train')
#     dataset_val = helicoid_Dataset(folds[fold]["val"], files, transform=transform, mode='val')
#     dataset_test = helicoid_Dataset(folds[fold]["test"], files, transform=transform, mode='test')

#     dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=128)
#     dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=128)
#     dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=128)

#     return dataloader_train, dataloader_val, dataloader_test


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