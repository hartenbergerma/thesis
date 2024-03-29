import os
import torch
import numpy as np
import spectral as sp
import pytorch_lightning as L
import matplotlib.pyplot as plt


from tqdm import tqdm
from torch import nn
from model import ClassificationModel
from dataloader import HelicoidDataModule
from hyperspectral_transforms import *
from lightning.pytorch.loggers import TensorBoardLogger


def train(model, train_loader, val_loader, log_dir, max_epochs):
    logger = TensorBoardLogger(log_dir, name="my_model")
    trainer = L.Trainer(logger=logger, max_epochs=max_epochs, devices=1)
    trainer.fit(model, train_loader, val_loader)
    trainer.save_checkpoint("./classification/model.ckpt")
    return model


def visualize_weights(model):
    layer_weights = []
    for layer in model.layers:
        if isinstance(layer, nn.Linear):
            layer_weights.append(layer.weight.detach().numpy())

    # individual weights
    plt.imshow(np.abs(layer_weights[0]), cmap="hot", aspect="auto", interpolation="none")
    cax = plt.colorbar()
    cax.set_label("Weight magnitude")
    plt.xlabel("Bands")
    # plt.xticks([0, 80, 180, 280, 380], ["520", "600", "700","800", "900"])
    plt.xticks([0,11,22,33,44,55,66,77])
    plt.ylabel("Hidden units")
    plt.yticks([])
    plt.savefig("./classification/weights.png")

    # mean weights
    mean_weights = np.mean(np.abs(layer_weights[0]), axis=0)
    plt.figure()
    plt.bar(np.arange(0,77), mean_weights)
    plt.xlabel("Bands")
    # plt.xticks([0, 80, 180, 280, 380], ["520", "600", "700","800", "900"])
    plt.xticks([0,11,22,33,44,55,66])
    plt.xlim(-1, 77)
    plt.vlines([10.5, 21.5, 32.5, 43.5, 54.5, 65.5], 0, 0.21, color='k', linestyles="dashed")
    plt.ylabel("Mean weight magnitude")
    plt.savefig("./classification/mean_weights.png")

def test_img(model, data_folder, files, transform=None):
    # get the model prediction
    dataloader = get_dataloader(data_folders=[data_folder], files=files, transform=transform, mode="test")
    y_pred = []
    for x, y in tqdm(dataloader):
        y_pred.append(model(x).detach().numpy())
    y_pred = np.concatenate(y_pred, axis=0)
    gt_map = np.load(os.path.join(data_folder, "gtMap.npy"))
    # img_shape = np.load(os.path.join(data_folder, "img_shape.npy"))
    img_shape = gt_map.shape[:2]


    pred = np.argmax(y_pred, axis=-1)
    pred_img = pred.reshape(img_shape[0], img_shape[1])
    gt_map = gt_map.reshape(img_shape[0], img_shape[1])

    # get accuracy for each class
    classes = {1: "Normal", 2: "Tumor", 3: "Blood"}
    for i, c in classes.items():
        idx_class = np.where(gt_map == i)
        accuracy = (pred_img[idx_class] == i-1).mean()
        print(f"Accuracy for {c}: {accuracy}")

    # visualize the prediction
    plt.figure()
    im = plt.imshow(pred_img)
    plt.colorbar(im)
    plt.savefig("./classification/prediction_img.png")

    # do majority voting within a 4x4 window
    window_size = 4
    pred_img_padded = np.pad(pred_img, ((window_size//2, window_size//2), (window_size//2, window_size//2)), mode="edge")
    pred_img_knn = np.zeros_like(pred_img)
    for i in range(window_size//2, img_shape[0]):
        for j in range(window_size//2, img_shape[1]):
            window = pred_img_padded[i-window_size//2:i+window_size//2+1, j-window_size//2:j+window_size//2+1]
            window_class_counts = [np.sum(window == c, axis=(0,1)) for c in range(0,3)]
            window_class = np.argmax(window_class_counts)
            pred_img_knn[i, j] = window_class

    # get accuracy for each class after majority voting
    for i, c in classes.items():
        idx_class = np.where(gt_map == i)
        accuracy = (pred_img_knn[idx_class] == i-1).mean()
        print(f"Accuracy for {c} after majority voting: {accuracy}")

    # visualize the prediction after majority voting
    plt.figure()
    im = plt.imshow(pred_img_knn)
    plt.colorbar(im)
    plt.savefig("./classification/prediction_img_knn.png")

            

def main():
    train_data_folders = ['own_labels/normal_tumor_blood/' + folder for folder in ["008-01", "008-02", "010-03", "012-02", "015-01", "016-04", "016-05", "017-01", "020-01"]]#, "039-01"]]
    val_data_folders = ['own_labels/normal_tumor_blood/012-01']
    # data_files = ["preprocessed.npy", "heatmaps_osp.npy", "heatmaps_osp_diff.npy", "heatmaps_osp_diff_mc.npy", "heatmaps_icem.npy", "heatmaps_icem_diff.npy", "heatmaps_icem_diff_mc.npy"]#, "pca_data.npy"] # add LMM
    data_files = ["preprocessed.npy"]
    train_loader = get_dataloader(data_folders=train_data_folders, files=data_files, batch_size=64, transform=ToTensor())
    val_loader = get_dataloader(data_folders=val_data_folders, files=data_files, batch_size=64, transform=ToTensor(), mode='val')
    # get first batch
    x, y = next(iter(train_loader))
    print(x.shape, y.shape)

    # # model = ClassificationModel(x.shape[1], 32, 3, 8, [1/0.47, 1/0.28, 1/0.25])
    # # model = ClassificationModel(x.shape[1], 24, 3, 1, [1/0.47, 1/0.29, 1/0.25], weight_decay=1e-3)
    # # # model = ClassificationModel(x.shape[1], 24, 2, 1, [1/0.47, 1/0.29, 1/0.25], weight_decay=1e-3)
    model = ClassificationModel(x.shape[1], 32, 3, 8, [1/0.47, 1/0.29, 1/0.25], weight_decay=0)

    train(model, train_loader, val_loader, "./classification/tb_logs", max_epochs=40)

    visualize_weights(model)
    # model = ClassificationModel.load_from_checkpoint("./classification/model.ckpt", input_dim=66, hidden_dim=12, output_dim=3, num_layers=4, loss_weight=[1/29903,1/10199,1/17890])

    # test_img(model, data_folder="own_labels/cem_global/039-01", files=data_files, transform=ToTensor())

if __name__ == "__main__":
    main()


# 24, 3,8, lr 1e-5, wd=1e-3