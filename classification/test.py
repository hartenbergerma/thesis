import os
import json
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch import nn
from model import ClassificationModel
from dataloader import HelicoidDataModule

from torchmetrics.classification import MulticlassAccuracy, MulticlassRecall, MulticlassPrecision, MulticlassF1Score


parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, required=True, help="Training mode: baseline, baseline_reduced, heatmap or heatmap_only", choices=["baseline", "heatmap", "baseline_reduced", "heatmap_only"])
parser.add_argument("--log_dir", type=str, required=True, help="Model checkpoint directory")
parser.add_argument("--folds", nargs='+', type=str, required=True, help="Fold to use for training", choices=["fold1", "fold2", "fold3", "fold4", "fold5"])
args = parser.parse_args()


def visualize_weights(model, save_dir):
    layer_weights = []
    for layer in model.layers:
        if isinstance(layer, nn.Linear):
            layer_weights.append(layer.weight.detach().cpu().numpy())

    # individual weights
    plt.imshow(np.abs(layer_weights[0]), cmap="hot", aspect="auto", interpolation="none")
    cax = plt.colorbar()
    cax.set_label("Weight magnitude")
    plt.xlabel("Bands")
    # plt.xticks([0, 80, 180, 280, 380], ["520", "600", "700","800", "900"])
    # plt.xticks([0,11,22,33,44,55,66,77])
    plt.ylabel("Hidden units")
    plt.yticks([])
    plt.savefig(os.path.join(save_dir, "weights.png"))

    # mean weights
    mean_weights = np.mean(np.abs(layer_weights[0]), axis=0)
    plt.figure()
    plt.bar(np.arange(0, mean_weights.shape[0]), mean_weights)
    plt.xlabel("Bands")
    # plt.xticks([0, 80, 180, 280, 380], ["520", "600", "700","800", "900"])
    # plt.xticks([0,11,22,33,44,55,66])
    # plt.xlim(-1, 77)
    # plt.vlines([10.5, 21.5, 32.5, 43.5, 54.5, 65.5], 0, 0.21, color='k', linestyles="dashed")
    plt.ylabel("Mean weight magnitude")
    plt.savefig(os.path.join(save_dir, "mean_weights.png"))

# def test_img(model, files, fold, save_dir):
#     dm_pred = HelicoidDataModule(files=files, fold=fold)
#     dm_pred.setup("predict")
#     dataloader = dm_pred.predict_dataloader()
#     y_pred = []
#     for x, y in tqdm(dataloader):
#         y_pred.append(model(x).detach().numpy())
#     y_pred = np.concatenate(y_pred, axis=0)
#     gt_map = np.load(os.path.join(data_folder, "gtMap.npy"))
#     img_shape = gt_map.shape[:2]


#     pred = np.argmax(y_pred, axis=-1)
#     pred_img = pred.reshape(img_shape[0], img_shape[1])
#     gt_map = gt_map.reshape(img_shape[0], img_shape[1])

#     # get accuracy for each class
#     classes = {1: "Normal", 2: "Tumor", 3: "Blood"}
#     for i, c in classes.items():
#         idx_class = np.where(gt_map == i)
#         accuracy = (pred_img[idx_class] == i-1).mean()
#         print(f"Accuracy for {c}: {accuracy}")

#     # visualize the prediction
#     plt.figure()
#     im = plt.imshow(pred_img)
#     plt.colorbar(im)
#     plt.savefig("./classification/prediction_img.png")

#     # do majority voting within a 4x4 window
#     window_size = 4
#     pred_img_padded = np.pad(pred_img, ((window_size//2, window_size//2), (window_size//2, window_size//2)), mode="edge")
#     pred_img_knn = np.zeros_like(pred_img)
#     for i in range(window_size//2, img_shape[0]):
#         for j in range(window_size//2, img_shape[1]):
#             window = pred_img_padded[i-window_size//2:i+window_size//2+1, j-window_size//2:j+window_size//2+1]
#             window_class_counts = [np.sum(window == c, axis=(0,1)) for c in range(0,3)]
#             window_class = np.argmax(window_class_counts)
#             pred_img_knn[i, j] = window_class

#     # get accuracy for each class after majority voting
#     for i, c in classes.items():
#         idx_class = np.where(gt_map == i)
#         accuracy = (pred_img_knn[idx_class] == i-1).mean()
#         print(f"Accuracy for {c} after majority voting: {accuracy}")

#     # visualize the prediction after majority voting
#     plt.figure()
#     im = plt.imshow(pred_img_knn)
#     plt.colorbar(im)
#     plt.savefig("./classification/prediction_img_knn.png")

def metrics(model, files, fold, save_dir):
    dm_test = HelicoidDataModule(files=files, fold=fold)
    dm_test.setup("test")
    dataloader = dm_test.test_dataloader()

    logits = []
    y_true = []
    for x, y in tqdm(dataloader):
        logits.append(model(x).detach().cpu())
        y_true.append(y.cpu())
    logits = torch.concatenate(logits, axis=0)
    y_true = torch.concatenate(y_true, axis=0)

    accuracy = MulticlassAccuracy(num_classes=4, average=None)
    precision = MulticlassPrecision(num_classes=4, average=None)
    recall = MulticlassRecall(num_classes=4, average=None)
    f1_score = MulticlassF1Score(num_classes=4, average=None)

    accuracy_macro = MulticlassAccuracy(num_classes=4, average="macro")
    precision_macro = MulticlassPrecision(num_classes=4, average="macro")
    recall_macro = MulticlassRecall(num_classes=4, average="macro")
    f1_score_macro = MulticlassF1Score(num_classes=4, average="macro")

    retults = {
        "accuracy": accuracy(logits, y_true).numpy().tolist(),
        "accuracy_macro": accuracy_macro(logits, y_true).numpy().tolist(),
        "precision": precision(logits, y_true).numpy().tolist(),
        "precision_macro": precision_macro(logits, y_true).numpy().tolist(),
        "recall": recall(logits, y_true).numpy().tolist(),
        "recall_macro": recall_macro(logits, y_true).numpy().tolist(),
        "f1_score": f1_score(logits, y_true).numpy().tolist(),
        "f1_score_macro": f1_score_macro(logits, y_true).numpy().tolist()
    }

    # save results as json
    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(retults, f, indent=4)



def main():

    if args.mode == "baseline":
        files = ["preprocessed.npy"]
    elif args.mode == "heatmap":
        files = ["preprocessed.npy", "heatmaps_osp.npy", "heatmaps_osp_diff.npy", "heatmaps_osp_diff_mc.npy", "heatmaps_icem.npy", "heatmaps_icem_diff.npy", "heatmaps_icem_diff_mc.npy"]
    elif args.mode == "heatmap_only":
        files = ["heatmaps_osp.npy", "heatmaps_osp_diff.npy", "heatmaps_osp_diff_mc.npy", "heatmaps_icem.npy", "heatmaps_icem_diff.npy", "heatmaps_icem_diff_mc.npy"]
    elif args.mode == "baseline_reduced":
        files = ["preprocessed_reduced"]
    
    for fold in args.folds:

        checkpoint_path = os.path.join(args.log_dir, f"{fold}.ckpt")
        save_dir = os.path.join(args.log_dir, "results", fold)
        os.makedirs(save_dir, exist_ok=True)

        model = ClassificationModel.load_from_checkpoint(checkpoint_path)
        model.eval()

        visualize_weights(model, save_dir)

        metrics(model, files, fold, save_dir)

        # test_img(model, files, fold, save_dir)




if __name__ == "__main__":
    main()