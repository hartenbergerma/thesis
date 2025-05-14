import os
import sys
import json
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt


from tqdm import tqdm
from torch import nn
from matplotlib.colors import LinearSegmentedColormap
from model import ClassificationModel
from dataloader import HelicoidDataModule

from torchmetrics.classification import MulticlassAccuracy, MulticlassRecall, MulticlassPrecision, MulticlassF1Score, MulticlassAUROC, Specificity

# add parent folder to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plotting_parameters import *


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

def get_predictions(model, dataloader):
    logits = []
    y_true = []
    for x, y in tqdm(dataloader):
        logits.append(model(x).detach().cpu())
        y_true.append(y.cpu())
    logits = torch.concatenate(logits, axis=0)
    y_true = torch.concatenate(y_true, axis=0)
    return logits, y_true

def get_metrics(pred, y_true, logits=True):
    accuracy = MulticlassAccuracy(num_classes=4, average=None)
    precision = MulticlassPrecision(num_classes=4, average=None)
    recall = MulticlassRecall(num_classes=4, average=None)
    f1_score = MulticlassF1Score(num_classes=4, average=None)
    specificity = Specificity(task="multiclass", num_classes=4, average=None)

    accuracy_macro = MulticlassAccuracy(num_classes=4, average="macro")
    precision_macro = MulticlassPrecision(num_classes=4, average="macro")
    recall_macro = MulticlassRecall(num_classes=4, average="macro")
    f1_score_macro = MulticlassF1Score(num_classes=4, average="macro")
    specificity_macro = Specificity(task="multiclass", num_classes=4, average="macro")

    results = {
        "accuracy": accuracy(pred, y_true).numpy().tolist(),
        "accuracy_macro": accuracy_macro(pred, y_true).numpy().tolist(),
        "precision": precision(pred, y_true).numpy().tolist(),
        "precision_macro": precision_macro(pred, y_true).numpy().tolist(),
        "recall": recall(pred, y_true).numpy().tolist(),
        "recall_macro": recall_macro(pred, y_true).numpy().tolist(),
        "f1_score": f1_score(pred, y_true).numpy().tolist(),
        "f1_score_macro": f1_score_macro(pred, y_true).numpy().tolist(),
        "specificity": specificity(pred, y_true).numpy().tolist(),
        "specificity_macro": specificity_macro(pred, y_true).numpy().tolist()
    }

    if logits:
        roc_auc = MulticlassAUROC(num_classes=4, average=None)
        roc_auc_macro = MulticlassAUROC(num_classes=4, average="macro")
        results["roc_auc"] = roc_auc(pred, y_true).numpy().tolist()
        results["roc_auc_macro"] = roc_auc_macro(pred, y_true).numpy().tolist()

    # accuracy from MultiClassAccuracy is the same as MulticlassRecall, so I calculate it manually
    pred = pred.numpy()
    y_true = y_true.numpy().squeeze()
    if logits:
        pred = np.argmax(pred, axis=-1)
    else:
        pred = pred.squeeze()
    accuracy = []
    for i in range(4):
        TP = np.sum((pred == i) & (y_true == i))
        TN = np.sum((pred != i) & (y_true != i))
        accuracy.append((TP + TN) / np.shape(y_true)[0])
    results["accuracy"] = accuracy

    return results
    

def test_img(model, files, fold, save_dir):
    dm = HelicoidDataModule(files=files, fold=fold)
    dm.setup("predict")
    dataloaders, img_shapes, img_ids = dm.predict_dataloader()
    for dataloader, img_shape, img_id in zip(dataloaders, img_shapes, img_ids):
        logits, y_true = get_predictions(model, dataloader)

        pred = np.argmax(logits, axis=-1)
        pred_img = pred.reshape(img_shape[0], img_shape[1])

        # visualize the prediction
        plt.figure()
        class_colors = [tum_blue_dark_2, tum_orange, tum_red, tum_grey_1]
        cmap = LinearSegmentedColormap.from_list("custom", class_colors, N=4)
        im = plt.imshow(pred_img, cmap, interpolation="none")
        plt.axis('off')
        plt.savefig(os.path.join(save_dir, f"{img_id}_prediction.png"), dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()

        # tumor heatmap
        plt.figure()

        im = plt.imshow(logits[:,1].reshape(img_shape[0],img_shape[1]), cmap=tum_cmap)
        plt.axis('off')
        plt.savefig(os.path.join(save_dir, f"{img_id}_prediction_tumor.png"), dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()

        # do majority voting within a 3x3 window
        window_size = 3
        pred_img_padded = np.pad(pred_img, ((window_size//2, window_size//2), (window_size//2, window_size//2)), mode="edge")
        pred_img_knn = np.zeros_like(pred_img)
        for i in range(window_size//2, img_shape[0]):
            for j in range(window_size//2, img_shape[1]):
                window = pred_img_padded[i-window_size//2:i+window_size//2+1, j-window_size//2:j+window_size//2+1]
                window_class_counts = [np.sum(window == c, axis=(0,1)) for c in range(0,4)]
                window_class = np.argmax(window_class_counts)
                pred_img_knn[i, j] = window_class

        # visualize the prediction
        plt.figure()
        im = plt.imshow(pred_img_knn, cmap, interpolation="none")
        plt.axis('off')
        plt.savefig(os.path.join(save_dir, f"{img_id}_prediction_knn.png"), dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()

        # convert to long tensor
        y_pred_knn = torch.LongTensor(pred_img_knn.flatten())
        idx = np.argwhere(y_true >= 0)
        metrics = get_metrics(y_pred_knn[idx], y_true[idx], logits=False)

        metrics["label_counts"] = [int(np.sum(y_true.cpu().numpy() == i)) for i in range(4)]
        print(metrics)

        # save results as json
        os.makedirs(os.path.join(save_dir, "knn_metrics"), exist_ok=True)
        with open(os.path.join(save_dir, "knn_metrics", f"{img_id}_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=4)


def test_lableled(model, files, fold, save_dir):
    dm_test = HelicoidDataModule(files=files, fold=fold)
    dm_test.setup("test")
    dataloader = dm_test.test_dataloader()
    logits, y_true = get_predictions(model, dataloader)

    metrics = get_metrics(logits, y_true)
    # save results as json
    with open(os.path.join(save_dir, f"{fold}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)


def main():

    if args.mode == "baseline":
        files = ["preprocessed.npy"]
    elif args.mode == "heatmap":
        # files = ["preprocessed.npy", "heatmaps_osp.npy", "heatmaps_osp_diff.npy", "heatmaps_osp_diff_mc.npy", "heatmaps_icem.npy", "heatmaps_icem_diff.npy", "heatmaps_icem_diff_mc.npy"]
        files = ["preprocessed.npy", "osp_absolute.npy", "osp_rel_lit.npy", "osp_rel_mc.npy", "cem_absolute.npy", "cem_rel_lit.npy", "cem_rel_mc.npy"]
    elif args.mode == "heatmap_only":
        # files = ["heatmaps_osp.npy", "heatmaps_osp_diff.npy", "heatmaps_osp_diff_mc.npy", "heatmaps_icem.npy", "heatmaps_icem_diff.npy", "heatmaps_icem_diff_mc.npy"]
        files = ["osp_absolute.npy", "osp_rel_lit.npy", "osp_rel_mc.npy", "cem_absolute.npy", "cem_rel_lit.npy", "cem_rel_mc.npy"]
    # elif args.mode == "baseline_reduced":
    #     files = ["preprocessed_reduced"]
    
    for fold in args.folds:

        checkpoint_path = os.path.join(args.log_dir, f"{fold}.ckpt")
        save_dir = os.path.join(args.log_dir, "results")
        os.makedirs(save_dir, exist_ok=True)

        model = ClassificationModel.load_from_checkpoint(checkpoint_path)
        model.eval()

        # visualize_weights(model, save_dir)

        test_lableled(model, files, fold, save_dir)

        # test_img(model, files, fold, save_dir)


if __name__ == "__main__":
    main()