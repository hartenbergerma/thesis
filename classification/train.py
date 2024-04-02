import argparse
import lightning.pytorch as pl
import matplotlib.pyplot as plt

from model import ClassificationModel
from dataloader import HelicoidDataModule
from hyperspectral_transforms import *
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint


parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, required=True, help="Training mode: baseline, baseline_reduced, heatmap or heatmap_only", choices=["baseline", "heatmap", "baseline_reduced", "heatmap_only"])
# add mandatory argument for log_dir
parser.add_argument("--log_dir", type=str, required=True, help="Directory to save logs")
parser.add_argument("--folds", nargs='+', type=str, required=True, help="Fold to use for training", choices=["fold1", "fold2", "fold3", "fold4", "fold5"])
parser.add_argument("--hidden_dim", type=int, required=True, help="Hidden dimension of the model")
parser.add_argument("--num_layers", type=int, required=True, help="Number of hidden layers")
parser.add_argument("--last_layer_dim", type=int, required=True, help="Dimension of the last hidden layer")
parser.add_argument("--lr", type=float, required=True, help="Learning rate")
parser.add_argument("--weight_decay", type=float, required=True, help="Weight decay")
args = parser.parse_args()


def train(config, dm):
    logger = TensorBoardLogger(config["log_dir"], name="my_model")
    model = ClassificationModel(input_dim=dm.sample_size(), output_dim=dm.num_classes(), loss_weight=dm.class_distribution(), config=config)

    early_stop_callback = EarlyStopping(monitor="val/val_loss", mode="min", min_delta=0.0, patience=5, verbose=False)
    checkpoint_callback = ModelCheckpoint(monitor="val/val_loss", mode="min", save_top_k=1, dirpath=config["log_dir"], filename=f"{dm.get_fold()}")
    trainer = pl.Trainer(logger=logger, max_epochs=config["num_epochs"], devices=1, callbacks=[early_stop_callback, checkpoint_callback])

    trainer.fit(model, dm.train_dataloader(), dm.val_dataloader())

    # log hyperparameters and metrics
    best_val_loss = early_stop_callback.best_score
    params = {
        "hidden_dim": config["hidden_dim"],
        "num_layers": config["num_layers"],
        "last_layer_dim": config["last_layer_dim"],
        "lr": config["lr"],
        "weight_decay": config["weight_decay"],
    }
    metrics = {"best_val_loss": best_val_loss}
    logger.log_hyperparams(params, metrics)

    return model


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
        dm = HelicoidDataModule(files=files, fold=fold)
        dm.setup("fit")

        config = {
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "last_layer_dim": args.last_layer_dim,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "num_epochs": 100,
            "log_dir": args.log_dir,
        }

        train(config, dm)

if __name__ == "__main__":
    main()

