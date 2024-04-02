### Baseline (just spectra)

CUDA_VISIBLE_DEVICES=7 python train.py \
    --mode baseline \
    --log_dir models/baseline/v1 \
    --folds fold1 fold2 fold3 fold4 fold5 \
    --hidden_dim 4 \
    --num_layers 3 \
    --last_layer_dim 4 \
    --lr 0.000049488 \
    --weight_decay 0.0037483

CUDA_VISIBLE_DEVICES=7 python train.py \
    --mode baseline \
    --log_dir models/baseline/v2 \
    --folds fold1 fold2 fold3 fold4 fold5 \
    --hidden_dim 17 \
    --num_layers 0 \
    --last_layer_dim 12 \
    --lr 0.0000061972 \
    --weight_decay 0.0018549

CUDA_VISIBLE_DEVICES=7 python train.py \
    --mode baseline \
    --log_dir models/baseline/v3 \
    --folds fold1 fold2 fold3 fold4 fold5 \
    --hidden_dim 19 \
    --num_layers 3 \
    --last_layer_dim 10 \
    --lr 0.000015197 \
    --weight_decay 0.00051274


### Baseline with heatmaps

CUDA_VISIBLE_DEVICES=5 python train.py \
    --mode heatmap \
    --log_dir models/heatmaps/v1 \
    --folds fold1 fold2 fold3 fold4 fold5 \
    --hidden_dim 9 \
    --num_layers 0 \
    --last_layer_dim 8 \
    --lr 0.000063855 \
    --weight_decay 0.00014437

CUDA_VISIBLE_DEVICES=5 python train.py \
    --mode heatmap \
    --log_dir models/heatmaps/v2 \
    --folds fold1 fold2 fold3 fold4 fold5 \
    --hidden_dim 20 \
    --num_layers 0 \
    --last_layer_dim 5 \
    --lr 0.000010001 \
    --weight_decay 0.0022435

CUDA_VISIBLE_DEVICES=5 python train.py \
    --mode heatmap \
    --log_dir models/heatmaps/v3 \
    --folds fold1 fold2 fold3 fold4 fold5 \
    --hidden_dim 25 \
    --num_layers 0 \
    --last_layer_dim 21 \
    --lr 0.000079276 \
    --weight_decay 0.0021629


### Baseline with reduced band count

CUDA_VISIBLE_DEVICES=1 python train.py \
    --mode baseline_reduced \
    --log_dir models/baseline_red/v1 \
    --folds fold1 fold2 fold3 fold4 fold5 \
    --hidden_dim 9 \
    --num_layers 0 \
    --last_layer_dim 8 \
    --lr 0.0000084509 \
    --weight_decay 0.000032581

CUDA_VISIBLE_DEVICES=1 python train.py \
    --mode baseline_reduced \
    --log_dir models/baseline_red/v2 \
    --folds fold1 fold2 fold3 fold4 fold5 \
    --hidden_dim 10 \
    --num_layers 0 \
    --last_layer_dim 4 \
    --lr 0.0000072034 \
    --weight_decay 0.00071457

CUDA_VISIBLE_DEVICES=1 python train.py \
    --mode baseline_reduced \
    --log_dir models/baseline_red/v3 \
    --folds fold1 fold2 fold3 fold4 fold5 \
    --hidden_dim 7 \
    --num_layers 2 \
    --last_layer_dim 4 \
    --lr 0.000060792 \
    --weight_decay 0.00020305


### Heatmaps only

CUDA_VISIBLE_DEVICES=7 python train.py \
    --mode heatmap_only \
    --log_dir models/heatmap_only/v1 \
    --folds fold1 fold2 fold3 fold4 fold5 \
    --hidden_dim 24 \
    --num_layers 3 \
    --last_layer_dim 7 \
    --lr 0.000092289 \
    --weight_decay 0.000012580

CUDA_VISIBLE_DEVICES=7 python train.py \
    --mode heatmap_only \
    --log_dir models/heatmap_only/v2 \
    --folds fold1 fold2 fold3 fold4 fold5 \
    --hidden_dim 23 \
    --num_layers 1 \
    --last_layer_dim 12 \
    --lr 0.000016322 \
    --weight_decay 0.00018688

CUDA_VISIBLE_DEVICES=7 python train.py \
    --mode heatmap_only \
    --log_dir models/heatmap_only/v3 \
    --folds fold1 fold2 fold3 fold4 fold5 \
    --hidden_dim 25 \
    --num_layers 1 \
    --last_layer_dim 16 \
    --lr 0.0000095301 \
    --weight_decay 0.00010428