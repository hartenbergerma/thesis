# ### Baseline (just spectra)

# CUDA_VISIBLE_DEVICES=1 python train.py \
#     --mode baseline \
#     --log_dir models/baseline/v1 \
#     --folds fold1 fold2 fold3 fold4 fold5 \
#     --hidden_dim 13 \
#     --num_layers 2 \
#     --last_layer_dim 11 \
#     --lr 0.00028839 \
#     --weight_decay 0.00012021

# CUDA_VISIBLE_DEVICES=1 python train.py \
#     --mode baseline \
#     --log_dir models/baseline/v2 \
#     --folds fold1 fold2 fold3 fold4 fold5 \
#     --hidden_dim 8 \
#     --num_layers 3 \
#     --last_layer_dim 7 \
#     --lr 0.00042422 \
#     --weight_decay 0.000078039

# CUDA_VISIBLE_DEVICES=1 python train.py \
#     --mode baseline \
#     --log_dir models/baseline/v3 \
#     --folds fold1 fold2 fold3 fold4 fold5 \
#     --hidden_dim 47 \
#     --num_layers 0 \
#     --last_layer_dim 15 \
#     --lr 0.00026821 \
#     --weight_decay 0.0057339


# ### Baseline with heatmaps

# CUDA_VISIBLE_DEVICES=2 python train.py \
#     --mode heatmap \
#     --log_dir models/heatmaps/v1 \
#     --folds fold1 fold2 fold3 fold4 fold5 \
#     --hidden_dim 31 \
#     --num_layers 0 \
#     --last_layer_dim 18 \
#     --lr 0.000044678 \
#     --weight_decay 0.00066603

# CUDA_VISIBLE_DEVICES=2 python train.py \
#     --mode heatmap \
#     --log_dir models/heatmaps/v2 \
#     --folds fold1 fold2 fold3 fold4 fold5 \
#     --hidden_dim 47 \
#     --num_layers 0 \
#     --last_layer_dim 15 \
#     --lr 0.00026821 \
#     --weight_decay 0.0057339

# CUDA_VISIBLE_DEVICES=2 python train.py \
#     --mode heatmap \
#     --log_dir models/heatmaps/v3 \
#     --folds fold1 fold2 fold3 fold4 fold5 \
#     --hidden_dim 42 \
#     --num_layers 0 \
#     --last_layer_dim 21 \
#     --lr 0.00023102 \
#     --weight_decay 0.0067802


# ### Baseline with reduced band count

# CUDA_VISIBLE_DEVICES=5 python train.py \
#     --mode baseline_reduced \
#     --log_dir models/baseline_red/v1 \
#     --folds fold1 fold2 fold3 fold4 fold5 \
#     --hidden_dim 42 \
#     --num_layers 0 \
#     --last_layer_dim 21 \
#     --lr 0.00023102 \
#     --weight_decay 0.0067802

# CUDA_VISIBLE_DEVICES=5 python train.py \
#     --mode baseline_reduced \
#     --log_dir models/baseline_red/v2 \
#     --folds fold1 fold2 fold3 fold4 fold5 \
#     --hidden_dim 47 \
#     --num_layers 0 \
#     --last_layer_dim 15 \
#     --lr 0.00026821 \
#     --weight_decay 0.0057339

# CUDA_VISIBLE_DEVICES=5 python train.py \
#     --mode baseline_reduced \
#     --log_dir models/baseline_red/v3 \
#     --folds fold1 fold2 fold3 fold4 fold5 \
#     --hidden_dim 28 \
#     --num_layers 0 \
#     --last_layer_dim 16 \
#     --lr 0.00038322 \
#     --weight_decay 0.00014136


### Heatmaps only

# CUDA_VISIBLE_DEVICES=7 python train.py \
#     --mode heatmap_only \
#     --log_dir models/heatmap_only/v1 \
#     --folds fold1 fold2 fold3 fold4 fold5 \
#     --hidden_dim 29 \
#     --num_layers 0 \
#     --last_layer_dim 29 \
#     --lr 0.00081690 \
#     --weight_decay 0.00031913

# CUDA_VISIBLE_DEVICES=7 python train.py \
#     --mode heatmap_only \
#     --log_dir models/heatmap_only/v2 \
#     --folds fold1 fold2 fold3 fold4 fold5 \
#     --hidden_dim 47 \
#     --num_layers 0 \
#     --last_layer_dim 15 \
#     --lr 0.00026821 \
#     --weight_decay 0.0057339

# CUDA_VISIBLE_DEVICES=7 python train.py \
#     --mode heatmap_only \
#     --log_dir models/heatmap_only/v3 \
#     --folds fold1 fold2 fold3 fold4 fold5 \
#     --hidden_dim 6 \
#     --num_layers 3 \
#     --last_layer_dim 6 \
#     --lr 0.00084019 \
#     --weight_decay 0.0016054


#-----------------------old models trained with unbalanced data-----------------------

# ### Baseline (just spectra)

# CUDA_VISIBLE_DEVICES=0 python train.py \
#     --mode baseline \
#     --log_dir models4/baseline/v1 \
#     --folds fold1 fold2 fold3 fold4 fold5 \
#     --hidden_dim 4 \
#     --last_layer_dim 4 \
#     --num_layers 3 \
#     --lr 0.000049488 \
#     --weight_decay 0.0037483

# CUDA_VISIBLE_DEVICES=0 python train.py \
#     --mode baseline \
#     --log_dir models4/baseline/v2 \
#     --folds fold1 fold2 fold3 fold4 fold5 \
#     --hidden_dim 17 \
#     --last_layer_dim 12 \
#     --num_layers 0 \
#     --lr 0.0000061972 \
#     --weight_decay 0.0018549

# CUDA_VISIBLE_DEVICES=0 python train.py \
#     --mode baseline \
#     --log_dir models4/baseline/v3 \
#     --folds fold1 fold2 fold3 fold4 fold5 \
#     --hidden_dim 19 \
#     --last_layer_dim 10 \
#     --num_layers 3 \
#     --lr 0.000015197 \
#     --weight_decay 0.00051274


# ### Baseline with heatmaps

# CUDA_VISIBLE_DEVICES=0 python train.py \
#     --mode heatmap \
#     --log_dir models4/heatmaps/v1 \
#     --folds fold1 fold2 fold3 fold4 fold5 \
#     --hidden_dim 9 \
#     --last_layer_dim 8 \
#     --num_layers 0 \
#     --lr 0.000063855 \
#     --weight_decay 0.00014437

# CUDA_VISIBLE_DEVICES=0 python train.py \
#     --mode heatmap \
#     --log_dir models4/heatmaps/v2 \
#     --folds fold1 fold2 fold3 fold4 fold5 \
#     --hidden_dim 20 \
#     --last_layer_dim 5 \
#     --num_layers 0 \
#     --lr 0.000010001 \
#     --weight_decay 0.0022435

# CUDA_VISIBLE_DEVICES=0 python train.py \
#     --mode heatmap \
#     --log_dir models4/heatmaps/v3 \
#     --folds fold1 fold2 fold3 fold4 fold5 \
#     --hidden_dim 25 \
#     --last_layer_dim 21 \
#     --num_layers 0 \
#     --lr 0.000079276 \
#     --weight_decay 0.0021629


# ### Baseline with reduced band count

# CUDA_VISIBLE_DEVICES=1 python train.py \
#     --mode baseline_reduced \
#     --log_dir models/baseline_red/v1 \
#     --folds fold1 fold2 fold3 fold4 fold5 \
#     --hidden_dim 9 \
#     --num_layers 0 \
#     --last_layer_dim 8 \
#     --lr 0.0000084509 \
#     --weight_decay 0.000032581

# CUDA_VISIBLE_DEVICES=1 python train.py \
#     --mode baseline_reduced \
#     --log_dir models/baseline_red/v2 \
#     --folds fold1 fold2 fold3 fold4 fold5 \
#     --hidden_dim 10 \
#     --num_layers 0 \
#     --last_layer_dim 4 \
#     --lr 0.0000072034 \
#     --weight_decay 0.00071457

# CUDA_VISIBLE_DEVICES=1 python train.py \
#     --mode baseline_reduced \
#     --log_dir models/baseline_red/v3 \
#     --folds fold1 fold2 fold3 fold4 fold5 \
#     --hidden_dim 7 \
#     --num_layers 2 \
#     --last_layer_dim 4 \
#     --lr 0.000060792 \
#     --weight_decay 0.00020305


# ### Heatmaps only

# CUDA_VISIBLE_DEVICES=7 python train.py \
#     --mode heatmap_only \
#     --log_dir models/heatmap_only/v1 \
#     --folds fold1 fold2 fold3 fold4 fold5 \
#     --hidden_dim 24 \
#     --num_layers 3 \
#     --last_layer_dim 7 \
#     --lr 0.000092289 \
#     --weight_decay 0.000012580

# CUDA_VISIBLE_DEVICES=7 python train.py \
#     --mode heatmap_only \
#     --log_dir models/heatmap_only/v2 \
#     --folds fold1 fold2 fold3 fold4 fold5 \
#     --hidden_dim 23 \
#     --num_layers 1 \
#     --last_layer_dim 12 \
#     --lr 0.000016322 \
#     --weight_decay 0.00018688

# CUDA_VISIBLE_DEVICES=7 python train.py \
#     --mode heatmap_only \
#     --log_dir models/heatmap_only/v3 \
#     --folds fold1 fold2 fold3 fold4 fold5 \
#     --hidden_dim 25 \
#     --num_layers 1 \
#     --last_layer_dim 16 \
#     --lr 0.0000095301 \
#     --weight_decay 0.00010428





### Baseline (just spectra)

CUDA_VISIBLE_DEVICES=0 python train.py \
    --mode baseline \
    --log_dir models5/baseline/v1 \
    --folds fold1 fold2 fold3 fold4 fold5 \
    --hidden_dim 4 \
    --last_layer_dim 4 \
    --num_layers 3 \
    --lr 5.612365099322873e-05 \
    --weight_decay 0.000714173793104537 \
    --batch_size 32

CUDA_VISIBLE_DEVICES=1 python train.py \
    --mode baseline \
    --log_dir models5/baseline/v2 \
    --folds fold1 fold2 fold3 fold4 fold5 \
    --hidden_dim 4 \
    --last_layer_dim 4 \
    --num_layers 2 \
    --lr 1.489449797127195e-05 \
    --weight_decay 5.4046235343238e-05 \
    --batch_size 32

CUDA_VISIBLE_DEVICES=1 python train.py \
    --mode baseline \
    --log_dir models5/baseline/v3 \
    --folds fold1 fold2 fold3 fold4 fold5 \
    --hidden_dim 4 \
    --last_layer_dim 4 \
    --num_layers 0 \
    --lr 1.6756813861488494e-05 \
    --weight_decay 0.0029548945587266834 \
    --batch_size 32


# ### Baseline with heatmaps

CUDA_VISIBLE_DEVICES=2 python train.py \
    --mode heatmap \
    --log_dir models5/heatmaps/v1 \
    --folds fold1 fold2 fold3 fold4 fold5 \
    --hidden_dim 43 \
    --last_layer_dim 39 \
    --num_layers 0 \
    --lr 3.174596194097272e-05 \
    --weight_decay 0.02865007549783739 \
    --batch_size 64

CUDA_VISIBLE_DEVICES=3 python train.py \
    --mode heatmap \
    --log_dir models5/heatmaps/v2 \
    --folds fold1 fold2 fold3 fold4 fold5 \
    --hidden_dim 47 \
    --last_layer_dim 15 \
    --num_layers 0 \
    --lr 1.969465059629719e-05 \
    --weight_decay 0.023408216129688644 \
    --batch_size 32

CUDA_VISIBLE_DEVICES=3 python train.py \
    --mode heatmap \
    --log_dir models5/heatmaps/v3 \
    --folds fold1 fold2 fold3 fold4 fold5 \
    --hidden_dim 9 \
    --last_layer_dim 7 \
    --num_layers 3 \
    --lr 6.901845408537136e-05 \
    --weight_decay 2.517141032510143e-05 \
    --batch_size 32