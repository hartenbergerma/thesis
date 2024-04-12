### Baseline (just spectra)

CUDA_VISIBLE_DEVICES=1 python test.py \
    --mode baseline \
    --log_dir ./models5/baseline/v1 \
    --folds fold1 fold2 fold3 fold4 fold5

CUDA_VISIBLE_DEVICES=1 python test.py \
    --mode baseline \
    --log_dir ./models5/baseline/v2 \
    --folds fold1 fold2 fold3 fold4 fold5

CUDA_VISIBLE_DEVICES=1 python test.py \
    --mode baseline \
    --log_dir ./models5/baseline/v3 \
    --folds fold1 fold2 fold3 fold4 fold5


### Baseline with heatmaps

CUDA_VISIBLE_DEVICES=0 python test.py \
    --mode heatmap \
    --log_dir ./models5/heatmaps/v1 \
    --folds fold1 fold2 fold3 fold4 fold5

CUDA_VISIBLE_DEVICES=1 python test.py \
    --mode heatmap \
    --log_dir ./models5/heatmaps/v2 \
    --folds fold1 fold2 fold3 fold4 fold5

CUDA_VISIBLE_DEVICES=1 python test.py \
    --mode heatmap \
    --log_dir ./models5/heatmaps/v3 \
    --folds fold1 fold2 fold3 fold4 fold5
    

### Baseline with reduced band count

DEVICES=0 python test.py \
    --mode baseline_reduced \
    --log_dir ./models/baseline_red/v1 \
    --folds fold1 fold2 fold3 fold4 fold5

DEVICES=0 python test.py \
    --mode baseline_reduced \
    --log_dir ./models/baseline_red/v2 \
    --folds fold1 fold2 fold3 fold4 fold5

DEVICES=0 python test.py \
    --mode baseline_reduced \
    --log_dir ./models/baseline_red/v3 \
    --folds fold1 fold2 fold3 fold4 fold5


### Heatmaps only

CUDA_VISIBLE_DEVICES=0 python test.py \
    --mode heatmap_only \
    --log_dir ./models/heatmap_only/v1 \
    --folds fold1 fold2 fold3 fold4 fold5

CUDA_VISIBLE_DEVICES=0 python test.py \
    --mode heatmap_only \
    --log_dir ./models/heatmap_only/v2 \
    --folds fold1 fold2 fold3 fold4 fold5

CUDA_VISIBLE_DEVICES=0 python test.py \
    --mode heatmap_only \
    --log_dir ./models/heatmap_only/v3 \
    --folds fold1 fold2 fold3 fold4 fold5