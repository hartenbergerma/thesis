# CUDA_VISIBLE_DEVICES=0 python test.py \
#     --mode heatmap_only \
#     --log_dir ./models/heatmap_only/v1 \
#     --folds fold2

# CUDA_VISIBLE_DEVICES=0 python test.py \
#     --mode heatmap \
#     --log_dir ./models/heatmaps/v1 \
#     --folds fold1 fold2 fold3

CUDA_VISIBLE_DEVICES=0 python test.py \
    --mode baseline \
    --log_dir ./models/baseline/v1 \
    --folds fold1 fold2

# DEVICES=0 python test.py \
#     --mode baseline_reduced \
#     --log_dir ./models/baseline_red/v1 \
#     --folds fold1 fold2