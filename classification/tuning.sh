CUDA_VISIBLE_DEVICES=3 python tuning.py \
    --mode baseline \
    --log_dir ./logs7/bl

CUDA_VISIBLE_DEVICES=7 python tuning.py \
    --mode heatmap \
    --log_dir ./logs7/hm

CUDA_VISIBLE_DEVICES=6 python tuning.py \
    --mode heatmap_only \
    --log_dir ./logs7/hm_only

CUDA_VISIBLE_DEVICES=5 python tuning.py \
    --mode baseline_reduced \
    --log_dir ./logs/bl_reduced