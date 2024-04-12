CUDA_VISIBLE_DEVICES=7 python tuning.py \
    --mode baseline \
    --log_dir ./logs4/bl

CUDA_VISIBLE_DEVICES=0 python tuning.py \
    --mode heatmap \
    --log_dir ./logs4/hm

CUDA_VISIBLE_DEVICES=3 python tuning.py \
    --mode heatmap_only \
    --log_dir ./logs4/hm_only

CUDA_VISIBLE_DEVICES=5 python tuning.py \
    --mode baseline_reduced \
    --log_dir ./logs/bl_reduced