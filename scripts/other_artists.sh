export HF_ENDPOINT=https://hf-mirror.com
CUDA_VISIBLE_DEVICES=0 python src/main_multi.py \
    --erase_type 'other_artists' \
    --target_concept 'Van Gogh, Claude Monet, Pablo Picasso' \
    --contents 'erase' \
    --mode 'retain' \
    --num_samples 1 --batch_size 1 \
    --save_root "results/other_artists"
