export HF_ENDPOINT=https://hf-mirror.com

CUDA_VISIBLE_DEVICES=1 python src/main.py \
    --erase_type 'style' \
    --target_concept 'Kelly McKernan' \
    --contents 'Van Gogh, Kelly McKernan' \
    --mode 'original, retain' \
    --num_samples 1 --batch_size 1 \
    --save_root "results"