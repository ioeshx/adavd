export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=1

python src/main.py \
    --erase_type 'style' \
    --target_concept 'Van Gogh' \
    --contents 'Van Gogh, Monet, Picasso' \
    --mode 'original, erase, retain' \
    --num_samples 10 --batch_size 10 \
    --save_root "results/compare-style"

python src/main.py \
    --erase_type 'style' \
    --target_concept 'Monet' \
    --contents 'Van Gogh, Monet, Picasso' \
    --mode 'original, erase, retain' \
    --num_samples 10 --batch_size 10 \
    --save_root "results/compare-style"

python src/main.py \
    --erase_type 'style' \
    --target_concept 'Picasso' \
    --contents 'Van Gogh, Monet, Picasso' \
    --mode 'original, erase, retain' \
    --num_samples 10 --batch_size 10 \
    --save_root "results/compare-style"