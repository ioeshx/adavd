CUDA_VISIBLE_DEVICES=0 python src/main_multi.py \
    --erase_type '3style' \
    --target_concept 'Van Gogh, Claude Monet, Pablo Picasso' \
    --contents 'erase' \
    --mode 'retain' \
    --num_samples 1 --batch_size 1 \
    --save_root "results/3style"

CUDA_VISIBLE_DEVICES=0 python src/main.py \
    --erase_type 'style' \
    --target_concept 'Van Gogh' \
    --contents 'Van Gogh, Kelly McKernan' \
    --mode 'retain' \
    --num_samples 1 --batch_size 1 \
    --save_root "results" \

mv results/Van\ Gogh/Van\ Gogh/original/* results/Van\ Gogh/origin
mv results/Van\ Gogh/Van\ Gogh/retain/* results/Van\ Gogh/erased

mv results/Van\ Gogh/Kelly\ McKernan/original/* results/Van\ Gogh/origin
mv results/Van\ Gogh/Kelly\ McKernan/retain/* results/Van\ Gogh/erased


python src/cal_lpips.py --dir1 results/Kelly\ McKernan/erased --dir2 results/Kelly\ McKernan/origin

python src/cal_lpips.py --dir1 results/Van\ Gogh/erased --dir2 results/Van\ Gogh/origin
