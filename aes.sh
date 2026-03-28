export HF_ENDPOINT=https://hf-mirror.com
python cal_aesthetic_score.py \
    --image_dir "/home/shx/Code/RECE/results/3style_erased" \
    --device "cuda"

python cal_aesthetic_score.py \
    --image_dir "/home/shx/Code/RECE/results/vanSnoopy_erased" \
    --device "cuda"

python cal_aesthetic_score.py \
    --image_dir "/home/shx/Code/AdaVD/results/3style/3style/erase/retain" \
    --device "cuda"

python cal_aesthetic_score.py \
    --image_dir "/home/shx/Code/AdaVD/results/van_snoopy/van_snoopy/erase/retain" \
    --device "cuda"


    