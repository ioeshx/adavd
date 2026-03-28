export HF_ENDPOINT=https://hf-mirror.com
python cal_lpips.py \
    --dir1 "/home/shx/Code/RECE/results/other_artists_origin" \
    --dir2 "/home/shx/Code/RECE/results/other_artists_erased"

python cal_lpips.py \
    --dir1 "/home/shx/Code/RECE/results/other_artists_origin" \
    --dir2 "/home/shx/Code/AdaVD/results/other_artists/other_artists/erase/retain" 


python cal_lpips.py \
    --dir1 "/home/shx/Code/RECE/results/3style_erased" \
    --dir2 "/home/shx/Code/RECE/results/3style_origin"

python cal_lpips.py \
    --dir1 "/home/shx/Code/RECE/results/vanSnoopy_origin" \
    --dir2 "/home/shx/Code/RECE/results/vanSnoopy_erased"

python cal_lpips.py \
    --dir1 "/home/shx/Code/AdaVD/results/3style/3style/erase/retain" \
    --dir2 "/home/shx/Code/RECE/results/3style_origin"

python cal_lpips.py \
    --dir1 "/home/shx/Code/AdaVD/results/van_snoopy/van_snoopy/erase/retain" \
    --dir2 "/home/shx/Code/RECE/results/vanSnoopy_origin"
