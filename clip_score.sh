export HF_ENDPOINT=https://hf-mirror.com



python cal_clip_score.py \
    --csv_path "/home/shx/Code/AdaVD/data/3style.csv" \
    --image_dir "/home/shx/Code/RECE/results/3style_erased" \
    --output_file "output/style3_rece_clip_score.txt" \
    --device "cuda:1"


python cal_clip_score.py \
    --csv_path "/home/shx/Code/AdaVD/data/van_snoopy.csv" \
    --image_dir "/home/shx/Code/RECE/results/vanSnoopy_erased" \
    --output_file "output/van_snoopy_rece_clip_score.txt" \
    --device "cuda:1"


python cal_clip_score.py \
    --csv_path "/home/shx/Code/AdaVD/data/3style.csv" \
    --image_dir "/home/shx/Code/AdaVD/results/3style/3style/erase/retain" \
    --output_file "output/style3_AdaVD_clip_score.txt" \
    --device "cuda:1"

python cal_clip_score.py \
    --csv_path "/home/shx/Code/AdaVD/data/van_snoopy.csv" \
    --image_dir "/home/shx/Code/AdaVD/results/van_snoopy/van_snoopy/erase/retain" \
    --output_file "output/van_snoopy_AdaVD_clip_score.txt" \
    --device "cuda:1"