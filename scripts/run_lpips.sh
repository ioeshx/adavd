for targets in "CIFAR100_top50" "CIFAR100_top75" "CIFAR100_top100" ; do
    echo "Calculating LPIPS for $targets"
    origin_path="results/compare-coco/${targets}/coco-1k/original"
    erased_path="results/compare-coco/${targets}/coco-1k/erase"

    python cal_lpips.py \
        --dir1 "${origin_path}" \
        --dir2 "${erased_path}" \

done