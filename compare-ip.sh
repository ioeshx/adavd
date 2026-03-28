export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0

python src/main.py \
    --erase_type 'instance' \
    --target_concept 'Snoopy' \
    --contents 'Snoopy, Mickey, Spongebob, Pikachu, Dog, Legislator' \
    --mode 'original, erase, retain' \
    --num_samples 10 --batch_size 10 \
    --save_root "results/compare-ip"

python src/main.py \
    --erase_type 'instance' \
    --target_concept 'Snoopy, Mickey' \
    --contents 'Snoopy, Mickey, Spongebob, Pikachu, Dog, Legislator' \
    --mode 'original, erase, retain' \
    --num_samples 10 --batch_size 10 \
    --save_root "results/compare-ip"

python src/main.py \
    --erase_type 'instance' \
    --target_concept 'Snoopy, Mickey, Spongebob' \
    --contents 'Snoopy, Mickey, Spongebob, Pikachu, Dog, Legislator' \
    --mode 'original, erase, retain' \
    --num_samples 10 --batch_size 10 \
    --save_root "results/compare-ip"
