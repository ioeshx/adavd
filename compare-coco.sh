export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=1

python src/main-coco.py \
  --save_root results/coco1k \
  --mode "original,erase,retain" \
  --guidance_scale 7.5 \
  --total_timesteps 20 \
  --num_samples 1 \
  --batch_size 1 \
  --target_cifar_topk 75 \
  --prompt_source coco-1k \
  --coco_prompt_path data/coco-1k.csv \
  --coco_prompt_column text \
  --save_root "results/compare-coco" \
  --target_record_batch_size 4


python src/main-coco.py  \
  --save_root results/coco1k \
  --mode "original,erase,retain" \
  --guidance_scale 7.5 \
  --total_timesteps 20 \
  --num_samples 1 \
  --batch_size 1 \
  --target_cifar_topk 50 \
  --prompt_source coco-1k \
  --coco_prompt_path data/coco-1k.csv \
  --coco_prompt_column text \
  --save_root "results/compare-coco" \
  --target_record_batch_size 4

python src/main-coco.py  \
  --save_root results/coco1k \
  --mode "original,erase,retain" \
  --guidance_scale 7.5 \
  --total_timesteps 20 \
  --num_samples 1 \
  --batch_size 1 \
  --target_cifar_topk 100 \
  --prompt_source coco-1k \
  --coco_prompt_path data/coco-1k.csv \
  --coco_prompt_column text \
  --save_root "results/compare-coco" \
  --target_record_batch_size 4