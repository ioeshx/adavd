import os
import argparse
import torch
from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip
from pathlib import Path
from PIL import Image
import numpy as np


args = argparse.ArgumentParser(description="Calculate Aesthetic Score for a given image.")
args.add_argument("--image_dir", type=str, required=True, help="Path to the image to evaluate")
args.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
args = args.parse_args()

# load model and preprocessor
model, preprocessor = convert_v2_5_from_siglip(
    low_cpu_mem_usage=True,
    trust_remote_code=True,)
model = model.to(torch.bfloat16).to(args.device)

scores = []
files = os.listdir(args.image_dir)
for f in files:
    img_path = os.path.join(args.image_dir, f)
    image = Image.open(img_path).convert("RGB")
    # preprocess image
    pixel_values = (
        preprocessor(images=image, return_tensors="pt")
        .pixel_values.to(torch.bfloat16)
        .to(args.device)
    )
    # predict aesthetic score
    with torch.inference_mode():
        score = model(pixel_values).logits.squeeze().float().cpu().numpy()
    scores.append(score)

print(f"""
Aesthetic Score:
Average = {np.mean(scores):.4f}
Std = {np.std(scores):.4f}
Total = {len(scores)}
""")