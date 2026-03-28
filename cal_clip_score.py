import os
import torch
import pandas as pd
import argparse
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

def get_image_path(image_dir, filename_base):
    # Try common extensions
    for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
        path = os.path.join(image_dir, f"{filename_base}{ext}")
        if os.path.exists(path):
            return path
    return None

def main():
    parser = argparse.ArgumentParser(description="Calculate CLIP Score for images in a directory corresponding to prompts in a CSV.")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the CSV file containing prompts")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing the generated images")
    parser.add_argument("--model_id", type=str, default="openai/clip-vit-large-patch14", help="HuggingFace model ID for CLIP")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference")
    parser.add_argument("--prompt_col", type=str, default="prompt", help="Column name for prompts in CSV")
    parser.add_argument("--filename_col", type=str, default="idx", help="Column name for image IDs/filenames in CSV")
    parser.add_argument("--output_file", type=str, default="clip_scores.json", help="Path to save results")
    
    args = parser.parse_args()
    
    print(f"Loading CSV from {args.csv_path}...")
    df = pd.read_csv(args.csv_path)
    
    if args.prompt_col not in df.columns:
        raise ValueError(f"Prompt column '{args.prompt_col}' not found in CSV. Available columns: {df.columns.tolist()}")
    
    # Filter rows where image exists
    valid_data = []
    print(f"Checking for images in {args.image_dir}...")
    
    for _, row in df.iterrows():
        file_id = row[args.filename_col]
        # Handle case where file_id might be an int or float in CSV but string in filesystem
        if isinstance(file_id, float):
            file_id = int(file_id)
        file_id = str(file_id)
        
        img_path = get_image_path(args.image_dir, file_id)
        if img_path:
            valid_data.append({
                "prompt": row[args.prompt_col],
                "image_path": img_path,
                "id": file_id
            })
        # Note: If image not found, we skip it. You might want to log this.
    
    print(f"Found {len(valid_data)} valid image-prompt pairs out of {len(df)} entries.")
    
    if len(valid_data) == 0:
        print("No images found. Exiting.")
        return

    print(f"Loading CLIP model: {args.model_id}...")
    model = CLIPModel.from_pretrained(args.model_id).to(args.device)
    processor = CLIPProcessor.from_pretrained(args.model_id)
    model.eval()
    
    scores = []
    all_results = []
    
    # Process in batches
    for i in tqdm(range(0, len(valid_data), args.batch_size), desc="Calculating CLIP Scores"):
        batch = valid_data[i : i + args.batch_size]
        
        prompts = [item["prompt"] for item in batch]
        # Truncate prompts if too long (77 tokens is CLIP limit)
        # processor handles truncation automatically usually, but good to be aware
        
        images = []
        for item in batch:
            try:
                img = Image.open(item["image_path"]).convert("RGB")
                images.append(img)
            except Exception as e:
                print(f"Error opening {item['image_path']}: {e}")
                # Use a dummy black image to keep batch size consistent or robust handling
                images.append(Image.new('RGB', (224, 224), (0, 0, 0)))

        try:
            inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True, truncation=True, max_length=77).to(args.device)
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            # CLIP 'logits_per_image' are cosine similarities * 100 usually
            # But standard metric is often just cosine similarity.
            # Let's compute cosine similarity manually to be sure.
            
            image_embeds = outputs.image_embeds
            text_embeds = outputs.text_embeds
            
            # Normalize
            image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
            text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
            
            # Diagonal contains the pairs we care about (image_i matches prompt_i)
            # We want element-wise dot product
            cosine_sims = (image_embeds * text_embeds).sum(dim=-1).cpu().numpy()
            
            # CLIP score often scaled by 100 in papers or 2.5 in Stable Diffusion eval
            # Here we store raw cosine similarity (0-1 range typically)
            
            for j, score in enumerate(cosine_sims):
                scores.append(float(score))
                all_results.append({
                    "id": batch[j]["id"],
                    "prompt": batch[j]["prompt"],
                    "clip_score": float(score)
                })
                
        except Exception as e:
            print(f"Error processing batch {i}: {e}")

    if not scores:
        print("No scores calculated.")
        return

    avg_score = np.mean(scores)
    std_score = np.std(scores)
    
    print("\n==========================================")
    print(f"Average CLIP Score (Cosine Sim): {avg_score:.4f}")
    print(f"Standard Deviation: {std_score:.4f}")
    print(f"Total processed: {len(scores)}")
    print("==========================================\n")
    
    # Save output
    output_data = {
        "summary": {
            "average_clip_score": float(avg_score),
            "std_clip_score": float(std_score),
            "image_dir": args.image_dir,
            "csv_path": args.csv_path,
            "model_id": args.model_id
        },
        "per_sample": all_results
    }
    
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(output_data, f, indent=4)
        print(f"Detailed results saved to {args.output_file}")

if __name__ == "__main__":
    main()
