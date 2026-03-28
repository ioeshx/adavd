import argparse
import os
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import lpips


def load_image(path):
    img = Image.open(path).convert('RGB')
    transform = transforms.Compose([
        # transforms.Resize((256, 256)), # Resize to standard size for LPIPS
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Normalize to [-1, 1]
    ])
    return transform(img).unsqueeze(0)

def main():
    parser = argparse.ArgumentParser(description='Calculate LPIPS score between two directories of images.')
    parser.add_argument('--dir1', type=str, required=True, help='Path to the first directory (e.g., generated images)')
    parser.add_argument('--dir2', type=str, required=True, help='Path to the second directory (e.g., reference/original images)')
    parser.add_argument('--gpu', action='store_true', default=True, help='Use GPU if available')
    
    args = parser.parse_args()

    if args.gpu and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    
    print(f"Using device: {device}")

    # Initialize LPIPS
    loss_fn = lpips.LPIPS(net='alex').to(device)

    # List images in dir1
    original_image = sorted([f for f in os.listdir(args.dir1) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    erased_image = sorted([f for f in os.listdir(args.dir2) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    print(f"Original images: {original_image}")
    print(f"Erased images: {erased_image}")
    
    lpips_values = []
    print(f"Calculating LPIPS between:\nDir1: {args.dir1}\nDir2: {args.dir2}")

    for f in original_image:
        path1 = os.path.join(args.dir1, f)
        path2 = os.path.join(args.dir2, f)
        
        if os.path.exists(path2):
            try:
                img1 = load_image(path1).to(device)
                img2 = load_image(path2).to(device)
                
                with torch.no_grad():
                    d = loss_fn(img1, img2)
                    lpips_values.append(d.item())
            except Exception as e:
                print(f"Error processing {f}: {e}")
        else:
            # print(f"Warning: {f} not found in reference directory.")
            pass

    if lpips_values:
        print(f"\nLPIPS Statistics:")
        print(f"Average LPIPS: {np.mean(lpips_values):.4f}")
        print(f"Std Dev: {np.std(lpips_values):.4f}")
        print(f"Count: {len(lpips_values)}")
        print(f"\nTotal Average LPIPS: {np.mean(lpips_values):.4f}")
    else:
        print("No matching image pairs found.")

if __name__ == '__main__':
    main()