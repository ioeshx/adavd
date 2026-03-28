import argparse
import os

# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import re
from typing import Dict, List, Tuple

import lpips
import numpy as np
import pandas as pd
import torch
import torch_fidelity
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer


IMAGE_EXTS = (".png", ".jpg", ".jpeg")


def list_images(folder: str) -> Dict[str, str]:
    return {
        name: os.path.join(folder, name)
        for name in os.listdir(folder)
        if name.lower().endswith(IMAGE_EXTS)
    }


def find_coco1k_roots(results_root: str, coco_subdir: str, erase_dir: str, original_dir: str) -> List[str]:
    roots = []
    for dirpath, dirnames, _ in os.walk(results_root):
        if os.path.basename(dirpath) != coco_subdir:
            continue
        if erase_dir in dirnames and original_dir in dirnames:
            roots.append(dirpath)
    return sorted(roots)


def prompt_from_filename(filename: str) -> str:
    stem = os.path.splitext(filename)[0]
    stem = re.sub(r"_\d+$", "", stem)
    return stem.replace("_", " ").strip()


def load_lpips_tensor(path: str) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    image = Image.open(path).convert("RGB")
    return transform(image).unsqueeze(0)


def calc_lpips(
    erase_files: Dict[str, str],
    original_files: Dict[str, str],
    device: str,
) -> Tuple[float, int]:
    names = sorted(set(erase_files.keys()) & set(original_files.keys()))
    if not names:
        return float("nan"), 0

    loss_fn = lpips.LPIPS(net="alex").to(device)
    values = []
    with torch.no_grad():
        for name in tqdm(names, desc="LPIPS", leave=False):
            img1 = load_lpips_tensor(erase_files[name]).to(device)
            img2 = load_lpips_tensor(original_files[name]).to(device)
            values.append(loss_fn(img1, img2).item())

    return float(np.mean(values)), len(values)


class ClipScoreCalculator:
    def __init__(self, version: str, device: str):
        self.device = device
        self.model = CLIPModel.from_pretrained(version).to(device)
        self.processor = CLIPProcessor.from_pretrained(version)
        self.tokenizer = CLIPTokenizer.from_pretrained(version)

    def score(self, image_paths: List[str], prompts: List[str], batch_size: int) -> float:
        assert len(image_paths) == len(prompts)
        if not image_paths:
            return float("nan")

        self.model.eval()
        scores = []
        with torch.no_grad():
            for start in tqdm(range(0, len(image_paths), batch_size), desc="CLIP", leave=False):
                batch_imgs = image_paths[start : start + batch_size]
                batch_txts = prompts[start : start + batch_size]

                images = [Image.open(p).convert("RGB") for p in batch_imgs]
                img_inputs = self.processor(images=images, return_tensors="pt").to(self.device)
                txt_inputs = self.tokenizer(
                    batch_txts,
                    padding=True,
                    truncation=True,
                    max_length=77,
                    return_tensors="pt",
                ).to(self.device)

                image_feats = self.model.get_image_features(**img_inputs)
                text_feats = self.model.get_text_features(**txt_inputs)
                image_feats = image_feats / image_feats.norm(dim=1, p=2, keepdim=True)
                text_feats = text_feats / text_feats.norm(dim=1, p=2, keepdim=True)
                batch_scores = (image_feats * text_feats).sum(-1)
                scores.extend(batch_scores.detach().cpu().tolist())

        return float(np.mean(scores))


def calc_fid(erase_dir: str, original_dir: str, device: str) -> float:
    metrics = torch_fidelity.calculate_metrics(
        input1=erase_dir,
        input2=original_dir,
        cuda=(device == "cuda"),
        fid=True,
        verbose=False,
    )
    return float(metrics["frechet_inception_distance"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate FID/LPIPS/CLIP Score for coco-1k erase images under results."
    )
    parser.add_argument("--results_root", type=str, default="results/compare-coco")
    parser.add_argument("--coco_subdir", type=str, default="coco-1k")
    parser.add_argument("--erase_dir", type=str, default="retain")
    parser.add_argument("--original_dir", type=str, default="original")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--clip_model", type=str, default="openai/clip-vit-large-patch14")
    parser.add_argument("--output_csv", type=str, default="output/coco1k_metrics.csv")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    roots = find_coco1k_roots(
        args.results_root,
        args.coco_subdir,
        args.erase_dir,
        args.original_dir,
    )

    if not roots:
        raise FileNotFoundError(
            f"No coco-1k roots found under {args.results_root} with {args.erase_dir}/{args.original_dir}."
        )

    print(f"Using device: {device}")
    print("Found roots:")
    for r in roots:
        print(f" - {r}")

    clip_calculator = ClipScoreCalculator(args.clip_model, device)
    all_rows = []

    for root in roots:
        erase_path = os.path.join(root, args.erase_dir)
        original_path = os.path.join(root, args.original_dir)
        erase_files = list_images(erase_path)
        original_files = list_images(original_path)
        pair_names = sorted(set(erase_files.keys()) & set(original_files.keys()))

        print(f"\nEvaluating: {root}")
        print(f"Erase images: {len(erase_files)}, Original images: {len(original_files)}, Paired: {len(pair_names)}")

        fid = calc_fid(erase_path, original_path, device)
        lpips_mean, pair_count = calc_lpips(erase_files, original_files, device)
        clip_prompts = [prompt_from_filename(n) for n in pair_names]
        clip_images = [erase_files[n] for n in pair_names]
        clip_score = clip_calculator.score(clip_images, clip_prompts, args.batch_size)

        row = {
            "root": root,
            "fid": fid,
            "lpips": lpips_mean,
            "clip_score": clip_score,
            "pair_count": pair_count,
        }
        all_rows.append(row)

        txt_path = os.path.join(root, "record_metrics_coco1k.txt")
        with open(txt_path, "a", encoding="utf-8") as f:
            f.write("***************************\n")
            f.write(f"Root: {root}\n")
            f.write(f"FID: {fid}\n")
            f.write(f"LPIPS: {lpips_mean}\n")
            f.write(f"CLIP Score: {clip_score}\n")
            f.write(f"Paired images: {pair_count}\n")

        print(f"FID={fid:.4f}, LPIPS={lpips_mean:.4f}, CLIP={clip_score * 100:.4f} (x100)")

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df = pd.DataFrame(all_rows)
    df.to_csv(args.output_csv, index=False)
    print(f"\nSaved summary CSV to: {args.output_csv}")
    print(df)


if __name__ == "__main__":
    main()
