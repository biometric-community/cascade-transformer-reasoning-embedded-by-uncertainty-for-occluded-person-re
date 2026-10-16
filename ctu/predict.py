"""Encode one image with CTU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from PIL import Image
from torchvision import transforms

from .models import build_ctu


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", default="outputs/checkpoints/best.pt")
    ap.add_argument("--image", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    device = args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu"
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    num_classes = int(ckpt.get("num_classes", 751))
    model = build_ctu(cfg, num_classes)
    model.load_state_dict(ckpt["model"], strict=False)
    model.to(device).eval()
    h, w = int(cfg["data"]["height"]), int(cfg["data"]["width"])
    tfm = transforms.Compose(
        [
            transforms.Resize((h, w)),
            transforms.ToTensor(),
            transforms.Normalize(cfg["data"]["mean"], cfg["data"]["std"]),
        ]
    )
    img = tfm(Image.open(args.image).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model(img, train=False)["feat"].cpu().squeeze(0)
    out = {"image": args.image, "dim": int(feat.numel())}
    Path("outputs/logs").mkdir(parents=True, exist_ok=True)
    Path("outputs/logs/predict.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
