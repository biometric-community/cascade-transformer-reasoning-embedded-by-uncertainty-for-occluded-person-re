"""Evaluate CTU on Market-1501."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from .models import build_ctu
from .train import build_eval_loader, extract_features, load_config
from .metrics import evaluate_rank


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", default="outputs/checkpoints/best.pt")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    cfg = load_config(args.config)
    device = args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu"
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    num_classes = int(ckpt.get("num_classes", 751))
    model = build_ctu(cfg, num_classes).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    q = extract_features(model, build_eval_loader(cfg, "query"), device)
    g = extract_features(model, build_eval_loader(cfg, "gallery"), device)
    metrics = evaluate_rank(
        q["feats"], q["pids"], q["cams"], g["feats"], g["pids"], g["cams"],
        ranks=list(cfg["eval"].get("ranks", [1, 5, 10])),
    )
    payload = {"metrics": metrics, "checkpoint": args.checkpoint, "dataset": "Market-1501"}
    Path(cfg["output"]["logs"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["output"]["logs"], "eval_results.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
