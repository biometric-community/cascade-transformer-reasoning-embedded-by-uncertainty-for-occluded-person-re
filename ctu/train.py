"""Train CTU on Market-1501."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
import yaml
from torch.cuda import amp
from torch.utils.data import DataLoader

from .data import Market1501, RandomIdentitySampler, resolve_market_root
from .data.market1501 import build_transforms
from .losses import BatchHardTriplet, ctu_total_loss
from .metrics import evaluate_rank
from .models import build_ctu


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_train_loader(cfg: dict) -> DataLoader:
    root = resolve_market_root(cfg)
    ds = Market1501(
        root,
        split="train",
        transform=build_transforms(cfg, train=True),
        max_samples=cfg["data"].get("max_train_samples"),
    )
    batch = int(cfg["train"]["batch_size"])
    ni = int(cfg["train"].get("num_instances", 4))
    sampler = RandomIdentitySampler(ds, num_instances=ni, batch_size=batch)
    return DataLoader(
        ds,
        batch_size=batch,
        sampler=sampler,
        num_workers=int(cfg["data"].get("num_workers", 4)),
        pin_memory=True,
        drop_last=True,
    )


def build_eval_loader(cfg: dict, split: str) -> DataLoader:
    root = resolve_market_root(cfg)
    key = "max_eval_samples"
    ds = Market1501(
        root,
        split=split,
        transform=build_transforms(cfg, train=False),
        max_samples=cfg["data"].get(key),
    )
    return DataLoader(
        ds,
        batch_size=int(cfg["eval"].get("batch_size", 64)),
        shuffle=False,
        num_workers=int(cfg["data"].get("num_workers", 4)),
        pin_memory=True,
    )


@torch.no_grad()
def extract_features(model, loader, device: str) -> Dict[str, Any]:
    model.eval()
    feats, pids, cams = [], [], []
    for batch in loader:
        img = batch["image"].to(device, non_blocking=True)
        out = model(img, train=False)
        feats.append(out["feat"].cpu())
        pids.append(batch["pid"].numpy())
        cams.append(batch["camid"].numpy())
    import numpy as np

    return {
        "feats": torch.cat(feats, 0),
        "pids": np.concatenate(pids),
        "cams": np.concatenate(cams),
    }


def run_eval(model, cfg, device: str) -> Dict[str, float]:
    q = extract_features(model, build_eval_loader(cfg, "query"), device)
    g = extract_features(model, build_eval_loader(cfg, "gallery"), device)
    return evaluate_rank(
        q["feats"],
        q["pids"],
        q["cams"],
        g["feats"],
        g["pids"],
        g["cams"],
        ranks=list(cfg["eval"].get("ranks", [1, 5, 10])),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Train CTU")
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    cfg = load_config(args.config)
    for k in ("checkpoints", "logs", "figures"):
        Path(cfg["output"][k]).mkdir(parents=True, exist_ok=True)

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    train_loader = build_train_loader(cfg)
    num_classes = train_loader.dataset.num_classes
    print(f"Market train: {len(train_loader.dataset)} imgs, {num_classes} ids", flush=True)

    model = build_ctu(cfg, num_classes).to(device)
    opt = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"].get("weight_decay", 5e-4)),
    )
    epochs = int(cfg["train"]["epochs"])
    warmup = int(cfg["train"].get("warmup_epochs", 10))
    ce = nn.CrossEntropyLoss(label_smoothing=float(cfg["train"].get("label_smoothing", 0.0)))
    tri = BatchHardTriplet(margin=float(cfg["train"].get("margin", 0.3)))
    scaler = amp.GradScaler(enabled=device.startswith("cuda"))
    history = []
    best_map = -1.0

    for epoch in range(1, epochs + 1):
        # linear warmup then cosine
        if epoch <= warmup:
            lr = float(cfg["train"]["lr"]) * epoch / max(warmup, 1)
        else:
            t = (epoch - warmup) / max(epochs - warmup, 1)
            lr = float(cfg["train"]["lr"]) * 0.5 * (1.0 + math.cos(math.pi * t))
        for g in opt.param_groups:
            g["lr"] = lr

        model.train()
        t0 = time.time()
        losses = []
        for step, batch in enumerate(train_loader, 1):
            img = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with amp.autocast(enabled=device.startswith("cuda")):
                out = model(img, labels=labels, train=True)
                loss = ctu_total_loss(out, labels, ce, tri)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.item()))
            if step % int(cfg["train"].get("log_every", 20)) == 0:
                print(
                    f"epoch {epoch}/{epochs} step {step} loss={loss.item():.4f} lr={lr:.2e}",
                    flush=True,
                )
        mean_loss = sum(losses) / max(len(losses), 1)
        row: Dict[str, Any] = {
            "epoch": epoch,
            "loss": mean_loss,
            "lr": lr,
            "sec": time.time() - t0,
        }
        eval_every = int(cfg["train"].get("eval_every", 10))
        if epoch % eval_every == 0 or epoch == epochs:
            metrics = run_eval(model, cfg, device)
            row.update(metrics)
            print(f"epoch {epoch} eval {metrics}", flush=True)
            payload = {
                "model": model.state_dict(),
                "epoch": epoch,
                "metrics": metrics,
                "num_classes": num_classes,
                "cfg": cfg,
            }
            ckpt = Path(cfg["output"]["checkpoints"])
            torch.save(payload, ckpt / f"epoch_{epoch}.pt")
            if metrics.get("mAP", -1) > best_map:
                best_map = metrics["mAP"]
                torch.save(payload, ckpt / "best.pt")
        history.append(row)
        print(f"epoch {epoch} mean_loss={mean_loss:.4f} ({row['sec']:.1f}s)", flush=True)

    summary = {
        "num_classes": num_classes,
        "train_size": len(train_loader.dataset),
        "device": device,
        "history": history,
        "config": args.config,
    }
    with_map = [h for h in history if "mAP" in h]
    if with_map:
        best = max(with_map, key=lambda x: x["mAP"])
        summary["final_metrics"] = {
            k: best[k] for k in ("mAP", "Rank-1", "Rank-5", "Rank-10") if k in best
        }
        summary["best_epoch"] = best["epoch"]
    out = Path(cfg["output"]["logs"]) / "train_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
