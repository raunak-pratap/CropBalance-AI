"""
disease_trainer.py
--------------------------
Fine-tunes the disease CNN on the PlantVillage dataset.

Dataset setup:
  1. Download PlantVillage from Kaggle:
     kaggle datasets download -d abdallahalidev/plantvillage-dataset
  2. Extract to: data/raw/plantvillage/
     Structure: data/raw/plantvillage/<ClassName>/<image.jpg>

Run:
    python disease/disease_trainer.py --epochs 30 --batch-size 32
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from loguru import logger
from tqdm import tqdm

from disease.disease_model import (
    build_disease_model, get_train_transform, get_inference_transform,
    DISEASE_CLASSES, NUM_CLASSES,
)


def train(args):
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = args.data_dir   # e.g. data/raw/plantvillage/

    # ── Datasets ───────────────────────────────
    full_ds = ImageFolder(data_root, transform=get_train_transform())
    n_train = int(0.80 * len(full_ds))
    n_val   = int(0.10 * len(full_ds))
    n_test  = len(full_ds) - n_train - n_val

    train_ds, val_ds, test_ds = random_split(
        full_ds, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42),
    )
    # Use inference transform for val/test (no augmentation)
    val_ds.dataset.transform  = get_inference_transform()
    test_ds.dataset.transform = get_inference_transform()

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0, pin_memory=False)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_dl  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=0)

    logger.info(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    logger.info(f"Classes found: {len(full_ds.classes)}")

    # ── Model ──────────────────────────────────
    model = build_disease_model(num_classes=len(full_ds.classes), pretrained=True)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # label smoothing reduces overconfidence
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-4,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_acc = 0.0
    save_path    = os.path.join("models", "saved", "disease_model.pt")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # ── Training loop ──────────────────────────
    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_loss, train_correct = 0.0, 0
        for X, y in tqdm(train_dl, desc=f"Epoch {epoch:03d}/{args.epochs} [train]", leave=False):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out  = model(X)
            loss = criterion(out, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss    += loss.item() * len(X)
            train_correct += (out.argmax(1) == y).sum().item()

        # Validate
        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for X, y in val_dl:
                X, y = X.to(device), y.to(device)
                out       = model(X)
                val_loss += criterion(out, y).item() * len(X)
                val_correct += (out.argmax(1) == y).sum().item()

        scheduler.step()

        tr_acc  = train_correct / len(train_ds) * 100
        val_acc = val_correct   / len(val_ds)   * 100
        logger.info(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_loss/len(train_ds):.4f} | train_acc={tr_acc:.1f}% | "
            f"val_loss={val_loss/len(val_ds):.4f} | val_acc={val_acc:.1f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch":       epoch,
                "val_acc":     val_acc,
                "model_state": model.state_dict(),
                "classes":     full_ds.classes,
            }, save_path)
            logger.info(f"  *** New best: {val_acc:.1f}% — checkpoint saved ***")

    # ── Test evaluation ────────────────────────
    ckpt  = torch.load(save_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    test_correct = 0
    with torch.no_grad():
        for X, y in test_dl:
            X, y = X.to(device), y.to(device)
            test_correct += (model(X).argmax(1) == y).sum().item()

    test_acc = test_correct / len(test_ds) * 100
    logger.info(f"Test accuracy: {test_acc:.2f}%")
    logger.info(f"Model saved -> {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",   default="data/raw/plantvillage")
    parser.add_argument("--epochs",     type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr",         type=float, default=1e-3)
    train(parser.parse_args())