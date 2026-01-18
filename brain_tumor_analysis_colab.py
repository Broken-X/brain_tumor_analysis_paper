"""
Brain Tumor MRI (4-class) — Robust Transfer Learning + Metrics + Grad-CAM Artifacts

This script is written to be run in Google Colab as a plain .py (or converted to a notebook).
It implements:
  - 4-class MRI classification (glioma/meningioma/pituitary/no_tumor)
  - Robustness evaluation across multiple train/test split ratios (80/20, 85/15, 90/10) and multiple seeds
  - Validation split + early stopping + best-checkpoint saving
  - Accuracy + macro/weighted precision/recall/F1 + confusion matrices + per-class report
  - Systematic Grad-CAM export for correct + incorrect examples per class (per model/split)
  - Reproducibility: fixed seeds, config + splits saved

Dataset:
  - Downloads Kaggle dataset "masoudnickparvar/brain-tumor-mri-dataset" via kagglehub.
  - Uses BOTH Training/ and Testing/ folders as the full pool, then creates custom stratified splits.

Outputs:
  results/
    config.json
    splits/*.npz
    metrics/summary.csv
    metrics/*classification_report.txt
    metrics/*confusion_matrix.png
    checkpoints/*.pth
    gradcam/*/*.png

Note:
  - This script trains multiple models and split configs; it can take time.
  - Control scope via CONFIG at the top (models, epochs, seeds, etc.).
"""

import os
import json
import csv
import shutil
import time
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import torchvision
from torchvision import transforms
from torchvision import models

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

import matplotlib.pyplot as plt
import kagglehub

# ---------------------------
# Config (edit this section)
# ---------------------------
@dataclass
class Config:
    # Data / splits
    img_size: int = 224
    split_ratios: Tuple[float, ...] = (0.80, 0.85, 0.90)  # train fraction; test = 1-train
    val_fraction_of_train: float = 0.10                   # validation fraction within the training portion
    seeds: Tuple[int, ...] = (0, 1)                       # increase for stronger robustness stats

    # Training
    batch_size: int = 32
    num_workers: int = 2
    max_epochs: int = 12
    patience: int = 3
    lr: float = 1e-4
    weight_decay: float = 1e-4
    freeze_backbone_epochs: int = 0  # warm start if set 1: train head only for first N epochs
    device: str = "cuda"             # "cuda" or "cpu"

    # Models to run
    models: Tuple[str, ...] = ("resnet50", "densenet121", "efficientnet_b0")

    # Grad-CAM exports
    gradcam_per_class_correct: int = 3
    gradcam_per_class_incorrect: int = 3

    # Optional: export a 10–20% per-class sample for physician review (paper methodology support)
    physician_review_fraction: float = 0.15  # 15% default within 10–20% range
    physician_review_max_per_class: int = 60
    physician_review_seed: int = 0

    # Reproducibility
    deterministic: bool = False  # True can slow training; set True if you want stricter determinism

CONFIG = Config()

# ----------------------
# Data (Kaggle download
# ----------------------
import kagglehub
DATA_ROOT = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")

# ----------
# Utilities
# ----------
def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
    else:
        torch.backends.cudnn.benchmark = True

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def now_ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S")

def get_model_stats(model: nn.Module, device: torch.device) -> Tuple[float, float]:
    """
    Calculates:
      1. params_M: Total parameters in Millions
      2. latency_ms: Average inference time per image in milliseconds (over 100 runs)
    """
    # Parameter Count (Millions)
    params_M = sum(p.numel() for p in model.parameters()) / 1e6

    # Latency Measurement
    model.eval()
    # Create a dummy input matching the input size
    dummy_input = torch.randn(1, 3, CONFIG.img_size, CONFIG.img_size).to(device)

    # Warmup (to ensure GPU is active and caches are loaded)
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)

    # Timing loop
    if device.type == 'cuda':
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        with torch.no_grad():
            for _ in range(100):
                _ = model(dummy_input)
        end_event.record()
        torch.cuda.synchronize()

        total_time_ms = start_event.elapsed_time(end_event)
        latency_ms = total_time_ms / 100.0
    else:
        # Fallback for CPU timing
        start_time = time.time()
        with torch.no_grad():
            for _ in range(100):
                _ = model(dummy_input)
        end_time = time.time()
        latency_ms = ((end_time - start_time) * 1000) / 100.0

    return params_M, latency_ms

def export_physician_review_sample(
    paths: List[str],
    labels: List[int],
    class_names: List[str],
    out_dir: Path,
    fraction: float,
    max_per_class: int,
    seed: int,
) -> Path:
    """Export a stratified sample (by class) to support external/physician verification.

    Creates: results/physician_review_sample/<class_name>/*.jpg and a CSV manifest.
    This is intentionally lightweight and reproducible via the provided seed.
    """
    rng = np.random.default_rng(seed)
    review_root = out_dir / "physician_review_sample"
    ensure_dir(review_root)

    rows: List[Dict[str, str]] = []
    for cls_idx, cls_name in enumerate(class_names):
        cls_paths = [p for p, y in zip(paths, labels) if y == cls_idx]
        if not cls_paths:
            continue
        n = int(round(len(cls_paths) * fraction))
        n = max(1, min(n, max_per_class, len(cls_paths)))
        chosen = rng.choice(len(cls_paths), size=n, replace=False)

        cls_out = review_root / cls_name
        ensure_dir(cls_out)

        for k, ci in enumerate(chosen):
            src = Path(cls_paths[int(ci)])
            # Keep extension; create deterministic-ish name
            dst = cls_out / f"{cls_name}_{k:03d}{src.suffix.lower()}"
            try:
                shutil.copy2(src, dst)
            except Exception:
                # If copy fails (permissions), still record original path
                dst = Path("")
            rows.append({
                "class": cls_name,
                "label": str(cls_idx),
                "source_path": str(src),
                "copied_path": str(dst) if str(dst) else "",
            })

    manifest = review_root / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["class", "label", "source_path", "copied_path"])
        w.writeheader()
        w.writerows(rows)

    return review_root

# ----------------------------
# Dataset: filepaths + labels
# ----------------------------
def list_image_files(root: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    files: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    return files


def build_samples_from_folders(data_root: Path) -> Tuple[List[str], List[int], List[str]]:
    """
    Build (paths, labels, class_names) from both Training/ and Testing/ subfolders.
    Assumes structure: <split>/<class_name>/*.jpg
    """
    train_dir = data_root / "Training"
    test_dir = data_root / "Testing"
    if not train_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(
            f"Expected Training/ and Testing/ under {data_root}. Found: "
            f"Training exists={train_dir.exists()}, Testing exists={test_dir.exists()}"
        )

    # Class names from folder names (union across both)
    class_names = sorted({p.name for p in train_dir.iterdir() if p.is_dir()} | {p.name for p in test_dir.iterdir() if p.is_dir()})
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    paths: List[str] = []
    labels: List[int] = []

    for base in [train_dir, test_dir]:
        for cls in class_names:
            cls_dir = base / cls
            if not cls_dir.exists():
                continue
            for img_path in list_image_files(cls_dir):
                paths.append(str(img_path))
                labels.append(class_to_idx[cls])

    if len(paths) == 0:
        raise RuntimeError("No images found. Check dataset structure and permissions.")

    return paths, labels, class_names


class MRIFilesDataset(Dataset):
    def __init__(self, paths: List[str], labels: List[int], transform: Optional[transforms.Compose] = None):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        y = self.labels[idx]
        return img, y, self.paths[idx]

# -----------
# Transforms
# -----------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

train_tfms = transforms.Compose([
    transforms.Resize((CONFIG.img_size, CONFIG.img_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

test_tfms = transforms.Compose([
    transforms.Resize((CONFIG.img_size, CONFIG.img_size)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# -------
# Models
# -------
def set_trainable(module: nn.Module, trainable: bool) -> None:
    for p in module.parameters():
        p.requires_grad = trainable


def build_classifier_head(in_features: int, num_classes: int) -> nn.Module:
    # Keep consistent with your Streamlit head (Linear -> ReLU -> Dropout -> Linear)
    return nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes),
    )


def get_model(name: str, num_classes: int) -> nn.Module:
    name = name.lower().strip()
    if name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        in_features = model.fc.in_features
        model.fc = build_classifier_head(in_features, num_classes)
        return model

    if name == "densenet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        in_features = model.classifier.in_features
        model.classifier = build_classifier_head(in_features, num_classes)
        return model

    if name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = model.classifier[-1].in_features
        # classifier is Sequential([Dropout, Linear]) by default; replace entirely
        model.classifier = build_classifier_head(in_features, num_classes)
        return model

    raise ValueError(f"Unknown model: {name}")


def get_target_layer(model_name: str, model: nn.Module) -> nn.Module:
    """
    Return the layer to hook for Grad-CAM.
    """
    model_name = model_name.lower()
    if model_name == "resnet50":
        return model.layer4[-1]
    if model_name == "densenet121":
        #return model.features[-1]
        #return model.features[-2][-1]
        return model.features.denseblock4
    if model_name == "efficientnet_b0":
        return model.features[-1]
    raise ValueError(f"No target layer mapping for {model_name}")


# -------------
# Train / Eval
# -------------
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict:
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    y_prob: List[np.ndarray] = []
    paths: List[str] = []

    for x, y, p in loader:
        x = x.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        pred = probs.argmax(axis=1)

        y_true.extend(y.numpy().tolist())
        y_pred.extend(pred.tolist())
        y_prob.extend([row for row in probs])
        paths.extend(list(p))

    acc = accuracy_score(y_true, y_pred)
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)

    return {
        "y_true": np.array(y_true),
        "y_pred": np.array(y_pred),
        "y_prob": np.array(y_prob),
        "paths": np.array(paths),
        "accuracy": float(acc),
        "precision_weighted": float(prec_w),
        "recall_weighted": float(rec_w),
        "f1_weighted": float(f1_w),
        "precision_macro": float(prec_m),
        "recall_macro": float(rec_m),
        "f1_macro": float(f1_m),
    }


def train_one_run(
    model_name: str,
    split_tag: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_classes: int,
    class_names: List[str],
    out_dir: Path,
) -> Tuple[nn.Module, Dict]:
    model = get_model(model_name, num_classes).to(device)

    # Freeze backbone initially (optional)
    if CONFIG.freeze_backbone_epochs > 0:
        # Freeze everything except classifier head
        set_trainable(model, False)
        if model_name == "resnet50":
            set_trainable(model.fc, True)
        elif model_name == "densenet121":
            set_trainable(model.classifier, True)
        elif model_name == "efficientnet_b0":
            set_trainable(model.classifier, True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=CONFIG.lr, weight_decay=CONFIG.weight_decay)

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    history = []

    for epoch in range(CONFIG.max_epochs):
        print(f"- Epoch: {epoch}")
        model.train()

        # Unfreeze after warm-up
        if epoch == CONFIG.freeze_backbone_epochs:
            set_trainable(model, True)
            optimizer = optim.AdamW(model.parameters(), lr=CONFIG.lr, weight_decay=CONFIG.weight_decay)

        train_losses = []
        for x, y, _ in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, y, _ in val_loader:
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                val_losses.append(loss.item())

        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        val_loss = float(np.mean(val_losses)) if val_losses else float("nan")

        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})

        # Early stopping
        improved = val_loss < best_val_loss - 1e-6
        if improved:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= CONFIG.patience:
                break

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    # Save checkpoint (loadable by Streamlit)
    ckpt_path = out_dir / "checkpoints" / f"{model_name}_{split_tag}.pth"
    torch.save(
        {
            "model_name": model_name,
            "split_tag": split_tag,
            "state_dict": model.state_dict(),
            "class_names": class_names,
            "img_size": CONFIG.img_size,
            "mean": IMAGENET_MEAN,
            "std": IMAGENET_STD,
            "head": "Linear->ReLU->Dropout(0.3)->Linear(512)",
            "timestamp": now_ts(),
            "config": asdict(CONFIG),
            "train_history": history,
        },
        ckpt_path,
    )

    return model, {"best_val_loss": best_val_loss, "history": history, "checkpoint": str(ckpt_path)}


# ---------
# Grad-CAM
# ---------
class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        def fwd_hook(_m, _inp, out):
            self.activations = out

        def bwd_hook(_m, _gin, gout):
            self.gradients = gout[0]

        self.h1 = self.target_layer.register_forward_hook(fwd_hook)
        self.h2 = self.target_layer.register_full_backward_hook(bwd_hook)

    def close(self):
        self.h1.remove()
        self.h2.remove()

    def generate(self, x: torch.Tensor, class_idx: int) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        score = logits[:, class_idx].sum()
        score.backward(retain_graph=True)

        # activations/gradients shape: [B, C, H, W]
        A = self.activations
        G = self.gradients
        weights = torch.mean(G, dim=(2, 3), keepdim=True)  # [B,C,1,1]
        cam = torch.sum(weights * A, dim=1)  # [B,H,W]
        cam = torch.relu(cam)
        cam = cam.detach().cpu().numpy()[0]

        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam


def unnormalize(img_t: torch.Tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD) -> np.ndarray:
    img = img_t.detach().cpu().numpy().transpose(1, 2, 0)
    img = (img * np.array(std)) + np.array(mean)
    img = np.clip(img, 0, 1)
    return img


def overlay_cam(img: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Overlay a Grad-CAM (which may be low-res e.g. 7x7) onto an RGB image (H,W,3).

    img: float RGB in [0,1], shape (H,W,3)
    cam: float in [0,1] (or any range), shape (h,w) or (H,W)
    """
    # Ensure cam is 2D
    if cam.ndim == 3:
        cam = cam[..., 0]

    H, W = img.shape[:2]
    if cam.shape != (H, W):
        # Upsample cam to image resolution
        cam_t = torch.tensor(cam, dtype=torch.float32)[None, None, :, :]
        cam_up = F.interpolate(cam_t, size=(H, W), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
        cam = cam_up

    # Normalize cam to [0,1] for stable visualization
    cam = cam - np.min(cam)
    cam = cam / (np.max(cam) + 1e-8)
    # cam -> RGB heatmap via matplotlib colormap
    cmap = plt.get_cmap("jet")
    heat = cmap(cam)[:, :, :3]  # drop alpha
    out = (1 - alpha) * img + alpha * heat
    return np.clip(out, 0, 1)


def export_gradcam_examples(
    model_name: str,
    model: nn.Module,
    dataset: MRIFilesDataset,
    eval_dict: Dict,
    class_names: List[str],
    split_tag: str,
    out_dir: Path,
    device: torch.device,
) -> None:
    model.eval()
    target_layer = get_target_layer(model_name, model)
    cammer = GradCAM(model, target_layer)

    y_true = eval_dict["y_true"]
    y_pred = eval_dict["y_pred"]
    y_prob = eval_dict["y_prob"]
    paths = eval_dict["paths"]

    # For each class, select top correct and top incorrect by confidence
    for cls_idx, cls_name in enumerate(class_names):
        # Correct predictions for this class
        correct_mask = (y_true == cls_idx) & (y_pred == cls_idx)
        incorrect_mask = (y_true == cls_idx) & (y_pred != cls_idx)

        def pick_top(mask: np.ndarray, n: int, kind: str) -> List[int]:
            idxs = np.where(mask)[0]
            if idxs.size == 0:
                return []
            if kind == "correct":
                # Rank by confidence in the true class.
                confs = y_prob[idxs, cls_idx]
            else:
                # Rank incorrect examples by confidence in the predicted class (most confident mistakes).
                pred_for_idxs = y_pred[idxs].astype(int)
                confs = y_prob[idxs, pred_for_idxs]
            order = np.argsort(-confs)
            return idxs[order][:n].tolist()

        correct_idxs = pick_top(correct_mask, CONFIG.gradcam_per_class_correct, kind="correct")
        incorrect_idxs = pick_top(incorrect_mask, CONFIG.gradcam_per_class_incorrect, kind="incorrect")

        for kind, picked in [("correct", correct_idxs), ("incorrect", incorrect_idxs)]:
            for rank, global_i in enumerate(picked, start=1):
                img_path = str(paths[global_i])

                orig_filename = Path(img_path).stem

                # Locate dataset index by filepath (dataset.paths is list[str])
                try:
                    ds_i = dataset.paths.index(img_path)
                except ValueError:
                    continue

                x_t, y, _ = dataset[ds_i]
                x = x_t.unsqueeze(0).to(device)

                pred_cls = int(y_pred[global_i])
                cam = cammer.generate(x, pred_cls)
                img = unnormalize(x_t)
                ov = overlay_cam(img, cam)

                fig = plt.figure(figsize=(10, 4))
                ax1 = fig.add_subplot(1, 2, 1)
                ax1.imshow(img)
                ax1.set_title(f"Original: {orig_filename}")
                ax1.axis("off")

                ax2 = fig.add_subplot(1, 2, 2)
                ax2.imshow(ov)
                ax2.set_title(f"Grad-CAM (pred: {class_names[pred_cls]})")
                ax2.axis("off")

                save_dir = out_dir / "gradcam" / f"{model_name}_{split_tag}" / cls_name / kind
                ensure_dir(save_dir)

                out_path = save_dir / f"{cls_name}_{kind}_{rank}_{orig_filename}.png"

                plt.tight_layout()
                plt.savefig(out_path, dpi=200)
                plt.close(fig)

    cammer.close()


# ----------
# Splitting
# ----------
def stratified_split_indices(labels: List[int], train_frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    from sklearn.model_selection import StratifiedShuffleSplit
    y = np.array(labels)
    sss = StratifiedShuffleSplit(n_splits=1, train_size=train_frac, random_state=seed)
    train_idx, test_idx = next(sss.split(np.zeros_like(y), y))
    return train_idx, test_idx


def split_train_val(train_idx: np.ndarray, labels: List[int], val_frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    from sklearn.model_selection import StratifiedShuffleSplit
    y_train = np.array(labels)[train_idx]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    tr_sub, val_sub = next(sss.split(np.zeros_like(y_train), y_train))
    return train_idx[tr_sub], train_idx[val_sub]


def make_loader(paths: List[str], labels: List[int], indices: np.ndarray, tfm, shuffle: bool) -> Tuple[MRIFilesDataset, DataLoader]:
    p = [paths[i] for i in indices]
    y = [labels[i] for i in indices]
    ds = MRIFilesDataset(p, y, transform=tfm)
    dl = DataLoader(
        ds,
        batch_size=CONFIG.batch_size,
        shuffle=shuffle,
        num_workers=CONFIG.num_workers,
        pin_memory=True,
    )
    return ds, dl


# -----------------
# Confusion Matrix
# -----------------
def plot_and_save_confusion_matrix(cm: np.ndarray, class_names: List[str], out_path: Path, title: str) -> None:
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(1, 1, 1)

    ax.imshow(cm, interpolation="nearest", cmap="Blues")

    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=12)
    ax.set_yticklabels(class_names, fontsize=12)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", # Conditional text color
                    fontweight="bold",     # Always bold
                    fontsize=18)           # Larger font size

    ax.set_ylabel("True label", fontsize=12)
    ax.set_xlabel("Predicted label", fontsize=12)
    plt.tight_layout()
    ensure_dir(out_path.parent)
    plt.savefig(out_path, dpi=200)
    plt.close(fig)

# -----
# Main
# -----
def main() -> None:
    device = torch.device(CONFIG.device if torch.cuda.is_available() and CONFIG.device == "cuda" else "cpu")

    out_dir = Path("results")
    ensure_dir(out_dir)
    ensure_dir(out_dir / "checkpoints")
    ensure_dir(out_dir / "metrics")
    ensure_dir(out_dir / "splits")
    ensure_dir(out_dir / "gradcam")

    # Save config
    (out_dir / "config.json").write_text(json.dumps(asdict(CONFIG), indent=2))

    # Build sample pool
    paths, labels, class_names = build_samples_from_folders(Path(DATA_ROOT))

    # Export a reproducible 10–20% per-class sample to support physician/external review (optional).
    try:
        export_physician_review_sample(
            paths=paths,
            labels=labels,
            class_names=class_names,
            out_dir=out_dir,
            fraction=CONFIG.physician_review_fraction,
            max_per_class=CONFIG.physician_review_max_per_class,
            seed=CONFIG.physician_review_seed,
        )
    except Exception as e:
        (out_dir / "metrics" / "physician_review_export_error.txt").write_text(str(e))
    num_classes = len(class_names)

    # Summary rows
    summary_rows = []

    for seed in CONFIG.seeds:
        set_seed(seed, deterministic=CONFIG.deterministic)

        for train_frac in CONFIG.split_ratios:
            split_tag = f"train{int(train_frac*100)}_seed{seed}"
            train_idx, test_idx = stratified_split_indices(labels, train_frac=train_frac, seed=seed)
            tr_idx, val_idx = split_train_val(train_idx, labels, val_frac=CONFIG.val_fraction_of_train, seed=seed)

            # Save split indices for reproducibility
            np.savez(out_dir / "splits" / f"{split_tag}.npz", train_idx=tr_idx, val_idx=val_idx, test_idx=test_idx)

            train_ds, train_loader = make_loader(paths, labels, tr_idx, train_tfms, shuffle=True)
            val_ds, val_loader = make_loader(paths, labels, val_idx, test_tfms, shuffle=False)
            test_ds, test_loader = make_loader(paths, labels, test_idx, test_tfms, shuffle=False)

            for model_name in CONFIG.models:
                run_tag = f"{model_name}_{split_tag}"
                print(f"\n=== RUN: {run_tag} on device={device} ===")

                # Train
                model, train_info = train_one_run(
                    model_name=model_name,
                    split_tag=split_tag,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    device=device,
                    num_classes=num_classes,
                    class_names=class_names,
                    out_dir=out_dir,
                )

                # Eval on test
                eval_dict = evaluate(model, test_loader, device=device)
                params_M, latency_ms = get_model_stats(model, device)

                y_true = eval_dict["y_true"]
                y_pred = eval_dict["y_pred"]

                # Calculate number of incorrect predictions
                num_incorrect = np.sum(y_true != y_pred)
                total_test_samples = len(y_true)
                print(f"    -> Incorrect Predictions: {num_incorrect} out of {total_test_samples} images")

                cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
                cm_path = out_dir / "metrics" / f"confusion_{run_tag}.png"
                plot_and_save_confusion_matrix(cm, class_names, cm_path, title=f"Confusion Matrix: {run_tag}")

                report = classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0)
                (out_dir / "metrics" / f"classification_report_{run_tag}.txt").write_text(report)

                # Grad-CAM exports (correct + incorrect)
                try:
                    export_gradcam_examples(
                        model_name=model_name,
                        model=model,
                        dataset=test_ds,
                        eval_dict=eval_dict,
                        class_names=class_names,
                        split_tag=split_tag,
                        out_dir=out_dir,
                        device=device,
                    )
                except Exception as e:
                    # Grad-CAM should not crash the run; log and continue
                    (out_dir / "metrics" / f"gradcam_error_{run_tag}.txt").write_text(str(e))

                row = {
                    "run": run_tag,
                    "model": model_name,
                    "train_frac": float(train_frac),
                    "seed": int(seed),
                    "num_train": int(len(tr_idx)),
                    "num_val": int(len(val_idx)),
                    "num_test": int(len(test_idx)),
                    "accuracy": eval_dict["accuracy"],
                    "num_incorrect": int(num_incorrect), # Added to CSV
                    "params_M": params_M,
                    "latency_ms": latency_ms,
                    "precision_weighted": eval_dict["precision_weighted"],
                    "recall_weighted": eval_dict["recall_weighted"],
                    "f1_weighted": eval_dict["f1_weighted"],
                    "precision_macro": eval_dict["precision_macro"],
                    "recall_macro": eval_dict["recall_macro"],
                    "f1_macro": eval_dict["f1_macro"],
                    "best_val_loss": float(train_info["best_val_loss"]),
                    "checkpoint": train_info["checkpoint"],
                    "confusion_matrix_png": str(cm_path),
                }
                summary_rows.append(row)

    # Save summary CSV
    summary_path = out_dir / "metrics" / "summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()) if summary_rows else [])
        writer.writeheader()
        for r in summary_rows:
            writer.writerow(r)

    print("\nDone.")
    print(f"- Config: {out_dir / 'config.json'}")
    print(f"- Summary: {summary_path}")
    print(f"- Checkpoints: {out_dir / 'checkpoints'}")
    print(f"- Grad-CAM: {out_dir / 'gradcam'}")


if __name__ == "__main__":
    main()

!zip -r /content/results.zip /content/results
from google.colab import files
files.download('/content/results.zip')