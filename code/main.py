#!/usr/bin/env python3
"""
ViT + loss functions for class-imbalanced pneumonia detection (RSNA).

Block 1: data download + splits + transforms
Block 2: datasets, model, loss definitions
Block 3: training loop, experiment runner, metric/figure saving
"""

#imports
import os
from pathlib import Path
import random
import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from torchvision import transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights

import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)

import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from PIL import Image
import pydicom


# --------------------------------------------
# GLOBALS & HYPERPARAMS
# --------------------------------------------

SEED = 42
VAL_SIZE = 0.25
TEST_FRACTION_FROM_TRAIN = 0.10
IMG_SIZE = 224
BATCH_SIZE = 32
LR = 3e-4
NUM_EPOCHS = 15
NUM_EPOCHS_TUNE = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# For CAF mix loss (weighted NLL + focal)
CAF_LAMBDA_DEFAULT = 0.8
FOCAL_ALPHA = [0.25, 0.75] # [no-pneumonia, pneumonia]
FOCAL_GAMMA = 2.0

# Where to save figures for the HTML blog
IMAGES_DIR = Path("./images")
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

print("Using device:", DEVICE)

# Reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# BLOCK 1: RSNA pneumonia via kagglehub, splits, transforms
def load_rsna_dataframe():
    """
    Download RSNA pneumonia dataset via kagglehub and build a patient-level
    dataframe with columns: patientId, target (0/1), path (DICOM path).
    """
    dataset_root = Path(
        kagglehub.dataset_download("parin30/rsna-pneumonia-detection")
    )
    print("Using dataset root:", dataset_root)

    train_images_dir = dataset_root / "drive-download-20240112T131344Z-002/stage_2_train_images"
    train_labels_csv = dataset_root / "drive-download-20240112T131344Z-002/stage_2_train_labels.csv"

    print("Train images dir:", train_images_dir)
    print("Train labels CSV:", train_labels_csv)

    if not train_images_dir.exists():
        raise FileNotFoundError(f"Train images dir not found: {train_images_dir}")
    if not train_labels_csv.exists():
        raise FileNotFoundError(f"Train labels CSV not found: {train_labels_csv}")

    labels_raw = pd.read_csv(train_labels_csv)

    # One row per patientId; patient is positive if ANY box has Target=1
    agg = (
        labels_raw.groupby("patientId")["Target"]
        .max()
        .reset_index()
        .rename(columns={"Target": "target"})
    )
    agg["path"] = agg["patientId"].apply(
        lambda pid: str(train_images_dir / f"{pid}.dcm")
    )

    df = agg[agg["path"].apply(os.path.exists)].reset_index(drop=True)

    print("Total images (after path check):", len(df))
    print("Class counts (0 = no pneumonia, 1 = pneumonia):")
    print(df["target"].value_counts())
    return df


def make_splits(df_full):
    """
    Stratified train/val/test splits at the patient level.
    No manual downsampling: we keep the natural class imbalance.
    """
    train_df, val_df = train_test_split(
        df_full,
        test_size=VAL_SIZE,
        stratify=df_full["target"],
        random_state=SEED,
    )

    train_df, test_df = train_test_split(
        train_df,
        test_size=TEST_FRACTION_FROM_TRAIN,
        stratify=train_df["target"],
        random_state=SEED,
    )

    print("\nSplit sizes (no downsampling):")
    print("  Train:", len(train_df), train_df["target"].value_counts())
    print("  Val  :", len(val_df),   val_df["target"].value_counts())
    print("  Test :", len(test_df),  test_df["target"].value_counts())

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def compute_class_weights(train_df):
    """
    Inverse-frequency class weights, rescaled so that mean weight ~ 1.

    Returns:
        class_weights_np: np.ndarray of shape [num_classes]
        oversample_factor: int, ~1 / positive_fraction
    """
    train_class_counts = train_df["target"].value_counts().sort_index()
    print("\nTrain class counts (for weighting):")
    print(train_class_counts)

    class_weights_np = 1.0 / train_class_counts.to_numpy()
    class_weights_np = class_weights_np / class_weights_np.mean()
    print("Class weights (approx inverse freq, mean≈1):", class_weights_np)

    pos_fraction = train_class_counts[1] / train_class_counts.sum()
    print(f"Positive fraction in train: {pos_fraction:.4f}")

    oversample_factor = int(round(1.0 / pos_fraction))
    oversample_factor = max(1, oversample_factor)

    return class_weights_np, oversample_factor


# Image transforms
base_train_transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

minority_train_transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

eval_transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)



###################################
###################################
###################################

# BLOCK 2: Dataset, ViT model, losses

class RsnaPneumoniaDataset(Dataset):
    """
    Patient-level RSNA dataset that loads chest X-ray DICOMs.

    DataFrame must have:
        - 'path': full .dcm path
        - 'target': 0 (no pneumonia) or 1 (pneumonia)
    """

    def __init__(
        self,
        df,
        base_transform=None,
        minority_transform=None,
        minority_label=1,
        apply_minority_aug=False,
    ):
        self.df = df.reset_index(drop=True)
        self.base_transform = base_transform
        self.minority_transform = minority_transform
        self.minority_label = minority_label
        self.apply_minority_aug = apply_minority_aug

    def __len__(self):
        return len(self.df)

    @staticmethod
    def _load_image(path: str) -> Image.Image:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".dcm":
            ds = pydicom.dcmread(path)
            img = ds.pixel_array.astype(np.float32)
            img -= img.min()
            img /= (img.max() + 1e-5)
            img = (img * 255).astype(np.uint8)
            img = Image.fromarray(img).convert("RGB")
        else:
            img = Image.open(path).convert("RGB")
        return img

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["path"]
        label = int(row["target"])

        img = self._load_image(img_path)

        if (
            self.apply_minority_aug
            and label == self.minority_label
            and self.minority_transform is not None
        ):
            img = self.minority_transform(img)
        elif self.base_transform is not None:
            img = self.base_transform(img)

        return img, label


def make_dataloaders(
    train_df,
    val_df,
    test_df,
    oversample=False,
    augment_minority=False,
    batch_size=BATCH_SIZE,
):
    """
    Construct train/val/test dataloaders with optional:
      - oversample: WeightedRandomSampler for balanced batches
      - augment_minority: use stronger transform for pneumonia-positive examples
    """
    train_dataset = RsnaPneumoniaDataset(
        train_df,
        base_transform=base_train_transform,
        minority_transform=minority_train_transform,
        apply_minority_aug=augment_minority,
        minority_label=1,
    )

    val_dataset = RsnaPneumoniaDataset(
        val_df,
        base_transform=eval_transform,
        minority_transform=None,
        apply_minority_aug=False,
        minority_label=1,
    )
    test_dataset = RsnaPneumoniaDataset(
        test_df,
        base_transform=eval_transform,
        minority_transform=None,
        apply_minority_aug=False,
        minority_label=1,
    )

    if oversample:
        cls_counts = train_df["target"].value_counts().to_dict()
        sample_weights = train_df["target"].map(
            lambda c: 1.0 / cls_counts[c]
        ).to_numpy()
        sample_weights = torch.DoubleTensor(sample_weights)
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=0,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, val_loader, test_loader


def build_model():
    """
    ViT-B/16 with ImageNet weights, 2-class head producing log-probabilities.
    """
    weights = ViT_B_16_Weights.IMAGENET1K_V1
    model = vit_b_16(weights=weights)
    model.heads = nn.Sequential(
        nn.Linear(model.hidden_dim, 2),
        nn.LogSoftmax(dim=1),
    )
    return model.to(DEVICE)


class FocalLoss(nn.Module):
    """
    Standard multi-class Focal Loss for log-probabilities.

    alpha: tensor of shape [num_classes], e.g. [0.25, 0.75]
    gamma: focusing parameter
    """

    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        if alpha is not None and not isinstance(alpha, torch.Tensor):
            alpha = torch.tensor(alpha, dtype=torch.float32)
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, log_probs, targets):
        if log_probs.ndim != 2:
            raise ValueError("log_probs must be (N, C)")
        if targets.ndim != 1:
            targets = targets.view(-1)

        probs = log_probs.exp()
        targets = targets.long()

        idx = torch.arange(log_probs.size(0), device=log_probs.device)
        log_pt = log_probs[idx, targets]
        pt = probs[idx, targets]

        if self.alpha is not None:
            alpha_t = self.alpha.to(log_probs.device)[targets]
            loss = -alpha_t * (1 - pt) ** self.gamma * log_pt
        else:
            loss = -(1 - pt) ** self.gamma * log_pt

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class MixedCAFWeightedLoss(nn.Module):
    """
    CAF-style mixed loss used in the 'caf_nll' experiment:

        L_mix = (1 - λ) * L_weighted_NLL  +  λ * L_focal

    where:
        - L_weighted_NLL = NLLLoss(weight=class_weights_tensor)
        - L_focal        = FocalLoss(alpha, gamma)
    """

    def __init__(self, lam, class_weights_tensor, focal_alpha=None, focal_gamma=2.0):
        super().__init__()
        self.lam = lam
        self.weighted_nll = nn.NLLLoss(weight=class_weights_tensor)
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, reduction="mean")

    def forward(self, log_probs, targets):
        loss_w = self.weighted_nll(log_probs, targets)
        loss_f = self.focal(log_probs, targets)
        return (1.0 - self.lam) * loss_w + self.lam * loss_f


###################################
###################################
###################################

# BLOCK 3: training, eval, experiment runner, figure saving

def plot_confusion_matrix(
    cm,
    class_names,
    title="Confusion Matrix",
    save_path=None,
):
    """
    Pretty confusion matrix with counts + row-normalized percentages.
    If save_path is given, save PNG instead of (or in addition to) showing.
    """
    cm = np.array(cm)
    cm_sum = cm.sum(axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100

    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Count", rotation=90)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            perc = cm_perc[i, j]
            ax.text(
                j,
                i,
                f"{count}\n{perc:.1f}%",
                ha="center",
                va="center",
                color="white" if count > thresh else "black",
                fontsize=10,
            )

    ax.set_ylim(len(class_names) - 0.5, -0.5)
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curve(
    y_true,
    y_scores,
    title="ROC Curve (Test)",
    save_path=None,
):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.03])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return roc_auc


def train_one_epoch(model, loader, optimizer, criterion, epoch, num_epochs):
    model.train()
    tot_loss, tot_correct, tot = 0.0, 0, 0
    pbar = tqdm(loader, desc=f"Train {epoch}/{num_epochs}", leave=False)
    for imgs, labels in pbar:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        out = model(imgs)  # log-probabilities
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        bs = imgs.size(0)
        tot_loss += loss.item() * bs
        preds = out.argmax(1)
        tot_correct += (preds == labels).sum().item()
        tot += bs

        pbar.set_postfix(
            loss=f"{tot_loss/tot:.4f}", acc=f"{tot_correct/tot:.4f}"
        )
    return tot_loss / tot, tot_correct / tot


def evaluate(model, loader, criterion, return_scores=False):
    model.eval()
    tot_loss, tot_correct, tot = 0.0, 0, 0
    y_true, y_pred, y_scores = [], [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            out = model(imgs)  # log-probabilities
            loss = criterion(out, labels)

            bs = imgs.size(0)
            tot_loss += loss.item() * bs
            preds = out.argmax(1)
            tot_correct += (preds == labels).sum().item()
            tot += bs

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

            if return_scores:
                probs = out.exp()[:, 1].cpu().numpy()
                y_scores.extend(probs)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_scores = np.array(y_scores) if return_scores else None

    if return_scores:
        return tot_loss / tot, tot_correct / tot, y_true, y_pred, y_scores
    return tot_loss / tot, tot_correct / tot, y_true, y_pred


def run_experiment(
    exp_name,
    train_df,
    val_df,
    test_df,
    class_weights_tensor,
    use_oversample=False,
    use_minority_aug=False,
    use_class_weighted=False,
    use_focal_loss=False,
    use_caf_nll=False,
):
    """
    Run a single experiment with specified imbalance / loss configuration.
    Returns a dict of summary metrics and saves CM + ROC PNGs to IMAGES_DIR.
    """
    print("\n" + "=" * 60)
    print(f"EXPERIMENT: {exp_name}")
    print("=" * 60)

    train_loader, val_loader, test_loader = make_dataloaders(
        train_df,
        val_df,
        test_df,
        oversample=use_oversample,
        augment_minority=use_minority_aug,
    )

    # CAF experiment: mixed weighted NLL + focal
    if use_caf_nll:
        lam = CAF_LAMBDA_DEFAULT
        print(f"[CAF] Using fixed λ = {lam:.1f} for MixedCAFWeightedLoss")

        model = build_model()
        focal_alpha = torch.tensor(FOCAL_ALPHA, dtype=torch.float32, device=DEVICE)
        criterion = MixedCAFWeightedLoss(
            lam=lam,
            class_weights_tensor=class_weights_tensor,
            focal_alpha=focal_alpha,
            focal_gamma=FOCAL_GAMMA,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        best_val_acc = 0.0
        best_path = Path(f"./best_vit_rsna_{exp_name}_lam_{lam:.1f}.pth")

        for epoch in range(1, NUM_EPOCHS + 1):
            tr_loss, tr_acc = train_one_epoch(
                model, train_loader, optimizer, criterion, epoch, NUM_EPOCHS
            )
            val_loss, val_acc, y_true_v, y_pred_v = evaluate(
                model, val_loader, criterion, return_scores=False
            )
            print(f"Epoch {epoch}/{NUM_EPOCHS} (CAF λ={lam:.1f})")
            print(f"  Train: loss {tr_loss:.4f} acc {tr_acc:.4f}")
            print(f"  Val  : loss {val_loss:.4f} acc {val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), best_path)
                print("  -> saved best CAF model")

        best_model = build_model()
        best_model.load_state_dict(torch.load(best_path, map_location=DEVICE))

        test_loss, test_acc, y_true_t, y_pred_t, y_score_t = evaluate(
            best_model, test_loader, criterion, return_scores=True
        )

        print("\nFINAL TEST RESULTS (CAF mix):", exp_name)
        print(f"  λ = {lam:.1f}")
        print(f"  Test loss {test_loss:.4f} acc {test_acc:.4f}")

        cm = confusion_matrix(y_true_t, y_pred_t)
        print("\nTest Confusion Matrix (raw):\n", cm)
        print(
            "\nTest Classification Report:\n",
            classification_report(
                y_true_t,
                y_pred_t,
                target_names=["no_pneumonia(0)", "pneumonia(1)"],
            ),
        )

        cm_path = IMAGES_DIR / f"{exp_name}_cm.png"
        roc_path = IMAGES_DIR / f"{exp_name}_roc.png"

        plot_confusion_matrix(
            cm,
            class_names=["no_pneumonia(0)", "pneumonia(1)"],
            title=f"{exp_name} (CAF λ={lam:.1f}) — Test Confusion Matrix",
            save_path=cm_path,
        )
        roc_auc = plot_roc_curve(
            y_true_t,
            y_score_t,
            title=f"{exp_name} (CAF λ={lam:.1f}) — ROC Curve (Test)",
            save_path=roc_path,
        )

        return {
            "name": f"{exp_name}_lam{lam:.1f}",
            "test_loss": float(test_loss),
            "test_acc": float(test_acc),
            "test_auc": float(roc_auc),
        }

    # Non-CAF experiments: plain NLL / class-weighted NLL / Focal
    model = build_model()

    if use_focal_loss:
        focal_alpha = torch.tensor(FOCAL_ALPHA, dtype=torch.float32, device=DEVICE)
        criterion = FocalLoss(alpha=focal_alpha, gamma=FOCAL_GAMMA, reduction="mean")
    elif use_class_weighted:
        criterion = nn.NLLLoss(weight=class_weights_tensor)
    else:
        criterion = nn.NLLLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    best_val_acc = 0.0
    best_path = Path(f"./best_vit_rsna_{exp_name}.pth")

    for epoch in range(1, NUM_EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, epoch, NUM_EPOCHS
        )
        val_loss, val_acc, y_true_v, y_pred_v = evaluate(
            model, val_loader, criterion, return_scores=False
        )

        print(f"Epoch {epoch}/{NUM_EPOCHS}")
        print(f"  Train: loss {tr_loss:.4f} acc {tr_acc:.4f}")
        print(f"  Val  : loss {val_loss:.4f} acc {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)
            print("  -> saved best model")

    best_model = build_model()
    best_model.load_state_dict(torch.load(best_path, map_location=DEVICE))

    test_loss, test_acc, y_true_t, y_pred_t, y_score_t = evaluate(
        best_model, test_loader, criterion, return_scores=True
    )

    print("\nFINAL TEST RESULTS:", exp_name)
    print(f"  Test loss {test_loss:.4f} acc {test_acc:.4f}")

    cm = confusion_matrix(y_true_t, y_pred_t)
    print("\nTest Confusion Matrix (raw):\n", cm)
    print(
        "\nTest Classification Report:\n",
        classification_report(
            y_true_t,
            y_pred_t,
            target_names=["no_pneumonia(0)", "pneumonia(1)"],
        ),
    )

    cm_path = IMAGES_DIR / f"{exp_name}_cm.png"
    roc_path = IMAGES_DIR / f"{exp_name}_roc.png"

    plot_confusion_matrix(
        cm,
        class_names=["no_pneumonia(0)", "pneumonia(1)"],
        title=f"{exp_name} — Test Confusion Matrix",
        save_path=cm_path,
    )
    roc_auc = plot_roc_curve(
        y_true_t,
        y_score_t,
        title=f"{exp_name} — ROC Curve (Test)",
        save_path=roc_path,
    )

    return {
        "name": exp_name,
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "test_auc": float(roc_auc),
    }


###################################
###################################
###################################

# MAIN: run all experiments


def main():
    # 1) Load data + splits
    df_full = load_rsna_dataframe()
    train_df, val_df, test_df = make_splits(df_full)

    # 2) Class weights for NLL / mixed loss, oversample factor (for optional combo)
    class_weights_np, oversample_factor = compute_class_weights(train_df)
    class_weights_tensor = torch.tensor(
        class_weights_np, dtype=torch.float32, device=DEVICE
    )

    results = []

    # 1) Basic ViT (no special imbalance handling)
    results.append(
        run_experiment(
            "baseline_ViT",
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            class_weights_tensor=class_weights_tensor,
            use_oversample=False,
            use_minority_aug=False,
            use_class_weighted=False,
            use_focal_loss=False,
            use_caf_nll=False,
        )
    )

    # 2) Oversample minority class (WeightedRandomSampler)
    results.append(
        run_experiment(
            "oversample_minority",
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            class_weights_tensor=class_weights_tensor,
            use_oversample=True,
            use_minority_aug=False,
            use_class_weighted=False,
            use_focal_loss=False,
            use_caf_nll=False,
        )
    )

    # 3) Augment minority class (stronger transforms for pneumonia-positive)
    results.append(
        run_experiment(
            "augment_minority",
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            class_weights_tensor=class_weights_tensor,
            use_oversample=False,
            use_minority_aug=True,
            use_class_weighted=False,
            use_focal_loss=False,
            use_caf_nll=False,
        )
    )

    # 4) Class-weighted NLLLoss
    results.append(
        run_experiment(
            "class_weighted_NLL",
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            class_weights_tensor=class_weights_tensor,
            use_oversample=False,
            use_minority_aug=False,
            use_class_weighted=True,
            use_focal_loss=False,
            use_caf_nll=False,
        )
    )

    # 5) Focal Loss
    results.append(
        run_experiment(
            "focal_loss",
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            class_weights_tensor=class_weights_tensor,
            use_oversample=False,
            use_minority_aug=False,
            use_class_weighted=False,
            use_focal_loss=True,
            use_caf_nll=False,
        )
    )

    # 6) CAF mixed loss (weighted NLL + focal)
    results.append(
        run_experiment(
            "caf_nll",
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            class_weights_tensor=class_weights_tensor,
            use_oversample=False,
            use_minority_aug=False,
            use_class_weighted=False,
            use_focal_loss=False,
            use_caf_nll=True,
        )
    )

    print("\n\n================ SUMMARY (Test metrics) ================")
    for r in results:
        print(
            f"{r['name']:>24s}  "
            f"loss={r['test_loss']:.4f}  acc={r['test_acc']:.4f}  auc={r['test_auc']:.4f}"
        )


if __name__ == "__main__":
    main()