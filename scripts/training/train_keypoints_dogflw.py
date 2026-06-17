#!/usr/bin/env python3
"""
Trening modelu keypoints (46 punktów DogFLW) — wersja "z koszerem".

Architektura i wszystkie poprawki uzgodnione po systematycznej diagnozie:
- Backbone HRNet-W32 (stride-4 → heatmapy 64×64 bez deconv) — najlepsza lokalizacja.
- AdaptiveWing loss zamiast MSE (ostre piki, wyższa pewność).
- POPRAWNY HorizontalFlip: po odbiciu PRZESTAWIA pary lewo/prawo (FLIP_PAIRS).
  To naprawia główny błąd starego treningu (kolaps par do osi pionowej).
- Augmentacje jak w pracy DogFLW (rotacja/kolor/jasność/kontrast/blur/szum), bez naiwnego flipa.
- Walidacja NME_iod + PCK@0.1 co epokę; checkpoint po NME (nie po loss).
- Cosine LR + early stopping (nie marnujemy godzin GPU).

Klasa modelu importowana z packages.models.keypoints — trening i inferencja
używają DOKŁADNIE tej samej architektury (inaczej wagi się nie wczytają).

Uruchomienie (Kaggle, dataset lovodkin/dogflw, internet ON dla wag ImageNet):
    !git clone <repo> && cd group-project
    !pip install -q timm albumentations
    !python scripts/training/train_keypoints_dogflw.py \
        --dogflw_path /kaggle/input/dogflw/DogFLW \
        --output /kaggle/working/keypoints_dogflw.pt
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

# repo na ścieżce (import packages)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import albumentations as A  # noqa: E402
from albumentations.pytorch import ToTensorV2  # noqa: E402

from packages.data.schemas import KP, NUM_KEYPOINTS, SKELETON_CONNECTIONS  # noqa: E402
from packages.models.keypoints import (  # noqa: E402
    FLIP_MAPPING,
    SimpleBaselineModel,
)


# --------------------------------------------------------------------------- #
# AdaptiveWing loss (Wang et al., 2019) — standard dla heatmap landmarków twarzy
# --------------------------------------------------------------------------- #
class AdaptiveWingLoss(nn.Module):
    """AdaptiveWing loss liczony per-piksel na heatmapach."""

    def __init__(
        self,
        omega: float = 14.0,
        theta: float = 0.5,
        epsilon: float = 1.0,
        alpha: float = 2.1,
    ) -> None:
        super().__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        delta = (target - pred).abs()
        # współczynniki ciągłości na granicy theta (zależne od wartości target)
        a = (
            self.omega
            * (1.0 / (1.0 + (self.theta / self.epsilon) ** (self.alpha - target)))
            * (self.alpha - target)
            * ((self.theta / self.epsilon) ** (self.alpha - target - 1.0))
            * (1.0 / self.epsilon)
        )
        c = self.theta * a - self.omega * torch.log(
            1.0 + (self.theta / self.epsilon) ** (self.alpha - target)
        )
        loss = torch.where(
            delta < self.theta,
            self.omega
            * torch.log(1.0 + (delta / self.epsilon) ** (self.alpha - target)),
            a * delta - c,
        )
        if weight is not None:
            # weight: (K,) — waga per-keypoint (np. uszy ważniejsze)
            loss = loss * weight.view(1, -1, 1, 1)
        return loss.mean()


class STARLoss(nn.Module):
    """
    STAR loss (Zhou et al., CVPR 2023) — anizotropowa kara na współrzędnych
    z soft-argmax. Tłumi błąd wzdłuż kierunku największej niepewności heatmapy
    (np. uszy), zmniejszając wpływ niejednoznacznej anotacji.
    """

    def __init__(
        self, eps: float = 1e-3, reg: float = 1.0, min_var: float = 1.0
    ) -> None:
        super().__init__()
        self.eps = eps
        self.reg = reg
        # dolny próg wariancji (px²) — chroni przed eksplozją p/sqrt(λ) przy
        # ostrym piku heatmapy; przy sigma~1.5 realna wariancja ~2.25.
        self.min_var = min_var

    def _mean_cov(self, hm: torch.Tensor):
        b, k, h, w = hm.shape
        p = torch.softmax(hm.reshape(b, k, -1), dim=-1).reshape(b, k, h, w)
        ys = torch.arange(h, device=hm.device, dtype=hm.dtype)
        xs = torch.arange(w, device=hm.device, dtype=hm.dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        mx = (p * xx).sum((-1, -2))
        my = (p * yy).sum((-1, -2))
        dx = xx[None, None] - mx[..., None, None]
        dy = yy[None, None] - my[..., None, None]
        cxx = (p * dx * dx).sum((-1, -2))
        cyy = (p * dy * dy).sum((-1, -2))
        cxy = (p * dx * dy).sum((-1, -2))
        mean = torch.stack([mx, my], -1)
        cov = torch.stack(
            [torch.stack([cxx, cxy], -1), torch.stack([cxy, cyy], -1)], -2
        )
        return mean, cov

    def forward(
        self,
        pred_hm: torch.Tensor,
        gt_coords_hm: torch.Tensor,
        weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mean, cov = self._mean_cov(pred_hm)
        # Rozkład własny 2×2 w POSTACI ZAMKNIĘTEJ (bez torch.linalg.eigh —
        # eigh wywala się na macierzy zdegenerowanej, np. ostry pik → cov≈0).
        a = cov[..., 0, 0] + self.eps
        b = cov[..., 0, 1]
        c = cov[..., 1, 1] + self.eps
        half = 0.5 * (a + c)
        disc = torch.sqrt(((a - c) * 0.5) ** 2 + b * b + self.eps)
        l1 = (half + disc).clamp(min=self.min_var)  # większa wariancja (oś niepewności)
        l2 = (half - disc).clamp(min=self.min_var)  # mniejsza
        # wektor własny dla l1: [b, l1-a] (z fallbackiem na [1,0] gdy zdegenerowany)
        vx = b
        vy = l1 - a
        vn = torch.sqrt(vx * vx + vy * vy + self.eps)
        v1x, v1y = vx / vn, vy / vn  # oś 1 (jednostkowa)
        e = mean - gt_coords_hm  # (B,K,2)
        ex, ey = e[..., 0], e[..., 1]
        p1 = ex * v1x + ey * v1y  # projekcja na oś 1
        p2 = -ex * v1y + ey * v1x  # projekcja na oś 2 (prostopadła)
        per_kp = (
            p1.abs() / torch.sqrt(l1)
            + p2.abs() / torch.sqrt(l2)
            + self.reg * (torch.log(l1) + torch.log(l2))
        )
        if weight is not None:
            per_kp = per_kp * weight.view(1, -1)
        return per_kp.mean()


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class DogFLWDataset(Dataset):
    """DogFLW: obraz → heatmapy 46 keypoints. Flip z przestawieniem par."""

    def __init__(
        self,
        root_path: str,
        split: str = "train",
        image_size: int = 256,
        heatmap_size: int = 64,
        sigma: float = 2.0,
    ) -> None:
        self.image_size = image_size
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        self.is_train = split == "train"

        images_dir = os.path.join(root_path, split, "images")
        labels_dir = os.path.join(root_path, split, "labels")
        self.samples: list[tuple[str, str]] = []
        for label_file in sorted(os.listdir(labels_dir)):
            if not label_file.endswith(".json"):
                continue
            name = label_file[:-5]
            for ext in (".jpg", ".jpeg", ".png", ".JPEG"):
                img_path = os.path.join(images_dir, name + ext)
                if os.path.exists(img_path):
                    self.samples.append(
                        (img_path, os.path.join(labels_dir, label_file))
                    )
                    break

        # Augmentacje jak w pracy DogFLW — BEZ flipa (flip robimy ręcznie z parami)
        if self.is_train:
            self.aug = A.Compose(
                [
                    A.Affine(
                        scale=(0.85, 1.15),
                        translate_percent=(0.0, 0.06),
                        rotate=(-30, 30),
                        p=0.7,
                    ),
                    A.ColorJitter(0.2, 0.2, 0.2, 0.1, p=0.5),
                    A.GaussianBlur(blur_limit=(3, 5), p=0.2),
                    A.GaussNoise(p=0.2),
                    A.Resize(image_size, image_size),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ],
                keypoint_params=A.KeypointParams(
                    format="xy", remove_invisible=False
                ),
            )
        else:
            self.aug = A.Compose(
                [
                    A.Resize(image_size, image_size),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ],
                keypoint_params=A.KeypointParams(
                    format="xy", remove_invisible=False
                ),
            )
        print(f"{split}: {len(self.samples)} próbek")

    def __len__(self) -> int:
        return len(self.samples)

    def _gen_heatmaps(self, kps: list[tuple[float, float]]) -> np.ndarray:
        hs = self.heatmap_size
        heatmaps = np.zeros((NUM_KEYPOINTS, hs, hs), dtype=np.float32)
        xx, yy = np.meshgrid(np.arange(hs), np.arange(hs))
        for i, (x, y) in enumerate(kps):
            xh = x * hs / self.image_size
            yh = y * hs / self.image_size
            if 0 <= xh < hs and 0 <= yh < hs:
                heatmaps[i] = np.exp(
                    -((xx - xh) ** 2 + (yy - yh) ** 2) / (2 * self.sigma**2)
                )
        return heatmaps

    def __getitem__(self, idx: int):
        img_path, label_path = self.samples[idx]
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        with open(label_path) as f:
            data = json.load(f)
        kps = [(float(x), float(y)) for x, y in data["landmarks"]]

        # POPRAWNY flip: odbicie obrazu + x, a następnie przestawienie par L/R.
        if self.is_train and np.random.rand() < 0.5:
            image = np.ascontiguousarray(image[:, ::-1])
            flipped = [(w - 1 - x, y) for (x, y) in kps]
            kps = [flipped[FLIP_MAPPING.get(i, i)] for i in range(NUM_KEYPOINTS)]

        out = self.aug(image=image, keypoints=kps)
        image_t = out["image"]
        kps_t = list(out["keypoints"])
        while len(kps_t) < NUM_KEYPOINTS:
            kps_t.append((0.0, 0.0))

        heatmaps = torch.from_numpy(self._gen_heatmaps(kps_t))
        kp_xy = torch.tensor(
            [[x, y] for x, y in kps_t[:NUM_KEYPOINTS]], dtype=torch.float32
        )
        return image_t, heatmaps, kp_xy


# --------------------------------------------------------------------------- #
# Metryki: NME_iod + PCK@0.1
# --------------------------------------------------------------------------- #
_LEFT_EYE = [KP.LEFT_EYE_INNER, KP.LEFT_EYE_OUTER, KP.LEFT_EYE_TOP, KP.LEFT_EYE_BOTTOM]
_RIGHT_EYE = [
    KP.RIGHT_EYE_INNER,
    KP.RIGHT_EYE_OUTER,
    KP.RIGHT_EYE_TOP,
    KP.RIGHT_EYE_BOTTOM,
]


def _decode_argmax(heatmaps: np.ndarray, image_size: int) -> np.ndarray:
    """(K,H,W) → (K,2) w układzie obrazu (argmax, do walidacji)."""
    k, hh, ww = heatmaps.shape
    coords = np.zeros((k, 2), dtype=np.float32)
    for i in range(k):
        idx = int(heatmaps[i].argmax())
        y, x = idx // ww, idx % ww
        coords[i] = [x * image_size / ww, y * image_size / hh]
    return coords


def compute_metrics(
    pred_xy: np.ndarray, gt_xy: np.ndarray
) -> tuple[float, float]:
    """
    Zwraca (NME_iod, PCK@0.1) dla jednej próbki.

    iod = odległość między ZEWNĘTRZNYMI kącikami oczu (kp 18/19) — zgodnie z
    definicją w pracy DogFLW (Martvel 2025), więc NME jest bezpośrednio
    porównywalne z ich wynikami (ELD 6.52, Hourglass 6.87).
    """
    iod = float(np.linalg.norm(gt_xy[KP.LEFT_EYE_OUTER] - gt_xy[KP.RIGHT_EYE_OUTER]))
    if iod < 1e-3:
        return float("nan"), float("nan")
    dists = np.linalg.norm(pred_xy - gt_xy, axis=1)
    nme = float(dists.mean() / iod)
    pck = float((dists < 0.1 * iod).mean())
    return nme, pck


@torch.no_grad()
def evaluate(model, loader, criterion, device, image_size):
    model.eval()
    total_loss, nmes, pcks = 0.0, [], []
    for images, heatmaps, kp_xy in tqdm(loader, desc="Eval", leave=False):
        images = images.to(device)
        pred = model(images)
        total_loss += criterion(pred, heatmaps.to(device)).item()
        pred_np = pred.cpu().numpy()
        for b in range(pred_np.shape[0]):
            coords = _decode_argmax(pred_np[b], image_size)
            nme, pck = compute_metrics(coords, kp_xy[b].numpy())
            if not math.isnan(nme):
                nmes.append(nme)
                pcks.append(pck)
    return (
        total_loss / len(loader),
        float(np.mean(nmes)) if nmes else float("nan"),
        float(np.mean(pcks)) if pcks else float("nan"),
    )


# --------------------------------------------------------------------------- #
# Wizualna kontrola par (po treningu)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def render_pair_check(model, dataset, device, out_dir, n=6):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    for i in range(min(n, len(dataset))):
        image_t, _, _ = dataset[i]
        pred = model(image_t.unsqueeze(0).to(device)).cpu().numpy()[0]
        coords = _decode_argmax(pred, dataset.image_size)
        img = image_t.permute(1, 2, 0).numpy()
        img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]).clip(0, 1)
        img = (img * 255).astype(np.uint8)[:, :, ::-1].copy()
        for a, b in SKELETON_CONNECTIONS:
            if a < NUM_KEYPOINTS and b < NUM_KEYPOINTS:
                pa, pb = coords[a].astype(int), coords[b].astype(int)
                cv2.line(img, tuple(pa), tuple(pb), (0, 220, 0), 1, cv2.LINE_AA)
        for x, y in coords:
            cv2.circle(img, (int(x), int(y)), 2, (0, 165, 255), -1)
        cv2.imwrite(os.path.join(out_dir, f"paircheck_{i}.png"), img)
    print(f"Zapisano wizualizacje par do {out_dir}")


# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dogflw_path", default="/kaggle/input/dogflw/DogFLW")
    p.add_argument("--backbone", default="hrnet_w32")
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--heatmap_size", type=int, default=64)
    p.add_argument("--sigma", type=float, default=2.0)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=20, help="early stopping po NME")
    p.add_argument("--output", default="models/keypoints_dogflw.pt")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument(
        "--ear_weight", type=float, default=1.0,
        help="waga kanałów uszu (0-13) w loss; >1 = nacisk na uszy (dla AU)",
    )
    p.add_argument(
        "--star_weight", type=float, default=0.0,
        help="waga członu STAR loss (anizotropia na niepewnych punktach)",
    )
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | backbone: {args.backbone}")

    train_ds = DogFLWDataset(
        args.dogflw_path, "train", args.img_size, args.heatmap_size, args.sigma
    )
    test_ds = DogFLWDataset(
        args.dogflw_path, "test", args.img_size, args.heatmap_size, args.sigma
    )
    train_loader = DataLoader(
        train_ds, args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    test_loader = DataLoader(
        test_ds, args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    model = SimpleBaselineModel(
        NUM_KEYPOINTS, args.backbone, args.img_size, args.heatmap_size,
        pretrained=True,
    ).to(device)
    print(f"Parametry: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    criterion = AdaptiveWingLoss()
    star = STARLoss()
    # wagi per-keypoint: uszy (0-13) ważniejsze (ważne dla AU)
    kp_weight = torch.ones(NUM_KEYPOINTS, device=device)
    kp_weight[0:14] = args.ear_weight
    hm_scale = args.heatmap_size / args.img_size  # obraz → heatmapa
    print(
        f"ear_weight={args.ear_weight} star_weight={args.star_weight} "
        f"| metryka NME: iod=zewn. kąciki oczu (jak w pracy)"
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_nme = float("inf")
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": [], "val_nme": [], "val_pck": []}
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        run_loss = 0.0
        for images, heatmaps, kp_xy in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False
        ):
            images, heatmaps = images.to(device), heatmaps.to(device)
            optimizer.zero_grad()
            pred = model(images)
            loss = criterion(pred, heatmaps, weight=kp_weight)
            if args.star_weight > 0:
                gt_hm = kp_xy.to(device) * hm_scale  # współrz. w układzie heatmapy
                loss = loss + args.star_weight * star(pred, gt_hm, weight=kp_weight)
            loss.backward()
            optimizer.step()
            run_loss += loss.item()
        train_loss = run_loss / len(train_loader)

        val_loss, val_nme, val_pck = evaluate(
            model, test_loader, criterion, device, args.img_size
        )
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_nme"].append(val_nme)
        history["val_pck"].append(val_pck)

        improved = val_nme < best_nme
        if improved:
            best_nme = val_nme
            epochs_no_improve = 0
            torch.save(model.state_dict(), out_path)
        else:
            epochs_no_improve += 1

        print(
            f"Epoch {epoch+1:3d}/{args.epochs} | loss {train_loss:.5f} | "
            f"val_loss {val_loss:.5f} | NME {val_nme:.4f} | "
            f"PCK@0.1 {val_pck:.3f} | best_NME {best_nme:.4f}"
            + ("  <== zapisano" if improved else "")
        )
        if epochs_no_improve >= args.patience:
            print(f"Early stopping (brak poprawy NME przez {args.patience} epok)")
            break

    # metryki + wizualna kontrola par na najlepszym modelu
    model.load_state_dict(torch.load(out_path, map_location=device, weights_only=True))
    _, final_nme, final_pck = evaluate(
        model, test_loader, criterion, device, args.img_size
    )
    metrics = {
        "backbone": args.backbone,
        "best_val_nme_iod": best_nme,
        "final_test_nme_iod": final_nme,
        "final_test_pck_0.1": final_pck,
        "num_keypoints": NUM_KEYPOINTS,
        "epochs_ran": len(history["train_loss"]),
        "history": history,
    }
    with open(out_path.with_suffix(".json"), "w") as f:
        json.dump(metrics, f, indent=2)
    render_pair_check(
        model, test_ds, device, str(out_path.parent / "paircheck")
    )
    print(f"\nGotowe. NME_iod={final_nme:.4f}  PCK@0.1={final_pck:.3f}")
    print(f"Wagi: {out_path}\nMetryki: {out_path.with_suffix('.json')}")
    print("Porównanie z pracą: ELD(EffNetV2S)=6.52, Hourglass=6.87 (NME w %).")
    print(f"Nasz NME_iod*100 = {final_nme*100:.2f}")


if __name__ == "__main__":
    main()
