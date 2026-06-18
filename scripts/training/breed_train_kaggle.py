#!/usr/bin/env python3
"""
Trening klasyfikatora ras psów na Kaggle (GPU) — EfficientNet-B4 @ 380.

Uruchamiany jako Kaggle kernel z podłączonym datasetem:
  jessicali9530/stanford-dogs-dataset

Cechy:
- crop po bounding boxie Stanford (zgodność domeny z inference = crop psa od YOLO),
- SquarePad jak w packages/models/breed.py,
- label smoothing + MixUp/CutMix + RandAugment + EMA + AMP,
- wybór najlepszego checkpointu po Top-1 (model EMA),
- deterministyczna, czysta breeds.json (kolejność alfabetyczna nazw ras),
- ewaluacja: Top-1/Top-5, per-class, najczęstsze pomyłki.

Wyjście (w /kaggle/working): best.pt, breeds.json, breed_metrics.json
"""

import subprocess
import sys

# Kaggle "latest" PyTorch bywa zbudowany bez kerneli dla GPU P100 (sm_60),
# co daje błąd "no kernel image is available for execution on the device".
# Instalujemy oficjalne wheele PyTorch (cu121) z pełnym wsparciem architektur
# (Pascal P100 ... Ada/Hopper) — działa na każdym przydzielonym GPU.
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q",
     "torch==2.4.1", "torchvision==0.19.1",
     "--index-url", "https://download.pytorch.org/whl/cu121"],
    check=False,
)

import json
import random
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

# timm bywa wstępnie zainstalowany; gwarantujemy obecność
try:
    import timm
except ImportError:  # pragma: no cover
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "timm"], check=True)
    import timm

from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy

# ----------------------------------------------------------------------------
# Konfiguracja
# ----------------------------------------------------------------------------
IMG_SIZE = 380
BATCH = 24
EPOCHS = 30
WARMUP_EPOCHS = 3
LR = 1e-3
WEIGHT_DECAY = 0.05
LABEL_SMOOTH = 0.1
BBOX_MARGIN = 0.12       # margines wokół bboxa psa
NUM_WORKERS = 2
SEED = 42
MODEL_NAME = "efficientnet_b4"

WORK = Path("/kaggle/working")
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# ----------------------------------------------------------------------------
# Lokalizacja danych
# ----------------------------------------------------------------------------
ROOT = Path("/kaggle/input")
images_dir = next((p for p in ROOT.rglob("Images") if p.is_dir()), None)
annot_dir = next((p for p in ROOT.rglob("Annotation") if p.is_dir()), None)
assert images_dir is not None, "Nie znaleziono katalogu Images"
print(f"Images: {images_dir}")
print(f"Annotation: {annot_dir}")


def breed_name(folder: str) -> str:
    """n02085620-Chihuahua -> Chihuahua."""
    name = folder.split("-", 1)[1] if "-" in folder else folder
    return name.replace("_", " ").strip()


# Kolejność klas: alfabetycznie wg nazwy rasy (deterministyczne, czyste)
breed_dirs = [d for d in images_dir.iterdir() if d.is_dir()]
pairs = sorted(((breed_name(d.name), d) for d in breed_dirs), key=lambda x: x[0].lower())
CLASSES = [p[0] for p in pairs]
DIRS = [p[1] for p in pairs]
NUM_CLASSES = len(CLASSES)
print(f"Liczba ras: {NUM_CLASSES}")

# Czysta breeds.json
breeds_json = {str(i): CLASSES[i] for i in range(NUM_CLASSES)}
(WORK / "breeds.json").write_text(
    json.dumps(breeds_json, indent=2, ensure_ascii=False), encoding="utf-8"
)


def load_bbox(folder_name: str, stem: str):
    """Wczytuje bbox z adnotacji Stanford (lub None)."""
    if annot_dir is None:
        return None
    f = annot_dir / folder_name / stem
    if not f.exists():
        return None
    try:
        root = ET.parse(f).getroot()
        b = root.find(".//bndbox")
        return (
            int(float(b.find("xmin").text)),
            int(float(b.find("ymin").text)),
            int(float(b.find("xmax").text)),
            int(float(b.find("ymax").text)),
        )
    except Exception:
        return None


# Lista próbek: (ścieżka, label, folder_name)
samples = []
for label, d in enumerate(DIRS):
    for img in list(d.glob("*.jpg")) + list(d.glob("*.JPEG")):
        samples.append((img, label, d.name))
print(f"Łącznie obrazów: {len(samples)}")

# Stratyfikowany podział 80/10/10
by_class = {}
for s in samples:
    by_class.setdefault(s[1], []).append(s)

train_s, val_s, test_s = [], [], []
rng = np.random.default_rng(SEED)
for label, items in by_class.items():
    idx = rng.permutation(len(items))
    n = len(items)
    n_tr = int(n * 0.8)
    n_va = int(n * 0.1)
    for j, i in enumerate(idx):
        if j < n_tr:
            train_s.append(items[i])
        elif j < n_tr + n_va:
            val_s.append(items[i])
        else:
            test_s.append(items[i])
print(f"Train/Val/Test: {len(train_s)}/{len(val_s)}/{len(test_s)}")


# ----------------------------------------------------------------------------
# Transformacje (SquarePad zgodny z inference)
# ----------------------------------------------------------------------------
class SquarePad:
    def __call__(self, im: Image.Image) -> Image.Image:
        w, h = im.size
        m = max(w, h)
        left = (m - w) // 2
        top = (m - h) // 2
        return ImageOps.expand(im, (left, top, m - w - left, m - h - top), fill=0)


train_tf = T.Compose([
    SquarePad(),
    T.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0), ratio=(0.85, 1.18)),
    T.RandomHorizontalFlip(0.5),
    T.RandAugment(num_ops=2, magnitude=9),
    T.ColorJitter(0.2, 0.2, 0.2, 0.05),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
    T.RandomErasing(p=0.25),
])

eval_tf = T.Compose([
    SquarePad(),
    T.Resize(int(IMG_SIZE * 1.14)),
    T.CenterCrop(IMG_SIZE),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])


class DogDataset(Dataset):
    def __init__(self, items, tf, use_bbox=True):
        self.items = items
        self.tf = tf
        self.use_bbox = use_bbox

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label, folder = self.items[i]
        img = Image.open(path).convert("RGB")
        if self.use_bbox:
            bb = load_bbox(folder, path.stem)
            if bb is not None:
                x0, y0, x1, y1 = bb
                bw, bh = x1 - x0, y1 - y0
                mx, my = int(bw * BBOX_MARGIN), int(bh * BBOX_MARGIN)
                x0 = max(0, x0 - mx)
                y0 = max(0, y0 - my)
                x1 = min(img.width, x1 + mx)
                y1 = min(img.height, y1 + my)
                if x1 > x0 and y1 > y0:
                    img = img.crop((x0, y0, x1, y1))
        return self.tf(img), label


train_loader = DataLoader(
    DogDataset(train_s, train_tf), batch_size=BATCH, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
)
val_loader = DataLoader(
    DogDataset(val_s, eval_tf), batch_size=BATCH, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True,
)
test_loader = DataLoader(
    DogDataset(test_s, eval_tf), batch_size=BATCH, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True,
)


# ----------------------------------------------------------------------------
# Model, EMA, optymalizator
# ----------------------------------------------------------------------------
model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=NUM_CLASSES).to(device)

try:
    from timm.utils import ModelEmaV2
    ema = ModelEmaV2(model, decay=0.9998)
except Exception:
    from timm.utils import ModelEmaV3
    ema = ModelEmaV3(model, decay=0.9998)

mixup_fn = Mixup(
    mixup_alpha=0.2, cutmix_alpha=1.0, prob=0.5, switch_prob=0.5,
    mode="batch", label_smoothing=LABEL_SMOOTH, num_classes=NUM_CLASSES,
)
train_criterion = SoftTargetCrossEntropy()
eval_criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scaler = torch.amp.GradScaler('cuda')

steps_per_epoch = len(train_loader)
total_steps = EPOCHS * steps_per_epoch
warmup_steps = WARMUP_EPOCHS * steps_per_epoch


def lr_at(step: int) -> float:
    if step < warmup_steps:
        return LR * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * LR * (1 + np.cos(np.pi * progress))


@torch.no_grad()
def evaluate(eval_model, loader, tta=False):
    eval_model.eval()
    correct1 = correct5 = total = 0
    per_correct = np.zeros(NUM_CLASSES)
    per_total = np.zeros(NUM_CLASSES)
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.amp.autocast('cuda'):
            out = eval_model(x)
            if tta:
                out = out + eval_model(torch.flip(x, dims=[3]))
        _, top5 = out.topk(5, dim=1)
        pred = top5[:, 0]
        correct1 += (pred == y).sum().item()
        correct5 += sum(y[i] in top5[i] for i in range(y.size(0)))
        total += y.size(0)
        for t, p in zip(y.cpu().numpy(), pred.cpu().numpy()):
            per_total[t] += 1
            confusion[t, p] += 1
            if t == p:
                per_correct[t] += 1
    top1 = correct1 / total
    top5 = correct5 / total
    return top1, top5, per_correct, per_total, confusion


# ----------------------------------------------------------------------------
# Trening
# ----------------------------------------------------------------------------
best_top1 = 0.0
step = 0
print("\n=== Trening ===")
for epoch in range(EPOCHS):
    model.train()
    running = 0.0
    for x, y in train_loader:
        for g in optimizer.param_groups:
            g["lr"] = lr_at(step)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        x, y_soft = mixup_fn(x, y)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            out = model(x)
            loss = train_criterion(out, y_soft)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        ema.update(model)
        running += loss.item()
        step += 1

    val_top1, val_top5, *_ = evaluate(ema.module, val_loader)
    print(
        f"Epoch {epoch + 1}/{EPOCHS}  loss={running / steps_per_epoch:.3f}  "
        f"lr={lr_at(step):.2e}  val_top1={val_top1 * 100:.2f}%  val_top5={val_top5 * 100:.2f}%",
        flush=True,
    )
    if val_top1 > best_top1:
        best_top1 = val_top1
        torch.save(ema.module.state_dict(), WORK / "best.pt")
        print(f"  -> nowy najlepszy (EMA) val_top1={best_top1 * 100:.2f}%", flush=True)


# ----------------------------------------------------------------------------
# Ewaluacja końcowa na teście (najlepszy model EMA + TTA)
# ----------------------------------------------------------------------------
print("\n=== Test (najlepszy model, TTA) ===")
best = timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES).to(device)
best.load_state_dict(torch.load(WORK / "best.pt", map_location=device, weights_only=True))
t1, t5, per_c, per_t, conf = evaluate(best, test_loader, tta=True)
print(f"TEST Top-1={t1 * 100:.2f}%  Top-5={t5 * 100:.2f}%")

per_class_acc = {
    str(i): float(per_c[i] / per_t[i]) if per_t[i] > 0 else 0.0
    for i in range(NUM_CLASSES)
}
# Najczęstsze pomyłki (poza przekątną)
conf_off = conf.copy()
np.fill_diagonal(conf_off, 0)
top_conf = []
flat = np.dstack(np.unravel_index(np.argsort(conf_off.ravel())[::-1], conf_off.shape))[0]
for t, p in flat[:25]:
    c = int(conf_off[t, p])
    if c <= 0:
        break
    top_conf.append([CLASSES[t], CLASSES[p], c])

metrics = {
    "model": MODEL_NAME,
    "img_size": IMG_SIZE,
    "num_classes": NUM_CLASSES,
    "test_top1": float(t1),
    "test_top5": float(t5),
    "best_val_top1": float(best_top1),
    "n_test": int(per_t.sum()),
    "per_class_accuracy": per_class_acc,
    "top_confusions": top_conf,
}
(WORK / "breed_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))

# Posortowane najgorsze rasy
worst = sorted(per_class_acc.items(), key=lambda x: x[1])[:15]
print("\n15 najgorszych ras:")
for cid, acc in worst:
    print(f"  {CLASSES[int(cid)]:30s} {acc * 100:5.1f}%")
print("\nNajczęstsze pomyłki:")
for a, b, c in top_conf[:10]:
    print(f"  {a} -> {b}: {c}")

print("\nGotowe. Zapisano: best.pt, breeds.json, breed_metrics.json")
