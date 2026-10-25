"""Experimentos controlados sobre el clasificador de tipo de keypoint (Unidad 4, Actividad 1).

Reutiliza los datos, el split por rally y las arquitecturas de
notebooks/unidad_3/actividad_3_unidad_3_transfer_learning.ipynb, pero mueve la augmentacion a la GPU:
cada parche se recorta UNA vez desde la imagen original con un margen de contexto (CONTEXT x el lado
del parche), y el jitter de escala/posicion, el flip, la rotacion y el color se aplican por lotes en la
GPU. En la Unidad 3 todo eso se hacia por muestra en CPU y la epoca tardaba ~26 s; aqui tarda ~1-2 s,
lo que hace viable una busqueda de hiperparametros dentro de ~1 h de GPU.

Todo experimento se registra en MLflow: parametros, metricas por epoca, metricas de sistema
(CPU/GPU/memoria), costo de entrenamiento e inferencia, y artefactos.
"""
import copy
import json
import math
import os
import platform
import random
import re
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torchvision.models import ResNet18_Weights, resnet18

KP_NAMES = ["hombro", "codo", "muneca", "raqueta", "cabeza"]
N_CLASSES = len(KP_NAMES)
PATCH_FRAC, MIN_PATCH, MAX_PATCH = 0.5, 24, 90  # mismo recorte que la Unidad 3
CONTEXT = 1.7  # margen del recorte almacenado: cubre el jitter "fuerte" (escala 1.25, desplazamiento 15%)
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
STAGES = ["stem", "layer1", "layer2", "layer3", "layer4"]
RALLY_RE = re.compile(r"^(.*_mp4)-(\d+)_jpg")

# Intensidades de augmentacion. "original" es exactamente la de la Unidad 3:
# escala x[0.85, 1.15], desplazamiento +-10% del lado, flip 0.5, rotacion +-12, ColorJitter(.3, .3, .2, .02)
AUG_LEVELS = {
    "ninguna": dict(scale=0.0, shift=0.0, flip=0.0, rot=0.0, bright=0.0, contrast=0.0, sat=0.0, hue=0.0),
    "suave": dict(scale=0.08, shift=0.05, flip=0.5, rot=6.0, bright=0.15, contrast=0.15, sat=0.1, hue=0.01),
    "original": dict(scale=0.15, shift=0.10, flip=0.5, rot=12.0, bright=0.3, contrast=0.3, sat=0.2, hue=0.02),
    "fuerte": dict(scale=0.25, shift=0.15, flip=0.5, rot=20.0, bright=0.4, contrast=0.4, sat=0.3, hue=0.05),
}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_repo_root(start=None):
    p = Path(start or Path.cwd()).resolve()
    for _ in range(6):
        if (p / "data.yaml").exists():
            return p
        p = p.parent
    raise FileNotFoundError("No se encontro data.yaml subiendo desde " + str(start))


# ---------------------------------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------------------------------
def load_keypoints(root):
    """Construye la tabla de parches (una fila por keypoint visible) con el split por rally de la U2/U3."""
    data_dir = root / "data" / "Padel_an_2-6"
    rows = []
    for orig_split in ["train", "valid", "test"]:
        for img_path in sorted((data_dir / orig_split / "images").glob("*.jpg")):
            lbl = data_dir / orig_split / "labels" / (img_path.stem + ".txt")
            if not lbl.exists():
                continue
            m = RALLY_RE.match(img_path.name)
            for line in open(lbl):
                tok = line.split()
                if not tok:
                    continue
                vals = list(map(float, tok))
                h, kpts = vals[4], vals[5:20]
                for i in range(N_CLASSES):
                    x, y, v = kpts[3 * i], kpts[3 * i + 1], kpts[3 * i + 2]
                    if v > 0:
                        rows.append(dict(image_path=str(img_path), image_id=img_path.stem,
                                         rally_id=m.group(1) if m else "unknown", x=x, y=y, bbox_h=h, label=i))
    kpt = pd.DataFrame(rows)
    splits = pd.read_csv(root / "data" / "processed" / "image_splits.csv")
    kpt["split"] = kpt.image_id.map(dict(zip(splits.image_id, splits.split)))
    assert kpt.split.notna().all(), "Hay imagenes sin split asignado"
    assert (kpt.groupby("rally_id").split.nunique() == 1).all(), "Fuga: un rally aparece en mas de un split"
    return kpt


def build_context_crops(kpt, ctx_px):
    """Recorta una vez, por keypoint, un cuadrado de CONTEXT x lado (relleno negro fuera de la imagen)."""
    images, out = {}, np.empty((len(kpt), ctx_px, ctx_px, 3), np.uint8)
    for k, r in enumerate(kpt.itertuples()):
        im = images.get(r.image_path)
        if im is None:
            im = images[r.image_path] = Image.open(r.image_path).convert("RGB")
        W, H = im.size
        side = float(np.clip(r.bbox_h * H * PATCH_FRAC, MIN_PATCH, MAX_PATCH)) * CONTEXT
        px, py = r.x * W, r.y * H
        box = (px - side / 2, py - side / 2, px + side / 2, py + side / 2)
        out[k] = np.asarray(im.crop(box).resize((ctx_px, ctx_px), Image.BILINEAR))
    return torch.from_numpy(out).permute(0, 3, 1, 2).contiguous()


class GPUPatches:
    """Parches de un split en la GPU (uint8, con contexto) + generacion de lotes aumentados."""

    def __init__(self, ctx_u8, labels, img_size, mean, std, device):
        self.x = ctx_u8.to(device)
        self.y = torch.tensor(np.array(labels), dtype=torch.long, device=device)
        self.img_size, self.device = img_size, device
        self.mean = torch.tensor(mean, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(std, device=device).view(1, 3, 1, 1)
        self._eval = None

    def __len__(self):
        return len(self.y)

    def _crop(self, x, scale, dx, dy, flip):
        # grid de salida -> coordenadas de entrada normalizadas [-1, 1] del recorte de contexto
        a = scale / CONTEXT
        theta = torch.zeros(len(x), 2, 3, device=x.device)
        theta[:, 0, 0] = a * flip
        theta[:, 1, 1] = a
        theta[:, 0, 2] = 2 * dx * scale / CONTEXT
        theta[:, 1, 2] = 2 * dy * scale / CONTEXT
        grid = F.affine_grid(theta, (len(x), 3, self.img_size, self.img_size), align_corners=False)
        return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

    def eval_batches(self, batch_size=512):
        """Recorte central sin augmentacion (equivale al recorte determinista de la U3), normalizado."""
        if self._eval is None:
            parts = []
            for i in range(0, len(self), batch_size):
                x = self.x[i:i + batch_size].float() / 255
                one = torch.ones(len(x), device=x.device)
                zero = torch.zeros(len(x), device=x.device)
                parts.append(((self._crop(x, one, zero, zero, one) - self.mean) / self.std).half())
            self._eval = torch.cat(parts)
        return self._eval

    def augmented(self, idx, aug, gen):
        x = self.x[idx].float() / 255
        n, dev = len(idx), x.device

        def unif(lo, hi):
            return lo + (hi - lo) * torch.rand(n, device=dev, generator=gen)

        scale = unif(1 - aug["scale"], 1 + aug["scale"])
        dx, dy = unif(-aug["shift"], aug["shift"]), unif(-aug["shift"], aug["shift"])
        flip = torch.where(torch.rand(n, device=dev, generator=gen) < aug["flip"], -1.0, 1.0)
        x = self._crop(x, scale, dx, dy, flip)
        if aug["rot"] > 0:  # como RandomRotation: rota el parche ya recortado, esquinas en negro
            ang = torch.deg2rad(unif(-aug["rot"], aug["rot"]))
            theta = torch.zeros(n, 2, 3, device=dev)
            theta[:, 0, 0], theta[:, 0, 1] = torch.cos(ang), -torch.sin(ang)
            theta[:, 1, 0], theta[:, 1, 1] = torch.sin(ang), torch.cos(ang)
            grid = F.affine_grid(theta, x.shape, align_corners=False)
            x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        x = color_jitter(x, aug, unif)
        return (x - self.mean) / self.std, self.y[idx]


def color_jitter(x, aug, unif):
    """ColorJitter vectorizado por muestra (brillo, contraste, saturacion, matiz)."""
    if aug["bright"] > 0:
        x = x * unif(1 - aug["bright"], 1 + aug["bright"]).view(-1, 1, 1, 1)
    gray_w = torch.tensor([0.299, 0.587, 0.114], device=x.device).view(1, 3, 1, 1)
    if aug["contrast"] > 0:
        m = (x * gray_w).sum(1, keepdim=True).mean((2, 3), keepdim=True)
        x = (x - m) * unif(1 - aug["contrast"], 1 + aug["contrast"]).view(-1, 1, 1, 1) + m
    if aug["sat"] > 0:
        g = (x * gray_w).sum(1, keepdim=True)
        x = (x - g) * unif(1 - aug["sat"], 1 + aug["sat"]).view(-1, 1, 1, 1) + g
    if aug["hue"] > 0:  # rotacion del plano de croma en YIQ
        ang = (2 * math.pi * unif(-aug["hue"], aug["hue"])).view(-1, 1, 1)
        to_yiq = torch.tensor([[0.299, 0.587, 0.114], [0.596, -0.274, -0.322], [0.211, -0.523, 0.312]],
                              device=x.device)
        yiq = torch.einsum("ij,njhw->nihw", to_yiq, x)
        i, q = yiq[:, 1], yiq[:, 2]
        c, s = torch.cos(ang), torch.sin(ang)
        yiq = torch.stack([yiq[:, 0], c * i - s * q, s * i + c * q], 1)
        x = torch.einsum("ij,njhw->nihw", torch.linalg.inv(to_yiq), yiq)
    return x.clamp(0, 1)


class PatchData:
    """Los tres splits en GPU para un tamano de entrada y una normalizacion dados."""

    def __init__(self, kpt, img_size, norm, device, train_stats=None):
        mean, std = (IMAGENET_MEAN, IMAGENET_STD) if norm == "imagenet" else train_stats
        ctx_px = int(math.ceil(img_size * CONTEXT))
        self.img_size, self.norm = img_size, norm
        self.splits = {}
        for s in ["train", "valid", "test"]:
            part = kpt[kpt.split == s].reset_index(drop=True)
            self.splits[s] = GPUPatches(build_context_crops(part, ctx_px), part.label.values, img_size,
                                        mean, std, device)
        self.frames = {s: kpt[kpt.split == s].reset_index(drop=True) for s in ["train", "valid", "test"]}

    def __getitem__(self, split):
        return self.splits[split]


def train_patch_stats(kpt, device, n=600, seed=42):
    """Media/desviacion por canal sobre parches de train (normalizacion de la CNN de la U2)."""
    sample = kpt[kpt.split == "train"].sample(n, random_state=seed).reset_index(drop=True)
    x = build_context_crops(sample, int(math.ceil(48 * CONTEXT))).float() / 255
    c0 = (x.shape[-1] - int(x.shape[-1] / CONTEXT)) // 2
    x = x[:, :, c0:x.shape[-1] - c0, c0:x.shape[-1] - c0]
    return tuple(x.mean((0, 2, 3)).tolist()), tuple(x.std((0, 2, 3)).tolist())


# ---------------------------------------------------------------------------------------------------
# Modelos (sin cambios respecto de la Unidad 3)
# ---------------------------------------------------------------------------------------------------
class TransferNet(nn.Module):
    def __init__(self, n_classes=N_CLASSES, p_drop=0.3):
        super().__init__()
        bb = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.blocks = nn.ModuleDict(dict(
            stem=nn.Sequential(bb.conv1, bb.bn1, bb.relu, bb.maxpool),
            layer1=bb.layer1, layer2=bb.layer2, layer3=bb.layer3, layer4=bb.layer4))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(p_drop), nn.Linear(bb.fc.in_features, n_classes))
        self.trainable = set()
        self.set_trainable([])

    def set_trainable(self, stages):
        self.trainable = set(stages)
        for name, block in self.blocks.items():
            for p in block.parameters():
                p.requires_grad = name in self.trainable
        return self

    def train(self, mode=True):
        super().train(mode)
        for name, block in self.blocks.items():  # bloques congelados: BN con estadisticas fijas
            if name not in self.trainable:
                block.eval()
        return self

    def features(self, x):
        for name in STAGES:
            x = self.blocks[name](x)
        return x

    def forward(self, x):
        return self.head(self.pool(self.features(x)))


class KptTypeCNN(nn.Module):
    """Linea base inicial: CNN desde cero de la Unidad 2."""

    def __init__(self, n_classes=N_CLASSES, p_drop=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1))
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(p_drop), nn.Linear(64, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


def count_params(model):
    return (sum(p.numel() for p in model.parameters()),
            sum(p.numel() for p in model.parameters() if p.requires_grad))


# ---------------------------------------------------------------------------------------------------
# Configuracion y entrenamiento
# ---------------------------------------------------------------------------------------------------
@dataclass
class Config:
    name: str
    arch: str = "resnet18"          # resnet18 | cnn
    recipe: str = "single"          # single (protocolo U4) | u3_two_phase (receta exacta de la U3)
    img_size: int = 112
    norm: str = "imagenet"          # imagenet | train
    optimizer: str = "adamw"        # adamw | sgd
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 1e-4
    momentum: float = 0.9
    epochs: int = 30
    schedule: str = "cosine"        # cosine (warmup lineal + coseno completo) | constant
    warmup_epochs: int = 2
    label_smoothing: float = 0.0
    dropout: float = 0.3
    aug: str = "original"
    mixup_alpha: float = 0.0
    batch_size: int = 64
    patience: int = 0               # 0 = sin parada temprana (presupuesto fijo de epocas)
    seed: int = 42

    def short(self):
        return {k: v for k, v in asdict(self).items() if k != "name"}


def make_model(cfg, device):
    if cfg.arch == "cnn":
        return KptTypeCNN(p_drop=cfg.dropout).to(device).to(memory_format=torch.channels_last)
    return TransferNet(p_drop=cfg.dropout).to(device).to(memory_format=torch.channels_last)


def param_groups(model, cfg, stages):
    if isinstance(model, KptTypeCNN):
        return [dict(params=model.parameters(), lr=cfg.lr_head)]
    groups = [dict(params=model.blocks[s].parameters(), lr=cfg.lr_backbone) for s in stages]
    groups.append(dict(params=model.head.parameters(), lr=cfg.lr_head))
    return groups


def make_optimizer(groups, cfg):
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(groups, momentum=cfg.momentum, nesterov=True, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "adam":
        return torch.optim.Adam(groups, weight_decay=cfg.weight_decay)
    return torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)


@torch.no_grad()
def predict(model, patches, tta=False, batch_size=512):
    model.eval()
    x_all, logits = patches.eval_batches(), []
    for i in range(0, len(x_all), batch_size):
        x = x_all[i:i + batch_size].float().contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x).float()
            if tta:
                out = (out.softmax(1) + model(torch.flip(x, dims=[3])).float().softmax(1)).log()
        logits.append(out)
    logits = torch.cat(logits)
    return patches.y.cpu().numpy(), logits.argmax(1).cpu().numpy(), logits


def evaluate(model, patches, criterion, tta=False):
    y, p, logits = predict(model, patches, tta=tta)
    loss = criterion(logits, patches.y).item() if not tta else float("nan")
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, average="macro", zero_division=0)
    return dict(loss=loss, acc=accuracy_score(y, p), f1=f1, precision=pr, recall=rc), y, p


def _run_epochs(model, opt, data, cfg, n_epochs, criterion, select_by, log_fn, sched_fn=None,
                patience=0, epoch_offset=0):
    """Bucle comun. select_by: 'f1' (protocolo U4) o 'loss' (receta U3). Devuelve historial y mejor estado."""
    train = data["train"]
    gen = torch.Generator(device=train.y.device).manual_seed(cfg.seed)
    aug = AUG_LEVELS[cfg.aug]
    n, bs = len(train), cfg.batch_size
    steps_per_epoch = math.ceil(n / bs)
    hist, best, stale, ref = [], None, 0, float("inf")
    for ep in range(1, n_epochs + 1):
        model.train()
        t0 = time.perf_counter()
        perm = torch.randperm(n, device=train.y.device, generator=gen)
        tot_loss, correct = 0.0, 0
        for b in range(steps_per_epoch):
            if sched_fn is not None:
                sched_fn(opt, (ep - 1) + b / steps_per_epoch)
            idx = perm[b * bs:(b + 1) * bs]
            x, y = train.augmented(idx, aug, gen)
            x = x.contiguous(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if cfg.mixup_alpha > 0:
                    lam = float(np.random.default_rng(cfg.seed * 1000 + ep * 100 + b).beta(cfg.mixup_alpha,
                                                                                         cfg.mixup_alpha))
                    j = torch.randperm(len(idx), device=x.device, generator=gen)
                    out = model(lam * x + (1 - lam) * x[j])
                    loss = lam * criterion(out.float(), y) + (1 - lam) * criterion(out.float(), y[j])
                else:
                    out = model(x)
                    loss = criterion(out.float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot_loss += loss.item() * len(idx)
            correct += (out.argmax(1) == y).sum().item()
        torch.cuda.synchronize()
        ep_time = time.perf_counter() - t0
        va, _, _ = evaluate(model, data["valid"], nn.CrossEntropyLoss())
        rec = dict(epoch=epoch_offset + ep, train_loss=tot_loss / n, train_acc=correct / n,
                   val_loss=va["loss"], val_acc=va["acc"], val_f1=va["f1"],
                   val_precision=va["precision"], val_recall=va["recall"],
                   lr=opt.param_groups[-1]["lr"], lr_group0=opt.param_groups[0]["lr"],
                   epoch_time_s=ep_time, train_throughput=n / ep_time)
        hist.append(rec)
        log_fn(rec)
        if select_by == "loss":
            # identico a fit() de la U3: mejor epoca por val_loss; paciencia sobre mejoras >= 1e-4
            better = best is None or va["loss"] < best["val_loss"]
            if ref - va["loss"] >= 1e-4:
                ref, stale = va["loss"], 0
            else:
                stale += 1
        else:
            # protocolo U4: mejor epoca por F1 macro en valid; desempate por menor val_loss
            better = (best is None or va["f1"] > best["val_f1"] + 1e-9
                      or (abs(va["f1"] - best["val_f1"]) <= 1e-9 and va["loss"] < best["val_loss"]))
        if better:
            best = dict(val_loss=va["loss"], val_f1=va["f1"], epoch=epoch_offset + ep,
                        state=copy.deepcopy(model.state_dict()))
        if patience and stale >= patience:
            break
    return hist, best


def cosine_with_warmup(base_lrs, warmup, total):
    def fn(opt, t):
        for g, lr0 in zip(opt.param_groups, base_lrs):
            if t < warmup:
                g["lr"] = lr0 * (t + 1e-3) / warmup
            else:
                g["lr"] = lr0 * 0.5 * (1 + math.cos(math.pi * (t - warmup) / max(1e-9, total - warmup)))
    return fn


def cosine_per_epoch(base_lrs, t_max):
    """CosineAnnealingLR(T_max) con paso por epoca, como en la U3."""
    def fn(opt, t):
        e = math.floor(t)
        for g, lr0 in zip(opt.param_groups, base_lrs):
            g["lr"] = lr0 * 0.5 * (1 + math.cos(math.pi * e / t_max))
    return fn


def fit_config(cfg, data, device, log_fn=lambda rec: None):
    """Entrena una configuracion. Devuelve (modelo con el mejor estado, historial, mejor epoca)."""
    seed_everything(cfg.seed)
    model = make_model(cfg, device)
    crit = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    if cfg.recipe == "u3_two_phase":
        # Fase 1 (linear probe): solo la cabeza, AdamW 1e-3, hasta 100 epocas, parada temprana 10 (val_loss)
        model.set_trainable([])
        opt = torch.optim.AdamW(model.head.parameters(), lr=cfg.lr_head, weight_decay=cfg.weight_decay)
        h1, b1 = _run_epochs(model, opt, data, cfg, 100, crit, "loss", log_fn, patience=10)
        model.load_state_dict(b1["state"])
        # Fase 2 (FT completo): todos los grupos a lr_backbone, coseno T_max=60, parada temprana 10
        model.set_trainable(STAGES)
        groups = param_groups(model, replace(cfg, lr_head=cfg.lr_backbone), STAGES)
        opt = make_optimizer(groups, cfg)
        sched = cosine_per_epoch([g["lr"] for g in opt.param_groups], 60)
        h2, b2 = _run_epochs(model, opt, data, cfg, 60, crit, "loss", log_fn, sched_fn=sched,
                             patience=10, epoch_offset=len(h1))
        model.load_state_dict(b2["state"])
        hist = pd.DataFrame(h1 + h2)
        hist["phase"] = ["probe"] * len(h1) + ["finetune"] * len(h2)
        return model, hist, b2["epoch"]
    if cfg.arch == "resnet18":
        model.set_trainable(STAGES)
    opt = make_optimizer(param_groups(model, cfg, STAGES), cfg)
    base = [g["lr"] for g in opt.param_groups]
    sched = cosine_with_warmup(base, cfg.warmup_epochs, cfg.epochs) if cfg.schedule == "cosine" else None
    select = "loss" if cfg.patience else "f1"
    hist, best = _run_epochs(model, opt, data, cfg, cfg.epochs, crit, select, log_fn, sched_fn=sched,
                             patience=cfg.patience)
    model.load_state_dict(best["state"])
    hist = pd.DataFrame(hist)
    hist["phase"] = "train"
    return model, hist, best["epoch"]


# ---------------------------------------------------------------------------------------------------
# Costo computacional
# ---------------------------------------------------------------------------------------------------
@torch.no_grad()
def inference_benchmark(model, img_size, device, n_iter=100, cpu_iter=30):
    """Latencia de inferencia con warmup y sincronizacion explicita (GPU) + referencia en CPU."""
    model.eval()
    res = {}
    for bs in [1, 64]:
        x = torch.randn(bs, 3, img_size, img_size, device=device).contiguous(memory_format=torch.channels_last)
        for _ in range(10):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                model(x)
        torch.cuda.synchronize()
        times = []
        for _ in range(n_iter):
            t0 = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                model(x)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        t = np.array(times) * 1000
        res[f"infer_gpu_bs{bs}_p50_ms"] = float(np.percentile(t, 50))
        res[f"infer_gpu_bs{bs}_p95_ms"] = float(np.percentile(t, 95))
        res[f"infer_gpu_bs{bs}_throughput"] = float(bs / (np.median(t) / 1000))
    cpu_model = copy.deepcopy(model).cpu().float()
    x = torch.randn(1, 3, img_size, img_size)
    for _ in range(3):
        cpu_model(x)
    times = []
    for _ in range(cpu_iter):
        t0 = time.perf_counter()
        cpu_model(x)
        times.append(time.perf_counter() - t0)
    t = np.array(times) * 1000
    res["infer_cpu_bs1_p50_ms"] = float(np.percentile(t, 50))
    res["infer_cpu_bs1_p95_ms"] = float(np.percentile(t, 95))
    return res


def hardware_tags():
    tags = dict(cpu=platform.processor() or platform.machine(), python=platform.python_version(),
                torch=torch.__version__, cuda=str(torch.version.cuda))
    if torch.cuda.is_available():
        tags["gpu"] = torch.cuda.get_device_name(0)
    try:
        tags["git_commit"] = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                                     text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        pass
    return tags


# ---------------------------------------------------------------------------------------------------
# Ejecucion registrada en MLflow
# ---------------------------------------------------------------------------------------------------
def setup_mlflow(root, experiment="unidad4_transfer_optim"):
    import logging
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    import mlflow
    logging.getLogger("mlflow").setLevel(logging.ERROR)  # sin los INFO del monitor de sistema en cada run
    mlflow.set_tracking_uri("sqlite:///" + (root / "mlflow.db").as_posix())
    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(experiment, artifact_location=(root / "mlruns").as_uri())
    mlflow.set_experiment(experiment)
    mlflow.set_system_metrics_sampling_interval(2)
    mlflow.set_system_metrics_samples_before_logging(1)
    return mlflow


def run_experiment(cfg, data, device, sweep, tags=None, log_weights=False, benchmark=False, log_model=False):
    """Entrena `cfg` dentro de un run anidado de MLflow y devuelve un dict con todo lo necesario."""
    import mlflow
    import matplotlib.pyplot as plt

    torch.cuda.reset_peak_memory_stats()
    with mlflow.start_run(run_name=cfg.name, nested=True, log_system_metrics=True) as run:
        mlflow.set_tags({**hardware_tags(), "sweep": sweep, **(tags or {})})
        mlflow.log_params(cfg.short())
        t0 = time.perf_counter()

        def log_fn(rec):
            mlflow.log_metrics({k: v for k, v in rec.items() if k != "epoch"}, step=int(rec["epoch"]))

        model, hist, best_ep = fit_config(cfg, data, device, log_fn)
        train_time = time.perf_counter() - t0
        crit = nn.CrossEntropyLoss()
        va, yv, pv = evaluate(model, data["valid"], crit)
        n_total, n_train = count_params(model)
        b = min(best_ep, len(hist)) - 1
        with tempfile.TemporaryDirectory() as d:
            ckpt = Path(d) / "model_state.pt"
            torch.save(model.state_dict(), ckpt)
            size_mb = ckpt.stat().st_size / 1e6
            summary = dict(best_epoch=best_ep, epochs_run=len(hist), val_acc_best=va["acc"], val_f1_best=va["f1"],
                           val_loss_best=va["loss"], val_precision_best=va["precision"],
                           val_recall_best=va["recall"], train_time_s=train_time,
                           # diagnostico de sobreajuste en la mejor epoca y despues de ella
                           train_loss_at_best=float(hist.train_loss.iloc[b]),
                           train_acc_at_best=float(hist.train_acc.iloc[b]),
                           val_loss_rise=float(hist.val_loss.iloc[-1] - hist.val_loss.min()),
                           **per_class_metrics(yv, pv, "val"),
                           # mediana: la 1a epoca incluye el calentamiento de cuDNN y la inflaria
                           median_epoch_time_s=float(hist.epoch_time_s.median()),
                           median_train_throughput=float(hist.train_throughput.median()),
                           vram_peak_mb=torch.cuda.max_memory_allocated() / 1e6,
                           params_total=n_total, params_trainable=n_train, checkpoint_mb=size_mb,
                           train_val_gap_acc=float(hist.train_acc.iloc[best_ep - 1] - hist.val_acc.iloc[best_ep - 1])
                           if best_ep <= len(hist) else float("nan"))
            if benchmark:
                summary.update(inference_benchmark(model, cfg.img_size, device))
            mlflow.log_metrics(summary)
            (Path(d) / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
            hist.to_csv(Path(d) / "history.csv", index=False)
            fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
            axes[0].plot(hist.epoch, hist.train_loss, label="train")
            axes[0].plot(hist.epoch, hist.val_loss, label="valid")
            axes[0].set_title("perdida"); axes[0].set_xlabel("epoca"); axes[0].legend()
            axes[1].plot(hist.epoch, hist.train_acc, label="train")
            axes[1].plot(hist.epoch, hist.val_acc, label="valid")
            axes[1].axvline(best_ep, ls=":", c="gray", label=f"mejor ({best_ep})")
            axes[1].set_title("exactitud"); axes[1].set_xlabel("epoca"); axes[1].legend()
            fig.suptitle(cfg.name); fig.tight_layout()
            fig.savefig(Path(d) / "curvas.png", dpi=110); plt.close(fig)
            pd.DataFrame(confusion_matrix(yv, pv, labels=range(N_CLASSES)), index=KP_NAMES,
                         columns=KP_NAMES).to_csv(Path(d) / "confusion_valid.csv")
            for f in ["config.json", "history.csv", "curvas.png", "confusion_valid.csv"]:
                mlflow.log_artifact(str(Path(d) / f))
            if log_weights:
                mlflow.log_artifact(str(ckpt))
        if log_model:
            import mlflow.pytorch
            # formato pickle: el 'pt2' por defecto de MLflow 3 exige trazar el grafo con un ejemplo de entrada
            mlflow.pytorch.log_model(copy.deepcopy(model).cpu().float(), name="model",
                                     serialization_format="pickle")
    # el monitor de sistema vuelca sus ultimas muestras al cerrar el run: el resumen va despues
    sysm = {f"sys_{k}": v for k, v in system_metric_stats(run.info.run_id).items() if np.isfinite(v)}
    log_to_run(run.info.run_id, sysm)
    return dict(cfg=cfg, model=model, hist=hist, run_id=run.info.run_id, **summary, **sysm)


def log_to_run(run_id, metrics=None, params=None, tags=None):
    """Agrega metricas/params/tags a un run ya cerrado (p. ej. las de test, calculadas al final)."""
    import mlflow
    with mlflow.start_run(run_id=run_id, nested=True):
        if metrics:
            mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
        if params:
            mlflow.log_params(params)
        if tags:
            mlflow.set_tags(tags)


def log_artifacts_to_run(run_id, figures=None, tables=None):
    """Sube figuras de matplotlib ({archivo.png: fig}) y DataFrames ({archivo.csv: df}) a un run."""
    import mlflow
    client = mlflow.tracking.MlflowClient()
    for name, fig in (figures or {}).items():
        client.log_figure(run_id, fig, name)
    for name, df in (tables or {}).items():
        client.log_text(run_id, df.to_csv(), name)


def per_class_metrics(y, p, prefix):
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, labels=range(N_CLASSES), zero_division=0)
    out = {}
    for i, n in enumerate(KP_NAMES):
        out.update({f"{prefix}_precision_{n}": pr[i], f"{prefix}_recall_{n}": rc[i], f"{prefix}_f1_{n}": f1[i]})
    return out


def log_test_eval(run_id, y, p, prefix="test"):
    """Metricas de test completas de un run: macro + IC95% bootstrap, por clase, errores y matriz de confusion."""
    m = test_metrics(y, p)
    metrics = {f"{prefix}_acc": m["acc"], f"{prefix}_acc_ci_lo": m["ci_lo"], f"{prefix}_acc_ci_hi": m["ci_hi"],
               f"{prefix}_f1_macro": m["f1_macro"], f"{prefix}_precision_macro": m["precision_macro"],
               f"{prefix}_recall_macro": m["recall_macro"], f"{prefix}_errors": int((y != p).sum()),
               **per_class_metrics(y, p, prefix)}
    log_to_run(run_id, metrics)
    cm = pd.DataFrame(confusion_matrix(y, p, labels=range(N_CLASSES)), index=KP_NAMES, columns=KP_NAMES)
    log_artifacts_to_run(run_id, tables={f"confusion_{prefix}.csv": cm})
    return m


def clone_as_reference(src_run_id, run_name, sweep, tags=None):
    """Run anidado en el barrido actual que replica un run previo (la incumbente reutilizada).

    Asi cada barrido tiene en MLflow TODAS sus variantes, incluida la que no se reentrena, y la vista
    Chart / Compare las compara juntas. Copia params y metricas (con su historia por epoca), no artefactos.
    """
    import mlflow
    from mlflow.entities import Metric
    client = mlflow.tracking.MlflowClient()
    src = client.get_run(src_run_id)
    with mlflow.start_run(run_name=run_name, nested=True) as run:
        keep = {k: v for k, v in src.data.tags.items() if not k.startswith("mlflow.")}
        mlflow.set_tags({**keep, **(tags or {}), "sweep": sweep, "reutilizada": "true",
                         "reutilizada_de": src_run_id})
        mlflow.log_params(src.data.params)
        hist = []
        for key in src.data.metrics:
            hist += [Metric(key, m.value, m.timestamp, m.step) for m in client.get_metric_history(src_run_id, key)]
        for i in range(0, len(hist), 1000):  # limite de log_batch
            client.log_batch(run.info.run_id, metrics=hist[i:i + 1000])
    return run.info.run_id


def system_metric_stats(run_id):
    """Media y maximo de cada metrica de sistema registrada por MLflow durante el run."""
    import mlflow
    client = mlflow.tracking.MlflowClient()
    out = {}
    for key in ["system/gpu_0_utilization_percentage", "system/gpu_0_memory_usage_megabytes",
                "system/gpu_0_power_usage_watts", "system/cpu_utilization_percentage",
                "system/system_memory_usage_megabytes"]:
        vals = [m.value for m in client.get_metric_history(run_id, key)]
        short = key.split("/")[-1]
        out[short + "_mean"] = float(np.mean(vals)) if vals else float("nan")
        out[short + "_max"] = float(np.max(vals)) if vals else float("nan")
    return out


def system_metric_series(run_id, key):
    import mlflow
    hist = mlflow.tracking.MlflowClient().get_metric_history(run_id, key)
    if not hist:
        return np.array([]), np.array([])
    t0 = hist[0].timestamp
    return np.array([(m.timestamp - t0) / 1000 for m in hist]), np.array([m.value for m in hist])


def lr_range_test(cfg, data, device, lr_min=1e-7, lr_max=1.0, steps=200, beta=0.9):
    """LR range test (Smith, 2017) sobre el MODELO COMPLETO, con la cabeza a lr_head/lr_backbone x el lr."""
    seed_everything(cfg.seed)
    model = make_model(cfg, device)
    model.set_trainable(STAGES)
    ratio = cfg.lr_head / cfg.lr_backbone
    opt = make_optimizer(param_groups(model, replace(cfg, lr_backbone=lr_min, lr_head=lr_min * ratio), STAGES), cfg)
    gamma = (lr_max / lr_min) ** (1 / steps)
    train, gen, aug, crit = data["train"], torch.Generator(device=device).manual_seed(cfg.seed), \
        AUG_LEVELS[cfg.aug], nn.CrossEntropyLoss()
    lrs, losses, avg, best = [], [], 0.0, float("inf")
    model.train()
    for i in range(steps):
        idx = torch.randint(0, len(train), (cfg.batch_size,), device=device, generator=gen)
        x, y = train.augmented(idx, aug, gen)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = crit(model(x.contiguous(memory_format=torch.channels_last)).float(), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        avg = beta * avg + (1 - beta) * loss.item()
        smooth = avg / (1 - beta ** (i + 1))
        lrs.append(opt.param_groups[0]["lr"])
        losses.append(smooth)
        best = min(best, smooth)
        if smooth > 4 * best or not math.isfinite(smooth):
            break
        for g in opt.param_groups:
            g["lr"] *= gamma
    return np.array(lrs), np.array(losses)


# ---------------------------------------------------------------------------------------------------
# Estadistica y analisis de errores
# ---------------------------------------------------------------------------------------------------
def bootstrap_ci(y, p, n=2000, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), size=(n, len(y)))
    return np.percentile((y[idx] == p[idx]).mean(1), [2.5, 97.5])


def mcnemar(y, p_a, p_b):
    """McNemar exacto. Devuelve (b acierta y a falla, a acierta y b falla, p-valor)."""
    a, b = p_a == y, p_b == y
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
    pval = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    return n01, n10, pval


def test_metrics(y, p):
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, average="macro", zero_division=0)
    lo, hi = bootstrap_ci(y, p)
    return dict(acc=accuracy_score(y, p), ci_lo=lo, ci_hi=hi, precision_macro=pr, recall_macro=rc, f1_macro=f1)


class GradCAM:
    """Grad-CAM (Selvaraju et al., 2017) sobre la salida de layer4 de TransferNet."""

    def __init__(self, model):
        self.model, self.acts, self.grads = model, None, None
        layer = model.blocks["layer4"]
        layer.register_forward_hook(lambda m, i, o: setattr(self, "acts", o))
        layer.register_full_backward_hook(lambda m, gi, go: setattr(self, "grads", go[0]))

    def __call__(self, x, cls):
        self.model.eval()
        x = x.unsqueeze(0).float().requires_grad_(True)
        out = self.model(x)
        self.model.zero_grad()
        out[0, cls].backward()
        w = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * self.acts).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        return (cam / (cam.max() + 1e-8)).detach().cpu().numpy()
