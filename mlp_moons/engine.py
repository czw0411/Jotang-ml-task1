"""训练引擎：训练/验证/测试、模型保存与加载、资源占用统计。"""

from __future__ import annotations

import ctypes
import random
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .models import MLP, count_parameters


# ----------------------------------------------------------------------------
# 随机种子与资源统计
# ----------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """固定 python / numpy / torch 的随机种子，保证实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class _PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def _psapi_call():
    """正确设置函数原型（否则 64 位句柄会被截断，调用直接失败）。"""
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    psapi.GetProcessMemoryInfo.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_PROCESS_MEMORY_COUNTERS),
        ctypes.c_ulong,
    ]
    psapi.GetProcessMemoryInfo.restype = ctypes.c_int

    def _call() -> _PROCESS_MEMORY_COUNTERS | None:
        counters = _PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(counters)
        ok = psapi.GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        )
        return counters if ok else None

    return _call


_read_memory = _psapi_call() if hasattr(ctypes, "WinDLL") else (lambda: None)


def memory_mb() -> tuple[float, float]:
    """返回 (当前进程占用 MB, 进程峰值占用 MB)。

    Windows 上用 psapi 读工作集；其它平台退化为 tracemalloc 无法提供时的 (-1, -1)。
    GPU 显存需要 torch.cuda.max_memory_allocated()，本机无 CUDA 设备。
    """
    try:
        counters = _read_memory()
        if counters is not None:
            return (
                counters.WorkingSetSize / 1024**2,
                counters.PeakWorkingSetSize / 1024**2,
            )
    except Exception:  # pragma: no cover - 非 Windows 平台
        pass
    return (-1.0, -1.0)


def gpu_memory_mb() -> float:
    """CUDA 峰值显存（MB）；没有 GPU 时返回 -1。"""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024**2
    return -1.0


# ----------------------------------------------------------------------------
# 配置
# ----------------------------------------------------------------------------
@dataclass
class TrainConfig:
    """一次训练的完整配置，便于对照实验中"只改一个变量"。"""

    hidden: int = 64
    n_hidden: int = 2
    activation: str = "relu"
    dropout: float = 0.0
    batchnorm: bool = False
    lr: float = 1e-2
    optimizer: str = "adam"
    momentum: float = 0.9
    weight_decay: float = 0.0
    batch_size: int = 64
    epochs: int = 200
    label_smoothing: float = 0.0
    class_weight: str | None = None  # None 或 "balanced"
    seed: int = 42
    zero_grad: bool = True  # 设为 False 用来演示梯度不清零的后果
    name: str = "baseline"
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def build_optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    name = cfg.optimizer.lower()
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if name == "sgd_momentum":
        return torch.optim.SGD(
            model.parameters(),
            lr=cfg.lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
        )
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if name == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    raise ValueError(f"未知优化器 {cfg.optimizer}")


# ----------------------------------------------------------------------------
# 训练 / 评估
# ----------------------------------------------------------------------------
@torch.no_grad()
def evaluate(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    criterion: nn.Module,
) -> dict:
    """在验证/测试集上评估：返回 loss、accuracy、概率、预测、logits。"""
    model.eval()
    logits = model(X)
    loss = criterion(logits, y).item()
    proba = torch.softmax(logits, dim=1)
    pred = logits.argmax(dim=1)
    return {
        "loss": loss,
        "acc": (pred == y).float().mean().item(),
        "logits": logits.cpu().numpy(),
        "proba": proba.cpu().numpy(),
        "pred": pred.cpu().numpy(),
        "y": y.cpu().numpy(),
    }


def train_model(cfg: TrainConfig, splits, device: str | torch.device = "cpu") -> dict:
    """完整训练流程：构造模型 -> 训练 N 轮 -> 每轮记录 train/val 指标。

    返回 dict，包含 model、history、统计信息（耗时、内存、参数量）等。
    """
    device = torch.device(device)
    set_seed(cfg.seed)

    Xtr = torch.from_numpy(splits.X_train_s).to(device)
    ytr = torch.from_numpy(splits.y_train).to(device)
    Xva = torch.from_numpy(splits.X_val_s).to(device)
    yva = torch.from_numpy(splits.y_val).to(device)
    Xte = torch.from_numpy(splits.X_test_s).to(device)
    yte = torch.from_numpy(splits.y_test).to(device)

    model = MLP(
        hidden=cfg.hidden,
        n_hidden=cfg.n_hidden,
        activation=cfg.activation,
        dropout=cfg.dropout,
        batchnorm=cfg.batchnorm,
    ).to(device)

    if cfg.class_weight == "balanced":
        counts = np.bincount(splits.y_train, minlength=2).astype(np.float32)
        w = counts.sum() / (2.0 * np.maximum(counts, 1.0))
        weight = torch.tensor(w, dtype=torch.float32, device=device)
    else:
        weight = None

    criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=cfg.label_smoothing)
    eval_criterion = nn.CrossEntropyLoss()  # 报告用的 loss 不带权重，方便横向比较
    optimizer = build_optimizer(model, cfg)

    n = Xtr.shape[0]
    gen = torch.Generator(device="cpu").manual_seed(cfg.seed)
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    mem_before = memory_mb()[0]
    t_start = time.perf_counter()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        perm = torch.randperm(n, generator=gen).to(device)
        epoch_loss, correct = 0.0, 0
        for start in range(0, n, cfg.batch_size):
            idx = perm[start : start + cfg.batch_size]
            logits = model(Xtr[idx])
            loss = criterion(logits, ytr[idx])

            if cfg.zero_grad:
                optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * idx.numel()
            correct += (logits.argmax(dim=1) == ytr[idx]).sum().item()

        val = evaluate(model, Xva, yva, eval_criterion)
        history["train_loss"].append(epoch_loss / n)
        history["train_acc"].append(correct / n)
        history["val_loss"].append(val["loss"])
        history["val_acc"].append(val["acc"])

    train_time = time.perf_counter() - t_start
    if device.type == "cuda":
        torch.cuda.synchronize()

    test = evaluate(model, Xte, yte, eval_criterion)
    train_eval = evaluate(model, Xtr, ytr, eval_criterion)

    stats = {
        "name": cfg.name,
        "params": count_parameters(model),
        "param_size_mb": count_parameters(model) * 4 / 1024**2,
        "train_time_s": train_time,
        "ms_per_epoch": train_time / cfg.epochs * 1000,
        "ram_delta_mb": memory_mb()[0] - mem_before,
        "ram_current_mb": memory_mb()[0],
        "ram_peak_mb": memory_mb()[1],
        "gpu_peak_mb": gpu_memory_mb(),
        "threads": torch.get_num_threads(),
        "device": str(device),
        "final_train_loss": history["train_loss"][-1],
        "final_train_acc": history["train_acc"][-1],
        "final_val_loss": history["val_loss"][-1],
        "final_val_acc": history["val_acc"][-1],
        "best_val_acc": max(history["val_acc"]),
        "best_val_epoch": int(np.argmax(history["val_acc"]) + 1),
        "test_loss": test["loss"],
        "test_acc": test["acc"],
        "train_eval_loss": train_eval["loss"],
        "train_eval_acc": train_eval["acc"],
        "generalization_gap": train_eval["acc"] - test["acc"],
    }

    return {
        "config": cfg,
        "model": model,
        "history": history,
        "stats": stats,
        "train": train_eval,
        "val": val if cfg.epochs else None,
        "test": test,
        "device": device,
        "splits": splits,
        "criterion": eval_criterion,
    }


# ----------------------------------------------------------------------------
# 保存 / 加载
# ----------------------------------------------------------------------------
def save_checkpoint(result: dict, path: str | Path) -> Path:
    """保存模型权重 + 配置 + 指标，方便复现与推理。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg: TrainConfig = result["config"]
    torch.save(
        {
            "state_dict": result["model"].state_dict(),
            "config": cfg.to_dict(),
            "architecture": {
                "in_dim": 2,
                "hidden": cfg.hidden,
                "n_hidden": cfg.n_hidden,
                "out_dim": 2,
                "activation": cfg.activation,
                "dropout": cfg.dropout,
                "batchnorm": cfg.batchnorm,
            },
            "stats": result["stats"],
            "history": result["history"],
        },
        path,
    )
    return path


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> dict:
    """加载 checkpoint：重建同结构模型并载入权重，返回 (model, meta)。"""
    device = torch.device(device)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    arch = ckpt["architecture"]
    model = MLP(
        in_dim=arch["in_dim"],
        hidden=arch["hidden"],
        n_hidden=arch["n_hidden"],
        out_dim=arch["out_dim"],
        activation=arch["activation"],
        dropout=arch["dropout"],
        batchnorm=arch["batchnorm"],
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return {"model": model, "meta": ckpt, "device": device}


@torch.no_grad()
def predict_with(model: nn.Module, X: np.ndarray, device: str | torch.device = "cpu") -> dict:
    """对任意 numpy 特征做推理（注意：调用方需保证特征已经用训练集 scaler 变换过）。"""
    device = torch.device(device)
    model.eval()
    logits = model(torch.from_numpy(X.astype(np.float32)).to(device))
    proba = torch.softmax(logits, dim=1)
    return {
        "logits": logits.cpu().numpy(),
        "proba": proba.cpu().numpy(),
        "pred": logits.argmax(dim=1).cpu().numpy(),
    }
