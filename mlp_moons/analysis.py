"""指标与错误分析工具：完整分类指标、错误样本明细、结果落盘。"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(result: dict, split: str = "test") -> dict:
    """一次性算齐常用指标（尤其是不均衡数据下比 accuracy 更有意义的那些）。"""
    d = result[split]
    y, pred, proba = d["y"], d["pred"], d["proba"]
    cm = confusion_matrix(y, pred, labels=[0, 1])
    return {
        "split": split,
        "n": int(len(y)),
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "precision_0": precision_score(y, pred, pos_label=0, zero_division=0),
        "recall_0": recall_score(y, pred, pos_label=0, zero_division=0),
        "f1_0": f1_score(y, pred, pos_label=0, zero_division=0),
        "precision_1": precision_score(y, pred, pos_label=1, zero_division=0),
        "recall_1": recall_score(y, pred, pos_label=1, zero_division=0),
        "f1_1": f1_score(y, pred, pos_label=1, zero_division=0),
        "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
        "roc_auc": roc_auc_score(y, proba[:, 1]) if len(np.unique(y)) > 1 else float("nan"),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }


def error_table(result: dict, split: str = "test", max_rows: int | None = None) -> list[dict]:
    """错误样本明细表：按 |logit margin| 从大到小排序（越靠前错得越"理直气壮"）。"""
    from .viz import error_samples

    err = error_samples(result, split)
    rows = []
    for k in range(err["n_wrong"]):
        rows.append(
            {
                "#": k + 1,
                "index_in_split": int(err["index"][k]),
                "x1": float(err["X"][k, 0]),
                "x2": float(err["X"][k, 1]),
                "y_true": int(err["y_true"][k]),
                "y_pred": int(err["y_pred"][k]),
                "p_true": float(err["p_true"][k]),
                "p_pred": float(err["confidence"][k]),
                "logit_margin_abs": float(err["margin"][k]),
                "dist_to_boundary": float(err["distance_to_boundary"][k]),
            }
        )
    return rows[:max_rows] if max_rows else rows


def stats_row(result: dict, group: str | None = None) -> dict:
    """把一次实验的配置 + 指标压成一行，方便写 CSV / 打表格。"""
    cfg = result["config"]
    row = {
        "group": group or cfg.name,
        "name": cfg.name,
        "hidden": cfg.hidden,
        "n_hidden": cfg.n_hidden,
        "activation": cfg.activation,
        "optimizer": cfg.optimizer,
        "lr": cfg.lr,
        "batch_size": cfg.batch_size,
        "weight_decay": cfg.weight_decay,
        "dropout": cfg.dropout,
        "epochs": cfg.epochs,
        "noise": cfg.extra.get("noise", 0.2),
        "n_train": cfg.extra.get("n_train", None),
        "label_noise": cfg.extra.get("label_noise", 0.0),
        "zero_grad": cfg.zero_grad,
    }
    row.update(result["stats"])
    row.update(
        {
            f"test_{k}": v
            for k, v in classification_metrics(result, "test").items()
            if k not in {"split", "n"}
        }
    )
    return row


def write_csv(rows: list[dict], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return path
    fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def read_csv(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_json(obj, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(str(type(o)))

    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=default)
    return path


def format_table(rows: list[dict], columns: list[str], path: str | Path | None = None) -> str:
    """生成 Markdown 表格，用于写进 README。"""
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, sep]
    for r in rows:
        cells = []
        for c in columns:
            v = r.get(c, "")
            if isinstance(v, float):
                cells.append(f"{v:.4f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    text = "\n".join(lines)
    if path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text + "\n", encoding="utf-8")
    return text
