"""可视化模块：数据划分、训练曲线、决策边界、混淆矩阵、错误样本、对照实验。"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from sklearn.metrics import confusion_matrix

from .engine import predict_with

CLASS_COLORS = np.array(["#2E6FD9", "#E4572E"])
SPLIT_COLORS = {"train": "#2E6FD9", "val": "#1B9E77", "test": "#D95F02"}


def _setup_fonts() -> str:
    """优先使用系统中文字体，避免图里出现方框；找不到就退回默认字体。"""
    from matplotlib import font_manager

    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Microsoft YaHei", "SimHei", "Noto Sans SC", "Source Han Sans SC"):
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return name
    return "DejaVu Sans"


FONT_IN_USE = _setup_fonts()


def _save(fig, path: str | Path, dpi: int = 140) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


# ----------------------------------------------------------------------------
# 1. 原始数据 + 三分划分
# ----------------------------------------------------------------------------
def plot_data_splits(splits, path: str | Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4))

    ax = axes[0]
    for label, color in [(0, CLASS_COLORS[0]), (1, CLASS_COLORS[1])]:
        m = splits.y_all == label
        ax.scatter(
            splits.X_all[m, 0],
            splits.X_all[m, 1],
            s=22,
            c=color,
            edgecolors="k",
            linewidths=0.3,
            label=f"class {label}",
        )
    ax.set_title(f"make_moons(n={len(splits.X_all)}, noise=0.2, seed=42)")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.legend()
    ax.grid(alpha=0.25)

    ax = axes[1]
    for (name, X, y) in [
        ("train", splits.X_train, splits.y_train),
        ("val", splits.X_val, splits.y_val),
        ("test", splits.X_test, splits.y_test),
    ]:
        ax.scatter(
            X[:, 0],
            X[:, 1],
            s=22,
            c=SPLIT_COLORS[name],
            marker={"train": "o", "val": "^", "test": "s"}[name],
            edgecolors="w",
            linewidths=0.4,
            alpha=0.85,
            label=f"{name} (n={len(X)})",
        )
    ax.set_title("60/20/20 stratified split (no overlap, no leakage)")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()
    return _save(fig, path)


# ----------------------------------------------------------------------------
# 2. loss / accuracy 曲线
# ----------------------------------------------------------------------------
def plot_curves(result: dict, path: str | Path, title: str | None = None) -> Path:
    h = result["history"]
    s = result["stats"]
    epochs = np.arange(1, len(h["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].plot(epochs, h["train_loss"], label="train", color="#2E6FD9", lw=1.8)
    axes[0].plot(epochs, h["val_loss"], label="val", color="#E4572E", lw=1.8)
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("cross-entropy loss")
    axes[0].set_title("loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, h["train_acc"], label="train", color="#2E6FD9", lw=1.8)
    axes[1].plot(epochs, h["val_acc"], label="val", color="#E4572E", lw=1.8)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].set_title("accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    default_title = (
        f"{s['name']} | test acc {s['test_acc']:.4f} | "
        f"{s['ms_per_epoch']:.1f} ms/epoch | {s['params']} params"
    )
    fig.suptitle(title or default_title, fontsize=11)
    fig.tight_layout()
    return _save(fig, path)


def plot_curves_multi(results: list[dict], path: str | Path, metric: str = "val_acc",
                      title: str = "ablation") -> Path:
    """把多次实验的同一条曲线画在一起，用于对照实验。"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    colors = plt.cm.viridis(np.linspace(0, 0.88, len(results)))
    for r, c in zip(results, colors):
        h = r["history"]
        epochs = np.arange(1, len(h["train_loss"]) + 1)
        axes[0].plot(epochs, h["val_loss"], label=r["config"].name, color=c, lw=1.6)
        axes[1].plot(epochs, h[metric], label=r["config"].name, color=c, lw=1.6)
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("val loss")
    axes[0].set_title("validation loss")
    axes[0].grid(alpha=0.3)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel(metric.replace("_", " "))
    axes[1].set_title(metric.replace("_", " "))
    axes[1].grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return _save(fig, path)


def plot_ablation_bars(stats_list: list[dict], path: str | Path, title: str,
                       metric: str = "test_acc") -> Path:
    """对照实验汇总：测试集指标 / 训练耗时 / 参数量。"""
    names = [s["name"] for s in stats_list]
    vals = [s[metric] for s in stats_list]
    times = [s["ms_per_epoch"] for s in stats_list]
    params = [s["params"] for s in stats_list]
    x = np.arange(len(names))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, data, ylabel, color, fmt in [
        (axes[0], vals, metric.replace("_", " "), "#2E6FD9", "{:.4f}"),
        (axes[1], times, "ms / epoch", "#E4572E", "{:,.1f}"),
        (axes[2], params, "# parameters", "#1B9E77", "{:,.0f}"),
    ]:
        bars = ax.bar(x, data, color=color, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25, axis="y")
        for b, v in zip(bars, data):
            ax.annotate(
                fmt.format(v),
                (b.get_x() + b.get_width() / 2, b.get_height()),
                ha="center",
                va="bottom",
                fontsize=7,
            )
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return _save(fig, path)


# ----------------------------------------------------------------------------
# 3. 决策边界
# ----------------------------------------------------------------------------
def decision_surface(result: dict, resolution: int = 400, pad: float = 0.6):
    """在原始特征空间生成 P(class=1) 网格（内部先做训练集的标准化）。"""
    splits = result["splits"]
    X = splits.X_all
    xx, yy = np.meshgrid(
        np.linspace(X[:, 0].min() - pad, X[:, 0].max() + pad, resolution),
        np.linspace(X[:, 1].min() - pad, X[:, 1].max() + pad, resolution),
    )
    grid = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)
    grid = splits.scaler.transform(grid).astype(np.float32)
    proba = predict_with(result["model"], grid, result["device"])["proba"][:, 1]
    return xx, yy, proba.reshape(xx.shape)


def plot_decision_boundary(result: dict, path: str | Path,
                           panels: tuple[str, ...] = ("train", "val", "test"),
                           resolution: int = 350) -> Path:
    splits = result["splits"]
    xx, yy, zz = decision_surface(result, resolution=resolution)
    data = {
        "train": (splits.X_train, splits.y_train),
        "val": (splits.X_val, splits.y_val),
        "test": (splits.X_test, splits.y_test),
    }

    fig, axes = plt.subplots(
        1, len(panels), figsize=(5.0 * len(panels), 4.8), squeeze=False, layout="constrained"
    )
    for ax, key in zip(axes[0], panels):
        cf = ax.contourf(xx, yy, zz, levels=20, cmap="RdBu_r", alpha=0.55, vmin=0, vmax=1)
        ax.contour(xx, yy, zz, levels=[0.5], colors="k", linewidths=1.3)
        X, y = data[key]
        pred = predict_with(result["model"], splits.scaler.transform(X).astype(np.float32),
                            result["device"])["pred"]
        wrong = pred != y
        ax.scatter(X[~wrong, 0], X[~wrong, 1], c=CLASS_COLORS[y[~wrong]], s=18,
                   edgecolors="k", linewidths=0.3)
        if wrong.any():
            ax.scatter(X[wrong, 0], X[wrong, 1], s=90, facecolors="none",
                       edgecolors="k", linewidths=1.2, label=f"wrong ({wrong.sum()})")
            ax.legend(fontsize=8)
        acc = (pred == y).mean()
        ax.set_title(f"{key}: acc={acc:.4f} (n={len(X)})")
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
    fig.colorbar(cf, ax=axes[0], shrink=0.85, label="P(class=1)")
    fig.suptitle(f"decision boundary | {result['config'].name}", fontsize=11)
    return _save(fig, path)


def plot_decision_boundary_multi(results: list[dict], path: str | Path,
                                 resolution: int = 300) -> Path:
    """多个模型（例如不同宽度）决策边界并排对比。"""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4.3 * n, 4.4), squeeze=False)
    for ax, r in zip(axes[0], results):
        xx, yy, zz = decision_surface(r, resolution=resolution)
        ax.contourf(xx, yy, zz, levels=20, cmap="RdBu_r", alpha=0.55, vmin=0, vmax=1)
        ax.contour(xx, yy, zz, levels=[0.5], colors="k", linewidths=1.2)
        X, y = r["splits"].X_train, r["splits"].y_train
        ax.scatter(X[:, 0], X[:, 1], c=CLASS_COLORS[y], s=10, edgecolors="k", linewidths=0.2)
        ax.set_title(
            f"{r['config'].name}\ntest acc={r['stats']['test_acc']:.4f}, "
            f"params={r['stats']['params']}",
            fontsize=9,
        )
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("decision boundary comparison (train points shown)", fontsize=11)
    fig.tight_layout()
    return _save(fig, path)


# ----------------------------------------------------------------------------
# 4. 混淆矩阵与错误样本
# ----------------------------------------------------------------------------
def plot_confusion(result: dict, path: str | Path, normalize: bool = True) -> Path:
    y_true = result["test"]["y"]
    y_pred = result["test"]["pred"]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_norm = cm / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), layout="constrained")
    for ax, mat, title, fmt in [
        (axes[0], cm, "counts", "d"),
        (axes[1], cm_norm, "row-normalized", ".3f"),
    ]:
        im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=mat.max())
        for i in range(2):
            for j in range(2):
                ax.text(j, i, format(mat[i, j], fmt), ha="center", va="center",
                        color="white" if mat[i, j] > mat.max() * 0.55 else "black")
        ax.set_xticks([0, 1], ["pred 0", "pred 1"])
        ax.set_yticks([0, 1], ["true 0", "true 1"])
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.85)
    fig.suptitle(
        f"test confusion matrix | {result['config'].name} | "
        f"acc={result['stats']['test_acc']:.4f}",
        fontsize=11,
    )
    return _save(fig, path)


def error_samples(result: dict, split: str = "test") -> dict:
    """找出被分错的样本，返回其坐标、真实/预测标签、置信度与 logit margin。"""
    d = result[split]
    X = {"train": result["splits"].X_train,
         "val": result["splits"].X_val,
         "test": result["splits"].X_test}[split]
    y, pred, proba, logits = d["y"], d["pred"], d["proba"], d["logits"]
    wrong = np.where(pred != y)[0]
    margin = logits[:, 1] - logits[:, 0]
    order = np.argsort(-np.abs(margin[wrong])) if len(wrong) else np.array([], dtype=int)
    return {
        "index": wrong[order],
        "X": X[wrong[order]],
        "y_true": y[wrong[order]],
        "y_pred": pred[wrong[order]],
        "confidence": proba[wrong[order], pred[wrong[order]]] if len(wrong) else np.array([]),
        "p_true": proba[wrong[order], y[wrong[order]]] if len(wrong) else np.array([]),
        "margin": np.abs(margin[wrong[order]]) if len(wrong) else np.array([]),
        "n_wrong": int(len(wrong)),
        "n_total": int(len(y)),
        "distance_to_boundary": np.abs(proba[wrong[order], 1] - 0.5) if len(wrong) else np.array([]),
    }


def plot_error_samples(result: dict, path: str | Path, split: str = "test") -> Path:
    """把错误样本单独标出来，并给出 P(class=1) 的分布。"""
    err = error_samples(result, split)
    splits = result["splits"]
    X = {"train": splits.X_train, "val": splits.X_val, "test": splits.X_test}[split]
    y = {"train": splits.y_train, "val": splits.y_val, "test": splits.y_test}[split]
    xx, yy, zz = decision_surface(result, resolution=350)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))
    ax = axes[0]
    ax.contourf(xx, yy, zz, levels=20, cmap="RdBu_r", alpha=0.5, vmin=0, vmax=1)
    ax.contour(xx, yy, zz, levels=[0.5], colors="k", linewidths=1.3)
    ax.scatter(X[:, 0], X[:, 1], c="#BBBBBB", s=12, edgecolors="none")
    if err["n_wrong"]:
        ax.scatter(err["X"][:, 0], err["X"][:, 1], c=CLASS_COLORS[err["y_true"]],
                   s=110, edgecolors="k", linewidths=1.1, zorder=5)
        for k, (xv, yv) in enumerate(err["X"]):
            ax.annotate(f"#{k+1}", (xv, yv), textcoords="offset points", xytext=(6, 6),
                        fontsize=7)
    ax.set_title(f"{split} set: {err['n_wrong']}/{err['n_total']} misclassified "
                 f"({err['n_wrong']/err['n_total']:.1%})")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")

    ax = axes[1]
    p1 = result[split]["proba"][:, 1]
    bins = np.linspace(0, 1, 41)
    for label, color in [(0, CLASS_COLORS[0]), (1, CLASS_COLORS[1])]:
        ax.hist(p1[y == label], bins=bins, alpha=0.6, color=color, label=f"true {label}")
    ax.axvline(0.5, color="k", ls="--", lw=1.2)
    if err["n_wrong"]:
        ax.scatter(err["p_true"], np.full(err["n_wrong"], 2.0), marker="v", s=45,
                   color="k", zorder=5, label="misclassified")
    ax.set_xlabel("P(class=1) predicted")
    ax.set_ylabel("count")
    ax.set_title("predicted probability distributions")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.suptitle(f"error analysis | {result['config'].name}", fontsize=11)
    fig.tight_layout()
    return _save(fig, path)


# ----------------------------------------------------------------------------
# 5. 混淆矩阵对比（例如不均衡数据 加权 vs 不加权）
# ----------------------------------------------------------------------------
def plot_confusion_compare(results: list[dict], path: str | Path, title: str) -> Path:
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5.0 * n, 4.3), squeeze=False, layout="constrained")
    for ax, r in zip(axes[0], results):
        cm = confusion_matrix(r["test"]["y"], r["test"]["pred"], labels=[0, 1])
        cmn = cm / cm.sum(axis=1, keepdims=True)
        im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]}\n({cmn[i, j]:.2f})", ha="center", va="center",
                        fontsize=8, color="white" if cmn[i, j] > 0.55 else "black")
        rec_minor = cmn[1, 1] if cm[1].sum() else float("nan")
        ax.set_xticks([0, 1], ["pred 0", "pred 1"])
        ax.set_yticks([0, 1], ["true 0", "true 1"])
        ax.set_title(f"{r['config'].name}\nacc={r['stats']['test_acc']:.4f}, "
                     f"minority recall={rec_minor:.4f}", fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(title, fontsize=11)
    return _save(fig, path)


# ----------------------------------------------------------------------------
# 6. 过拟合演示
# ----------------------------------------------------------------------------
def plot_overfit(results: list[dict], path: str | Path) -> Path:
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.6), squeeze=False)
    for ax, r in zip(axes[0], results):
        h, s = r["history"], r["stats"]
        epochs = np.arange(1, len(h["train_loss"]) + 1)
        ax.plot(epochs, h["train_loss"], color="#2E6FD9", label="train loss", lw=1.7)
        ax.plot(epochs, h["val_loss"], color="#E4572E", label="val loss", lw=1.7)
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax2 = ax.twinx()
        ax2.plot(epochs, h["train_acc"], color="#2E6FD9", ls=":", lw=1.5, label="train acc")
        ax2.plot(epochs, h["val_acc"], color="#E4572E", ls=":", lw=1.5, label="val acc")
        ax2.set_ylabel("accuracy")
        ax2.set_ylim(0.5, 1.02)
        ax.set_title(
            f"{r['config'].name}\ntrain acc={s['final_train_acc']:.3f} "
            f"val acc={s['final_val_acc']:.3f} gap={s['generalization_gap']:.3f}",
            fontsize=9,
        )
        lines = [
            Line2D([], [], color="#2E6FD9", lw=1.7, label="train loss"),
            Line2D([], [], color="#E4572E", lw=1.7, label="val loss"),
            Line2D([], [], color="#2E6FD9", ls=":", lw=1.5, label="train acc"),
            Line2D([], [], color="#E4572E", ls=":", lw=1.5, label="val acc"),
        ]
        ax.legend(handles=lines, fontsize=7, loc="center right")
    fig.suptitle("overfitting demonstration", fontsize=11)
    fig.tight_layout()
    return _save(fig, path)
