"""一键跑完 Jotang ML Task 1 的全部实验，并把图/表/指标落盘。

用法::

    python run_all.py --out outputs                # 全部实验（默认 CPU）
    python run_all.py --steps baseline,ablation    # 只跑部分
    python run_all.py --epochs 200 --device cpu

实验内容:
    baseline   : 6:2:2 划分 + MLP 训练/验证/测试 + 保存/加载校验
    ablation   : 7 组单变量对照实验（宽度/深度/激活/lr/优化器/batch/噪声）
    grad       : zero_grad() 的作用（梯度清零 vs 不清零）
    imbalance  : 类别不均衡 + class weight 的对比
    overfit    : 小样本 / 标签噪声下故意过拟合，以及正则化的缓解
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from mlp_moons.analysis import (
    classification_metrics,
    error_table,
    format_table,
    read_csv,
    stats_row,
    write_csv,
    write_json,
)
from mlp_moons.data import SEED, make_imbalanced_splits, make_splits
from mlp_moons.engine import (
    TrainConfig,
    load_checkpoint,
    predict_with,
    save_checkpoint,
    set_seed,
    train_model,
)
from mlp_moons.viz import (
    plot_ablation_bars,
    plot_confusion,
    plot_confusion_compare,
    plot_curves,
    plot_curves_multi,
    plot_data_splits,
    plot_decision_boundary,
    plot_decision_boundary_multi,
    plot_error_samples,
    plot_overfit,
)

BASE = TrainConfig()  # hidden=64, n_hidden=2, relu, adam, lr=1e-2, batch=64, epochs=200


# ----------------------------------------------------------------------------
# 工具
# ----------------------------------------------------------------------------
def run(cfg: TrainConfig, splits, device: str = "cpu") -> dict:
    t0 = time.perf_counter()
    result = train_model(cfg, splits, device=device)
    result["wall_s"] = time.perf_counter() - t0
    s = result["stats"]
    print(
        f"  [{cfg.name:<16}] test_acc={s['test_acc']:.4f} best_val={s['best_val_acc']:.4f}"
        f"@{s['best_val_epoch']:<4d} gap={s['generalization_gap']:+.4f}"
        f" params={s['params']:>6d} {s['ms_per_epoch']:6.1f} ms/epoch"
        f" ram={s['ram_peak_mb']:6.0f}MB"
    )
    return result


def make_cfg(**overrides) -> TrainConfig:
    """从 BASE 出发，只覆盖指定的字段（保证一次只改一个变量）。"""
    cfg = replace(BASE, **{k: v for k, v in overrides.items() if k in TrainConfig.__dataclass_fields__})
    extra = dict(BASE.extra)
    extra.update({k: v for k, v in overrides.items() if k not in TrainConfig.__dataclass_fields__})
    cfg.extra = extra
    return cfg


# ----------------------------------------------------------------------------
# 1) baseline
# ----------------------------------------------------------------------------
def step_baseline(splits, out: Path, device: str):
    print("[1/5] baseline")
    plot_data_splits(splits, out / "figures" / "01_data_splits.png")
    result = run(make_cfg(name="baseline"), splits, device)

    plot_curves(result, out / "figures" / "02_baseline_curves.png")
    plot_decision_boundary(result, out / "figures" / "03_decision_boundary.png")
    plot_confusion(result, out / "figures" / "04_confusion_matrix.png")
    plot_error_samples(result, out / "figures" / "05_error_samples.png")

    ckpt = save_checkpoint(result, out / "models" / "baseline.pt")
    loaded = load_checkpoint(ckpt, device=device)
    X_test = splits.X_test_s
    pred_orig = predict_with(result["model"], X_test, device)["pred"]
    pred_load = predict_with(loaded["model"], X_test, device)["pred"]
    identical = bool(np.array_equal(pred_orig, pred_load))
    print(f"  save/load 一致性: {identical} -> {ckpt}")

    metrics = {k: classification_metrics(result, k) for k in ("train", "val", "test")}
    rows = [{"split": k, **v} for k, v in metrics.items()]
    write_csv(rows, out / "metrics" / "baseline_metrics.csv")
    write_json(
        {
            "config": result["config"].to_dict(),
            "stats": result["stats"],
            "metrics": metrics,
            "error_samples": error_table(result, "test"),
            "checkpoint_reload_identical": identical,
            "history": result["history"],
        },
        out / "metrics" / "baseline.json",
    )
    return result


# ----------------------------------------------------------------------------
# 2) 对照实验
# ----------------------------------------------------------------------------
ABLATIONS: dict[str, dict] = {
    "width": {
        "title": "隐藏层宽度 (hidden width)",
        "variants": [
            dict(name="w4", hidden=4),
            dict(name="w16", hidden=16),
            dict(name="w64 (base)", hidden=64),
            dict(name="w256", hidden=256),
        ],
    },
    "depth": {
        "title": "隐藏层数 (depth)",
        "variants": [
            dict(name="d1", n_hidden=1, hidden=64),
            dict(name="d2 (base)", n_hidden=2, hidden=64),
            dict(name="d4", n_hidden=4, hidden=64),
        ],
    },
    "activation": {
        "title": "激活函数 (activation)",
        "variants": [
            dict(name="relu (base)", activation="relu"),
            dict(name="tanh", activation="tanh"),
            dict(name="gelu", activation="gelu"),
            dict(name="sigmoid", activation="sigmoid"),
        ],
    },
    "activation_deep": {
        "title": "激活函数 x 深度 (activation, n_hidden=4)",
        "variants": [
            dict(name="d4+relu", n_hidden=4, activation="relu"),
            dict(name="d4+tanh", n_hidden=4, activation="tanh"),
            dict(name="d4+gelu", n_hidden=4, activation="gelu"),
            dict(name="d4+sigmoid", n_hidden=4, activation="sigmoid"),
        ],
    },
    "lr": {
        "title": "学习率 (learning rate)",
        "variants": [
            dict(name="lr=1e-4", lr=1e-4),
            dict(name="lr=1e-3", lr=1e-3),
            dict(name="lr=1e-2 (base)", lr=1e-2),
            dict(name="lr=1e-1", lr=1e-1),
        ],
    },
    "optimizer": {
        "title": "优化器 (optimizer)",
        "variants": [
            dict(name="sgd", optimizer="sgd"),
            dict(name="sgd+mom", optimizer="sgd_momentum"),
            dict(name="adam (base)", optimizer="adam"),
            dict(name="rmsprop", optimizer="rmsprop"),
        ],
    },
    "batch": {
        "title": "batch size",
        "variants": [
            dict(name="bs=16", batch_size=16),
            dict(name="bs=64 (base)", batch_size=64),
            dict(name="bs=256", batch_size=256),
            dict(name="bs=900 (full)", batch_size=900),
        ],
    },
    "noise": {
        "title": "数据噪声 (make_moons noise)",
        "variants": [
            dict(name="noise=0.00", hidden=64, noise=0.00),
            dict(name="noise=0.10", hidden=64, noise=0.10),
            dict(name="noise=0.20 (base)", hidden=64, noise=0.20),
            dict(name="noise=0.35", hidden=64, noise=0.35),
            dict(name="noise=0.50", hidden=64, noise=0.50),
        ],
    },
}


def step_ablation(splits, out: Path, device: str, groups: list[str] | None = None):
    print("[2/5] 对照实验（每组只改一个变量）")
    all_rows, results_by_group = [], {}
    selected = groups or list(ABLATIONS)
    for group in selected:
        spec = ABLATIONS[group]
        print(f"  group={group}: {spec['title']}")
        results = []
        for v in spec["variants"]:
            overrides = dict(v)
            noise = overrides.pop("noise", 0.2)
            group_splits = splits if np.isclose(noise, 0.2) else make_splits(
                n_samples=1500, noise=float(noise), seed=SEED
            )
            cfg = make_cfg(noise=float(noise), **overrides)
            res = run(cfg, group_splits, device)
            res["stats"]["group"] = group
            res["stats"]["noise"] = float(noise)
            results.append(res)
            all_rows.append(stats_row(res, group))

        results_by_group[group] = results
        plot_curves_multi(results, out / "figures" / f"ablation_{group}_curves.png",
                          metric="val_acc", title=spec["title"])
        plot_ablation_bars([r["stats"] for r in results],
                           out / "figures" / f"ablation_{group}_bars.png",
                           title=spec["title"], metric="test_acc")
        if group in {"width", "depth", "noise"}:
            plot_decision_boundary_multi(results,
                                         out / "figures" / f"ablation_{group}_boundary.png")
        write_csv([stats_row(r, group) for r in results], out / "metrics" / f"ablation_{group}.csv")

    # 汇总：把所有已存在的分组结果合并（支持分多次运行不同分组）
    merged: dict[tuple[str, str], dict] = {}
    for f in sorted((out / "metrics").glob("ablation_*.csv")):
        for r in read_csv(f):
            merged[(r.get("group", ""), r.get("name", ""))] = r
    if merged:
        write_csv(list(merged.values()), out / "metrics" / "all_runs.csv")

    # 给 README 用的紧凑表格
    cols = ["name", "params", "ms_per_epoch", "final_train_acc", "final_val_acc",
            "test_acc", "generalization_gap"]
    for group, results in results_by_group.items():
        format_table([stats_row(r, group) for r in results], cols,
                     out / "tables" / f"ablation_{group}.md")
    return results_by_group


# ----------------------------------------------------------------------------
# 2b) 多随机种子：量化"实验噪声"，判断组间差异是否显著
# ----------------------------------------------------------------------------
def step_seeds(splits, out: Path, device: str, seeds: tuple[int, ...] = (0, 1, 2, 3, 4)):
    print("[2b] 多种子重复实验（baseline 配置）")
    results = []
    for s in seeds:
        res = run(make_cfg(name=f"seed={s}", seed=s, zero_grad=True), splits, device)
        results.append(res)

    accs = np.array([r["stats"]["test_acc"] for r in results])
    val_accs = np.array([r["stats"]["best_val_acc"] for r in results])
    print(
        f"  test acc: mean={accs.mean():.4f} std={accs.std(ddof=1):.4f} "
        f"min={accs.min():.4f} max={accs.max():.4f} | "
        f"val acc: mean={val_accs.mean():.4f} std={val_accs.std(ddof=1):.4f}"
    )
    rows = [stats_row(r, "seeds") for r in results]
    rows.append(
        {
            "group": "seeds",
            "name": "mean +/- std",
            "test_acc": float(accs.mean()),
            "final_val_acc": float(val_accs.mean()),
            "final_train_acc": float(np.mean([r["stats"]["final_train_acc"] for r in results])),
            "generalization_gap": float(np.mean([r["stats"]["generalization_gap"] for r in results])),
            "params": results[0]["stats"]["params"],
            "ms_per_epoch": float(np.mean([r["stats"]["ms_per_epoch"] for r in results])),
            "test_acc_std": float(accs.std(ddof=1)),
        }
    )
    write_csv(rows, out / "metrics" / "seeds.csv")
    format_table(rows, ["name", "params", "ms_per_epoch", "final_train_acc",
                        "final_val_acc", "test_acc", "generalization_gap"],
                 out / "tables" / "seeds.md")
    plot_ablation_bars([r["stats"] for r in results], out / "figures" / "seeds_bars.png",
                       title="baseline 配置在 5 个随机种子下的表现", metric="test_acc")
    return results


# ----------------------------------------------------------------------------
# 3) zero_grad 实验
# ----------------------------------------------------------------------------
def step_grad(splits, out: Path, device: str):
    print("[3/5] zero_grad 的作用")
    res_ok = run(make_cfg(name="with zero_grad"), splits, device)
    res_bad = run(make_cfg(name="without zero_grad", zero_grad=False), splits, device)
    plot_curves_multi([res_ok, res_bad], out / "figures" / "grad_accumulation.png",
                      metric="train_loss", title="optimizer.zero_grad() 的影响（train loss / val acc）")
    write_csv([stats_row(res_ok, "grad"), stats_row(res_bad, "grad")],
              out / "metrics" / "grad_zero_grad.csv")
    return [res_ok, res_bad]


# ----------------------------------------------------------------------------
# 4) 类别不均衡
# ----------------------------------------------------------------------------
def step_imbalance(out: Path, device: str, n_samples: int = 4000, ratio: float = 0.1):
    print("[4/5] 类别不均衡")
    spl = make_imbalanced_splits(n_samples=n_samples, minority_ratio=ratio, noise=0.2, seed=SEED)
    print(f"  数据: {spl.summary()}")
    plain = run(make_cfg(name="imbalanced / plain CE"), spl, device)
    weighted = run(make_cfg(name="imbalanced / class weight", class_weight="balanced"), spl, device)

    plot_confusion_compare([plain, weighted], out / "figures" / "imbalance_confusion.png",
                           title="类别不均衡：普通交叉熵 vs 类别加权")
    plot_curves_multi([plain, weighted], out / "figures" / "imbalance_curves.png",
                      metric="val_acc", title="类别不均衡下的 val accuracy（具有欺骗性）")

    rows = []
    for tag, r in [("plain", plain), ("weighted", weighted)]:
        m = classification_metrics(r, "test")
        rows.append({"variant": tag, **{k: v for k, v in m.items() if k != "split"}})
    write_csv(rows, out / "metrics" / "imbalance.csv")
    format_table(
        rows,
        ["variant", "accuracy", "balanced_accuracy", "recall_1", "precision_1",
         "f1_1", "macro_f1", "roc_auc", "fn", "fp"],
        out / "tables" / "imbalance.md",
    )
    return plain, weighted


# ----------------------------------------------------------------------------
# 5) 故意过拟合
# ----------------------------------------------------------------------------
def step_overfit(splits, out: Path, device: str, epochs: int = 2000, n_train: int = 30):
    print("[5/5] 过拟合演示")
    set_seed(SEED)
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(splits.X_train), size=n_train, replace=False)
    X_small = splits.X_train[idx]
    y_small = splits.y_train[idx].copy()

    y_flipped = y_small.copy()
    n_flip = max(1, int(0.2 * n_train))
    flip_idx = rng.choice(n_train, size=n_flip, replace=False)
    y_flipped[flip_idx] = 1 - y_flipped[flip_idx]

    small = replace(splits, X_train=X_small, y_train=y_small)
    noisy = replace(splits, X_train=X_small, y_train=y_flipped)

    big = dict(hidden=256, n_hidden=3, epochs=epochs, lr=3e-3)
    res_small = run(make_cfg(name=f"small data (n={n_train})", **big), small, device)
    res_noisy = run(make_cfg(name=f"20% label noise (n={n_train})", **big), noisy, device)
    res_reg = run(
        make_cfg(name="label noise + weight decay", weight_decay=5e-2, dropout=0.2, **big),
        noisy, device,
    )

    plot_overfit([res_small, res_noisy, res_reg], out / "figures" / "overfit_curves.png")
    plot_decision_boundary_multi([res_small, res_noisy, res_reg],
                                 out / "figures" / "overfit_boundary.png")
    cols = ["name", "params", "epochs", "final_train_acc", "final_val_acc", "final_val_loss",
            "test_acc", "generalization_gap"]
    rows = [stats_row(r, "overfit") for r in [res_small, res_noisy, res_reg]]
    write_csv(rows, out / "metrics" / "overfit.csv")
    format_table(rows, cols, out / "tables" / "overfit.md")
    return res_small, res_noisy, res_reg


# ----------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description="Jotang ML Task 1 runner")
    p.add_argument("--out", type=Path, default=Path("outputs"))
    p.add_argument("--device", default="cpu")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--overfit-epochs", type=int, default=2000)
    p.add_argument(
        "--steps",
        default="baseline,ablation,seeds,grad,imbalance,overfit",
        help="逗号分隔: baseline,ablation,seeds,grad,imbalance,overfit",
    )
    p.add_argument(
        "--ablation-groups",
        default="",
        help="只跑指定的分组（逗号分隔），留空表示全部；见 run_all.ABLATIONS",
    )
    args = p.parse_args()

    global BASE
    BASE = replace(BASE, epochs=args.epochs)
    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    out = args.out
    for sub in ("figures", "metrics", "models", "tables"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(torch.get_num_threads())
    print(f"torch {torch.__version__} | device={args.device} | threads={torch.get_num_threads()}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    splits = make_splits(n_samples=1500, noise=0.2, seed=SEED)
    print(f"数据划分: {splits.summary()}")

    t_all = time.perf_counter()

    # 预热：第一次运行包含 lazy init，计时不准
    warm = run(make_cfg(name="warmup", epochs=3), splits, args.device)
    del warm

    if "baseline" in steps:
        step_baseline(splits, out, args.device)
    if "ablation" in steps:
        groups = [g.strip() for g in args.ablation_groups.split(",") if g.strip()]
        step_ablation(splits, out, args.device, groups or None)
    if "seeds" in steps:
        step_seeds(splits, out, args.device)
    if "grad" in steps:
        step_grad(splits, out, args.device)
    if "imbalance" in steps:
        step_imbalance(out, args.device)
    if "overfit" in steps:
        step_overfit(splits, out, args.device, epochs=args.overfit_epochs)

    print(f"\n全部完成，用时 {time.perf_counter() - t_all:.1f}s -> {out.resolve()}")


if __name__ == "__main__":
    main()
