"""在独立子进程里逐个跑配置，测量"干净的"峰值内存与每轮耗时。

为什么要单开进程：同一个 Python 进程里连着跑很多实验，进程峰值工作集只会单调上升，
后面的实验会继承前面实验的内存峰值，横向比较就不公平了。
每个配置用一个新的子进程，跑完立刻记录该进程的峰值 RSS，再做差。

用法::

    python scripts/measure_resources.py --out outputs/metrics/resources.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mlp_moons.data import SEED, make_splits  # noqa: E402

CONFIGS = [
    {"name": "import only (no training)", "mode": "import_only"},
    {"name": "w16", "hidden": 16},
    {"name": "w64 (base)", "hidden": 64},
    {"name": "w256", "hidden": 256},
    {"name": "d1", "n_hidden": 1},
    {"name": "d4", "n_hidden": 4},
    {"name": "bs16", "batch_size": 16},
    {"name": "bs256", "batch_size": 256},
    {"name": "bs900 (full)", "batch_size": 900},
]


def worker(spec: dict) -> None:
    """子进程入口：训练一个配置，把统计信息以 JSON 打到 stdout。"""
    from mlp_moons.engine import TrainConfig, memory_mb, train_model

    if spec.get("mode") == "import_only":
        cur, peak = memory_mb()
        print(json.dumps({"name": spec["name"], "peak_rss_mb": peak, "mode": "import_only"}))
        return

    splits = make_splits(n_samples=1500, noise=0.2, seed=SEED)
    kwargs = {
        k: v
        for k, v in spec.items()
        if k in TrainConfig.__dataclass_fields__ and k != "name"
    }
    cfg = TrainConfig(name=spec["name"], **kwargs)
    result = train_model(cfg, splits, device="cpu")
    stats = dict(result["stats"])
    stats["mode"] = "train"
    stats["peak_rss_mb_final"] = memory_mb()[1]
    print(json.dumps(stats))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("outputs/metrics/resources.csv"))
    p.add_argument("--worker", default=None, help=argparse.SUPPRESS)
    p.add_argument("--spec", default=None, help=argparse.SUPPRESS)
    args = p.parse_args()

    if args.worker is not None:
        worker(json.loads(args.worker))
        return

    rows = []
    for spec in CONFIGS:
        proc = subprocess.run(
            [sys.executable, __file__, "--worker", json.dumps(spec)],
            capture_output=True,
            text=True,
        )
        line = [l for l in proc.stdout.splitlines() if l.strip().startswith("{")]
        if not line:
            print(f"!! {spec['name']} 失败\n{proc.stderr[-800:]}")
            continue
        rows.append(json.loads(line[-1]))
        print(f"  {spec['name']:<26} done")

    base = next((r["peak_rss_mb_final" if r["mode"] == "train" else "peak_rss_mb"]
                 for r in rows if r["mode"] == "import_only"), None)
    for r in rows:
        peak = r.get("peak_rss_mb_final", r.get("peak_rss_mb"))
        r["peak_rss_mb"] = peak
        r["rss_overhead_vs_import_mb"] = None if base is None else peak - base

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["name", "mode", "peak_rss_mb", "rss_overhead_vs_import_mb", "ms_per_epoch",
              "params", "param_size_mb", "train_time_s", "test_acc"]
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
