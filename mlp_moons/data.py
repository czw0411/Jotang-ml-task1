"""数据模块：用 sklearn.datasets.make_moons 生成二分类数据并做三分划分。

数据完全由接口现场生成，不需要下载任何文件。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42


@dataclass
class Splits:
    """保存三分数据集以及标准化器。

    X_* 为 float32 的原始特征，X_*_s 为只用训练集拟合的 StandardScaler 变换后的特征。
    """

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    scaler: StandardScaler
    X_all: np.ndarray
    y_all: np.ndarray

    @property
    def X_train_s(self) -> np.ndarray:
        return self.scaler.transform(self.X_train).astype(np.float32)

    @property
    def X_val_s(self) -> np.ndarray:
        return self.scaler.transform(self.X_val).astype(np.float32)

    @property
    def X_test_s(self) -> np.ndarray:
        return self.scaler.transform(self.X_test).astype(np.float32)

    def summary(self) -> str:
        return (
            f"train={self.X_train.shape[0]} val={self.X_val.shape[0]} "
            f"test={self.X_test.shape[0]} | "
            f"train balance={np.bincount(self.y_train).tolist()} "
            f"val balance={np.bincount(self.y_val).tolist()} "
            f"test balance={np.bincount(self.y_test).tolist()}"
        )


def make_splits(
    n_samples: int = 1500,
    noise: float = 0.2,
    seed: int = SEED,
    val_size: float = 0.2,
    test_size: float = 0.2,
) -> Splits:
    """生成双月数据并按 6:2:2 划分训练/验证/测试集。

    关键点：先切出 test，再从剩余数据里切 val，且三次划分都 stratify=y，
    保证各子集类别比例一致、互不重叠（无数据泄漏）。
    StandardScaler 只用训练集拟合，验证/测试集只做 transform。
    """
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    X = X.astype(np.float32)
    y = y.astype(np.int64)

    holdout = val_size + test_size
    X_train, X_hold, y_train, y_hold = train_test_split(
        X, y, test_size=holdout, random_state=seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_hold,
        y_hold,
        test_size=test_size / holdout,
        random_state=seed,
        stratify=y_hold,
    )

    scaler = StandardScaler().fit(X_train)  # 只用训练集拟合，避免数据泄漏

    return Splits(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        scaler=scaler,
        X_all=X,
        y_all=y,
    )


def make_imbalanced_splits(
    n_samples: int = 2000,
    minority_ratio: float = 0.05,
    noise: float = 0.2,
    seed: int = SEED,
    val_size: float = 0.2,
    test_size: float = 0.2,
) -> Splits:
    """构造类别极不均衡的数据集：class 0 占多数，class 1 只保留 minority_ratio。

    做法仍是 make_moons 生成，然后把 class 1 随机下采样到指定比例。
    """
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    rng = np.random.default_rng(seed)

    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    n_minority = max(2, int(len(idx0) * minority_ratio / (1 - minority_ratio)))
    idx1_keep = rng.choice(idx1, size=n_minority, replace=False)

    keep = np.concatenate([idx0, idx1_keep])
    rng.shuffle(keep)
    X, y = X[keep].astype(np.float32), y[keep].astype(np.int64)

    holdout = val_size + test_size
    X_train, X_hold, y_train, y_hold = train_test_split(
        X, y, test_size=holdout, random_state=seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_hold,
        y_hold,
        test_size=test_size / holdout,
        random_state=seed,
        stratify=y_hold,
    )
    scaler = StandardScaler().fit(X_train)
    return Splits(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        scaler=scaler,
        X_all=X,
        y_all=y,
    )
