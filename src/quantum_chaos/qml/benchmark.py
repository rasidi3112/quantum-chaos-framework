from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats import wilcoxon
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .quantum_kernel import QuantumKernel


@dataclass(frozen=True)
class KernelModel:
    name: str
    factory: Callable[[float], QuantumKernel] | None
    grid: Sequence[float]


def _quantum_accuracy(model: KernelModel, X_tr, y_tr, X_te, y_te, C_grid, seed) -> float:
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    best: tuple[float, float, float] = (-np.inf, float(model.grid[0]), float(C_grid[0]))
    assert model.factory is not None
    for param in model.grid:
        kernel = model.factory(param)
        K = kernel.kernel_matrix(X_tr)
        for C in C_grid:
            scores = []
            for tr, va in cv.split(X_tr, y_tr):
                clf = SVC(kernel="precomputed", C=C).fit(K[np.ix_(tr, tr)], y_tr[tr])
                scores.append(clf.score(K[np.ix_(va, tr)], y_tr[va]))
            if np.mean(scores) > best[0]:
                best = (float(np.mean(scores)), float(param), float(C))
    kernel = model.factory(best[1])
    clf = SVC(kernel="precomputed", C=best[2]).fit(kernel.kernel_matrix(X_tr), y_tr)
    return float(clf.score(kernel.kernel_matrix(X_te, X_tr), y_te))


def _rbf_accuracy(X_tr, y_tr, X_te, y_te, C_grid, gamma_grid, seed) -> float:
    search = GridSearchCV(
        SVC(kernel="rbf"),
        {"C": list(C_grid), "gamma": list(gamma_grid)},
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=seed),
    )
    search.fit(X_tr, y_tr)
    return float(search.score(X_te, y_te))


def compare_with_rbf_baseline(
    dataset: Callable[[int], tuple[np.ndarray, np.ndarray]],
    models: Sequence[KernelModel],
    seeds: Sequence[int],
    C_grid: Sequence[float] = (0.1, 1.0, 10.0),
    gamma_grid: Sequence[float] = (0.1, 0.5, 1.0, 2.0),
    test_size: float = 0.3,
) -> dict:
    accuracies: dict[str, list[float]] = {"rbf": []}
    for m in models:
        accuracies[m.name] = []
    for seed in seeds:
        X, y = dataset(seed)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y
        )
        scaler = StandardScaler().fit(X_tr)
        X_tr, X_te = scaler.transform(X_tr), scaler.transform(X_te)
        accuracies["rbf"].append(_rbf_accuracy(X_tr, y_tr, X_te, y_te, C_grid, gamma_grid, seed))
        for m in models:
            accuracies[m.name].append(_quantum_accuracy(m, X_tr, y_tr, X_te, y_te, C_grid, seed))
    summary = {}
    base = np.array(accuracies["rbf"])
    for name, acc in accuracies.items():
        a = np.array(acc)
        row = {"mean": float(a.mean()), "std": float(a.std(ddof=1)), "accuracies": acc}
        if name != "rbf":
            diff = a - base
            row["mean_difference_vs_rbf"] = float(diff.mean())
            row["wilcoxon_p"] = (
                float(wilcoxon(a, base, zero_method="zsplit").pvalue) if np.any(diff != 0) else 1.0
            )
        summary[name] = row
    return {"n_seeds": len(seeds), "results": summary}
