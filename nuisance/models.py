"""B0, M1 and C1 (PREREGISTRATION.md section 5), fitted once per patient-level outer fold.

B0 calls DermaFair's frozen `lab.logreg` and `lab.class_weights` directly. M1 and C1 use the same head
(StandardScaler + LogisticRegression, C = 0.003) with K training views per image: the clean row keeps half of the
image's class weight and the K views share the other half, and the scaler is fit on the clean images only.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from . import spec


def class_weights(y: np.ndarray) -> np.ndarray:
    from lab import class_weights as frozen  # DermaFair, commit 89be54e

    return frozen(np.asarray(y).astype(int))


def fit_b0(X: np.ndarray, y: np.ndarray):
    """DermaFair's final-model recipe, unchanged."""
    from lab import logreg  # DermaFair, commit 89be54e

    y = np.asarray(y).astype(int)
    return logreg(X, y, class_weights(y), C=spec.C)


def fit_augmented(X: np.ndarray, y: np.ndarray, V: np.ndarray):
    """M1 / C1: B0's head trained on clean rows plus V[i] (K views of image i), weights 0.5 * w_i and 0.5 * w_i / K."""
    y = np.asarray(y).astype(int)
    n, k, d = V.shape
    w = class_weights(y)
    scaler = StandardScaler().fit(X)  # clean training images only, as in B0
    Xa = np.concatenate([X, V.reshape(n * k, d)])
    ya = np.concatenate([y, np.repeat(y, k)])  # V.reshape keeps each image's K views together, in order
    wa = np.concatenate([spec.CLEAN_WEIGHT * w, np.repeat((1.0 - spec.CLEAN_WEIGHT) * w / k, k)])
    model = LogisticRegression(C=spec.C, max_iter=spec.MAX_ITER).fit(scaler.transform(Xa), ya, sample_weight=wa)
    return lambda Z: model.predict_proba(scaler.transform(Z))[:, 1]


def fit_folds(X: np.ndarray, y: np.ndarray, folds: np.ndarray, V: np.ndarray | None = None) -> list:
    """One predictor per outer fold (ascending fold id), each trained on the images of the other folds."""
    out = []
    for k in np.unique(folds):
        tr = folds != k
        out.append(fit_b0(X[tr], y[tr]) if V is None else fit_augmented(X[tr], y[tr], V[tr]))
    return out


def out_of_fold(predictors: list, folds: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Each internal image scored by the model of its own fold, which never trained on it."""
    out = np.full(len(Z), np.nan)
    for k, f in zip(np.unique(folds), predictors):
        te = folds == k
        out[te] = f(Z[te])
    return out


def external(predictors: list, Z: np.ndarray) -> np.ndarray:
    """External images: the five fold models' probabilities averaged, as DermaFair does."""
    return np.mean([f(Z) for f in predictors], 0)
