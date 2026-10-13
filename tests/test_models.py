"""Q4: B0 is DermaFair's frozen head; M1/C1 differ from it only by the training views. Synthetic features only."""

from __future__ import annotations

import numpy as np
from conftest import needs_base

from nuisance import models as M
from nuisance import spec

pytestmark = needs_base


def _data(n=240, d=20, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.3).astype(int)
    X = rng.normal(size=(n, d)) + 0.8 * y[:, None] * (np.arange(d) < 4)
    return X, y, np.arange(n) % 5


def test_class_weights_are_balanced():
    _, y, _ = _data()
    w = M.class_weights(y)
    assert np.isclose(w.sum(), len(y)) and np.isclose(w[y == 1].sum(), w[y == 0].sum())


def test_views_identical_to_the_clean_image_reproduce_b0():
    X, y, _ = _data()
    V = np.repeat(X[:, None, :], spec.K_VIEWS, axis=1)  # every view equals its clean image
    Z = np.random.default_rng(1).normal(size=(50, X.shape[1]))
    assert np.allclose(M.fit_b0(X, y)(Z), M.fit_augmented(X, y, V)(Z), atol=1e-4)


def test_views_change_the_model():
    X, y, _ = _data()
    V = X[:, None, :] + np.random.default_rng(2).normal(0, 2, (len(X), spec.K_VIEWS, X.shape[1]))
    Z = np.random.default_rng(3).normal(size=(50, X.shape[1]))
    assert not np.allclose(M.fit_b0(X, y)(Z), M.fit_augmented(X, y, V)(Z), atol=1e-3)


def test_out_of_fold_uses_the_model_of_each_images_fold():
    folds = np.array([0, 1, 2, 3, 4, 0, 1])
    predictors = [lambda Z, k=k: np.full(len(Z), float(k)) for k in range(5)]
    assert M.out_of_fold(predictors, folds, np.zeros((7, 2))).tolist() == folds.tolist()


def test_fit_folds_never_trains_on_the_held_out_fold():
    X, y, folds = _data()
    X_poisoned = X.copy()
    X_poisoned[folds == 2] += 1000.0  # if fold 2's model saw these rows its scaler would be dominated by them
    a = M.fit_folds(X, y, folds)[2](X[:10])
    b = M.fit_folds(X_poisoned, y, folds)[2](X[:10])
    assert np.allclose(a, b)


def test_external_averages_the_fold_models():
    predictors = [lambda Z, k=k: np.full(len(Z), float(k)) for k in range(5)]
    assert np.allclose(M.external(predictors, np.zeros((3, 2))), 2.0)
