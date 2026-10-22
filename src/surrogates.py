"""Quantile surrogates for choosing the next reaction conditions.

TabPFN is tried once, at import. If that import or a tiny fit fails, later
calls use a random forest and do not touch TabPFN again.
"""

from __future__ import annotations

import inspect
import os
import warnings

import numpy as np

# Flipped off if the TabPFN smoke test is too slow or crashes.
_ALLOW_TABPFN = True

SURROGATE_NAME: str = "rf"

_Z90 = 1.2816


def _as_2d(X) -> np.ndarray:
    arr = np.asarray(X, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"X must be 2-dimensional, got shape {arr.shape}")
    return arr


def _as_1d(y) -> np.ndarray:
    if y is None:
        return np.zeros((0,), dtype=float)
    return np.asarray(y, dtype=float).ravel()


def _seed_int(seed) -> int:
    if seed is None:
        return 0
    return int(seed)


def _constant_band(y_hist: np.ndarray, n_cand: int):
    mean = float(np.mean(y_hist)) if y_hist.size else 0.0
    return (
        np.full(n_cand, mean - 20.0, dtype=float),
        np.full(n_cand, mean, dtype=float),
        np.full(n_cand, mean + 20.0, dtype=float),
    )


def _needs_constant_band(y_hist: np.ndarray) -> bool:
    if y_hist.size < 2:
        return True
    return float(np.max(y_hist) - np.min(y_hist)) <= 1e-9


def _split_quantiles(raw, n: int):
    if isinstance(raw, (list, tuple)) and len(raw) == 3:
        cols = [np.asarray(part, dtype=float).reshape(-1) for part in raw]
        if all(col.shape == (n,) for col in cols):
            return cols[0], cols[1], cols[2]
    arr = np.squeeze(np.asarray(raw, dtype=float))
    if arr.ndim == 2 and arr.shape == (n, 3):
        return arr[:, 0].copy(), arr[:, 1].copy(), arr[:, 2].copy()
    if arr.ndim == 2 and arr.shape == (3, n):
        return arr[0].copy(), arr[1].copy(), arr[2].copy()
    raise ValueError(f"Unrecognized quantile array shape {getattr(arr, 'shape', None)} for n={n}")


def _coerce_mean_std(mu, sd, n: int):
    mu = np.asarray(mu, dtype=float).reshape(-1)
    sd = np.asarray(sd, dtype=float).reshape(-1)
    if mu.shape != (n,) or sd.shape != (n,):
        raise ValueError(f"mean/std shapes {mu.shape}, {sd.shape} != ({n},)")
    sd = np.maximum(sd, 0.0)
    return mu - _Z90 * sd, mu.copy(), mu + _Z90 * sd


def _mean_std_from_output(out, n: int):
    mu = None
    sd = None
    if isinstance(out, dict):
        mu = out.get("mean", out.get("loc"))
        if "std" in out:
            sd = out["std"]
        elif "scale" in out:
            sd = out["scale"]
        elif "variance" in out:
            sd = np.sqrt(np.maximum(np.asarray(out["variance"], dtype=float), 0.0))
    elif isinstance(out, (tuple, list)) and len(out) >= 2 and not isinstance(out, (str, bytes)):
        mu, sd = out[0], out[1]
    else:
        if hasattr(out, "mean"):
            mu = getattr(out, "mean")
        if hasattr(out, "std"):
            sd = getattr(out, "std")
        elif hasattr(out, "variance"):
            sd = np.sqrt(np.maximum(np.asarray(getattr(out, "variance"), dtype=float), 0.0))
    if mu is None or sd is None:
        raise ValueError("prediction output has no mean and std")
    return _coerce_mean_std(mu, sd, n)


def _predict_mean_std(model, X_cand: np.ndarray):
    n = int(X_cand.shape[0])
    predict = model.predict
    params = {}
    try:
        params = inspect.signature(predict).parameters
    except (TypeError, ValueError):
        params = {}
    has_output_type = "output_type" in params or any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
    )
    if has_output_type:
        for output_type in ("full", "main"):
            try:
                out = predict(X_cand, output_type=output_type)
                return _mean_std_from_output(out, n)
            except Exception:
                continue
    mu = np.asarray(predict(X_cand), dtype=float).reshape(-1)
    if mu.shape != (n,):
        raise ValueError(f"mean shape {mu.shape} != ({n},)")
    sd = np.zeros(n, dtype=float)
    return _coerce_mean_std(mu, sd, n)


def _tabpfn_fit_predict(X_hist: np.ndarray, y_hist: np.ndarray, X_cand: np.ndarray, seed):
    from tabpfn import TabPFNRegressor

    # Never open a browser or wait on stdin if weights are gated.
    os.environ.setdefault("TABPFN_NO_BROWSER", "1")
    model = TabPFNRegressor(device="cpu", random_state=_seed_int(seed))
    model.fit(X_hist, y_hist)
    n = int(X_cand.shape[0])
    try:
        raw = model.predict(X_cand, output_type="quantiles", quantiles=[0.1, 0.5, 0.9])
        return _split_quantiles(raw, n)
    except (TypeError, ValueError):
        return _predict_mean_std(model, X_cand)


def _rf_fit_predict(X_hist: np.ndarray, y_hist: np.ndarray, X_cand: np.ndarray, seed):
    from sklearn.ensemble import RandomForestRegressor

    rf = RandomForestRegressor(n_estimators=300, random_state=_seed_int(seed))
    rf.fit(X_hist, y_hist)
    tree_preds = np.stack([tree.predict(X_cand) for tree in rf.estimators_], axis=0)
    q10, q50, q90 = np.percentile(tree_preds, [10, 50, 90], axis=0)
    return (
        np.asarray(q10, dtype=float).ravel(),
        np.asarray(q50, dtype=float).ravel(),
        np.asarray(q90, dtype=float).ravel(),
    )


def _tabpfn_token_present() -> bool:
    """True when TabPFN has a saved API key and will not open a browser."""
    if os.environ.get("TABPFN_TOKEN", "").strip():
        return True
    home = os.path.expanduser("~")
    for path in (
        os.path.join(home, ".cache", "tabpfn", "auth_token"),
        os.path.join(home, ".tabpfn", "token"),
    ):
        try:
            if os.path.isfile(path) and os.path.getsize(path) > 0:
                return True
        except OSError:
            continue
    return False


def _probe_tabpfn() -> bool:
    """Import TabPFN and fit a tiny set once. Failure selects the forest path.

    Current TabPFN builds download gated weights. Without a saved API token
    that fit opens a browser, so a missing token is treated as failure and
    the package is not imported.
    """
    if not _ALLOW_TABPFN:
        return False
    os.environ.setdefault("TABPFN_NO_BROWSER", "1")
    if not _tabpfn_token_present():
        return False
    try:
        from tabpfn import TabPFNRegressor  # noqa: F401
    except Exception:
        return False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rng = np.random.default_rng(0)
            X = rng.normal(size=(8, 3))
            y = X[:, 0].copy()
            q10, q50, q90 = _tabpfn_fit_predict(X[:6], y[:6], X[6:], seed=0)
        if q10.shape != (2,) or q50.shape != (2,) or q90.shape != (2,):
            return False
        finite = np.isfinite(q10) & np.isfinite(q50) & np.isfinite(q90)
        return bool(np.all(finite))
    except Exception:
        return False


if _probe_tabpfn():
    SURROGATE_NAME = "tabpfn"
else:
    SURROGATE_NAME = "rf"


def fit_predict_quantiles(X_hist, y_hist, X_cand, seed):
    """Return (q10, q50, q90) for each candidate row.

    Each array has shape (n_cand,). Yields are not clipped to 0-100.
    """
    X_cand_arr = _as_2d(X_cand)
    y = _as_1d(y_hist)
    n = int(X_cand_arr.shape[0])
    if _needs_constant_band(y):
        return _constant_band(y, n)

    X_hist_arr = _as_2d(X_hist)
    if X_hist_arr.shape[0] != y.shape[0]:
        raise ValueError(
            f"X_hist rows ({X_hist_arr.shape[0]}) != y_hist length ({y.shape[0]})"
        )
    if X_hist_arr.shape[1] != X_cand_arr.shape[1]:
        raise ValueError(
            f"X_hist has {X_hist_arr.shape[1]} features, X_cand has {X_cand_arr.shape[1]}"
        )

    if SURROGATE_NAME == "tabpfn":
        try:
            return _tabpfn_fit_predict(X_hist_arr, y, X_cand_arr, seed)
        except Exception:
            # A later TabPFN failure must not break the caller. Stay on the forest.
            return _rf_fit_predict(X_hist_arr, y, X_cand_arr, seed)
    return _rf_fit_predict(X_hist_arr, y, X_cand_arr, seed)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 4))
    y = X[:, 0] * 10.0

    q10, q50, q90 = fit_predict_quantiles(X[:20], y[:20], X[20:], seed=0)
    assert q10.shape == q50.shape == q90.shape == (10,)
    assert np.all(np.isfinite(q10) & np.isfinite(q50) & np.isfinite(q90))
    assert np.all(q10 <= q50 + 1e-8) and np.all(q50 <= q90 + 1e-8)
    corr = float(np.corrcoef(X[20:, 0], q50)[0, 1])
    assert corr > 0.3, corr

    q10b, q50b, q90b = fit_predict_quantiles(
        X[:20].tolist(), y[:20].tolist(), X[20:].tolist(), seed=0
    )
    assert q10b.shape == (10,)
    assert np.allclose(q50, q50b)

    y_const = np.full(5, 42.0)
    a, b, c = fit_predict_quantiles(X[:5], y_const, X[5:8], seed=1)
    assert np.allclose(a, 22.0) and np.allclose(b, 42.0) and np.allclose(c, 62.0)

    y_near = np.array([7.0, 7.0 + 1e-12, 7.0])
    a, b, c = fit_predict_quantiles(X[:3], y_near, X[3:6], seed=2)
    assert np.allclose(a, np.mean(y_near) - 20.0)
    assert np.allclose(b, np.mean(y_near))
    assert np.allclose(c, np.mean(y_near) + 20.0)

    a, b, c = fit_predict_quantiles(np.zeros((0, 3)), [], np.zeros((4, 3)), seed=0)
    assert np.allclose(a, -20.0) and np.allclose(b, 0.0) and np.allclose(c, 20.0)

    a, b, c = fit_predict_quantiles(X[:1], [15.0], X[1:4], seed=0)
    assert np.allclose(a, -5.0) and np.allclose(b, 15.0) and np.allclose(c, 35.0)

    y_hi = np.array([150.0, 180.0, 200.0, 170.0])
    a, b, c = fit_predict_quantiles(X[:4], y_hi, X[4:7], seed=3)
    assert a.shape == (3,)
    assert np.all(np.isfinite(a) & np.isfinite(b) & np.isfinite(c))

    print("surrogates self-check ok", SURROGATE_NAME, "corr", round(corr, 3))
