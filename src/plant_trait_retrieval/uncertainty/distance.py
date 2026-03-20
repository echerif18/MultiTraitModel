from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None


@dataclass
class DistanceUncertaintyConfig:
    n_neighbors: int = 50
    metric: str = "euclidean"


class DistanceUncertaintyEstimator:
    """Distance-based uncertainty using kNN in feature space.

    Uncertainty score is the mean distance to k nearest training samples.
    A linear calibration maps distances to expected absolute residuals.
    """

    def __init__(self, cfg: DistanceUncertaintyConfig | None = None) -> None:
        self.cfg = cfg or DistanceUncertaintyConfig()
        self.nn_model: NearestNeighbors | None = None
        self.calibrator: LinearRegression | None = None

    def fit(
        self,
        train_features: np.ndarray,
        calib_features: np.ndarray,
        calib_residuals: np.ndarray,
    ) -> "DistanceUncertaintyEstimator":
        self.nn_model = NearestNeighbors(
            n_neighbors=self.cfg.n_neighbors,
            metric=self.cfg.metric,
            algorithm="auto",
        )
        self.nn_model.fit(train_features)

        calib_unc = self.score(calib_features)
        target = np.nanmean(np.abs(calib_residuals), axis=1)
        finite = np.isfinite(target) & np.isfinite(calib_unc)
        if finite.sum() >= 3:
            self.calibrator = LinearRegression()
            self.calibrator.fit(calib_unc[finite].reshape(-1, 1), target[finite])
        else:
            self.calibrator = None
        return self

    def score(self, features: np.ndarray) -> np.ndarray:
        if self.nn_model is None:
            raise RuntimeError("Call fit() before score().")
        dists, _ = self.nn_model.kneighbors(features, return_distance=True)
        return dists.mean(axis=1)

    def predict_interval_scale(self, features: np.ndarray) -> np.ndarray:
        raw = self.score(features)
        if self.calibrator is None:
            return raw
        pred = self.calibrator.predict(raw.reshape(-1, 1))
        return np.clip(pred, a_min=1e-8, a_max=None)


def _as_float32_contiguous(x: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(x.astype(np.float32))


def normalize_l2_rows(x: np.ndarray) -> np.ndarray:
    out = x.copy()
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    out /= norms
    return out


def faiss_knn(
    train: np.ndarray,
    query: np.ndarray,
    k: int,
    metric: str = "l2",
    gpu: int = -1,
    require_faiss: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """FAISS kNN search; falls back to sklearn if FAISS is unavailable."""
    train = _as_float32_contiguous(train)
    query = _as_float32_contiguous(query)
    k = min(max(1, int(k)), len(train))

    use_ip = metric.lower() in {"ip", "cos", "cosine"}
    if faiss is not None:
        d = train.shape[1]
        index = faiss.IndexFlatIP(d) if use_ip else faiss.IndexFlatL2(d)

        if gpu >= 0:
            try:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, gpu, index)
            except Exception as e:
                if require_faiss:
                    raise RuntimeError(
                        f"GPU FAISS requested (gpu={gpu}) but unavailable in this environment."
                    ) from e

        index.add(train)
        distances, indices = index.search(query, k)
        return distances, indices

    if require_faiss:
        raise RuntimeError(
            "FAISS requested but unavailable. Install faiss-cpu/faiss-gpu."
        )

    metric_name = "cosine" if use_ip else "euclidean"
    dmat = pairwise_distances(query, train, metric=metric_name)
    idx = np.argpartition(dmat, kth=k - 1, axis=1)[:, :k]
    d_sorted = np.take_along_axis(dmat, idx, axis=1)
    return d_sorted.astype(np.float32), idx.astype(np.int64)


def _summary_from_knn(distances: np.ndarray, quantile: float | None, average: bool) -> np.ndarray:
    if average:
        return np.mean(distances, axis=1)
    q = 0.5 if quantile is None else float(quantile)
    return np.quantile(distances, q, axis=1)


def disun_distance_features(
    train_embed: np.ndarray,
    query_embed: np.ndarray,
    train_spectra: np.ndarray,
    query_spectra: np.ndarray,
    n_neighbors: int = 50,
    quantile: float = 0.5,
    normalize_vectors: bool = True,
    normalize_by_train: bool = True,
    use_average: bool = False,
    gpu: int = -1,
    require_faiss: bool = False,
) -> dict[str, np.ndarray]:
    """Compute Dis_UN-like distance features in embedding and spectral spaces."""
    te = _as_float32_contiguous(train_embed)
    qe = _as_float32_contiguous(query_embed)
    ts = _as_float32_contiguous(train_spectra)
    qs = _as_float32_contiguous(query_spectra)

    if normalize_vectors:
        te = normalize_l2_rows(te)
        qe = normalize_l2_rows(qe)
        ts = normalize_l2_rows(ts)
        qs = normalize_l2_rows(qs)

    # L2 returns squared distances in FAISS IndexFlatL2
    d_emb_l2, _ = faiss_knn(te, qe, n_neighbors, metric="l2", gpu=gpu, require_faiss=require_faiss)
    d_sp_l2, _ = faiss_knn(ts, qs, n_neighbors, metric="l2", gpu=gpu, require_faiss=require_faiss)
    d_emb_l2 = np.sqrt(np.clip(d_emb_l2, 0.0, None))
    d_sp_l2 = np.sqrt(np.clip(d_sp_l2, 0.0, None))

    d_emb_ip, _ = faiss_knn(te, qe, n_neighbors, metric="ip", gpu=gpu, require_faiss=require_faiss)
    d_sp_ip, _ = faiss_knn(ts, qs, n_neighbors, metric="ip", gpu=gpu, require_faiss=require_faiss)

    emb_euc = _summary_from_knn(d_emb_l2, quantile=quantile, average=use_average)
    sp_euc = _summary_from_knn(d_sp_l2, quantile=quantile, average=use_average)
    emb_cos = 1.0 - _summary_from_knn(d_emb_ip, quantile=quantile, average=use_average)
    sp_cos = 1.0 - _summary_from_knn(d_sp_ip, quantile=quantile, average=use_average)
    emb_sp = np.arccos(np.clip(1.0 - emb_cos, -1.0, 1.0))
    sp_sp = np.arccos(np.clip(1.0 - sp_cos, -1.0, 1.0))

    feats: dict[str, np.ndarray] = {
        "qu50_dist_EmbLEuc": emb_euc.astype(np.float32),
        "qu50_dist_EmbLCos": emb_cos.astype(np.float32),
        "qu50_dist_EmbLSp": emb_sp.astype(np.float32),
        "qu50_dist_SpEuc": sp_euc.astype(np.float32),
        "qu50_dist_SpCos": sp_cos.astype(np.float32),
        "qu50_dist_SpSp": sp_sp.astype(np.float32),
    }

    if normalize_by_train:
        d_te_l2, _ = faiss_knn(te, te, n_neighbors, metric="l2", gpu=gpu, require_faiss=require_faiss)
        d_ts_l2, _ = faiss_knn(ts, ts, n_neighbors, metric="l2", gpu=gpu, require_faiss=require_faiss)
        d_te_l2 = np.sqrt(np.clip(d_te_l2, 0.0, None))
        d_ts_l2 = np.sqrt(np.clip(d_ts_l2, 0.0, None))
        d_te_ip, _ = faiss_knn(te, te, n_neighbors, metric="ip", gpu=gpu, require_faiss=require_faiss)
        d_ts_ip, _ = faiss_knn(ts, ts, n_neighbors, metric="ip", gpu=gpu, require_faiss=require_faiss)

        emb_euc_norm = max(float(np.mean(d_te_l2)), 1e-8)
        sp_euc_norm = max(float(np.mean(d_ts_l2)), 1e-8)
        emb_cos_norm = max(float(1.0 - np.mean(d_te_ip)), 1e-8)
        sp_cos_norm = max(float(1.0 - np.mean(d_ts_ip)), 1e-8)
        emb_sp_norm = max(float(np.arccos(np.clip(np.mean(d_te_ip), -1.0, 1.0))), 1e-8)
        sp_sp_norm = max(float(np.arccos(np.clip(np.mean(d_ts_ip), -1.0, 1.0))), 1e-8)

        feats["qu50_dist_EmbLEuc_nor"] = (emb_euc / emb_euc_norm).astype(np.float32)
        feats["qu50_dist_EmbLCos_nor"] = (emb_cos / emb_cos_norm).astype(np.float32)
        feats["qu50_dist_EmbLSp_nor"] = (emb_sp / emb_sp_norm).astype(np.float32)
        feats["qu50_dist_SpEuc_nor"] = (sp_euc / sp_euc_norm).astype(np.float32)
        feats["qu50_dist_SpCos_nor"] = (sp_cos / sp_cos_norm).astype(np.float32)
        feats["qu50_dist_SpSp_nor"] = (sp_sp / sp_sp_norm).astype(np.float32)

    return feats
