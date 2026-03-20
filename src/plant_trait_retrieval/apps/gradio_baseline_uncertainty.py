from __future__ import annotations

import tempfile
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.cross_decomposition import PLSRegression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge

from ..uncertainty.distance import DistanceUncertaintyConfig, DistanceUncertaintyEstimator


def _infer_wavelength_cols(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c.isdigit() and 350 <= int(c) <= 2600]
    return sorted(cols, key=lambda c: int(c))


def _fit_predict(
    train_df: pd.DataFrame,
    infer_df: pd.DataFrame,
    trait_col: str,
    model_kind: str,
    n_components: int,
    n_neighbors: int,
) -> pd.DataFrame:
    wl_cols = _infer_wavelength_cols(train_df)
    if not wl_cols:
        raise ValueError("No wavelength columns found in training file.")

    x_train = train_df[wl_cols].to_numpy(dtype=np.float32)
    x_infer = infer_df[wl_cols].to_numpy(dtype=np.float32)
    y_train = train_df[trait_col].to_numpy(dtype=np.float32)

    imp_x = SimpleImputer(strategy="median")
    x_train = imp_x.fit_transform(x_train)
    x_infer = imp_x.transform(x_infer)

    finite = np.isfinite(y_train)
    if finite.sum() < 20:
        raise ValueError("Not enough finite labels for selected trait.")

    x_fit = x_train[finite]
    y_fit = y_train[finite]

    if model_kind == "PLSR":
        n_comp = max(2, min(n_components, x_fit.shape[0] - 1, x_fit.shape[1] - 1))
        model = PLSRegression(n_components=n_comp)
    else:
        model = Ridge(alpha=1.0)

    model.fit(x_fit, y_fit)
    pred_train = model.predict(x_fit).reshape(-1)
    pred_infer = model.predict(x_infer).reshape(-1)

    residual = pred_train - y_fit
    unc = DistanceUncertaintyEstimator(DistanceUncertaintyConfig(n_neighbors=n_neighbors))
    unc.fit(train_features=x_fit, calib_features=x_fit, calib_residuals=residual[:, None])
    unc_infer = unc.predict_interval_scale(x_infer)

    out = infer_df.copy()
    out[f"pred_{trait_col}"] = pred_infer
    out["uncertainty"] = unc_infer
    return out


def _plot_maps(df: pd.DataFrame, trait_col: str):
    pred_col = f"pred_{trait_col}"
    if "lat" not in df.columns or "lon" not in df.columns:
        return None, None

    fig_pred = px.scatter_mapbox(
        df,
        lat="lat",
        lon="lon",
        color=pred_col,
        color_continuous_scale="Viridis",
        zoom=3,
        title=f"Prediction map ({trait_col})",
        height=500,
    )
    fig_pred.update_layout(mapbox_style="open-street-map")
    fig_unc = px.scatter_mapbox(
        df,
        lat="lat",
        lon="lon",
        color="uncertainty",
        size="uncertainty",
        color_continuous_scale="Magma",
        zoom=3,
        title="Uncertainty overlay map",
        height=500,
    )
    fig_unc.update_layout(mapbox_style="open-street-map")
    return fig_pred, fig_unc


def run_app(
    train_file,
    infer_file,
    trait_col,
    model_kind,
    n_components,
    n_neighbors,
):
    train_df = pd.read_csv(train_file.name)
    infer_df = pd.read_csv(infer_file.name)

    out = _fit_predict(
        train_df=train_df,
        infer_df=infer_df,
        trait_col=trait_col,
        model_kind=model_kind,
        n_components=int(n_components),
        n_neighbors=int(n_neighbors),
    )
    pred_map, unc_map = _plot_maps(out, trait_col)

    tmp = Path(tempfile.mkdtemp()) / "predictions_with_uncertainty.csv"
    out.to_csv(tmp, index=False)
    return out.head(500), pred_map, unc_map, str(tmp)


def launch() -> None:
    with gr.Blocks(title="Baseline + Uncertainty Mapper") as demo:
        gr.Markdown("# Plant Trait Prediction + Distance-Based Uncertainty")
        gr.Markdown("Upload labeled data to fit a baseline model, then upload inference data to map predictions and uncertainty.")

        with gr.Row():
            train_file = gr.File(label="Labeled CSV (must contain wavelengths + target trait)", file_types=[".csv"])
            infer_file = gr.File(label="Inference CSV (wavelengths, optional lat/lon)", file_types=[".csv"])

        with gr.Row():
            trait_col = gr.Textbox(label="Trait column name", value="trait_0")
            model_kind = gr.Dropdown(choices=["PLSR", "Ridge"], value="PLSR", label="Baseline model")
            n_components = gr.Slider(2, 80, value=25, step=1, label="PLSR components")
            n_neighbors = gr.Slider(5, 200, value=50, step=1, label="Uncertainty k-neighbors")

        run_btn = gr.Button("Run Prediction + Uncertainty")

        out_df = gr.Dataframe(label="Predictions (preview)")
        pred_map = gr.Plot(label="Prediction map")
        unc_map = gr.Plot(label="Uncertainty map")
        out_file = gr.File(label="Download full CSV")

        run_btn.click(
            run_app,
            inputs=[train_file, infer_file, trait_col, model_kind, n_components, n_neighbors],
            outputs=[out_df, pred_map, unc_map, out_file],
        )

    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    launch()
