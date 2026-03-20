from __future__ import annotations

import tempfile
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge

from ..ssl import SpectralMAE, encode_spectra, pretrain_mae


def _wl_cols(df: pd.DataFrame, wl_min: int, wl_max: int) -> list[str]:
    cols = [c for c in df.columns if c.isdigit() and wl_min <= int(c) <= wl_max]
    return sorted(cols, key=lambda c: int(c))


def _fit_predict(
    train_df: pd.DataFrame,
    infer_df: pd.DataFrame,
    unlabeled_df: pd.DataFrame | None,
    trait_col: str,
    model_name: str,
    wl_min: int,
    wl_max: int,
):
    cols = _wl_cols(train_df, wl_min, wl_max)
    if len(cols) < 30:
        raise ValueError("Selected wavelength range is too narrow or not present in input files.")

    x_train = train_df[cols].to_numpy(dtype=np.float32)
    x_infer = infer_df[cols].to_numpy(dtype=np.float32)
    y_train = train_df[trait_col].to_numpy(dtype=np.float32)

    imp = SimpleImputer(strategy="median")
    x_train = imp.fit_transform(x_train)
    x_infer = imp.transform(x_infer)

    finite = np.isfinite(y_train)
    x_fit = x_train[finite]
    y_fit = y_train[finite]

    if model_name == "PLSR":
        n_comp = max(2, min(25, x_fit.shape[0] - 1, x_fit.shape[1] - 1))
        model = PLSRegression(n_components=n_comp)
        model.fit(x_fit, y_fit)
        pred = model.predict(x_infer).reshape(-1)

    elif model_name == "Ridge (Supervised)":
        model = Ridge(alpha=1.0)
        model.fit(x_fit, y_fit)
        pred = model.predict(x_infer)

    elif model_name == "MAE + Linear Probe":
        pool = x_fit
        if unlabeled_df is not None:
            ul_cols = _wl_cols(unlabeled_df, wl_min, wl_max)
            if len(ul_cols) == len(cols):
                ul_x = unlabeled_df[ul_cols].to_numpy(dtype=np.float32)
                pool = imp.transform(ul_x)

        mae = SpectralMAE(input_length=len(cols), emb_dim=256, mask_ratio=0.4)
        pretrain_mae(mae, unlabeled_spectra=pool, epochs=20, batch_size=256, lr=1e-3, device="cpu")
        z_fit = encode_spectra(mae.encoder, x_fit, device="cpu")
        z_infer = encode_spectra(mae.encoder, x_infer, device="cpu")
        model = Ridge(alpha=1.0)
        model.fit(z_fit, y_fit)
        pred = model.predict(z_infer)

    else:
        raise ValueError(f"Unknown model: {model_name}")

    out = infer_df.copy()
    out[f"pred_{trait_col}"] = pred.astype(np.float32)
    return out


def run(train_file, infer_file, unlabeled_file, trait_col, model_name, wl_min, wl_max):
    train_df = pd.read_csv(train_file.name)
    infer_df = pd.read_csv(infer_file.name)
    unlabeled_df = pd.read_csv(unlabeled_file.name) if unlabeled_file is not None else None

    out = _fit_predict(
        train_df=train_df,
        infer_df=infer_df,
        unlabeled_df=unlabeled_df,
        trait_col=trait_col,
        model_name=model_name,
        wl_min=int(wl_min),
        wl_max=int(wl_max),
    )

    tmp = Path(tempfile.mkdtemp()) / "model_zoo_predictions.csv"
    out.to_csv(tmp, index=False)
    return out.head(500), str(tmp)


def launch() -> None:
    with gr.Blocks(title="Model Zoo - Sensor-Aware") as demo:
        gr.Markdown("# Flexible Model Zoo for Hyperspectral Trait Prediction")
        gr.Markdown("Choose model type, explicit sensor wavelength range, and run predictions.")

        with gr.Row():
            train_file = gr.File(label="Labeled CSV", file_types=[".csv"])
            infer_file = gr.File(label="Inference CSV", file_types=[".csv"])
            unlabeled_file = gr.File(label="Optional unlabeled CSV (for MAE)", file_types=[".csv"])

        with gr.Row():
            trait_col = gr.Textbox(label="Trait column", value="trait_0")
            model_name = gr.Dropdown(
                choices=["PLSR", "Ridge (Supervised)", "MAE + Linear Probe"],
                value="MAE + Linear Probe",
                label="Model",
            )

        with gr.Row():
            wl_min = gr.Number(value=400, label="Sensor wavelength min (nm)")
            wl_max = gr.Number(value=2450, label="Sensor wavelength max (nm)")

        run_btn = gr.Button("Run Inference")
        out_df = gr.Dataframe(label="Predictions (preview)")
        out_file = gr.File(label="Download predictions CSV")

        run_btn.click(
            run,
            inputs=[train_file, infer_file, unlabeled_file, trait_col, model_name, wl_min, wl_max],
            outputs=[out_df, out_file],
        )

    demo.launch(server_name="0.0.0.0", server_port=7861)


if __name__ == "__main__":
    launch()
