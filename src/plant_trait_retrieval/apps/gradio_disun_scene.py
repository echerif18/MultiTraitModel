# from __future__ import annotations

# import tempfile
# from pathlib import Path

# import gradio as gr
# import matplotlib as mpl
# import numpy as np
# import pandas as pd
# import folium
# from branca.element import Element, MacroElement, Template

# from omegaconf import OmegaConf

# from ..experiments.uncertainty_scene_infer import run_inference


# DEFAULT_TRAIT_CHOICES = [
#     "Anth_area_ug_cm2",
#     "Boron_area_mg_cm2",
#     "C_area_mg_cm2",
#     "Ca_area_mg_cm2",
#     "Car_area_ug_cm2",
#     "Cellulose_mg_cm2",
#     "Chl_area_ug_cm2",
#     "Cu_area_mg_cm2",
#     "EWT_mg_cm2",
#     "Fiber_mg_cm2",
#     "LAI_m2_m2",
#     "LMA_g_m2",
#     "Lignin_mg_cm2",
#     "Mg_area_mg_cm2",
#     "Mn_area_mg_cm2",
#     "NSC_mg_cm2",
#     "N_area_mg_cm2",
#     "P_area_mg_cm2",
#     "Potassium_area_mg_cm2",
#     "S_area_mg_cm2",
# ]


# # ---------------------------------------------------------------------------
# # Basic helpers
# # ---------------------------------------------------------------------------

# def _resolve_pred_col(df: pd.DataFrame, trait_name: str) -> str:
#     pred_cols = [c for c in df.columns if c.startswith("pred_")]
#     if not pred_cols:
#         raise ValueError("No prediction columns found in output.")
#     if trait_name:
#         if trait_name in pred_cols:
#             return trait_name
#         wanted = f"pred_{trait_name}"
#         if wanted in pred_cols:
#             return wanted
#     return pred_cols[0]


# def _grid_from_df(df: pd.DataFrame, col: str) -> np.ndarray:
#     h = int(df["row"].max()) + 1
#     w = int(df["col"].max()) + 1
#     grid = np.full((h, w), np.nan, dtype=np.float32)
#     grid[df["row"].to_numpy(int), df["col"].to_numpy(int)] = df[col].to_numpy(np.float32)
#     return grid


# PRED_COLORBAR_THRESHOLD = 0.05   # prediction: q05-q95
# UNC_THRE = 0.90                 # uncertainty: vmin=q10, vmax=q90


# def _pred_norm_bounds(arr: np.ndarray, thre: float = PRED_COLORBAR_THRESHOLD) -> tuple[float, float]:
#     """
#     Prediction stretch: vmin = q(thre), vmax = q(1-thre).
#     Low values map to the pale end of the colormap, high values to the dark end.
#     """
#     finite = arr[np.isfinite(arr)]
#     if finite.size == 0:
#         return 0.0, 1.0
#     vmin = float(np.nanpercentile(finite, thre * 100))
#     vmax = float(np.nanpercentile(finite, (1 - thre) * 100))
#     return vmin, vmax if vmax > vmin else vmin + 1e-6


# def _unc_norm_bounds(arr: np.ndarray, thre: float = UNC_THRE) -> tuple[float, float]:
#     """
#     Uncertainty stretch exactly as original project style:
#       vmax = quantile(thre)
#       vmin = quantile(1-thre)
#     Low uncertainty maps to the pale end, high uncertainty to the dark end.
#     """
#     finite = arr[np.isfinite(arr)]
#     if finite.size == 0:
#         return 0.0, 1.0
#     vmin = float(np.nanpercentile(finite, (1 - thre) * 100))
#     vmax = float(np.nanpercentile(finite, thre * 100))
#     if vmax <= vmin:
#         vmax = vmin + 1e-6
#     return vmin, vmax


# def _rgba_from_grid(
#     grid: np.ndarray,
#     cmap_name: str,
#     is_uncertainty: bool = False,
#     thre: float | None = None,
# ) -> tuple[np.ndarray, float, float]:
#     """
#     Return (RGBA uint8 H×W×4, vmin, vmax).

#     Prediction: q05-q95.
#     Uncertainty: vmin=q(1-thre), vmax=q(thre), with default thre=0.98.
#     """
#     if is_uncertainty:
#         vmin, vmax = _unc_norm_bounds(grid, UNC_THRE if thre is None else thre)
#     else:
#         vmin, vmax = _pred_norm_bounds(grid, PRED_COLORBAR_THRESHOLD if thre is None else thre)

#     norm = np.clip((grid - vmin) / (vmax - vmin), 0.0, 1.0)
#     cmap = mpl.colormaps.get_cmap(cmap_name)
#     rgba = (cmap(norm) * 255.0).astype(np.uint8)
#     rgba[..., 3] = np.where(np.isfinite(grid), 210, 0).astype(np.uint8)
#     return rgba, vmin, vmax


# def _fmt(v: float) -> str:
#     if v == 0.0:
#         return "0"
#     return f"{v:.2e}" if (abs(v) >= 10_000 or abs(v) < 0.001) else f"{v:.4g}"


# # ---------------------------------------------------------------------------
# # Projection: warp grid to WGS-84 using rasterio
# # ---------------------------------------------------------------------------

# def _warp_grid_to_wgs84(
#     grid: np.ndarray,
#     src_transform,
#     src_crs_wkt: str,
# ) -> tuple[np.ndarray, list]:
#     """
#     Reproject a 2-D float32 grid to EPSG:4326.

#     src_crs_wkt must be a WKT string (not an EPSG integer/CRS object) so that
#     we never touch the PROJ EPSG registry, which may be broken/outdated.

#     Returns
#     -------
#     warped_grid : np.ndarray  (float32, NaN for nodata)
#     bounds      : [[south, west], [north, east]]  in WGS-84 degrees
#     """
#     import rasterio
#     from rasterio.crs import CRS
#     from rasterio.warp import reproject, Resampling, calculate_default_transform
#     from rasterio.env import Env

#     h, w = grid.shape

#     with Env(GTIFF_SRS_SOURCE="GEOKEYS", PROJ_NETWORK="OFF"):
#         src_crs = CRS.from_wkt(src_crs_wkt)
#         dst_crs = CRS.from_wkt(
#             'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],'
#             'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433],'
#             'AUTHORITY["EPSG","4326"]]'
#         )

#         left   = src_transform.c
#         top    = src_transform.f
#         right  = left + w * src_transform.a
#         bottom = top  + h * src_transform.e

#         dst_transform, dst_w, dst_h = calculate_default_transform(
#             src_crs, dst_crs, w, h,
#             left=left, bottom=bottom, right=right, top=top,
#         )

#         NODATA = -9999.0
#         src_data = np.where(np.isnan(grid), NODATA, grid).astype(np.float32)
#         dst_data = np.full((dst_h, dst_w), NODATA, dtype=np.float32)

#         reproject(
#             source=src_data,
#             destination=dst_data,
#             src_transform=src_transform,
#             src_crs=src_crs,
#             dst_transform=dst_transform,
#             dst_crs=dst_crs,
#             src_nodata=NODATA,
#             dst_nodata=NODATA,
#             resampling=Resampling.bilinear,
#         )

#     warped = np.where(dst_data == NODATA, np.nan, dst_data)

#     west  = dst_transform.c
#     north = dst_transform.f
#     east  = west  + dst_w * dst_transform.a
#     south = north + dst_h * dst_transform.e

#     bounds = [[float(south), float(west)], [float(north), float(east)]]
#     return warped, bounds


# def _resolve_grids_and_bounds(
#     df: pd.DataFrame,
#     pred_col: str,
#     src_crs_override: str | None,
#     scene_tif_path: str | None = None,
#     output_pred_tif_path: str | None = None,
# ) -> tuple[np.ndarray, np.ndarray, list, bool]:
#     """
#     Return (pred_grid_wgs84, unc_grid_wgs84, bounds, has_geo).

#     All CRS operations use WKT strings (never EPSG integer lookups) so that
#     a broken/outdated PROJ database does not cause failures.

#     Priority:
#       1. x/y columns in df  — projected pixel coords, use tif WKT for CRS
#       2. scene_tif_path     — input tif has definitive CRS + geotransform
#       3. output_pred_tif_path / pred_tif_path column — secondary tif fallbacks
#       4. lat/lon columns    — WGS-84 direct or projected with src_crs_override
#       5. Pixel-space fallback
#     """
#     import rasterio
#     from rasterio.crs import CRS
#     from rasterio.env import Env
#     from rasterio.transform import Affine
#     from pyproj import CRS as pCRS, Transformer

#     WGS84_WKT = (
#         'GEOGCS["WGS 84",DATUM["WGS_1984",' +
#         'SPHEROID["WGS 84",6378137,298.257223563]],' +
#         'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433],' +
#         'AUTHORITY["EPSG","4326"]]'
#     )

#     pred_grid = _grid_from_df(df, pred_col)
#     # Resolve per-trait uncertainty column: prefer un_<trait> matching pred_col,
#     # fall back to uncertainty_mean (aggregate across all traits).
#     _trait_name = pred_col.replace("pred_", "", 1)
#     _unc_col_candidate = f"un_{_trait_name}"
#     _unc_col = _unc_col_candidate if _unc_col_candidate in df.columns else "uncertainty_mean"
#     unc_grid  = _grid_from_df(df, _unc_col)
#     print(f"[uncertainty] using column: {_unc_col}")
#     h, w = pred_grid.shape

#     def _wkt_from_tif(tif_path: str) -> tuple[str, object]:
#         """Return (crs_wkt, affine_transform) reading CRS from GeoTIFF keys,
#            bypassing the EPSG registry entirely."""
#         with Env(GTIFF_SRS_SOURCE="GEOKEYS", PROJ_NETWORK="OFF"):
#             with rasterio.open(tif_path) as src:
#                 transform = src.transform
#                 if src_crs_override and src_crs_override.strip():
#                     # User-supplied CRS as WKT or EPSG string
#                     wkt = pCRS.from_user_input(src_crs_override.strip()).to_wkt()
#                 elif src.crs is not None:
#                     wkt = src.crs.to_wkt()
#                 else:
#                     raise ValueError(f"No CRS in tif and no override: {tif_path}")
#         return wkt, transform

#     def _try_tif(tif_path: str, label: str):
#         try:
#             wkt, src_transform = _wkt_from_tif(tif_path)
#             pred_w, bounds = _warp_grid_to_wgs84(pred_grid, src_transform, wkt)
#             unc_w,  _      = _warp_grid_to_wgs84(unc_grid,  src_transform, wkt)
#             print(f"[projection] OK via {label}: bounds={bounds}")
#             return pred_w, unc_w, bounds, True
#         except Exception as exc:
#             print(f"[projection] {label} failed: {exc}")
#             return None

#     # ---- 1. x/y columns (projected UTM/CRS coords, always written by inference) ----
#     # This is the most direct path: x=easting, y=northing per pixel.
#     # We derive the affine from these + row/col, get WKT from any available tif.
#     if "x" in df.columns and "y" in df.columns:
#         try:
#             x_s = pd.to_numeric(df["x"], errors="coerce")
#             y_s = pd.to_numeric(df["y"], errors="coerce")
#             valid = np.isfinite(x_s) & np.isfinite(y_s) &                     np.isfinite(df["row"]) & np.isfinite(df["col"])
#             if valid.sum() > 10:
#                 x_v = x_s[valid].to_numpy()
#                 y_v = y_s[valid].to_numpy()
#                 row_v = df.loc[valid, "row"].to_numpy(int)
#                 col_v = df.loc[valid, "col"].to_numpy(int)

#                 # Fit pixel spacing: x = x0 + col*dx,  y = y0 + row*dy (dy < 0)
#                 dx = np.polyfit(col_v, x_v, 1)[0]
#                 dy = np.polyfit(row_v, y_v, 1)[0]
#                 # top-left corner of pixel (0,0)
#                 x0 = float(np.median(x_v - col_v * dx))
#                 y0 = float(np.median(y_v - row_v * dy))
#                 src_transform = Affine(abs(dx), 0.0, x0,
#                                        0.0, -abs(dy), y0)

#                 # Get CRS WKT — try tif files in order; never use EPSG lookup
#                 wkt = None
#                 for tif_candidate in [scene_tif_path, output_pred_tif_path]:
#                     if tif_candidate and Path(tif_candidate).exists():
#                         try:
#                             wkt, _ = _wkt_from_tif(tif_candidate)
#                             break
#                         except Exception:
#                             pass

#                 # Also try pred_tif_path column
#                 if wkt is None and "pred_tif_path" in df.columns:
#                     v = df["pred_tif_path"].dropna().astype(str)
#                     if len(v) > 0 and Path(v.iloc[0]).exists():
#                         try:
#                             wkt, _ = _wkt_from_tif(v.iloc[0])
#                         except Exception:
#                             pass

#                 # If src_crs_override given, use it directly
#                 if src_crs_override and src_crs_override.strip():
#                     wkt = pCRS.from_user_input(src_crs_override.strip()).to_wkt()

#                 if wkt is None:
#                     raise ValueError("Could not obtain CRS WKT from any tif file")

#                 pred_w, bounds = _warp_grid_to_wgs84(pred_grid, src_transform, wkt)
#                 unc_w,  _      = _warp_grid_to_wgs84(unc_grid,  src_transform, wkt)
#                 print(f"[projection] OK via x/y columns: dx={dx:.2f}m dy={dy:.2f}m bounds={bounds}")
#                 return pred_w, unc_w, bounds, True
#         except Exception as exc:
#             print(f"[projection] x/y columns path failed: {exc}")

#     # ---- 2. Input scene tif -----------------------------------------------
#     if scene_tif_path and Path(scene_tif_path).exists():
#         result = _try_tif(scene_tif_path, "scene_tif (input)")
#         if result:
#             return result

#     # ---- 3. Output prediction tif / pred_tif_path column ------------------
#     if output_pred_tif_path and Path(output_pred_tif_path).exists():
#         result = _try_tif(output_pred_tif_path, "output_pred_tif")
#         if result:
#             return result

#     if "pred_tif_path" in df.columns:
#         vals = df["pred_tif_path"].dropna().astype(str)
#         if len(vals) > 0 and Path(vals.iloc[0]).exists():
#             result = _try_tif(vals.iloc[0], "pred_tif_path column")
#             if result:
#                 return result

#     # ---- 4. lat/lon columns (WGS-84 direct) --------------------------------
#     if "lat" in df.columns and "lon" in df.columns:
#         try:
#             lat_v = pd.to_numeric(df["lat"], errors="coerce").dropna().to_numpy()
#             lon_v = pd.to_numeric(df["lon"], errors="coerce").dropna().to_numpy()
#             lat_v = lat_v[np.isfinite(lat_v)]
#             lon_v = lon_v[np.isfinite(lon_v)]
#             if len(lat_v) > 4:
#                 bounds = [[float(lat_v.min()), float(lon_v.min())],
#                           [float(lat_v.max()), float(lon_v.max())]]
#                 print(f"[projection] OK via lat/lon WGS84 columns: bounds={bounds}")
#                 return pred_grid, unc_grid, bounds, True
#         except Exception as exc:
#             print(f"[projection] lat/lon path failed: {exc}")

#     # ---- 5. Pixel-space fallback -------------------------------------------
#     print("[projection] WARNING: no geo info found, using pixel-space fallback")
#     print(f"[projection] df columns: {list(df.columns)}")
#     bounds = [[0.0, 0.0], [float(h), float(w)]]
#     return pred_grid, unc_grid, bounds, False


# # ---------------------------------------------------------------------------
# # Colorbar legend  (values baked via f-string — Branca `this.attr` is unreliable)
# # ---------------------------------------------------------------------------

# def _add_colorbars(
#     m: folium.Map,
#     pred_vmin: float, pred_vmax: float,
#     unc_vmin: float,  unc_vmax: float,
#     pred_label: str = "Prediction",
# ) -> None:
#     template = f"""
#     {{% macro html(this, kwargs) %}}
#     <div style="position:fixed;bottom:40px;left:16px;z-index:9999;
#                 background:rgba(255,255,255,0.93);padding:8px 12px;
#                 border:1px solid #888;border-radius:7px;font-size:12px;
#                 font-family:monospace;box-shadow:2px 2px 6px rgba(0,0,0,0.2);
#                 min-width:220px;pointer-events:none;">
#       <div style="font-weight:700;margin-bottom:4px;white-space:nowrap;
#                   overflow:hidden;text-overflow:ellipsis;max-width:220px;">
#         {pred_label}
#       </div>
#       <div style="width:210px;height:14px;
#                   background:linear-gradient(to right,#f7fcf5,#74c476,#00441b);
#                   border-radius:3px;margin-bottom:3px;"></div>
#       <div style="display:flex;justify-content:space-between;color:#333;">
#         <span>{_fmt(pred_vmin)}</span>
#         <span>{_fmt(pred_vmax)}</span>
#       </div>
#     </div>
#     <div style="position:fixed;bottom:40px;right:16px;z-index:9999;
#                 background:rgba(255,255,255,0.93);padding:8px 12px;
#                 border:1px solid #888;border-radius:7px;font-size:12px;
#                 font-family:monospace;box-shadow:2px 2px 6px rgba(0,0,0,0.2);
#                 min-width:220px;pointer-events:none;">
#       <div style="font-weight:700;margin-bottom:4px;">Uncertainty</div>
#       <div style="width:210px;height:14px;
#                   background:linear-gradient(to right,#fff5eb,#fdae6b,#a63603);
#                   border-radius:3px;margin-bottom:3px;"></div>
#       <div style="display:flex;justify-content:space-between;color:#333;">
#         <span>{_fmt(unc_vmin)}</span>
#         <span>{_fmt(unc_vmax)}</span>
#       </div>
#     </div>
#     {{% endmacro %}}
#     """
#     el = MacroElement()
#     el._template = Template(template)
#     el.add_to(m)


# # ---------------------------------------------------------------------------
# # Swipe control — pure inline JS, no external CDN dependency
# #
# # WHY NOT SideBySideLayers / custom CDN:
# #   Gradio renders gr.HTML content inside an iframe. The iframe sandbox may
# #   block requests to external CDN URLs (jsdelivr etc.), so the leaflet-
# #   side-by-side plugin script simply never loads and the swipe bar never
# #   appears.  The solution is to embed all swipe logic inline so zero
# #   network requests are needed.
# # ---------------------------------------------------------------------------

# def _add_inline_swipe(m: folium.Map) -> None:
#     """
#     Inject a draggable vertical swipe bar that clips the RIGHT/TOP image overlay.
#     Works entirely inline — no external scripts required.

#     The map's overlayPane contains the two <img class='leaflet-image-layer'>
#     elements in DOM order (first-added = index 0 = left, last-added = index 1 = right).
#     We apply CSS clip-path on the right image to reveal it only past the bar.
#     """
#     map_var = m.get_name()

#     js = f"""
# <script>
# (function() {{
#   function initSwipe() {{
#     var map = window['{map_var}'];
#     if (!map) {{ setTimeout(initSwipe, 150); return; }}
#     var container = map.getContainer();
#     var pane = map.getPanes().overlayPane;
#     if (!container || !pane) {{ setTimeout(initSwipe, 150); return; }}

#     /* Wait until both image overlay <img> elements exist in the DOM */
#     var attempts = 0;
#     function waitForLayers() {{
#       var imgs = pane.querySelectorAll('img.leaflet-image-layer');
#       if (imgs.length < 2 && attempts++ < 30) {{
#         setTimeout(waitForLayers, 200);
#         return;
#       }}
#       buildBar();
#     }}

#     function buildBar() {{
#       /* ---- Divider bar ---- */
#       var bar = document.createElement('div');
#       bar.id = 'swipe-divider';
#       bar.style.cssText = [
#         'position:absolute', 'top:0', 'bottom:0', 'left:50%',
#         'width:5px', 'transform:translateX(-50%)',
#         'background:rgba(255,255,255,0.92)',
#         'box-shadow:0 0 10px rgba(0,0,0,0.75)',
#         'cursor:ew-resize', 'z-index:1000',
#         'pointer-events:auto'
#       ].join(';');

#       /* ---- Handle icon ---- */
#       var handle = document.createElement('div');
#       handle.innerHTML = '&#8596;';
#       handle.style.cssText = [
#         'position:absolute', 'top:50%', 'left:50%',
#         'transform:translate(-50%,-50%)',
#         'background:#fff', 'border:2px solid #444', 'border-radius:50%',
#         'width:34px', 'height:34px', 'line-height:30px', 'text-align:center',
#         'font-size:17px', 'font-weight:bold',
#         'box-shadow:0 2px 8px rgba(0,0,0,0.5)',
#         'user-select:none', 'pointer-events:none'
#       ].join(';');
#       bar.appendChild(handle);

#       /* Ensure the map container is the offset parent for absolute positioning */
#       if (getComputedStyle(container).position === 'static') {{
#         container.style.position = 'relative';
#       }}
#       container.appendChild(bar);

#       var pct = 50;   /* current divider position as % of container width */

#       /* clip the RIGHT overlay (index 1) to show only its left portion */
#       function applyClip(p) {{
#         pct = p;
#         var imgs = pane.querySelectorAll('img.leaflet-image-layer');
#         if (imgs.length < 2) return;
#         /* inset(top right bottom left) — clip everything LEFT of the bar */
#         var clip = 'inset(0 0 0 ' + p + '%)';
#         imgs[1].style.clipPath        = clip;
#         imgs[1].style.webkitClipPath  = clip;
#       }}

#       function pctFromEvent(e) {{
#         var r   = container.getBoundingClientRect();
#         var x   = e.touches ? e.touches[0].clientX : e.clientX;
#         return Math.max(0, Math.min(100, (x - r.left) / r.width * 100));
#       }}

#       /* Dragging */
#       var dragging = false;
#       bar.addEventListener('mousedown',  function(e) {{ dragging = true; e.preventDefault(); e.stopPropagation(); }});
#       bar.addEventListener('touchstart', function(e) {{ dragging = true; e.stopPropagation(); }}, {{passive:true}});
#       document.addEventListener('mouseup',   function()  {{ dragging = false; }});
#       document.addEventListener('touchend',  function()  {{ dragging = false; }});
#       document.addEventListener('mousemove', function(e) {{
#         if (!dragging) return;
#         var p = pctFromEvent(e);
#         bar.style.left = p + '%';
#         applyClip(p);
#       }});
#       document.addEventListener('touchmove', function(e) {{
#         if (!dragging) return;
#         var p = pctFromEvent(e);
#         bar.style.left = p + '%';
#         applyClip(p);
#       }}, {{passive:true}});

#       /* Re-apply clip after any map repaint (pan / zoom repositions <img> elements) */
#       map.on('moveend zoomend layeradd', function() {{
#         setTimeout(function() {{ applyClip(pct); }}, 60);
#       }});

#       applyClip(50);
#     }}

#     waitForLayers();
#   }}

#   if (document.readyState === 'loading') {{
#     document.addEventListener('DOMContentLoaded', initSwipe);
#   }} else {{
#     initSwipe();
#   }}
# }})();
# </script>
# """
#     m.get_root().html.add_child(Element(js))


# # ---------------------------------------------------------------------------
# # Map builder
# # ---------------------------------------------------------------------------

# def _build_folium_overlay_map(
#     df: pd.DataFrame,
#     trait_name: str,
#     show_prediction: bool,
#     show_uncertainty: bool,
#     enable_swipe: bool,
#     swipe_upper_layer: str,
#     src_crs_override: str | None,
#     scene_tif_path: str | None = None,
#     output_pred_tif_path: str | None = None,
# ) -> str:
#     if "row" not in df.columns or "col" not in df.columns:
#         return "<h3>No raster indices (`row`, `col`) in output — raster inference required.</h3>"

#     pred_col = _resolve_pred_col(df, trait_name=trait_name)

#     # Warp grids to WGS-84 using the best available geo reference
#     pred_grid, unc_grid, bounds, has_geo = _resolve_grids_and_bounds(
#         df, pred_col, src_crs_override,
#         scene_tif_path=scene_tif_path,
#         output_pred_tif_path=output_pred_tif_path,
#     )

#     # Prediction/uncertainty: normal q05–q95 stretch
#     pred_rgba, pred_vmin, pred_vmax = _rgba_from_grid(pred_grid, "Greens",  is_uncertainty=False)
#     unc_rgba,  unc_vmin,  unc_vmax  = _rgba_from_grid(unc_grid,  "Oranges", is_uncertainty=True)

#     center = [(bounds[0][0] + bounds[1][0]) / 2.0,
#               (bounds[0][1] + bounds[1][1]) / 2.0]
#     m = folium.Map(
#         location=center,
#         zoom_start=12 if has_geo else 1,
#         tiles="OpenStreetMap",
#         control_scale=True,
#     )

#     # -----------------------------------------------------------------------
#     # Image overlays
#     # mercator_project=False  ← intentional.
#     #
#     # mercator_project=True uses Folium's mercator_transform() which does a
#     # row-by-row interpolation. When the grid contains NaN rows (scan-line
#     # gaps common in HSI data), the interpolation smears those NaN rows into
#     # visible horizontal stripes across the overlay.
#     #
#     # Instead we use rasterio.warp.reproject() (in _resolve_grids_and_bounds)
#     # which correctly reprojects the data grid — including proper handling of
#     # nodata pixels — before we ever create the RGBA image.  The reprojected
#     # grid is already in geographic (WGS-84) coordinates, so the ImageOverlay
#     # can be placed directly without any further warping.
#     # -----------------------------------------------------------------------
#     swipe_ready = enable_swipe and show_prediction and show_uncertainty
#     upper = str(swipe_upper_layer).strip().lower()
#     right_is_pred = upper.startswith("pred")

#     # When swiping, the LEFT layer must be added first, RIGHT layer second
#     # (our JS clips imgs[1] = the second/right layer).
#     if swipe_ready:
#         if right_is_pred:
#             first_rgba,  first_name  = unc_rgba,  "Uncertainty"
#             second_rgba, second_name = pred_rgba, f"Prediction ({pred_col})"
#         else:
#             first_rgba,  first_name  = pred_rgba, f"Prediction ({pred_col})"
#             second_rgba, second_name = unc_rgba,  "Uncertainty"
#     else:
#         first_rgba,  first_name  = pred_rgba, f"Prediction ({pred_col})"
#         second_rgba, second_name = unc_rgba,  "Uncertainty"

#     ol_first = folium.raster_layers.ImageOverlay(
#         image=first_rgba,
#         bounds=bounds,
#         name=first_name,
#         opacity=0.85 if (show_prediction if not (swipe_ready and right_is_pred) else show_uncertainty) else 0.0,
#         mercator_project=False,
#         interactive=True,
#         cross_origin=False,
#         zindex=3,
#     )
#     ol_second = folium.raster_layers.ImageOverlay(
#         image=second_rgba,
#         bounds=bounds,
#         name=second_name,
#         opacity=0.85 if (show_uncertainty if not (swipe_ready and right_is_pred) else show_prediction) else 0.0,
#         mercator_project=False,
#         interactive=True,
#         cross_origin=False,
#         zindex=4,
#     )
#     ol_first.add_to(m)
#     ol_second.add_to(m)

#     if swipe_ready:
#         _add_inline_swipe(m)
#     else:
#         folium.LayerControl(collapsed=False).add_to(m)

#     m.fit_bounds(bounds)
#     _add_colorbars(
#         m,
#         pred_vmin=pred_vmin, pred_vmax=pred_vmax,
#         unc_vmin=unc_vmin,   unc_vmax=unc_vmax,
#         pred_label=f"Prediction ({pred_col})",
#     )
#     return m._repr_html_()


# # ---------------------------------------------------------------------------
# # Artifact resolution
# # ---------------------------------------------------------------------------

# def _resolve_uncertainty_artifacts(cfg):
#     q_path = Path(str(cfg.inference.quantile_models_path))
#     d_path = Path(str(cfg.inference.distance_reference_npz))
#     if q_path.exists() and d_path.exists():
#         return str(q_path), str(d_path)

#     root = (
#         q_path.parents[1]
#         if len(q_path.parents) >= 2
#         else Path("results/experiments/study2_uncertainty_1522")
#     )
#     t1 = root / "target_1"
#     if (t1 / "quantile_models.pkl").exists() and (t1 / "distance_reference.npz").exists():
#         return str(t1 / "quantile_models.pkl"), str(t1 / "distance_reference.npz")

#     for c in sorted(root.glob("target_*")):
#         q, d = c / "quantile_models.pkl", c / "distance_reference.npz"
#         if q.exists() and d.exists():
#             return str(q), str(d)

#     raise FileNotFoundError(
#         f"Could not find uncertainty artifacts under {root}. "
#         "Run uncertainty stage=distance first."
#     )


# # ---------------------------------------------------------------------------
# # Gradio callbacks
# # ---------------------------------------------------------------------------

# def run_app(
#     scene_csv_file,
#     scene_tif_file,
#     sensor_bands_file,
#     src_crs_override,
#     trait_name,
#     show_prediction,
#     show_uncertainty,
#     enable_swipe,
#     swipe_upper_layer,
# ):
#     tmp_out = Path(tempfile.mkdtemp()) / "scene_predictions_uncertainty.csv"
#     cfg = OmegaConf.load(
#         Path(__file__).resolve().parents[3]
#         / "configs" / "experiments" / "scene_infer_uncertainty.yaml"
#     )

#     if scene_tif_file is not None and sensor_bands_file is not None:
#         cfg.inference.input_type = "tif"
#         cfg.inference.scene_tif = scene_tif_file.name
#         cfg.inference.sensor_bands_csv = sensor_bands_file.name
#     elif scene_csv_file is not None:
#         cfg.inference.input_type = "csv"
#         cfg.inference.scene_csv = scene_csv_file.name
#     else:
#         raise ValueError("Provide either Scene CSV or (Scene TIFF + Sensor Bands CSV).")

#     cfg.inference.output_path = str(tmp_out)
#     cfg.inference.output_pred_tif = str(
#         tmp_out.with_name(tmp_out.stem + "_predictions_20traits.tif")
#     )
#     cfg.inference.output_unc_tif = str(
#         tmp_out.with_name(tmp_out.stem + "_uncertainty_20traits.tif")
#     )

#     q, d = _resolve_uncertainty_artifacts(cfg)
#     cfg.inference.quantile_models_path = q
#     cfg.inference.distance_reference_npz = d
#     run_inference(cfg)

#     df = pd.read_csv(tmp_out)
#     pred_cols = [c for c in df.columns if c.startswith("pred_")]
#     trait_choices = [c.replace("pred_", "", 1) for c in pred_cols]
#     selected_trait = (
#         str(trait_name) if str(trait_name) in trait_choices
#         else (trait_choices[0] if trait_choices else "")
#     )

#     crs = str(src_crs_override).strip() or None

#     # Resolve tif paths for projection
#     _scene_tif = str(scene_tif_file.name) if scene_tif_file is not None else None
#     _output_pred_tif = str(tmp_out.with_name(tmp_out.stem + "_predictions_20traits.tif"))

#     overlay_html = _build_folium_overlay_map(
#         df,
#         trait_name=selected_trait,
#         show_prediction=bool(show_prediction),
#         show_uncertainty=bool(show_uncertainty),
#         enable_swipe=bool(enable_swipe),
#         swipe_upper_layer=str(swipe_upper_layer),
#         src_crs_override=crs,
#         scene_tif_path=_scene_tif,
#         output_pred_tif_path=_output_pred_tif,
#     )

#     pred_tif = Path(str(cfg.inference.output_pred_tif))
#     unc_tif  = Path(str(cfg.inference.output_unc_tif))
#     # Store csv path + scene_tif path together so viz-only can reproject correctly
#     cached = str(tmp_out) + "|" + (_scene_tif or "") + "|" + _output_pred_tif
#     return (
#         overlay_html,
#         gr.update(choices=trait_choices or DEFAULT_TRAIT_CHOICES, value=selected_trait or None),
#         str(pred_tif) if pred_tif.exists() else None,
#         str(unc_tif)  if unc_tif.exists()  else None,
#         cached,
#     )


# def update_visualization_only(
#     cached_out_csv,
#     src_crs_override,
#     trait_name,
#     show_prediction,
#     show_uncertainty,
#     enable_swipe,
#     swipe_upper_layer,
# ):
#     if not cached_out_csv:
#         return "<h3>Run inference first, then tweak visualization controls.</h3>"

#     # Unpack cached string: "csv_path|scene_tif_path|output_pred_tif_path"
#     parts = str(cached_out_csv).split("|")
#     csv_path          = parts[0] if len(parts) > 0 else ""
#     scene_tif_path    = parts[1] if len(parts) > 1 and parts[1] else None
#     output_pred_tif   = parts[2] if len(parts) > 2 and parts[2] else None

#     out_path = Path(csv_path)
#     if not out_path.exists():
#         return f"<h3>Cached inference output not found: {out_path}</h3>"
#     df = pd.read_csv(out_path)
#     crs = str(src_crs_override).strip() or None
#     return _build_folium_overlay_map(
#         df,
#         trait_name=str(trait_name),
#         show_prediction=bool(show_prediction),
#         show_uncertainty=bool(show_uncertainty),
#         enable_swipe=bool(enable_swipe),
#         swipe_upper_layer=str(swipe_upper_layer),
#         src_crs_override=crs,
#         scene_tif_path=scene_tif_path,
#         output_pred_tif_path=output_pred_tif,
#     )


# # ---------------------------------------------------------------------------
# # Gradio UI
# # ---------------------------------------------------------------------------

# def launch() -> None:
#     with gr.Blocks(title="Dis_UN Scene Inference") as demo:
#         gr.Markdown("# HSI Scene Prediction + Dis_UN Uncertainty")
#         gr.Markdown(
#             "Infer multi-trait predictions and Dis_UN uncertainty from CSV spectra "
#             "or raw HSI TIFF + sensor band list."
#         )

#         with gr.Row():
#             scene_csv_file = gr.File(
#                 label="Scene CSV (1522 spectra + optional lat/lon)",
#                 file_types=[".csv"],
#             )
#         with gr.Row():
#             scene_tif_file = gr.File(
#                 label="Scene HSI TIFF (raw image)", file_types=[".tif", ".tiff"]
#             )
#             sensor_bands_file = gr.File(
#                 label="Sensor Bands CSV (column: band/wavelength)", file_types=[".csv"]
#             )
#         with gr.Row():
#             src_crs_override = gr.Textbox(
#                 label="Source CRS override (e.g. EPSG:32610) — leave blank to use TIFF CRS metadata",
#                 value="",
#                 placeholder="EPSG:32610",
#             )
#         with gr.Row():
#             trait_name = gr.Dropdown(
#                 label="Trait for visualization",
#                 choices=DEFAULT_TRAIT_CHOICES,
#                 value="EWT_mg_cm2",
#                 allow_custom_value=False,
#             )
#         with gr.Row():
#             show_prediction = gr.Checkbox(label="Show prediction layer", value=True)
#             show_uncertainty = gr.Checkbox(label="Show uncertainty layer", value=True)
#             enable_swipe = gr.Checkbox(label="Enable swipe comparison", value=True)
#             swipe_upper_layer = gr.Radio(
#                 label="Right panel (swipe)",
#                 choices=["Uncertainty", "Prediction"],
#                 value="Uncertainty",
#             )

#         run_btn = gr.Button("▶  Run Scene Inference", variant="primary")
#         viz_btn = gr.Button("↻  Update Visualization (no re-run)")

#         overlay_map  = gr.HTML(label="Map viewer")
#         pred_tif_file = gr.File(label="Download prediction GeoTIFF (20 traits)")
#         unc_tif_file  = gr.File(label="Download uncertainty GeoTIFF (20 traits)")
#         cached_out_csv = gr.State(value="")

#         run_btn.click(
#             run_app,
#             inputs=[
#                 scene_csv_file, scene_tif_file, sensor_bands_file,
#                 src_crs_override, trait_name,
#                 show_prediction, show_uncertainty, enable_swipe, swipe_upper_layer,
#             ],
#             outputs=[overlay_map, trait_name, pred_tif_file, unc_tif_file, cached_out_csv],
#         )
#         viz_btn.click(
#             update_visualization_only,
#             inputs=[
#                 cached_out_csv, src_crs_override, trait_name,
#                 show_prediction, show_uncertainty, enable_swipe, swipe_upper_layer,
#             ],
#             outputs=[overlay_map],
#         )

#     demo.launch(server_name="0.0.0.0", server_port=7862)


# if __name__ == "__main__":
#     launch()


# from __future__ import annotations

# import tempfile
# from pathlib import Path

# import gradio as gr
# import matplotlib as mpl
# import numpy as np
# import pandas as pd
# import folium
# from branca.element import Element, MacroElement, Template

# from omegaconf import OmegaConf

# from ..experiments.uncertainty_scene_infer import run_inference


# DEFAULT_TRAIT_CHOICES = [
#     "Anth_area_ug_cm2",
#     "Boron_area_mg_cm2",
#     "C_area_mg_cm2",
#     "Ca_area_mg_cm2",
#     "Car_area_ug_cm2",
#     "Cellulose_mg_cm2",
#     "Chl_area_ug_cm2",
#     "Cu_area_mg_cm2",
#     "EWT_mg_cm2",
#     "Fiber_mg_cm2",
#     "LAI_m2_m2",
#     "LMA_g_m2",
#     "Lignin_mg_cm2",
#     "Mg_area_mg_cm2",
#     "Mn_area_mg_cm2",
#     "NSC_mg_cm2",
#     "N_area_mg_cm2",
#     "P_area_mg_cm2",
#     "Potassium_area_mg_cm2",
#     "S_area_mg_cm2",
# ]


# # ---------------------------------------------------------------------------
# # Basic helpers
# # ---------------------------------------------------------------------------

# def _resolve_pred_col(df: pd.DataFrame, trait_name: str) -> str:
#     pred_cols = [c for c in df.columns if c.startswith("pred_")]
#     if not pred_cols:
#         raise ValueError("No prediction columns found in output.")
#     if trait_name:
#         if trait_name in pred_cols:
#             return trait_name
#         wanted = f"pred_{trait_name}"
#         if wanted in pred_cols:
#             return wanted
#     return pred_cols[0]


# def _grid_from_df(df: pd.DataFrame, col: str) -> np.ndarray:
#     h = int(df["row"].max()) + 1
#     w = int(df["col"].max()) + 1
#     grid = np.full((h, w), np.nan, dtype=np.float32)
#     grid[df["row"].to_numpy(int), df["col"].to_numpy(int)] = df[col].to_numpy(np.float32)
#     return grid


# # ---------------------------------------------------------------------------
# # Colormap stretch
# # ---------------------------------------------------------------------------

# PRED_THRESHOLD = 0.05   # prediction: vmin=q05, vmax=q95
# UNC_THRESHOLD  = 0.90   # uncertainty: vmin=q10, vmax=q90  (matches original style)


# def _pred_norm_bounds(arr: np.ndarray) -> tuple[float, float]:
#     """q05–q95 stretch for prediction."""
#     finite = arr[np.isfinite(arr)]
#     if finite.size == 0:
#         return 0.0, 1.0
#     vmin = float(np.nanpercentile(finite, PRED_THRESHOLD * 100))
#     vmax = float(np.nanpercentile(finite, (1 - PRED_THRESHOLD) * 100))
#     return vmin, vmax if vmax > vmin else vmin + 1e-6


# def _unc_norm_bounds(arr: np.ndarray) -> tuple[float, float]:
#     """
#     Uncertainty stretch matching original project formula:
#         maxv = quantile(thre)        ← LOW quantile  → colormap bright end
#         minv = quantile(1 - thre)    ← HIGH quantile → colormap dark end
#     High-uncertainty pixels dominate visually (dark/saturated).
#     """
#     finite = arr[np.isfinite(arr)]
#     if finite.size == 0:
#         return 0.0, 1.0
#     vmin = float(np.nanpercentile(finite, (1 - UNC_THRESHOLD) * 100))
#     vmax = float(np.nanpercentile(finite, UNC_THRESHOLD * 100))
#     if vmax <= vmin:
#         vmax = vmin + 1e-6
#     return vmin, vmax


# def _rgba_from_grid(
#     grid: np.ndarray,
#     cmap_name: str,
#     is_uncertainty: bool = False,
# ) -> tuple[np.ndarray, float, float]:
#     """Return (RGBA uint8 H×W×4, vmin, vmax)."""
#     vmin, vmax = _unc_norm_bounds(grid) if is_uncertainty else _pred_norm_bounds(grid)
#     norm = np.clip((grid - vmin) / (vmax - vmin), 0.0, 1.0)
#     cmap = mpl.colormaps.get_cmap(cmap_name)
#     rgba = (cmap(norm) * 255.0).astype(np.uint8)
#     rgba[..., 3] = np.where(np.isfinite(grid), 210, 0).astype(np.uint8)
#     return rgba, vmin, vmax


# def _fmt(v: float) -> str:
#     if v == 0.0:
#         return "0"
#     return f"{v:.2e}" if (abs(v) >= 10_000 or abs(v) < 0.001) else f"{v:.4g}"


# # ---------------------------------------------------------------------------
# # Projection: warp grid to WGS-84 using rasterio
# # Uses WKT strings throughout — never EPSG integer lookups — so a
# # broken/outdated PROJ database (conda env mismatch) does not crash.
# # ---------------------------------------------------------------------------

# def _warp_grid_to_wgs84(
#     grid: np.ndarray,
#     src_transform,
#     src_crs_wkt: str,
# ) -> tuple[np.ndarray, list]:
#     """
#     Reproject a 2-D float32 grid from src_crs_wkt to WGS-84.

#     Returns
#     -------
#     warped_grid : np.ndarray  (float32, NaN for nodata)
#     bounds      : [[south, west], [north, east]]  in degrees
#     """
#     import rasterio
#     from rasterio.crs import CRS
#     from rasterio.warp import reproject, Resampling, calculate_default_transform
#     from rasterio.env import Env

#     h, w = grid.shape

#     WGS84_WKT = (
#         'GEOGCS["WGS 84",DATUM["WGS_1984",'
#         'SPHEROID["WGS 84",6378137,298.257223563]],'
#         'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433],'
#         'AUTHORITY["EPSG","4326"]]'
#     )

#     with Env(GTIFF_SRS_SOURCE="GEOKEYS", PROJ_NETWORK="OFF"):
#         src_crs = CRS.from_wkt(src_crs_wkt)
#         dst_crs = CRS.from_wkt(WGS84_WKT)

#         left   = src_transform.c
#         top    = src_transform.f
#         right  = left + w * src_transform.a
#         bottom = top  + h * src_transform.e   # e is negative

#         dst_transform, dst_w, dst_h = calculate_default_transform(
#             src_crs, dst_crs, w, h,
#             left=left, bottom=bottom, right=right, top=top,
#         )

#         NODATA = -9999.0
#         src_data = np.where(np.isnan(grid), NODATA, grid).astype(np.float32)
#         dst_data = np.full((dst_h, dst_w), NODATA, dtype=np.float32)

#         reproject(
#             source=src_data, destination=dst_data,
#             src_transform=src_transform, src_crs=src_crs,
#             dst_transform=dst_transform, dst_crs=dst_crs,
#             src_nodata=NODATA, dst_nodata=NODATA,
#             resampling=Resampling.bilinear,
#         )

#     warped = np.where(dst_data == NODATA, np.nan, dst_data)
#     west  = dst_transform.c
#     north = dst_transform.f
#     east  = west  + dst_w * dst_transform.a
#     south = north + dst_h * dst_transform.e

#     return warped, [[float(south), float(west)], [float(north), float(east)]]


# def _resolve_grids_and_bounds(
#     df: pd.DataFrame,
#     pred_col: str,
#     src_crs_override: str | None,
#     scene_tif_path: str | None = None,
#     output_pred_tif_path: str | None = None,
# ) -> tuple[np.ndarray, np.ndarray, list, bool]:
#     """
#     Return (pred_grid_wgs84, unc_grid_wgs84, bounds, has_geo).

#     Priority:
#       1. x/y columns  — projected pixel coords always written by inference.
#          Affine derived via polyfit; WKT read from any available tif.
#       2. scene_tif_path (INPUT tif) — definitive CRS + geotransform.
#       3. output_pred_tif_path / pred_tif_path column — secondary fallbacks.
#       4. lat/lon columns — assumed WGS-84.
#       5. Pixel-space fallback.
#     """
#     import rasterio
#     from rasterio.crs import CRS
#     from rasterio.env import Env
#     from rasterio.transform import Affine

#     pred_grid = _grid_from_df(df, pred_col)

#     # Per-trait uncertainty column: prefer un_<trait>, fall back to uncertainty_mean
#     _trait = pred_col.replace("pred_", "", 1)
#     _unc_col = f"un_{_trait}" if f"un_{_trait}" in df.columns else "uncertainty_mean"
#     unc_grid = _grid_from_df(df, _unc_col)
#     print(f"[uncertainty] using column: {_unc_col}")

#     h, w = pred_grid.shape

#     def _wkt_from_tif(tif_path: str) -> tuple[str, object]:
#         """Read (wkt, affine) from a tif using GEOKEYS — bypasses EPSG registry."""
#         with Env(GTIFF_SRS_SOURCE="GEOKEYS", PROJ_NETWORK="OFF"):
#             with rasterio.open(tif_path) as src:
#                 xform = src.transform
#                 if src_crs_override and src_crs_override.strip():
#                     wkt = CRS.from_user_input(src_crs_override.strip()).to_wkt()
#                 elif src.crs is not None:
#                     wkt = src.crs.to_wkt()
#                 else:
#                     raise ValueError(f"No CRS in tif and no override: {tif_path}")
#         return wkt, xform

#     def _try_tif(tif_path: str, label: str):
#         try:
#             wkt, xform = _wkt_from_tif(tif_path)
#             pw, bounds = _warp_grid_to_wgs84(pred_grid, xform, wkt)
#             uw, _      = _warp_grid_to_wgs84(unc_grid,  xform, wkt)
#             print(f"[projection] OK via {label}: bounds={bounds}")
#             return pw, uw, bounds, True
#         except Exception as exc:
#             print(f"[projection] {label} failed: {exc}")
#             return None

#     # ── 1. x/y columns ────────────────────────────────────────────────────
#     if "x" in df.columns and "y" in df.columns:
#         try:
#             x_s = pd.to_numeric(df["x"], errors="coerce")
#             y_s = pd.to_numeric(df["y"], errors="coerce")
#             valid = np.isfinite(x_s) & np.isfinite(y_s) & \
#                     np.isfinite(df["row"]) & np.isfinite(df["col"])
#             if valid.sum() > 10:
#                 x_v = x_s[valid].to_numpy()
#                 y_v = y_s[valid].to_numpy()
#                 row_v = df.loc[valid, "row"].to_numpy(int)
#                 col_v = df.loc[valid, "col"].to_numpy(int)

#                 dx = np.polyfit(col_v, x_v, 1)[0]
#                 dy = np.polyfit(row_v, y_v, 1)[0]
#                 x0 = float(np.median(x_v - col_v * dx))
#                 y0 = float(np.median(y_v - row_v * dy))
#                 xform = Affine(abs(dx), 0.0, x0, 0.0, -abs(dy), y0)

#                 # Get WKT from any available tif (GEOKEYS, no EPSG lookup)
#                 wkt = None
#                 for cand in [scene_tif_path, output_pred_tif_path]:
#                     if cand and Path(cand).exists():
#                         try:
#                             wkt, _ = _wkt_from_tif(cand)
#                             break
#                         except Exception:
#                             pass
#                 if wkt is None and "pred_tif_path" in df.columns:
#                     vals = df["pred_tif_path"].dropna().astype(str)
#                     if len(vals) > 0 and Path(vals.iloc[0]).exists():
#                         try:
#                             wkt, _ = _wkt_from_tif(vals.iloc[0])
#                         except Exception:
#                             pass
#                 if src_crs_override and src_crs_override.strip():
#                     with Env(GTIFF_SRS_SOURCE="GEOKEYS", PROJ_NETWORK="OFF"):
#                         wkt = CRS.from_user_input(src_crs_override.strip()).to_wkt()
#                 if wkt is None:
#                     raise ValueError("No CRS WKT obtainable from any tif file")

#                 pw, bounds = _warp_grid_to_wgs84(pred_grid, xform, wkt)
#                 uw, _      = _warp_grid_to_wgs84(unc_grid,  xform, wkt)
#                 print(f"[projection] OK via x/y cols: dx={dx:.2f}m bounds={bounds}")
#                 return pw, uw, bounds, True
#         except Exception as exc:
#             print(f"[projection] x/y columns failed: {exc}")

#     # ── 2. Input scene tif ────────────────────────────────────────────────
#     if scene_tif_path and Path(scene_tif_path).exists():
#         r = _try_tif(scene_tif_path, "scene_tif (input)")
#         if r:
#             return r

#     # ── 3. Output pred tif / pred_tif_path column ────────────────────────
#     if output_pred_tif_path and Path(output_pred_tif_path).exists():
#         r = _try_tif(output_pred_tif_path, "output_pred_tif")
#         if r:
#             return r
#     if "pred_tif_path" in df.columns:
#         vals = df["pred_tif_path"].dropna().astype(str)
#         if len(vals) > 0 and Path(vals.iloc[0]).exists():
#             r = _try_tif(vals.iloc[0], "pred_tif_path column")
#             if r:
#                 return r

#     # ── 4. lat/lon columns (WGS-84 assumed) ──────────────────────────────
#     if "lat" in df.columns and "lon" in df.columns:
#         try:
#             lat_v = pd.to_numeric(df["lat"], errors="coerce").dropna().to_numpy()
#             lon_v = pd.to_numeric(df["lon"], errors="coerce").dropna().to_numpy()
#             lat_v = lat_v[np.isfinite(lat_v)]
#             lon_v = lon_v[np.isfinite(lon_v)]
#             if len(lat_v) > 4:
#                 bounds = [[float(lat_v.min()), float(lon_v.min())],
#                           [float(lat_v.max()), float(lon_v.max())]]
#                 print(f"[projection] OK via lat/lon: bounds={bounds}")
#                 return pred_grid, unc_grid, bounds, True
#         except Exception as exc:
#             print(f"[projection] lat/lon failed: {exc}")

#     # ── 5. Pixel-space fallback ──────────────────────────────────────────
#     print("[projection] WARNING: fallback to pixel space")
#     print(f"[projection] df columns: {list(df.columns)}")
#     return pred_grid, unc_grid, [[0.0, 0.0], [float(h), float(w)]], False


# # ---------------------------------------------------------------------------
# # Colorbar legend  (values baked via f-string — Branca this.attr is unreliable)
# # ---------------------------------------------------------------------------

# def _add_colorbars(
#     m: folium.Map,
#     pred_vmin: float, pred_vmax: float,
#     unc_vmin: float,  unc_vmax: float,
#     pred_label: str = "Prediction",
# ) -> None:
#     template = f"""
#     {{% macro html(this, kwargs) %}}
#     <div style="position:fixed;bottom:40px;left:16px;z-index:9999;
#                 background:rgba(255,255,255,0.93);padding:8px 12px;
#                 border:1px solid #888;border-radius:7px;font-size:12px;
#                 font-family:monospace;box-shadow:2px 2px 6px rgba(0,0,0,0.2);
#                 min-width:220px;pointer-events:none;">
#       <div style="font-weight:700;margin-bottom:4px;white-space:nowrap;
#                   overflow:hidden;text-overflow:ellipsis;max-width:220px;">
#         {pred_label}
#       </div>
#       <div style="width:210px;height:14px;
#                   background:linear-gradient(to right,#f7fcf5,#74c476,#00441b);
#                   border-radius:3px;margin-bottom:3px;"></div>
#       <div style="display:flex;justify-content:space-between;color:#333;">
#         <span>{_fmt(pred_vmin)}</span>
#         <span style="color:#888;font-size:10px;">q05–q95</span>
#         <span>{_fmt(pred_vmax)}</span>
#       </div>
#     </div>
#     <div style="position:fixed;bottom:40px;right:220px;z-index:9999;
#                 background:rgba(255,255,255,0.93);padding:8px 12px;
#                 border:1px solid #888;border-radius:7px;font-size:12px;
#                 font-family:monospace;box-shadow:2px 2px 6px rgba(0,0,0,0.2);
#                 min-width:220px;pointer-events:none;">
#       <div style="font-weight:700;margin-bottom:4px;">Uncertainty</div>
#       <div style="width:210px;height:14px;
#                   background:linear-gradient(to right,#fff5eb,#fdae6b,#a63603);
#                   border-radius:3px;margin-bottom:3px;"></div>
#       <div style="display:flex;justify-content:space-between;color:#333;">
#         <span>{_fmt(unc_vmin)}</span>
#         <span style="color:#888;font-size:10px;">q10–q90</span>
#         <span>{_fmt(unc_vmax)}</span>
#       </div>
#     </div>
#     {{% endmacro %}}
#     """
#     el = MacroElement()
#     el._template = Template(template)
#     el.add_to(m)


# # ---------------------------------------------------------------------------
# # On-map control panel + swipe bar (all inline JS, no CDN)
# #
# # The floating panel (top-right) contains:
# #   • Prediction layer toggle  (checkbox)
# #   • Uncertainty layer toggle (checkbox) — uncertainty always on top (zindex 4)
# #   • Swipe compare toggle     (checkbox) — shows/hides the divider bar
# #   • Basemap switcher         (radio: OSM / Light / Satellite / None)
# #
# # The swipe bar is always in the DOM; the toggle just shows/hides it.
# # No external CDN scripts are loaded — works inside Gradio's sandboxed iframe.
# # ---------------------------------------------------------------------------

# def _add_map_controls(m: folium.Map, pred_col: str) -> None:
#     map_var = m.get_name()

#     js = f"""
# <script>
# (function() {{
#   function init() {{
#     var map = window['{map_var}'];
#     if (!map) {{ setTimeout(init, 150); return; }}
#     var container = map.getContainer();
#     var pane      = map.getPanes().overlayPane;
#     if (!container || !pane) {{ setTimeout(init, 150); return; }}
#     if (getComputedStyle(container).position === 'static')
#       container.style.position = 'relative';

#     /* wait for both image overlays to exist */
#     var attempts = 0;
#     function waitForLayers() {{
#       var imgs = pane.querySelectorAll('img.leaflet-image-layer');
#       if (imgs.length < 2 && attempts++ < 40) {{ setTimeout(waitForLayers, 200); return; }}
#       buildUI();
#     }}

#     /* basemap definitions */
#     var basemaps = {{
#       'OpenStreetMap': L.tileLayer(
#         'https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',
#         {{attribution:'&copy; OpenStreetMap contributors', maxZoom:19}}),
#       'Light': L.tileLayer(
#         'https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png',
#         {{attribution:'&copy; CartoDB', subdomains:'abcd', maxZoom:19}}),
#       'Satellite': L.tileLayer(
#         'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}',
#         {{attribution:'Esri World Imagery', maxZoom:19}}),
#       'None': null
#     }};
#     var activeBasemap = basemaps['OpenStreetMap'];

#     function buildUI() {{
#       /* ── swipe bar ─────────────────────────────────────────────────── */
#       var bar = document.createElement('div');
#       bar.style.cssText =
#         'position:absolute;top:0;bottom:0;left:50%;width:5px;' +
#         'transform:translateX(-50%);background:rgba(255,255,255,0.92);' +
#         'box-shadow:0 0 10px rgba(0,0,0,0.75);cursor:ew-resize;' +
#         'z-index:1000;pointer-events:auto;';
#       var handle = document.createElement('div');
#       handle.innerHTML = '&#8596;';
#       handle.style.cssText =
#         'position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);' +
#         'background:#fff;border:2px solid #444;border-radius:50%;' +
#         'width:34px;height:34px;line-height:30px;text-align:center;' +
#         'font-size:17px;font-weight:bold;' +
#         'box-shadow:0 2px 8px rgba(0,0,0,.5);user-select:none;pointer-events:none;';
#       bar.appendChild(handle);
#       container.appendChild(bar);

#       var swipeOn = true;
#       var pct     = 50;

#       function imgs() {{ return pane.querySelectorAll('img.leaflet-image-layer'); }}

#       function applyClip(p) {{
#         pct = p;
#         var ii = imgs(); if (ii.length < 2) return;
#         var c = 'inset(0 0 0 ' + p + '%)';
#         ii[1].style.clipPath = c; ii[1].style.webkitClipPath = c;
#       }}
#       function clearClip() {{
#         var ii = imgs(); if (ii.length < 2) return;
#         ii[1].style.clipPath = ''; ii[1].style.webkitClipPath = '';
#       }}

#       var drag = false;
#       function pctFromE(e) {{
#         var r = container.getBoundingClientRect();
#         var x = e.touches ? e.touches[0].clientX : e.clientX;
#         return Math.max(0, Math.min(100, (x - r.left) / r.width * 100));
#       }}
#       bar.addEventListener('mousedown',  function(e){{ drag=true; e.preventDefault(); e.stopPropagation(); }});
#       bar.addEventListener('touchstart', function(e){{ drag=true; e.stopPropagation(); }}, {{passive:true}});
#       document.addEventListener('mouseup',   function(){{ drag=false; }});
#       document.addEventListener('touchend',  function(){{ drag=false; }});
#       document.addEventListener('mousemove', function(e){{
#         if(!drag) return; var p=pctFromE(e); bar.style.left=p+'%'; applyClip(p);
#       }});
#       document.addEventListener('touchmove', function(e){{
#         if(!drag) return; var p=pctFromE(e); bar.style.left=p+'%'; applyClip(p);
#       }}, {{passive:true}});
#       map.on('moveend zoomend layeradd', function(){{
#         setTimeout(function(){{ if(swipeOn) applyClip(pct); }}, 60);
#       }});
#       applyClip(50);

#       /* ── floating control panel ────────────────────────────────────── */
#       var panel = document.createElement('div');
#       panel.style.cssText =
#         'position:absolute;top:12px;right:12px;z-index:1100;' +
#         'background:rgba(18,20,32,0.92);color:#e8eaf6;border-radius:10px;' +
#         'padding:14px 16px;font-family:Segoe UI,Arial,sans-serif;font-size:13px;' +
#         'box-shadow:0 4px 20px rgba(0,0,0,0.65);min-width:195px;' +
#         'backdrop-filter:blur(8px);border:1px solid rgba(255,255,255,0.08);';

#       /* helper: checkbox row */
#       function cbRow(label, checked, color, onChange) {{
#         var d  = document.createElement('div');
#         d.style.cssText = 'display:flex;align-items:center;gap:9px;margin-bottom:8px;cursor:pointer;';
#         var cb = document.createElement('input');
#         cb.type='checkbox'; cb.checked=checked;
#         cb.style.cssText = 'width:15px;height:15px;cursor:pointer;accent-color:'+color+';flex-shrink:0;';
#         cb.addEventListener('change', function(){{ onChange(cb.checked); }});
#         var sp = document.createElement('span');
#         sp.textContent = label; sp.style.userSelect='none';
#         d.appendChild(cb); d.appendChild(sp);
#         d.addEventListener('click', function(e){{
#           if(e.target!==cb){{ cb.checked=!cb.checked; cb.dispatchEvent(new Event('change')); }}
#         }});
#         return d;
#       }}

#       /* Section label helper */
#       function sectionLabel(text) {{
#         var d = document.createElement('div');
#         d.textContent = text;
#         d.style.cssText = 'font-size:10px;color:#7c84a8;letter-spacing:.08em;' +
#           'text-transform:uppercase;margin:6px 0 6px 0;font-weight:600;';
#         return d;
#       }}

#       /* Separator */
#       function sep() {{
#         var d = document.createElement('div');
#         d.style.cssText = 'border-top:1px solid rgba(255,255,255,0.1);margin:8px 0;';
#         return d;
#       }}

#       panel.appendChild(sectionLabel('Layers'));

#       /* Prediction toggle */
#       panel.appendChild(cbRow('Prediction', true, '#74c476', function(on){{
#         var ii = imgs(); if(!ii.length) return;
#         ii[0].style.opacity = on ? '1' : '0';
#       }}));

#       /* Uncertainty toggle */
#       panel.appendChild(cbRow('Uncertainty', true, '#fd8d3c', function(on){{
#         var ii = imgs(); if(ii.length<2) return;
#         ii[1].style.opacity = on ? '1' : '0';
#       }}));

#       panel.appendChild(sep());
#       panel.appendChild(sectionLabel('Swipe'));

#       /* Swipe toggle */
#       panel.appendChild(cbRow('Compare mode', true, '#9fa8d5', function(on){{
#         swipeOn = on;
#         bar.style.display = on ? '' : 'none';
#         if(on) applyClip(pct); else clearClip();
#       }}));

#       panel.appendChild(sep());
#       panel.appendChild(sectionLabel('Basemap'));

#       /* Basemap radio buttons */
#       ['OpenStreetMap','Light','Satellite','None'].forEach(function(name){{
#         var d  = document.createElement('div');
#         d.style.cssText = 'display:flex;align-items:center;gap:9px;margin-bottom:6px;cursor:pointer;';
#         var rb = document.createElement('input');
#         rb.type='radio'; rb.name='bm_{map_var}'; rb.value=name;
#         rb.checked = (name === 'OpenStreetMap');
#         rb.style.cssText = 'cursor:pointer;accent-color:#9fa8d5;flex-shrink:0;';
#         rb.addEventListener('change', function(){{
#           if(activeBasemap) map.removeLayer(activeBasemap);
#           activeBasemap = basemaps[name];
#           if(activeBasemap) activeBasemap.addTo(map);
#         }});
#         var sp = document.createElement('span');
#         sp.textContent = name; sp.style.userSelect='none';
#         d.appendChild(rb); d.appendChild(sp);
#         d.addEventListener('click', function(e){{
#           if(e.target!==rb){{ rb.checked=true; rb.dispatchEvent(new Event('change')); }}
#         }});
#         panel.appendChild(d);
#       }});

#       container.appendChild(panel);
#     }}

#     waitForLayers();
#   }}

#   if (document.readyState === 'loading')
#     document.addEventListener('DOMContentLoaded', init);
#   else
#     init();
# }})();
# </script>
# """
#     m.get_root().html.add_child(Element(js))


# # ---------------------------------------------------------------------------
# # Map builder
# # ---------------------------------------------------------------------------

# def _build_folium_overlay_map(
#     df: pd.DataFrame,
#     trait_name: str,
#     src_crs_override: str | None,
#     scene_tif_path: str | None = None,
#     output_pred_tif_path: str | None = None,
# ) -> str:
#     if "row" not in df.columns or "col" not in df.columns:
#         return "<h3>No raster indices (row/col) in output — raster inference required.</h3>"

#     pred_col = _resolve_pred_col(df, trait_name=trait_name)

#     pred_grid, unc_grid, bounds, has_geo = _resolve_grids_and_bounds(
#         df, pred_col, src_crs_override,
#         scene_tif_path=scene_tif_path,
#         output_pred_tif_path=output_pred_tif_path,
#     )

#     pred_rgba, pred_vmin, pred_vmax = _rgba_from_grid(pred_grid, "Greens",  is_uncertainty=False)
#     unc_rgba,  unc_vmin,  unc_vmax  = _rgba_from_grid(unc_grid,  "Oranges", is_uncertainty=True)

#     center = [(bounds[0][0] + bounds[1][0]) / 2.0,
#               (bounds[0][1] + bounds[1][1]) / 2.0]
#     m = folium.Map(
#         location=center,
#         zoom_start=12 if has_geo else 1,
#         tiles="OpenStreetMap",
#         control_scale=True,
#     )

#     # Prediction = bottom (zindex 3), Uncertainty always on top (zindex 4).
#     # mercator_project=False: grid already reprojected by rasterio — no further
#     # warping needed, and avoids the horizontal-stripe artefact caused by
#     # mercator_transform interpolating across NaN scan-line gaps.
#     folium.raster_layers.ImageOverlay(
#         image=pred_rgba, bounds=bounds, name="Prediction",
#         opacity=0.85, mercator_project=False,
#         interactive=True, cross_origin=False, zindex=3,
#     ).add_to(m)
#     folium.raster_layers.ImageOverlay(
#         image=unc_rgba, bounds=bounds, name="Uncertainty",
#         opacity=0.85, mercator_project=False,
#         interactive=True, cross_origin=False, zindex=4,
#     ).add_to(m)

#     # On-map floating panel: layer toggles + swipe bar + basemap switcher
#     _add_map_controls(m, pred_col=pred_col)

#     m.fit_bounds(bounds)
#     _add_colorbars(
#         m,
#         pred_vmin=pred_vmin, pred_vmax=pred_vmax,
#         unc_vmin=unc_vmin,   unc_vmax=unc_vmax,
#         pred_label=f"Prediction ({pred_col})",
#     )
#     return m._repr_html_()


# # ---------------------------------------------------------------------------
# # Artifact resolution
# # ---------------------------------------------------------------------------

# def _resolve_uncertainty_artifacts(cfg):
#     q_path = Path(str(cfg.inference.quantile_models_path))
#     d_path = Path(str(cfg.inference.distance_reference_npz))
#     if q_path.exists() and d_path.exists():
#         return str(q_path), str(d_path)

#     root = (
#         q_path.parents[1]
#         if len(q_path.parents) >= 2
#         else Path("results/experiments/study2_uncertainty_1522")
#     )
#     t1 = root / "target_1"
#     if (t1 / "quantile_models.pkl").exists() and (t1 / "distance_reference.npz").exists():
#         return str(t1 / "quantile_models.pkl"), str(t1 / "distance_reference.npz")

#     for c in sorted(root.glob("target_*")):
#         q, d = c / "quantile_models.pkl", c / "distance_reference.npz"
#         if q.exists() and d.exists():
#             return str(q), str(d)

#     raise FileNotFoundError(
#         f"Could not find uncertainty artifacts under {root}. "
#         "Run uncertainty stage=distance first."
#     )


# # ---------------------------------------------------------------------------
# # Core inference callback
# # ---------------------------------------------------------------------------

# def run_app(
#     scene_csv_file,
#     scene_tif_file,
#     sensor_bands_file,
#     src_crs_override,
#     trait_name,
# ):
#     """Run inference and return (overlay_html, trait_update, pred_tif, unc_tif, cached)."""
#     tmp_out = Path(tempfile.mkdtemp()) / "scene_predictions_uncertainty.csv"
#     cfg = OmegaConf.load(
#         Path(__file__).resolve().parents[3]
#         / "configs" / "experiments" / "scene_infer_uncertainty.yaml"
#     )

#     if scene_tif_file is not None and sensor_bands_file is not None:
#         cfg.inference.input_type = "tif"
#         cfg.inference.scene_tif = scene_tif_file.name
#         cfg.inference.sensor_bands_csv = sensor_bands_file.name
#     elif scene_csv_file is not None:
#         cfg.inference.input_type = "csv"
#         cfg.inference.scene_csv = scene_csv_file.name
#     else:
#         raise ValueError("Provide either Scene CSV or (Scene TIFF + Sensor Bands CSV).")

#     cfg.inference.output_path      = str(tmp_out)
#     cfg.inference.output_pred_tif  = str(tmp_out.with_name(tmp_out.stem + "_predictions_20traits.tif"))
#     cfg.inference.output_unc_tif   = str(tmp_out.with_name(tmp_out.stem + "_uncertainty_20traits.tif"))

#     q, d = _resolve_uncertainty_artifacts(cfg)
#     cfg.inference.quantile_models_path   = q
#     cfg.inference.distance_reference_npz = d
#     run_inference(cfg)

#     df = pd.read_csv(tmp_out)
#     pred_cols     = [c for c in df.columns if c.startswith("pred_")]
#     trait_choices = [c.replace("pred_", "", 1) for c in pred_cols]
#     selected      = str(trait_name) if str(trait_name) in trait_choices \
#                     else (trait_choices[0] if trait_choices else "")

#     crs        = str(src_crs_override).strip() or None
#     _scene_tif = str(scene_tif_file.name) if scene_tif_file is not None else None
#     _pred_tif  = str(tmp_out.with_name(tmp_out.stem + "_predictions_20traits.tif"))

#     overlay_html = _build_folium_overlay_map(
#         df,
#         trait_name=selected,
#         src_crs_override=crs,
#         scene_tif_path=_scene_tif,
#         output_pred_tif_path=_pred_tif,
#     )

#     pred_tif = Path(cfg.inference.output_pred_tif)
#     unc_tif  = Path(cfg.inference.output_unc_tif)
#     # Pack paths into one state string
#     cached = str(tmp_out) + "|" + (_scene_tif or "") + "|" + _pred_tif
#     return (
#         overlay_html,
#         gr.update(choices=trait_choices or DEFAULT_TRAIT_CHOICES, value=selected or None),
#         str(pred_tif) if pred_tif.exists() else None,
#         str(unc_tif)  if unc_tif.exists()  else None,
#         cached,
#     )


# def _refresh_map(cached_out_csv: str, src_crs_override: str, trait_name: str) -> str:
#     """Re-render map for a new trait without re-running inference."""
#     if not cached_out_csv:
#         return ""
#     parts           = str(cached_out_csv).split("|")
#     csv_path        = parts[0]
#     scene_tif_path  = parts[1] if len(parts) > 1 and parts[1] else None
#     pred_tif_path   = parts[2] if len(parts) > 2 and parts[2] else None

#     out_path = Path(csv_path)
#     if not out_path.exists():
#         return f"<h3>Cached CSV not found: {out_path}</h3>"

#     df  = pd.read_csv(out_path)
#     crs = str(src_crs_override).strip() or None
#     return _build_folium_overlay_map(
#         df,
#         trait_name=str(trait_name),
#         src_crs_override=crs,
#         scene_tif_path=scene_tif_path,
#         output_pred_tif_path=pred_tif_path,
#     )


# # ---------------------------------------------------------------------------
# # Gradio UI
# # ---------------------------------------------------------------------------

# def launch() -> None:
#     with gr.Blocks(
#         title="Dis_UN Scene Inference",
#         css="""
#         /* ── base ───────────────────────────────────── */
#         .gradio-container { max-width: 1100px !important; margin: 0 auto; }

#         /* ── upload card ────────────────────────────── */
#         #upload-card {
#             background: #1c1f2e;
#             border: 1px solid #2e3352;
#             border-radius: 12px;
#             padding: 20px 24px 16px;
#         }

#         /* ── run button ─────────────────────────────── */
#         #run-btn {
#             font-size: 15px !important;
#             font-weight: 700 !important;
#             padding: 11px 0 !important;
#             border-radius: 8px !important;
#             margin-top: 10px;
#         }

#         /* ── trait row ──────────────────────────────── */
#         #trait-row { margin-top: 18px; margin-bottom: 6px; }

#         /* ── map fills width ────────────────────────── */
#         #map-output iframe, #map-output { width: 100% !important; }

#         /* ── download row ───────────────────────────── */
#         #dl-row { margin-top: 10px; }
#         """,
#     ) as demo:

#         gr.Markdown("## 🌿 HSI Scene Inference — Dis_UN Uncertainty")
#         gr.Markdown(
#             "Upload your scene TIFF and sensor bands CSV, then click **Run Inference**. "
#             "The map and download files will appear when processing is complete."
#         )

#         # ── STATE ─────────────────────────────────────────────────────────
#         cached_out_csv = gr.State(value="")

#         # ═══════════════════════════════════════════════════════════════════
#         # PHASE 1 — always visible: upload + run
#         # ═══════════════════════════════════════════════════════════════════
#         with gr.Group(elem_id="upload-card"):
#             with gr.Row():
#                 scene_tif_file = gr.File(
#                     label="📂  Scene HSI TIFF",
#                     file_types=[".tif", ".tiff"],
#                     scale=2,
#                 )
#                 sensor_bands_file = gr.File(
#                     label="📋  Sensor Bands CSV",
#                     file_types=[".csv"],
#                     scale=1,
#                 )
#             with gr.Row():
#                 scene_csv_file = gr.File(
#                     label="📋  Scene CSV  (alternative to TIFF)",
#                     file_types=[".csv"],
#                     scale=2,
#                 )
#                 src_crs_override = gr.Textbox(
#                     label="CRS override  (optional, e.g. EPSG:32633)",
#                     value="",
#                     placeholder="Leave blank — auto-detected from TIFF",
#                     scale=1,
#                 )

#         run_btn = gr.Button("▶  Run Inference", variant="primary", elem_id="run-btn")

#         # ═══════════════════════════════════════════════════════════════════
#         # PHASE 2 — hidden until inference completes
#         # ═══════════════════════════════════════════════════════════════════
#         with gr.Group(visible=False) as results_panel:

#             # Trait selector + refresh button
#             with gr.Row(elem_id="trait-row"):
#                 trait_name = gr.Dropdown(
#                     label="🌱  Trait",
#                     choices=DEFAULT_TRAIT_CHOICES,
#                     value="EWT_mg_cm2",
#                     allow_custom_value=False,
#                     scale=4,
#                 )
#                 refresh_btn = gr.Button("↻  Refresh map", scale=1)

#             # Map (all controls rendered inside the map HTML itself)
#             overlay_map = gr.HTML(elem_id="map-output")

#             # Downloads — hidden until files are ready
#             with gr.Row(elem_id="dl-row", visible=False) as dl_row:
#                 pred_tif_file = gr.File(
#                     label="⬇  Prediction GeoTIFF (20 traits)",
#                     interactive=False,
#                 )
#                 unc_tif_file = gr.File(
#                     label="⬇  Uncertainty GeoTIFF (20 traits)",
#                     interactive=False,
#                 )

#         # ── WIRING ────────────────────────────────────────────────────────

#         def _run_and_reveal(
#             scene_csv, scene_tif, sensor_bands, src_crs, trait,
#             progress=gr.Progress(track_tqdm=True),
#         ):
#             progress(0.0, desc="Starting inference…")
#             overlay_html, trait_upd, pred_path, unc_path, cached = run_app(
#                 scene_csv, scene_tif, sensor_bands, src_crs, trait,
#             )
#             progress(1.0, desc="Done ✓")
#             has_files = bool(pred_path or unc_path)
#             return (
#                 overlay_html,           # overlay_map
#                 trait_upd,              # trait_name
#                 pred_path,              # pred_tif_file
#                 unc_path,               # unc_tif_file
#                 cached,                 # cached_out_csv
#                 gr.update(visible=True),        # results_panel
#                 gr.update(visible=has_files),   # dl_row
#             )

#         run_btn.click(
#             _run_and_reveal,
#             inputs=[scene_csv_file, scene_tif_file, sensor_bands_file,
#                     src_crs_override, trait_name],
#             outputs=[overlay_map, trait_name, pred_tif_file, unc_tif_file,
#                      cached_out_csv, results_panel, dl_row],
#         )

#         # Refresh map without re-running inference
#         def _do_refresh(cached, src_crs, trait):
#             return _refresh_map(cached, src_crs, trait)

#         refresh_btn.click(
#             _do_refresh,
#             inputs=[cached_out_csv, src_crs_override, trait_name],
#             outputs=[overlay_map],
#         )

#         # Auto-refresh when trait dropdown changes (after inference has run)
#         trait_name.change(
#             _do_refresh,
#             inputs=[cached_out_csv, src_crs_override, trait_name],
#             outputs=[overlay_map],
#         )

#     demo.launch(server_name="0.0.0.0", server_port=7862)


# if __name__ == "__main__":
#     launch()


############################################################
# from __future__ import annotations

# import logging
# import queue
# import re
# import tempfile
# import threading
# from pathlib import Path

# import gradio as gr
# import matplotlib as mpl
# import numpy as np
# import pandas as pd
# import folium
# from branca.element import Element, MacroElement, Template

# from omegaconf import OmegaConf

# from ..experiments.uncertainty_scene_infer import run_inference


# DEFAULT_TRAIT_CHOICES = [
#     "Anth_area_ug_cm2",
#     "Boron_area_mg_cm2",
#     "C_area_mg_cm2",
#     "Ca_area_mg_cm2",
#     "Car_area_ug_cm2",
#     "Cellulose_mg_cm2",
#     "Chl_area_ug_cm2",
#     "Cu_area_mg_cm2",
#     "EWT_mg_cm2",
#     "Fiber_mg_cm2",
#     "LAI_m2_m2",
#     "LMA_g_m2",
#     "Lignin_mg_cm2",
#     "Mg_area_mg_cm2",
#     "Mn_area_mg_cm2",
#     "NSC_mg_cm2",
#     "N_area_mg_cm2",
#     "P_area_mg_cm2",
#     "Potassium_area_mg_cm2",
#     "S_area_mg_cm2",
# ]


# # ---------------------------------------------------------------------------
# # Basic helpers
# # ---------------------------------------------------------------------------

# def _resolve_pred_col(df: pd.DataFrame, trait_name: str) -> str:
#     pred_cols = [c for c in df.columns if c.startswith("pred_")]
#     if not pred_cols:
#         raise ValueError("No prediction columns found in output.")
#     if trait_name:
#         if trait_name in pred_cols:
#             return trait_name
#         wanted = f"pred_{trait_name}"
#         if wanted in pred_cols:
#             return wanted
#     return pred_cols[0]


# def _grid_from_df(df: pd.DataFrame, col: str) -> np.ndarray:
#     h = int(df["row"].max()) + 1
#     w = int(df["col"].max()) + 1
#     grid = np.full((h, w), np.nan, dtype=np.float32)
#     grid[df["row"].to_numpy(int), df["col"].to_numpy(int)] = df[col].to_numpy(np.float32)
#     return grid


# # ---------------------------------------------------------------------------
# # Colormap stretch
# # ---------------------------------------------------------------------------

# PRED_THRESHOLD = 0.05   # prediction: vmin=q05, vmax=q95
# UNC_THRESHOLD  = 0.90   # uncertainty: vmin=q10, vmax=q90  (matches original style)


# def _pred_norm_bounds(arr: np.ndarray) -> tuple[float, float]:
#     """q05–q95 stretch for prediction."""
#     finite = arr[np.isfinite(arr)]
#     if finite.size == 0:
#         return 0.0, 1.0
#     vmin = float(np.nanpercentile(finite, PRED_THRESHOLD * 100))
#     vmax = float(np.nanpercentile(finite, (1 - PRED_THRESHOLD) * 100))
#     return vmin, vmax if vmax > vmin else vmin + 1e-6


# def _unc_norm_bounds(arr: np.ndarray) -> tuple[float, float]:
#     """
#     Uncertainty stretch matching original project formula:
#         maxv = quantile(thre)        ← LOW quantile  → colormap bright end
#         minv = quantile(1 - thre)    ← HIGH quantile → colormap dark end
#     High-uncertainty pixels dominate visually (dark/saturated).
#     """
#     finite = arr[np.isfinite(arr)]
#     if finite.size == 0:
#         return 0.0, 1.0
#     vmin = float(np.nanpercentile(finite, (1 - UNC_THRESHOLD) * 100))
#     vmax = float(np.nanpercentile(finite, UNC_THRESHOLD * 100))
#     if vmax <= vmin:
#         vmax = vmin + 1e-6
#     return vmin, vmax


# def _rgba_from_grid(
#     grid: np.ndarray,
#     cmap_name: str,
#     is_uncertainty: bool = False,
# ) -> tuple[np.ndarray, float, float]:
#     """Return (RGBA uint8 H×W×4, vmin, vmax)."""
#     vmin, vmax = _unc_norm_bounds(grid) if is_uncertainty else _pred_norm_bounds(grid)
#     norm = np.clip((grid - vmin) / (vmax - vmin), 0.0, 1.0)
#     cmap = mpl.colormaps.get_cmap(cmap_name)
#     rgba = (cmap(norm) * 255.0).astype(np.uint8)
#     rgba[..., 3] = np.where(np.isfinite(grid), 210, 0).astype(np.uint8)
#     return rgba, vmin, vmax


# def _fmt(v: float) -> str:
#     if v == 0.0:
#         return "0"
#     return f"{v:.2e}" if (abs(v) >= 10_000 or abs(v) < 0.001) else f"{v:.4g}"


# # ---------------------------------------------------------------------------
# # Projection: warp grid to WGS-84 using rasterio
# # Uses WKT strings throughout — never EPSG integer lookups — so a
# # broken/outdated PROJ database (conda env mismatch) does not crash.
# # ---------------------------------------------------------------------------

# def _warp_grid_to_wgs84(
#     grid: np.ndarray,
#     src_transform,
#     src_crs_wkt: str,
# ) -> tuple[np.ndarray, list]:
#     """
#     Reproject a 2-D float32 grid from src_crs_wkt to WGS-84.

#     Returns
#     -------
#     warped_grid : np.ndarray  (float32, NaN for nodata)
#     bounds      : [[south, west], [north, east]]  in degrees
#     """
#     import rasterio
#     from rasterio.crs import CRS
#     from rasterio.warp import reproject, Resampling, calculate_default_transform
#     from rasterio.env import Env

#     h, w = grid.shape

#     WGS84_WKT = (
#         'GEOGCS["WGS 84",DATUM["WGS_1984",'
#         'SPHEROID["WGS 84",6378137,298.257223563]],'
#         'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433],'
#         'AUTHORITY["EPSG","4326"]]'
#     )

#     with Env(GTIFF_SRS_SOURCE="GEOKEYS", PROJ_NETWORK="OFF"):
#         src_crs = CRS.from_wkt(src_crs_wkt)
#         dst_crs = CRS.from_wkt(WGS84_WKT)

#         left   = src_transform.c
#         top    = src_transform.f
#         right  = left + w * src_transform.a
#         bottom = top  + h * src_transform.e   # e is negative

#         dst_transform, dst_w, dst_h = calculate_default_transform(
#             src_crs, dst_crs, w, h,
#             left=left, bottom=bottom, right=right, top=top,
#         )

#         NODATA = -9999.0
#         src_data = np.where(np.isnan(grid), NODATA, grid).astype(np.float32)
#         dst_data = np.full((dst_h, dst_w), NODATA, dtype=np.float32)

#         reproject(
#             source=src_data, destination=dst_data,
#             src_transform=src_transform, src_crs=src_crs,
#             dst_transform=dst_transform, dst_crs=dst_crs,
#             src_nodata=NODATA, dst_nodata=NODATA,
#             resampling=Resampling.bilinear,
#         )

#     warped = np.where(dst_data == NODATA, np.nan, dst_data)
#     west  = dst_transform.c
#     north = dst_transform.f
#     east  = west  + dst_w * dst_transform.a
#     south = north + dst_h * dst_transform.e

#     return warped, [[float(south), float(west)], [float(north), float(east)]]


# def _resolve_grids_and_bounds(
#     df: pd.DataFrame,
#     pred_col: str,
#     src_crs_override: str | None,
#     scene_tif_path: str | None = None,
#     output_pred_tif_path: str | None = None,
# ) -> tuple[np.ndarray, np.ndarray, list, bool]:
#     """
#     Return (pred_grid_wgs84, unc_grid_wgs84, bounds, has_geo).

#     Priority:
#       1. x/y columns  — projected pixel coords always written by inference.
#          Affine derived via polyfit; WKT read from any available tif.
#       2. scene_tif_path (INPUT tif) — definitive CRS + geotransform.
#       3. output_pred_tif_path / pred_tif_path column — secondary fallbacks.
#       4. lat/lon columns — assumed WGS-84.
#       5. Pixel-space fallback.
#     """
#     import rasterio
#     from rasterio.crs import CRS
#     from rasterio.env import Env
#     from rasterio.transform import Affine

#     pred_grid = _grid_from_df(df, pred_col)

#     # Per-trait uncertainty column: prefer un_<trait>, fall back to uncertainty_mean
#     _trait = pred_col.replace("pred_", "", 1)
#     _unc_col = f"un_{_trait}" if f"un_{_trait}" in df.columns else "uncertainty_mean"
#     unc_grid = _grid_from_df(df, _unc_col)
#     print(f"[uncertainty] using column: {_unc_col}")

#     h, w = pred_grid.shape

#     def _wkt_from_tif(tif_path: str) -> tuple[str, object]:
#         """Read (wkt, affine) from a tif using GEOKEYS — bypasses EPSG registry."""
#         with Env(GTIFF_SRS_SOURCE="GEOKEYS", PROJ_NETWORK="OFF"):
#             with rasterio.open(tif_path) as src:
#                 xform = src.transform
#                 if src_crs_override and src_crs_override.strip():
#                     wkt = CRS.from_user_input(src_crs_override.strip()).to_wkt()
#                 elif src.crs is not None:
#                     wkt = src.crs.to_wkt()
#                 else:
#                     raise ValueError(f"No CRS in tif and no override: {tif_path}")
#         return wkt, xform

#     def _try_tif(tif_path: str, label: str):
#         try:
#             wkt, xform = _wkt_from_tif(tif_path)
#             pw, bounds = _warp_grid_to_wgs84(pred_grid, xform, wkt)
#             uw, _      = _warp_grid_to_wgs84(unc_grid,  xform, wkt)
#             print(f"[projection] OK via {label}: bounds={bounds}")
#             return pw, uw, bounds, True
#         except Exception as exc:
#             print(f"[projection] {label} failed: {exc}")
#             return None

#     # ── 1. x/y columns ────────────────────────────────────────────────────
#     if "x" in df.columns and "y" in df.columns:
#         try:
#             x_s = pd.to_numeric(df["x"], errors="coerce")
#             y_s = pd.to_numeric(df["y"], errors="coerce")
#             valid = np.isfinite(x_s) & np.isfinite(y_s) & \
#                     np.isfinite(df["row"]) & np.isfinite(df["col"])
#             if valid.sum() > 10:
#                 x_v = x_s[valid].to_numpy()
#                 y_v = y_s[valid].to_numpy()
#                 row_v = df.loc[valid, "row"].to_numpy(int)
#                 col_v = df.loc[valid, "col"].to_numpy(int)

#                 dx = np.polyfit(col_v, x_v, 1)[0]
#                 dy = np.polyfit(row_v, y_v, 1)[0]
#                 x0 = float(np.median(x_v - col_v * dx))
#                 y0 = float(np.median(y_v - row_v * dy))
#                 xform = Affine(abs(dx), 0.0, x0, 0.0, -abs(dy), y0)

#                 # Get WKT from any available tif (GEOKEYS, no EPSG lookup)
#                 wkt = None
#                 for cand in [scene_tif_path, output_pred_tif_path]:
#                     if cand and Path(cand).exists():
#                         try:
#                             wkt, _ = _wkt_from_tif(cand)
#                             break
#                         except Exception:
#                             pass
#                 if wkt is None and "pred_tif_path" in df.columns:
#                     vals = df["pred_tif_path"].dropna().astype(str)
#                     if len(vals) > 0 and Path(vals.iloc[0]).exists():
#                         try:
#                             wkt, _ = _wkt_from_tif(vals.iloc[0])
#                         except Exception:
#                             pass
#                 if src_crs_override and src_crs_override.strip():
#                     with Env(GTIFF_SRS_SOURCE="GEOKEYS", PROJ_NETWORK="OFF"):
#                         wkt = CRS.from_user_input(src_crs_override.strip()).to_wkt()
#                 if wkt is None:
#                     raise ValueError("No CRS WKT obtainable from any tif file")

#                 pw, bounds = _warp_grid_to_wgs84(pred_grid, xform, wkt)
#                 uw, _      = _warp_grid_to_wgs84(unc_grid,  xform, wkt)
#                 print(f"[projection] OK via x/y cols: dx={dx:.2f}m bounds={bounds}")
#                 return pw, uw, bounds, True
#         except Exception as exc:
#             print(f"[projection] x/y columns failed: {exc}")

#     # ── 2. Input scene tif ────────────────────────────────────────────────
#     if scene_tif_path and Path(scene_tif_path).exists():
#         r = _try_tif(scene_tif_path, "scene_tif (input)")
#         if r:
#             return r

#     # ── 3. Output pred tif / pred_tif_path column ────────────────────────
#     if output_pred_tif_path and Path(output_pred_tif_path).exists():
#         r = _try_tif(output_pred_tif_path, "output_pred_tif")
#         if r:
#             return r
#     if "pred_tif_path" in df.columns:
#         vals = df["pred_tif_path"].dropna().astype(str)
#         if len(vals) > 0 and Path(vals.iloc[0]).exists():
#             r = _try_tif(vals.iloc[0], "pred_tif_path column")
#             if r:
#                 return r

#     # ── 4. lat/lon columns (WGS-84 assumed) ──────────────────────────────
#     if "lat" in df.columns and "lon" in df.columns:
#         try:
#             lat_v = pd.to_numeric(df["lat"], errors="coerce").dropna().to_numpy()
#             lon_v = pd.to_numeric(df["lon"], errors="coerce").dropna().to_numpy()
#             lat_v = lat_v[np.isfinite(lat_v)]
#             lon_v = lon_v[np.isfinite(lon_v)]
#             if len(lat_v) > 4:
#                 bounds = [[float(lat_v.min()), float(lon_v.min())],
#                           [float(lat_v.max()), float(lon_v.max())]]
#                 print(f"[projection] OK via lat/lon: bounds={bounds}")
#                 return pred_grid, unc_grid, bounds, True
#         except Exception as exc:
#             print(f"[projection] lat/lon failed: {exc}")

#     # ── 5. Pixel-space fallback ──────────────────────────────────────────
#     print("[projection] WARNING: fallback to pixel space")
#     print(f"[projection] df columns: {list(df.columns)}")
#     return pred_grid, unc_grid, [[0.0, 0.0], [float(h), float(w)]], False


# # ---------------------------------------------------------------------------
# # Colorbar legend  (values baked via f-string — Branca this.attr is unreliable)
# # ---------------------------------------------------------------------------

# def _add_colorbars(
#     m: folium.Map,
#     pred_vmin: float, pred_vmax: float,
#     unc_vmin: float,  unc_vmax: float,
#     pred_label: str = "Prediction",
# ) -> None:
#     template = f"""
#     {{% macro html(this, kwargs) %}}
#     <div style="position:fixed;bottom:40px;left:16px;z-index:9999;
#                 background:rgba(255,255,255,0.93);padding:8px 12px;
#                 border:1px solid #888;border-radius:7px;font-size:12px;
#                 font-family:monospace;box-shadow:2px 2px 6px rgba(0,0,0,0.2);
#                 min-width:220px;pointer-events:none;">
#       <div style="font-weight:700;margin-bottom:4px;white-space:nowrap;
#                   overflow:hidden;text-overflow:ellipsis;max-width:220px;">
#         {pred_label}
#       </div>
#       <div style="width:210px;height:14px;
#                   background:linear-gradient(to right,#f7fcf5,#74c476,#00441b);
#                   border-radius:3px;margin-bottom:3px;"></div>
#       <div style="display:flex;justify-content:space-between;color:#333;">
#         <span>{_fmt(pred_vmin)}</span>
#         <span style="color:#888;font-size:10px;">q05–q95</span>
#         <span>{_fmt(pred_vmax)}</span>
#       </div>
#     </div>
#     <div style="position:fixed;bottom:40px;right:220px;z-index:9999;
#                 background:rgba(255,255,255,0.93);padding:8px 12px;
#                 border:1px solid #888;border-radius:7px;font-size:12px;
#                 font-family:monospace;box-shadow:2px 2px 6px rgba(0,0,0,0.2);
#                 min-width:220px;pointer-events:none;">
#       <div style="font-weight:700;margin-bottom:4px;">Uncertainty</div>
#       <div style="width:210px;height:14px;
#                   background:linear-gradient(to right,#fff5eb,#fdae6b,#a63603);
#                   border-radius:3px;margin-bottom:3px;"></div>
#       <div style="display:flex;justify-content:space-between;color:#333;">
#         <span>{_fmt(unc_vmin)}</span>
#         <span style="color:#888;font-size:10px;">q10–q90</span>
#         <span>{_fmt(unc_vmax)}</span>
#       </div>
#     </div>
#     {{% endmacro %}}
#     """
#     el = MacroElement()
#     el._template = Template(template)
#     el.add_to(m)


# # ---------------------------------------------------------------------------
# # On-map control panel + swipe bar (all inline JS, no CDN)
# #
# # The floating panel (top-right) contains:
# #   • Prediction layer toggle  (checkbox)
# #   • Uncertainty layer toggle (checkbox) — uncertainty always on top (zindex 4)
# #   • Swipe compare toggle     (checkbox) — shows/hides the divider bar
# #   • Basemap switcher         (radio: OSM / Light / Satellite / None)
# #
# # The swipe bar is always in the DOM; the toggle just shows/hides it.
# # No external CDN scripts are loaded — works inside Gradio's sandboxed iframe.
# # ---------------------------------------------------------------------------

# def _add_map_controls(m: folium.Map, pred_col: str) -> None:
#     map_var = m.get_name()

#     js = f"""
# <script>
# (function() {{
#   function init() {{
#     var map = window['{map_var}'];
#     if (!map) {{ setTimeout(init, 150); return; }}
#     var container = map.getContainer();
#     var pane      = map.getPanes().overlayPane;
#     if (!container || !pane) {{ setTimeout(init, 150); return; }}
#     if (getComputedStyle(container).position === 'static')
#       container.style.position = 'relative';

#     /* wait for both image overlays to exist */
#     var attempts = 0;
#     function waitForLayers() {{
#       var imgs = pane.querySelectorAll('img.leaflet-image-layer');
#       if (imgs.length < 2 && attempts++ < 40) {{ setTimeout(waitForLayers, 200); return; }}
#       buildUI();
#     }}

#     /* basemap definitions */
#     var basemaps = {{
#       'OpenStreetMap': L.tileLayer(
#         'https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',
#         {{attribution:'&copy; OpenStreetMap contributors', maxZoom:19}}),
#       'Light': L.tileLayer(
#         'https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png',
#         {{attribution:'&copy; CartoDB', subdomains:'abcd', maxZoom:19}}),
#       'Satellite': L.tileLayer(
#         'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}',
#         {{attribution:'Esri World Imagery', maxZoom:19}}),
#       'None': null
#     }};
#     var activeBasemap = basemaps['OpenStreetMap'];

#     function buildUI() {{
#       /* ── swipe bar ─────────────────────────────────────────────────── */
#       var bar = document.createElement('div');
#       bar.style.cssText =
#         'position:absolute;top:0;bottom:0;left:50%;width:5px;' +
#         'transform:translateX(-50%);background:rgba(255,255,255,0.92);' +
#         'box-shadow:0 0 10px rgba(0,0,0,0.75);cursor:ew-resize;' +
#         'z-index:1000;pointer-events:auto;';
#       var handle = document.createElement('div');
#       handle.innerHTML = '&#8596;';
#       handle.style.cssText =
#         'position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);' +
#         'background:#fff;border:2px solid #444;border-radius:50%;' +
#         'width:34px;height:34px;line-height:30px;text-align:center;' +
#         'font-size:17px;font-weight:bold;' +
#         'box-shadow:0 2px 8px rgba(0,0,0,.5);user-select:none;pointer-events:none;';
#       bar.appendChild(handle);
#       container.appendChild(bar);

#       var swipeOn = true;
#       var pct     = 50;

#       function imgs() {{ return pane.querySelectorAll('img.leaflet-image-layer'); }}

#       function applyClip(p) {{
#         pct = p;
#         var ii = imgs(); if (ii.length < 2) return;
#         var c = 'inset(0 0 0 ' + p + '%)';
#         ii[1].style.clipPath = c; ii[1].style.webkitClipPath = c;
#       }}
#       function clearClip() {{
#         var ii = imgs(); if (ii.length < 2) return;
#         ii[1].style.clipPath = ''; ii[1].style.webkitClipPath = '';
#       }}

#       var drag = false;
#       function pctFromE(e) {{
#         var r = container.getBoundingClientRect();
#         var x = e.touches ? e.touches[0].clientX : e.clientX;
#         return Math.max(0, Math.min(100, (x - r.left) / r.width * 100));
#       }}
#       bar.addEventListener('mousedown',  function(e){{ drag=true; e.preventDefault(); e.stopPropagation(); }});
#       bar.addEventListener('touchstart', function(e){{ drag=true; e.stopPropagation(); }}, {{passive:true}});
#       document.addEventListener('mouseup',   function(){{ drag=false; }});
#       document.addEventListener('touchend',  function(){{ drag=false; }});
#       document.addEventListener('mousemove', function(e){{
#         if(!drag) return; var p=pctFromE(e); bar.style.left=p+'%'; applyClip(p);
#       }});
#       document.addEventListener('touchmove', function(e){{
#         if(!drag) return; var p=pctFromE(e); bar.style.left=p+'%'; applyClip(p);
#       }}, {{passive:true}});
#       map.on('moveend zoomend layeradd', function(){{
#         setTimeout(function(){{ if(swipeOn) applyClip(pct); }}, 60);
#       }});
#       applyClip(50);

#       /* ── floating control panel ────────────────────────────────────── */
#       var panel = document.createElement('div');
#       panel.style.cssText =
#         'position:absolute;top:12px;right:12px;z-index:1100;' +
#         'background:rgba(18,20,32,0.92);color:#e8eaf6;border-radius:10px;' +
#         'padding:14px 16px;font-family:Segoe UI,Arial,sans-serif;font-size:13px;' +
#         'box-shadow:0 4px 20px rgba(0,0,0,0.65);min-width:195px;' +
#         'backdrop-filter:blur(8px);border:1px solid rgba(255,255,255,0.08);';

#       /* helper: checkbox row */
#       function cbRow(label, checked, color, onChange) {{
#         var d  = document.createElement('div');
#         d.style.cssText = 'display:flex;align-items:center;gap:9px;margin-bottom:8px;cursor:pointer;';
#         var cb = document.createElement('input');
#         cb.type='checkbox'; cb.checked=checked;
#         cb.style.cssText = 'width:15px;height:15px;cursor:pointer;accent-color:'+color+';flex-shrink:0;';
#         cb.addEventListener('change', function(){{ onChange(cb.checked); }});
#         var sp = document.createElement('span');
#         sp.textContent = label; sp.style.userSelect='none';
#         d.appendChild(cb); d.appendChild(sp);
#         d.addEventListener('click', function(e){{
#           if(e.target!==cb){{ cb.checked=!cb.checked; cb.dispatchEvent(new Event('change')); }}
#         }});
#         return d;
#       }}

#       /* Section label helper */
#       function sectionLabel(text) {{
#         var d = document.createElement('div');
#         d.textContent = text;
#         d.style.cssText = 'font-size:10px;color:#7c84a8;letter-spacing:.08em;' +
#           'text-transform:uppercase;margin:6px 0 6px 0;font-weight:600;';
#         return d;
#       }}

#       /* Separator */
#       function sep() {{
#         var d = document.createElement('div');
#         d.style.cssText = 'border-top:1px solid rgba(255,255,255,0.1);margin:8px 0;';
#         return d;
#       }}

#       panel.appendChild(sectionLabel('Layers'));

#       /* Prediction toggle */
#       panel.appendChild(cbRow('Prediction', true, '#74c476', function(on){{
#         var ii = imgs(); if(!ii.length) return;
#         ii[0].style.opacity = on ? '1' : '0';
#       }}));

#       /* Uncertainty toggle */
#       panel.appendChild(cbRow('Uncertainty', true, '#fd8d3c', function(on){{
#         var ii = imgs(); if(ii.length<2) return;
#         ii[1].style.opacity = on ? '1' : '0';
#       }}));

#       panel.appendChild(sep());
#       panel.appendChild(sectionLabel('Swipe'));

#       /* Swipe toggle */
#       panel.appendChild(cbRow('Compare mode', true, '#9fa8d5', function(on){{
#         swipeOn = on;
#         bar.style.display = on ? '' : 'none';
#         if(on) applyClip(pct); else clearClip();
#       }}));

#       panel.appendChild(sep());
#       panel.appendChild(sectionLabel('Basemap'));

#       /* Basemap radio buttons */
#       ['OpenStreetMap','Light','Satellite','None'].forEach(function(name){{
#         var d  = document.createElement('div');
#         d.style.cssText = 'display:flex;align-items:center;gap:9px;margin-bottom:6px;cursor:pointer;';
#         var rb = document.createElement('input');
#         rb.type='radio'; rb.name='bm_{map_var}'; rb.value=name;
#         rb.checked = (name === 'OpenStreetMap');
#         rb.style.cssText = 'cursor:pointer;accent-color:#9fa8d5;flex-shrink:0;';
#         rb.addEventListener('change', function(){{
#           if(activeBasemap) map.removeLayer(activeBasemap);
#           activeBasemap = basemaps[name];
#           if(activeBasemap) activeBasemap.addTo(map);
#         }});
#         var sp = document.createElement('span');
#         sp.textContent = name; sp.style.userSelect='none';
#         d.appendChild(rb); d.appendChild(sp);
#         d.addEventListener('click', function(e){{
#           if(e.target!==rb){{ rb.checked=true; rb.dispatchEvent(new Event('change')); }}
#         }});
#         panel.appendChild(d);
#       }});

#       container.appendChild(panel);
#     }}

#     waitForLayers();
#   }}

#   if (document.readyState === 'loading')
#     document.addEventListener('DOMContentLoaded', init);
#   else
#     init();
# }})();
# </script>
# """
#     m.get_root().html.add_child(Element(js))


# # ---------------------------------------------------------------------------
# # Map builder
# # ---------------------------------------------------------------------------

# def _build_folium_overlay_map(
#     df: pd.DataFrame,
#     trait_name: str,
#     src_crs_override: str | None,
#     scene_tif_path: str | None = None,
#     output_pred_tif_path: str | None = None,
# ) -> str:
#     if "row" not in df.columns or "col" not in df.columns:
#         return "<h3>No raster indices (row/col) in output — raster inference required.</h3>"

#     pred_col = _resolve_pred_col(df, trait_name=trait_name)

#     pred_grid, unc_grid, bounds, has_geo = _resolve_grids_and_bounds(
#         df, pred_col, src_crs_override,
#         scene_tif_path=scene_tif_path,
#         output_pred_tif_path=output_pred_tif_path,
#     )

#     pred_rgba, pred_vmin, pred_vmax = _rgba_from_grid(pred_grid, "Greens",  is_uncertainty=False)
#     unc_rgba,  unc_vmin,  unc_vmax  = _rgba_from_grid(unc_grid,  "Oranges", is_uncertainty=True)

#     center = [(bounds[0][0] + bounds[1][0]) / 2.0,
#               (bounds[0][1] + bounds[1][1]) / 2.0]
#     m = folium.Map(
#         location=center,
#         zoom_start=12 if has_geo else 1,
#         tiles="OpenStreetMap",
#         control_scale=True,
#     )

#     # Prediction = bottom (zindex 3), Uncertainty always on top (zindex 4).
#     # mercator_project=False: grid already reprojected by rasterio — no further
#     # warping needed, and avoids the horizontal-stripe artefact caused by
#     # mercator_transform interpolating across NaN scan-line gaps.
#     folium.raster_layers.ImageOverlay(
#         image=pred_rgba, bounds=bounds, name="Prediction",
#         opacity=0.85, mercator_project=False,
#         interactive=True, cross_origin=False, zindex=3,
#     ).add_to(m)
#     folium.raster_layers.ImageOverlay(
#         image=unc_rgba, bounds=bounds, name="Uncertainty",
#         opacity=0.85, mercator_project=False,
#         interactive=True, cross_origin=False, zindex=4,
#     ).add_to(m)

#     # On-map floating panel: layer toggles + swipe bar + basemap switcher
#     _add_map_controls(m, pred_col=pred_col)

#     m.fit_bounds(bounds)
#     _add_colorbars(
#         m,
#         pred_vmin=pred_vmin, pred_vmax=pred_vmax,
#         unc_vmin=unc_vmin,   unc_vmax=unc_vmax,
#         pred_label=f"Prediction ({pred_col})",
#     )
#     return m._repr_html_()


# # ---------------------------------------------------------------------------
# # Artifact resolution
# # ---------------------------------------------------------------------------

# def _resolve_uncertainty_artifacts(cfg):
#     q_path = Path(str(cfg.inference.quantile_models_path))
#     d_path = Path(str(cfg.inference.distance_reference_npz))
#     if q_path.exists() and d_path.exists():
#         return str(q_path), str(d_path)

#     root = (
#         q_path.parents[1]
#         if len(q_path.parents) >= 2
#         else Path("results/experiments/study2_uncertainty_1522")
#     )
#     t1 = root / "target_1"
#     if (t1 / "quantile_models.pkl").exists() and (t1 / "distance_reference.npz").exists():
#         return str(t1 / "quantile_models.pkl"), str(t1 / "distance_reference.npz")

#     for c in sorted(root.glob("target_*")):
#         q, d = c / "quantile_models.pkl", c / "distance_reference.npz"
#         if q.exists() and d.exists():
#             return str(q), str(d)

#     raise FileNotFoundError(
#         f"Could not find uncertainty artifacts under {root}. "
#         "Run uncertainty stage=distance first."
#     )


# # ---------------------------------------------------------------------------
# # Progress tracking
# # ---------------------------------------------------------------------------

# # Maps substrings found in inference log messages to (progress_fraction, label).
# # Fractions are calibrated to real wall-clock proportions from observed runs.
# _LOG_PROGRESS = [
#     ("Starting scene inference",        0.02, "Initialising inference pipeline…"),
#     ("Loaded model",                    0.05, "Model loaded · preparing scene…"),
#     ("Loading TIFF scene",              0.07, "Loading hyperspectral image…"),
#     ("TIFF prepared",                   0.12, "Scene prepared · running trait prediction…"),
#     ("[predict] batch",                 None, None),   # handled per-batch below
#     ("Prediction complete",             0.46, "Traits predicted · extracting spectral embeddings…"),
#     ("Loaded quantile models",          0.48, "Quantile models loaded…"),
#     ("Loaded distance reference",       0.50, "Computing spectral distances  ⏳  (~1–2 min)…"),
#     ("[embedding] batch",               None, None),   # handled per-batch below
#     ("Embedding extraction complete",   0.72, "Embeddings done · computing uncertainty distances…"),
#     ("Distance features computed",      0.88, "Distances computed · predicting uncertainty…"),
#     ("Uncertainty prediction complete", 0.92, "Uncertainty ready · saving outputs…"),
#     ("Saved GeoTIFF",                   0.95, "GeoTIFFs saved · building map…"),
#     ("Saved scene prediction",          0.97, "Results saved · rendering visualisation…"),
# ]

# _BATCH_RE = re.compile(r"\[(predict|embedding)\] batch\s+(\d+)/(\d+)")

# # Phase windows for per-batch interpolation
# _BATCH_WINDOWS = {
#     "predict":   (0.12, 0.46),   # 34 % of total
#     "embedding": (0.50, 0.72),   # 22 % of total
# }


# class _LogQueueHandler(logging.Handler):
#     """Forwards every log record message to a queue.Queue."""
#     def __init__(self, q: queue.Queue):
#         super().__init__()
#         self.q = q

#     def emit(self, record: logging.LogRecord) -> None:
#         try:
#             self.q.put_nowait(record.getMessage())
#         except Exception:
#             pass


# def _parse_progress(msg: str) -> tuple[float, str] | None:
#     """
#     Return (fraction 0-1, label) for a recognised log message, or None.
#     Batch messages are interpolated smoothly within their phase window.
#     """
#     m = _BATCH_RE.search(msg)
#     if m:
#         phase = m.group(1)
#         n, total = int(m.group(2)), int(m.group(3))
#         lo, hi = _BATCH_WINDOWS.get(phase, (0, 1))
#         frac = lo + (n / total) * (hi - lo)
#         verb = "Predicting traits" if phase == "predict" else "Extracting embeddings"
#         return frac, f"{verb}  {n}/{total} batches…"

#     for substring, frac, label in _LOG_PROGRESS:
#         if frac is not None and substring in msg:
#             return frac, label

#     return None


# # ---------------------------------------------------------------------------
# # Core inference callback (non-generator; called from the generator wrapper)
# # ---------------------------------------------------------------------------

# def _build_results(
#     scene_csv_file,
#     scene_tif_file,
#     sensor_bands_file,
#     src_crs_override,
#     trait_name,
# ) -> tuple:
#     """Run full inference and return (overlay_html, trait_upd, pred_path, unc_path, cached)."""
#     tmp_out = Path(tempfile.mkdtemp()) / "scene_predictions_uncertainty.csv"
#     cfg = OmegaConf.load(
#         Path(__file__).resolve().parents[3]
#         / "configs" / "experiments" / "scene_infer_uncertainty.yaml"
#     )

#     if scene_tif_file is not None and sensor_bands_file is not None:
#         cfg.inference.input_type = "tif"
#         cfg.inference.scene_tif = scene_tif_file.name
#         cfg.inference.sensor_bands_csv = sensor_bands_file.name
#     elif scene_csv_file is not None:
#         cfg.inference.input_type = "csv"
#         cfg.inference.scene_csv = scene_csv_file.name
#     else:
#         raise ValueError("Provide either Scene CSV or (Scene TIFF + Sensor Bands CSV).")

#     cfg.inference.output_path      = str(tmp_out)
#     cfg.inference.output_pred_tif  = str(tmp_out.with_name(tmp_out.stem + "_predictions_20traits.tif"))
#     cfg.inference.output_unc_tif   = str(tmp_out.with_name(tmp_out.stem + "_uncertainty_20traits.tif"))

#     q_art, d_art = _resolve_uncertainty_artifacts(cfg)
#     cfg.inference.quantile_models_path   = q_art
#     cfg.inference.distance_reference_npz = d_art
#     run_inference(cfg)

#     df = pd.read_csv(tmp_out)
#     pred_cols     = [c for c in df.columns if c.startswith("pred_")]
#     trait_choices = [c.replace("pred_", "", 1) for c in pred_cols]
#     selected      = (str(trait_name) if str(trait_name) in trait_choices
#                      else (trait_choices[0] if trait_choices else ""))

#     crs        = str(src_crs_override).strip() or None
#     _scene_tif = str(scene_tif_file.name) if scene_tif_file is not None else None
#     _pred_tif  = str(tmp_out.with_name(tmp_out.stem + "_predictions_20traits.tif"))

#     overlay_html = _build_folium_overlay_map(
#         df,
#         trait_name=selected,
#         src_crs_override=crs,
#         scene_tif_path=_scene_tif,
#         output_pred_tif_path=_pred_tif,
#     )

#     pred_tif = Path(cfg.inference.output_pred_tif)
#     unc_tif  = Path(cfg.inference.output_unc_tif)
#     cached   = str(tmp_out) + "|" + (_scene_tif or "") + "|" + _pred_tif
#     return (
#         overlay_html,
#         gr.update(choices=trait_choices or DEFAULT_TRAIT_CHOICES, value=selected or None),
#         str(pred_tif) if pred_tif.exists() else None,
#         str(unc_tif)  if unc_tif.exists()  else None,
#         cached,
#     )


# def _refresh_map(cached_out_csv: str, src_crs_override: str, trait_name: str) -> str:
#     """Re-render map for a new trait without re-running inference."""
#     if not cached_out_csv:
#         return ""
#     parts          = str(cached_out_csv).split("|")
#     csv_path       = parts[0]
#     scene_tif_path = parts[1] if len(parts) > 1 and parts[1] else None
#     pred_tif_path  = parts[2] if len(parts) > 2 and parts[2] else None

#     out_path = Path(csv_path)
#     if not out_path.exists():
#         return f"<h3>Cached CSV not found: {out_path}</h3>"

#     df  = pd.read_csv(out_path)
#     crs = str(src_crs_override).strip() or None
#     return _build_folium_overlay_map(
#         df,
#         trait_name=str(trait_name),
#         src_crs_override=crs,
#         scene_tif_path=scene_tif_path,
#         output_pred_tif_path=pred_tif_path,
#     )


# # ---------------------------------------------------------------------------
# # Gradio UI  —  Biodiversity / Earth Observation theme
# # ---------------------------------------------------------------------------

# _CSS = """
# /* ════════════════════════════════════════════════════════════════════
#    ROOT & PAGE
#    ════════════════════════════════════════════════════════════════════ */
# :root {
#     --bio-bg:          #0b1a12;
#     --bio-surface:     #112318;
#     --bio-surface2:    #172e1e;
#     --bio-border:      #1f4028;
#     --bio-border2:     #2d5c3a;
#     --bio-green:       #3a9e5f;
#     --bio-green-light: #52c97a;
#     --bio-green-pale:  #a8e6bc;
#     --bio-amber:       #e8a94a;
#     --bio-amber-pale:  #f5d48e;
#     --bio-text:        #d4edda;
#     --bio-text-muted:  #7aad8a;
#     --bio-text-dim:    #4a7558;
#     --bio-radius:      10px;
#     --bio-radius-lg:   16px;
#     --bio-shadow:      0 4px 24px rgba(0,0,0,0.55);
# }

# /* full-page background */
# body, .gradio-container, gradio-app {
#     background: var(--bio-bg) !important;
#     font-family: 'Segoe UI', system-ui, -apple-system, sans-serif !important;
# }

# .gradio-container {
#     max-width: 1120px !important;
#     margin: 0 auto !important;
#     padding: 0 16px 40px !important;
# }

# /* ════════════════════════════════════════════════════════════════════
#    HEADER  (rendered as gr.HTML)
#    ════════════════════════════════════════════════════════════════════ */
# #app-header {
#     background: linear-gradient(135deg, #0e2a18 0%, #163522 60%, #1a3d28 100%);
#     border: 1px solid var(--bio-border2);
#     border-radius: var(--bio-radius-lg);
#     padding: 28px 32px 22px;
#     margin-bottom: 20px;
#     position: relative;
#     overflow: hidden;
# }
# #app-header::before {               /* subtle leaf-vein texture */
#     content: '';
#     position: absolute;
#     inset: 0;
#     background:
#         radial-gradient(ellipse 60% 40% at 90% 20%, rgba(58,158,95,0.08) 0%, transparent 70%),
#         radial-gradient(ellipse 40% 60% at 10% 80%, rgba(58,158,95,0.06) 0%, transparent 70%);
#     pointer-events: none;
# }
# #app-header h1 {
#     color: var(--bio-green-pale) !important;
#     font-size: 1.75rem !important;
#     font-weight: 700 !important;
#     margin: 0 0 6px !important;
#     letter-spacing: -0.02em;
# }
# #app-header p {
#     color: var(--bio-text-muted) !important;
#     font-size: 0.9rem !important;
#     margin: 0 !important;
#     line-height: 1.5;
# }
# #app-header .badge {
#     display: inline-block;
#     background: rgba(58,158,95,0.18);
#     border: 1px solid rgba(82,201,122,0.35);
#     color: var(--bio-green-light);
#     font-size: 0.72rem;
#     font-weight: 600;
#     letter-spacing: .06em;
#     text-transform: uppercase;
#     padding: 2px 9px;
#     border-radius: 20px;
#     margin-bottom: 10px;
# }

# /* ════════════════════════════════════════════════════════════════════
#    SECTION LABELS
#    ════════════════════════════════════════════════════════════════════ */
# .section-label {
#     color: var(--bio-text-dim) !important;
#     font-size: 0.7rem !important;
#     font-weight: 700 !important;
#     letter-spacing: .1em !important;
#     text-transform: uppercase !important;
#     margin: 0 0 10px !important;
#     display: flex;
#     align-items: center;
#     gap: 7px;
# }
# .section-label::before {
#     content: '';
#     display: inline-block;
#     width: 3px; height: 13px;
#     background: var(--bio-green);
#     border-radius: 2px;
# }

# /* ════════════════════════════════════════════════════════════════════
#    UPLOAD CARD
#    ════════════════════════════════════════════════════════════════════ */
# #upload-card {
#     background: var(--bio-surface) !important;
#     border: 1px solid var(--bio-border) !important;
#     border-radius: var(--bio-radius-lg) !important;
#     padding: 22px 24px 18px !important;
#     box-shadow: var(--bio-shadow) !important;
# }

# /* Gradio file-upload widget overrides */
# #upload-card .file-preview,
# #upload-card [data-testid="file-upload"] {
#     background: var(--bio-surface2) !important;
#     border: 1px dashed var(--bio-border2) !important;
#     border-radius: var(--bio-radius) !important;
#     color: var(--bio-text-muted) !important;
#     transition: border-color .2s, background .2s;
# }
# #upload-card [data-testid="file-upload"]:hover {
#     border-color: var(--bio-green) !important;
#     background: #1a3521 !important;
# }

# /* input labels */
# #upload-card label span,
# #upload-card .label-wrap span {
#     color: var(--bio-text-muted) !important;
#     font-size: 0.8rem !important;
#     font-weight: 600 !important;
# }

# /* textbox */
# #upload-card input[type=text], #upload-card textarea {
#     background: var(--bio-surface2) !important;
#     border: 1px solid var(--bio-border2) !important;
#     border-radius: var(--bio-radius) !important;
#     color: var(--bio-text) !important;
#     font-size: 0.85rem !important;
# }
# #upload-card input[type=text]:focus, #upload-card textarea:focus {
#     border-color: var(--bio-green) !important;
#     box-shadow: 0 0 0 3px rgba(58,158,95,0.18) !important;
#     outline: none !important;
# }

# /* ════════════════════════════════════════════════════════════════════
#    RUN BUTTON
#    ════════════════════════════════════════════════════════════════════ */
# #run-btn {
#     background: linear-gradient(135deg, #2d8a50 0%, #3a9e5f 100%) !important;
#     color: #fff !important;
#     font-size: 15px !important;
#     font-weight: 700 !important;
#     letter-spacing: .03em !important;
#     padding: 13px 0 !important;
#     border-radius: var(--bio-radius) !important;
#     border: none !important;
#     margin-top: 14px !important;
#     box-shadow: 0 4px 14px rgba(58,158,95,0.35) !important;
#     transition: all .2s !important;
# }
# #run-btn:hover {
#     background: linear-gradient(135deg, #3aad64 0%, #4dbd74 100%) !important;
#     box-shadow: 0 6px 20px rgba(58,158,95,0.50) !important;
#     transform: translateY(-1px) !important;
# }
# #run-btn:active { transform: translateY(0) !important; }

# /* ════════════════════════════════════════════════════════════════════
#    RESULTS PANEL
#    ════════════════════════════════════════════════════════════════════ */
# #results-panel {
#     margin-top: 22px;
# }

# /* trait row */
# #trait-row {
#     background: var(--bio-surface) !important;
#     border: 1px solid var(--bio-border) !important;
#     border-radius: var(--bio-radius-lg) !important;
#     padding: 14px 18px !important;
#     margin-bottom: 14px !important;
#     box-shadow: var(--bio-shadow) !important;
#     align-items: flex-end !important;
# }
# #trait-row label span,
# #trait-row .label-wrap span {
#     color: var(--bio-text-muted) !important;
#     font-size: 0.78rem !important;
#     font-weight: 600 !important;
#     text-transform: uppercase;
#     letter-spacing: .05em;
# }
# #trait-row select, #trait-row .wrap {
#     background: var(--bio-surface2) !important;
#     border: 1px solid var(--bio-border2) !important;
#     border-radius: var(--bio-radius) !important;
#     color: var(--bio-text) !important;
# }

# /* refresh button */
# #refresh-btn {
#     background: var(--bio-surface2) !important;
#     color: var(--bio-green-light) !important;
#     border: 1px solid var(--bio-border2) !important;
#     border-radius: var(--bio-radius) !important;
#     font-weight: 600 !important;
#     font-size: 13px !important;
#     transition: all .2s !important;
# }
# #refresh-btn:hover {
#     background: #1f3d28 !important;
#     border-color: var(--bio-green) !important;
# }

# /* map container */
# #map-output {
#     border-radius: var(--bio-radius-lg) !important;
#     overflow: hidden !important;
#     border: 1px solid var(--bio-border2) !important;
#     box-shadow: var(--bio-shadow) !important;
# }
# #map-output iframe, #map-output > div {
#     width: 100% !important;
#     border-radius: var(--bio-radius-lg) !important;
# }

# /* ════════════════════════════════════════════════════════════════════
#    DOWNLOAD ROW
#    ════════════════════════════════════════════════════════════════════ */
# #dl-row {
#     margin-top: 14px !important;
# }
# #dl-row .file-preview,
# #dl-row [data-testid="file-upload"] {
#     background: var(--bio-surface) !important;
#     border: 1px solid var(--bio-border) !important;
#     border-radius: var(--bio-radius) !important;
#     color: var(--bio-text-muted) !important;
# }
# #dl-row label span {
#     color: var(--bio-text-muted) !important;
#     font-size: 0.78rem !important;
#     font-weight: 600 !important;
# }

# /* ════════════════════════════════════════════════════════════════════
#    PROGRESS BAR
#    ════════════════════════════════════════════════════════════════════ */
# .progress-bar-wrap .progress-bar { background: var(--bio-green) !important; }
# .progress-bar-wrap { background: var(--bio-border) !important; border-radius: 4px !important; }

# /* ════════════════════════════════════════════════════════════════════
#    SCROLLBAR
#    ════════════════════════════════════════════════════════════════════ */
# ::-webkit-scrollbar { width: 6px; height: 6px; }
# ::-webkit-scrollbar-track { background: var(--bio-bg); }
# ::-webkit-scrollbar-thumb { background: var(--bio-border2); border-radius: 3px; }
# ::-webkit-scrollbar-thumb:hover { background: var(--bio-green); }
# """

# _HEADER_HTML = """
# <div id="app-header">
#   <div class="badge">🛰 Hyperspectral · Biodiversity · Earth Observation</div>
#   <h1>BioDis-UN &nbsp;·&nbsp; Plant Trait Mapping</h1>
#   <p>
#     Airborne hyperspectral scene inference with calibrated uncertainty estimation.
#     Upload a raw HSI scene TIFF and sensor band list to retrieve spatially explicit
#     plant functional traits and their prediction confidence across the landscape.
#   </p>
# </div>
# """


# def launch() -> None:
#     with gr.Blocks(
#         title="BioDis-UN — Plant Trait Mapping",
#         css=_CSS,
#         theme=gr.themes.Base(),
#     ) as demo:

#         # ── App header ────────────────────────────────────────────────────
#         gr.HTML(_HEADER_HTML)

#         # ── STATE ─────────────────────────────────────────────────────────
#         cached_out_csv = gr.State(value="")

#         # ═════════════════════════════════════════════════════════════════
#         # PHASE 1  —  always visible: data upload
#         # ═════════════════════════════════════════════════════════════════
#         gr.HTML('<p class="section-label">Scene Data Input</p>')

#         with gr.Group(elem_id="upload-card"):
#             with gr.Row():
#                 scene_tif_file = gr.File(
#                     label="Scene HSI TIFF  (.tif / .tiff)",
#                     file_types=[".tif", ".tiff"],
#                     scale=2,
#                 )
#                 sensor_bands_file = gr.File(
#                     label="Sensor Bands CSV  (wavelength list)",
#                     file_types=[".csv"],
#                     scale=1,
#                 )
#             with gr.Row():
#                 scene_csv_file = gr.File(
#                     label="Scene CSV  (alternative to TIFF — 1522 spectra)",
#                     file_types=[".csv"],
#                     scale=2,
#                 )
#                 src_crs_override = gr.Textbox(
#                     label="CRS override  (optional — e.g. EPSG:32633)",
#                     value="",
#                     placeholder="Auto-detected from TIFF metadata",
#                     scale=1,
#                 )

#         run_btn = gr.Button(
#             "▶  Run Scene Inference",
#             variant="primary",
#             elem_id="run-btn",
#         )

#         # ═════════════════════════════════════════════════════════════════
#         # PHASE 2  —  hidden until inference completes
#         # ═════════════════════════════════════════════════════════════════
#         with gr.Group(visible=False, elem_id="results-panel") as results_panel:

#             gr.HTML('<p class="section-label" style="margin-top:8px;">Trait Visualisation</p>')

#             with gr.Row(elem_id="trait-row"):
#                 trait_name = gr.Dropdown(
#                     label="Plant Functional Trait",
#                     choices=DEFAULT_TRAIT_CHOICES,
#                     value="EWT_mg_cm2",
#                     allow_custom_value=False,
#                     scale=5,
#                 )
#                 refresh_btn = gr.Button(
#                     "↻  Refresh map",
#                     elem_id="refresh-btn",
#                     scale=1,
#                 )

#             # Map — layer controls, swipe bar and basemap switcher are
#             # all rendered inside the Folium HTML (see _add_map_controls)
#             overlay_map = gr.HTML(elem_id="map-output")

#             # Downloads — revealed only when GeoTIFFs are written
#             gr.HTML('<p class="section-label" style="margin-top:6px;">Export Results</p>')
#             with gr.Row(elem_id="dl-row", visible=False) as dl_row:
#                 pred_tif_file = gr.File(
#                     label="⬇  Prediction GeoTIFF  (all 20 traits)",
#                     interactive=False,
#                 )
#                 unc_tif_file = gr.File(
#                     label="⬇  Uncertainty GeoTIFF  (all 20 traits)",
#                     interactive=False,
#                 )

#         # ── WIRING ────────────────────────────────────────────────────────

#         def _run_and_reveal(scene_csv, scene_tif, sensor_bands, src_crs, trait):
#             """
#             Generator — yields intermediate progress updates to Gradio while
#             running _build_results() in a background thread.

#             Strategy
#             --------
#             • Attach a QueueHandler to the root logger before starting the
#               inference thread so every log record emitted by run_inference
#               (predictions, embeddings, distance calc) is captured.
#             • Poll the queue in a tight loop, parse each message with
#               _parse_progress(), and yield a gr.Progress update + a status
#               HTML snippet so the user sees a live progress bar AND a text
#               description of what is happening.
#             • The distance-calculation phase (the 60-second silent gap) is
#               covered by the "Computing spectral distances ⏳" label at 50 %
#               and the "Distances computed" label at 88 % — the bar will stay
#               at 50 % during that phase but the label tells the user why.
#             • After the thread finishes, yield the final map outputs and
#               reveal the results panel.
#             """
#             # ── result container shared with the thread ────────────────
#             result_box: list = [None]
#             error_box:  list = [None]

#             # ── log capture ───────────────────────────────────────────
#             log_q: queue.Queue = queue.Queue()
#             handler = _LogQueueHandler(log_q)
#             root_logger = logging.getLogger()
#             root_logger.addHandler(handler)

#             # ── blank outputs for intermediate yields ─────────────────
#             # (keeps Gradio happy — all 7 outputs must be present every yield)
#             _BLANK = (
#                 "",           # overlay_map
#                 gr.update(),  # trait_name
#                 None,         # pred_tif_file
#                 None,         # unc_tif_file
#                 "",           # cached_out_csv
#                 gr.update(visible=False),  # results_panel
#                 gr.update(visible=False),  # dl_row
#             )

#             def _worker():
#                 try:
#                     result_box[0] = _build_results(
#                         scene_csv, scene_tif, sensor_bands, src_crs, trait
#                     )
#                 except Exception as exc:
#                     error_box[0] = exc

#             thread = threading.Thread(target=_worker, daemon=True)
#             thread.start()

#             # ── live progress loop ────────────────────────────────────
#             last_frac  = 0.0
#             last_label = "Starting inference pipeline…"

#             def _status_html(frac: float, label: str) -> str:
#                 pct = int(frac * 100)
#                 # Green fill + animated shimmer on the active portion
#                 return f"""
# <div style="margin:6px 0 2px;font-family:Segoe UI,sans-serif;">
#   <div style="display:flex;justify-content:space-between;
#               font-size:12px;color:#7aad8a;margin-bottom:5px;">
#     <span>{label}</span>
#     <span style="color:#3a9e5f;font-weight:700;">{pct}%</span>
#   </div>
#   <div style="height:8px;background:#1f4028;border-radius:4px;overflow:hidden;">
#     <div style="
#       height:100%;width:{pct}%;background:linear-gradient(90deg,#2d8a50,#52c97a);
#       border-radius:4px;transition:width .4s ease;
#       box-shadow:0 0 8px rgba(82,201,122,0.5);
#       position:relative;overflow:hidden;">
#       <div style="
#         position:absolute;inset:0;
#         background:linear-gradient(90deg,transparent 0%,rgba(255,255,255,0.25) 50%,transparent 100%);
#         animation:shimmer 1.4s infinite;
#         background-size:200% 100%;"></div>
#     </div>
#   </div>
# </div>
# <style>
# @keyframes shimmer{{from{{background-position:200% 0}}to{{background-position:-200% 0}}}}
# </style>
# """

#             while thread.is_alive():
#                 # drain everything that arrived since last poll
#                 drained = False
#                 while True:
#                     try:
#                         msg = log_q.get_nowait()
#                         parsed = _parse_progress(msg)
#                         if parsed:
#                             frac, label = parsed
#                             if frac > last_frac:
#                                 last_frac, last_label = frac, label
#                                 drained = True
#                     except queue.Empty:
#                         break

#                 if drained:
#                     yield (
#                         _status_html(last_frac, last_label),
#                         *_BLANK[1:],
#                     )

#                 thread.join(timeout=0.25)

#             # ── drain any final messages ──────────────────────────────
#             while not log_q.empty():
#                 try:
#                     msg = log_q.get_nowait()
#                     parsed = _parse_progress(msg)
#                     if parsed:
#                         frac, label = parsed
#                         if frac > last_frac:
#                             last_frac, last_label = frac, label
#                 except queue.Empty:
#                     break

#             root_logger.removeHandler(handler)

#             # ── propagate errors ──────────────────────────────────────
#             if error_box[0] is not None:
#                 raise error_box[0]

#             # ── final map-building step ───────────────────────────────
#             yield (
#                 _status_html(0.99, "Rendering map…"),
#                 *_BLANK[1:],
#             )

#             overlay_html, trait_upd, pred_path, unc_path, cached = result_box[0]

#             # ── reveal results ────────────────────────────────────────
#             yield (
#                 overlay_html,
#                 trait_upd,
#                 pred_path,
#                 unc_path,
#                 cached,
#                 gr.update(visible=True),
#                 gr.update(visible=bool(pred_path or unc_path)),
#             )

#         run_btn.click(
#             _run_and_reveal,
#             inputs=[scene_csv_file, scene_tif_file, sensor_bands_file,
#                     src_crs_override, trait_name],
#             outputs=[overlay_map, trait_name, pred_tif_file, unc_tif_file,
#                      cached_out_csv, results_panel, dl_row],
#         )

#         def _do_refresh(cached, src_crs, trait):
#             return _refresh_map(cached, src_crs, trait)

#         refresh_btn.click(
#             _do_refresh,
#             inputs=[cached_out_csv, src_crs_override, trait_name],
#             outputs=[overlay_map],
#         )

#         trait_name.change(
#             _do_refresh,
#             inputs=[cached_out_csv, src_crs_override, trait_name],
#             outputs=[overlay_map],
#         )

#     demo.launch(server_name="0.0.0.0", server_port=7862)


# if __name__ == "__main__":
#     launch()

#################################

from __future__ import annotations

import logging
import queue
import re
import tempfile
import threading
from pathlib import Path

import gradio as gr
import matplotlib as mpl
import numpy as np
import pandas as pd
import folium
from branca.element import Element, MacroElement, Template

from omegaconf import OmegaConf

from ..experiments.uncertainty_scene_infer import run_inference


DEFAULT_TRAIT_CHOICES = [
    "Anth_area_ug_cm2",
    "Boron_area_mg_cm2",
    "C_area_mg_cm2",
    "Ca_area_mg_cm2",
    "Car_area_ug_cm2",
    "Cellulose_mg_cm2",
    "Chl_area_ug_cm2",
    "Cu_area_mg_cm2",
    "EWT_mg_cm2",
    "Fiber_mg_cm2",
    "LAI_m2_m2",
    "LMA_g_m2",
    "Lignin_mg_cm2",
    "Mg_area_mg_cm2",
    "Mn_area_mg_cm2",
    "NSC_mg_cm2",
    "N_area_mg_cm2",
    "P_area_mg_cm2",
    "Potassium_area_mg_cm2",
    "S_area_mg_cm2",
]


# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------

def _resolve_pred_col(df: pd.DataFrame, trait_name: str) -> str:
    pred_cols = [c for c in df.columns if c.startswith("pred_")]
    if not pred_cols:
        raise ValueError("No prediction columns found in output.")
    if trait_name:
        if trait_name in pred_cols:
            return trait_name
        wanted = f"pred_{trait_name}"
        if wanted in pred_cols:
            return wanted
    return pred_cols[0]


def _grid_from_df(df: pd.DataFrame, col: str) -> np.ndarray:
    h = int(df["row"].max()) + 1
    w = int(df["col"].max()) + 1
    grid = np.full((h, w), np.nan, dtype=np.float32)
    grid[df["row"].to_numpy(int), df["col"].to_numpy(int)] = df[col].to_numpy(np.float32)
    return grid


# ---------------------------------------------------------------------------
# Colormap stretch
# ---------------------------------------------------------------------------

PRED_THRESHOLD = 0.05   # prediction: vmin=q05, vmax=q95
UNC_THRESHOLD  = 0.90   # uncertainty: vmin=q10, vmax=q90  (matches original style)


def _pred_norm_bounds(arr: np.ndarray) -> tuple[float, float]:
    """q05–q95 stretch for prediction."""
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin = float(np.nanpercentile(finite, PRED_THRESHOLD * 100))
    vmax = float(np.nanpercentile(finite, (1 - PRED_THRESHOLD) * 100))
    return vmin, vmax if vmax > vmin else vmin + 1e-6


def _unc_norm_bounds(arr: np.ndarray) -> tuple[float, float]:
    """
    Uncertainty stretch matching original project formula:
        maxv = quantile(thre)        ← LOW quantile  → colormap bright end
        minv = quantile(1 - thre)    ← HIGH quantile → colormap dark end
    High-uncertainty pixels dominate visually (dark/saturated).
    """
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin = float(np.nanpercentile(finite, (1 - UNC_THRESHOLD) * 100))
    vmax = float(np.nanpercentile(finite, UNC_THRESHOLD * 100))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    return vmin, vmax


def _rgba_from_grid(
    grid: np.ndarray,
    cmap_name: str,
    is_uncertainty: bool = False,
) -> tuple[np.ndarray, float, float]:
    """Return (RGBA uint8 H×W×4, vmin, vmax)."""
    vmin, vmax = _unc_norm_bounds(grid) if is_uncertainty else _pred_norm_bounds(grid)
    norm = np.clip((grid - vmin) / (vmax - vmin), 0.0, 1.0)
    cmap = mpl.colormaps.get_cmap(cmap_name)
    rgba = (cmap(norm) * 255.0).astype(np.uint8)
    rgba[..., 3] = np.where(np.isfinite(grid), 210, 0).astype(np.uint8)
    return rgba, vmin, vmax


def _fmt(v: float) -> str:
    if v == 0.0:
        return "0"
    return f"{v:.2e}" if (abs(v) >= 10_000 or abs(v) < 0.001) else f"{v:.4g}"


# ---------------------------------------------------------------------------
# Projection: warp grid to WGS-84 using rasterio
# Uses WKT strings throughout — never EPSG integer lookups — so a
# broken/outdated PROJ database (conda env mismatch) does not crash.
# ---------------------------------------------------------------------------

def _warp_grid_to_wgs84(
    grid: np.ndarray,
    src_transform,
    src_crs_wkt: str,
) -> tuple[np.ndarray, list]:
    """
    Reproject a 2-D float32 grid from src_crs_wkt to WGS-84.

    Returns
    -------
    warped_grid : np.ndarray  (float32, NaN for nodata)
    bounds      : [[south, west], [north, east]]  in degrees
    """
    import rasterio
    from rasterio.crs import CRS
    from rasterio.warp import reproject, Resampling, calculate_default_transform
    from rasterio.env import Env

    h, w = grid.shape

    WGS84_WKT = (
        'GEOGCS["WGS 84",DATUM["WGS_1984",'
        'SPHEROID["WGS 84",6378137,298.257223563]],'
        'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433],'
        'AUTHORITY["EPSG","4326"]]'
    )

    with Env(GTIFF_SRS_SOURCE="GEOKEYS", PROJ_NETWORK="OFF"):
        src_crs = CRS.from_wkt(src_crs_wkt)
        dst_crs = CRS.from_wkt(WGS84_WKT)

        left   = src_transform.c
        top    = src_transform.f
        right  = left + w * src_transform.a
        bottom = top  + h * src_transform.e   # e is negative

        dst_transform, dst_w, dst_h = calculate_default_transform(
            src_crs, dst_crs, w, h,
            left=left, bottom=bottom, right=right, top=top,
        )

        NODATA = -9999.0
        src_data = np.where(np.isnan(grid), NODATA, grid).astype(np.float32)
        dst_data = np.full((dst_h, dst_w), NODATA, dtype=np.float32)

        reproject(
            source=src_data, destination=dst_data,
            src_transform=src_transform, src_crs=src_crs,
            dst_transform=dst_transform, dst_crs=dst_crs,
            src_nodata=NODATA, dst_nodata=NODATA,
            resampling=Resampling.bilinear,
        )

    warped = np.where(dst_data == NODATA, np.nan, dst_data)
    west  = dst_transform.c
    north = dst_transform.f
    east  = west  + dst_w * dst_transform.a
    south = north + dst_h * dst_transform.e

    return warped, [[float(south), float(west)], [float(north), float(east)]]


def _resolve_grids_and_bounds(
    df: pd.DataFrame,
    pred_col: str,
    src_crs_override: str | None,
    scene_tif_path: str | None = None,
    output_pred_tif_path: str | None = None,
) -> tuple[np.ndarray, np.ndarray, list, bool]:
    """
    Return (pred_grid_wgs84, unc_grid_wgs84, bounds, has_geo).

    Priority:
      1. x/y columns  — projected pixel coords always written by inference.
         Affine derived via polyfit; WKT read from any available tif.
      2. scene_tif_path (INPUT tif) — definitive CRS + geotransform.
      3. output_pred_tif_path / pred_tif_path column — secondary fallbacks.
      4. lat/lon columns — assumed WGS-84.
      5. Pixel-space fallback.
    """
    import rasterio
    from rasterio.crs import CRS
    from rasterio.env import Env
    from rasterio.transform import Affine

    pred_grid = _grid_from_df(df, pred_col)

    # Per-trait uncertainty column: prefer un_<trait>, fall back to uncertainty_mean
    _trait = pred_col.replace("pred_", "", 1)
    _unc_col = f"un_{_trait}" if f"un_{_trait}" in df.columns else "uncertainty_mean"
    unc_grid = _grid_from_df(df, _unc_col)
    print(f"[uncertainty] using column: {_unc_col}")

    h, w = pred_grid.shape

    def _wkt_from_tif(tif_path: str) -> tuple[str, object]:
        """Read (wkt, affine) from a tif using GEOKEYS — bypasses EPSG registry."""
        with Env(GTIFF_SRS_SOURCE="GEOKEYS", PROJ_NETWORK="OFF"):
            with rasterio.open(tif_path) as src:
                xform = src.transform
                if src_crs_override and src_crs_override.strip():
                    wkt = CRS.from_user_input(src_crs_override.strip()).to_wkt()
                elif src.crs is not None:
                    wkt = src.crs.to_wkt()
                else:
                    raise ValueError(f"No CRS in tif and no override: {tif_path}")
        return wkt, xform

    def _try_tif(tif_path: str, label: str):
        try:
            wkt, xform = _wkt_from_tif(tif_path)
            pw, bounds = _warp_grid_to_wgs84(pred_grid, xform, wkt)
            uw, _      = _warp_grid_to_wgs84(unc_grid,  xform, wkt)
            print(f"[projection] OK via {label}: bounds={bounds}")
            return pw, uw, bounds, True
        except Exception as exc:
            print(f"[projection] {label} failed: {exc}")
            return None

    # ── 1. x/y columns ────────────────────────────────────────────────────
    if "x" in df.columns and "y" in df.columns:
        try:
            x_s = pd.to_numeric(df["x"], errors="coerce")
            y_s = pd.to_numeric(df["y"], errors="coerce")
            valid = np.isfinite(x_s) & np.isfinite(y_s) & \
                    np.isfinite(df["row"]) & np.isfinite(df["col"])
            if valid.sum() > 10:
                x_v = x_s[valid].to_numpy()
                y_v = y_s[valid].to_numpy()
                row_v = df.loc[valid, "row"].to_numpy(int)
                col_v = df.loc[valid, "col"].to_numpy(int)

                dx = np.polyfit(col_v, x_v, 1)[0]
                dy = np.polyfit(row_v, y_v, 1)[0]
                x0 = float(np.median(x_v - col_v * dx))
                y0 = float(np.median(y_v - row_v * dy))
                xform = Affine(abs(dx), 0.0, x0, 0.0, -abs(dy), y0)

                # Get WKT from any available tif (GEOKEYS, no EPSG lookup)
                wkt = None
                for cand in [scene_tif_path, output_pred_tif_path]:
                    if cand and Path(cand).exists():
                        try:
                            wkt, _ = _wkt_from_tif(cand)
                            break
                        except Exception:
                            pass
                if wkt is None and "pred_tif_path" in df.columns:
                    vals = df["pred_tif_path"].dropna().astype(str)
                    if len(vals) > 0 and Path(vals.iloc[0]).exists():
                        try:
                            wkt, _ = _wkt_from_tif(vals.iloc[0])
                        except Exception:
                            pass
                if src_crs_override and src_crs_override.strip():
                    with Env(GTIFF_SRS_SOURCE="GEOKEYS", PROJ_NETWORK="OFF"):
                        wkt = CRS.from_user_input(src_crs_override.strip()).to_wkt()
                if wkt is None:
                    raise ValueError("No CRS WKT obtainable from any tif file")

                pw, bounds = _warp_grid_to_wgs84(pred_grid, xform, wkt)
                uw, _      = _warp_grid_to_wgs84(unc_grid,  xform, wkt)
                print(f"[projection] OK via x/y cols: dx={dx:.2f}m bounds={bounds}")
                return pw, uw, bounds, True
        except Exception as exc:
            print(f"[projection] x/y columns failed: {exc}")

    # ── 2. Input scene tif ────────────────────────────────────────────────
    if scene_tif_path and Path(scene_tif_path).exists():
        r = _try_tif(scene_tif_path, "scene_tif (input)")
        if r:
            return r

    # ── 3. Output pred tif / pred_tif_path column ────────────────────────
    if output_pred_tif_path and Path(output_pred_tif_path).exists():
        r = _try_tif(output_pred_tif_path, "output_pred_tif")
        if r:
            return r
    if "pred_tif_path" in df.columns:
        vals = df["pred_tif_path"].dropna().astype(str)
        if len(vals) > 0 and Path(vals.iloc[0]).exists():
            r = _try_tif(vals.iloc[0], "pred_tif_path column")
            if r:
                return r

    # ── 4. lat/lon columns (WGS-84 assumed) ──────────────────────────────
    if "lat" in df.columns and "lon" in df.columns:
        try:
            lat_v = pd.to_numeric(df["lat"], errors="coerce").dropna().to_numpy()
            lon_v = pd.to_numeric(df["lon"], errors="coerce").dropna().to_numpy()
            lat_v = lat_v[np.isfinite(lat_v)]
            lon_v = lon_v[np.isfinite(lon_v)]
            if len(lat_v) > 4:
                bounds = [[float(lat_v.min()), float(lon_v.min())],
                          [float(lat_v.max()), float(lon_v.max())]]
                print(f"[projection] OK via lat/lon: bounds={bounds}")
                return pred_grid, unc_grid, bounds, True
        except Exception as exc:
            print(f"[projection] lat/lon failed: {exc}")

    # ── 5. Pixel-space fallback ──────────────────────────────────────────
    print("[projection] WARNING: fallback to pixel space")
    print(f"[projection] df columns: {list(df.columns)}")
    return pred_grid, unc_grid, [[0.0, 0.0], [float(h), float(w)]], False


# ---------------------------------------------------------------------------
# Colorbar legend  (values baked via f-string — Branca this.attr is unreliable)
# ---------------------------------------------------------------------------

def _add_colorbars(
    m: folium.Map,
    pred_vmin: float, pred_vmax: float,
    unc_vmin: float,  unc_vmax: float,
    pred_label: str = "Prediction",
) -> None:
    template = f"""
    {{% macro html(this, kwargs) %}}
    <div style="position:fixed;bottom:40px;left:16px;z-index:9999;
                background:rgba(255,255,255,0.93);padding:8px 12px;
                border:1px solid #888;border-radius:7px;font-size:12px;
                font-family:monospace;box-shadow:2px 2px 6px rgba(0,0,0,0.2);
                min-width:220px;pointer-events:none;">
      <div style="font-weight:700;margin-bottom:4px;white-space:nowrap;
                  overflow:hidden;text-overflow:ellipsis;max-width:220px;">
        {pred_label}
      </div>
      <div style="width:210px;height:14px;
                  background:linear-gradient(to right,#f7fcf5,#74c476,#00441b);
                  border-radius:3px;margin-bottom:3px;"></div>
      <div style="display:flex;justify-content:space-between;color:#333;">
        <span>{_fmt(pred_vmin)}</span>
        <span style="color:#888;font-size:10px;">q05–q95</span>
        <span>{_fmt(pred_vmax)}</span>
      </div>
    </div>
    <div style="position:fixed;bottom:40px;right:220px;z-index:9999;
                background:rgba(255,255,255,0.93);padding:8px 12px;
                border:1px solid #888;border-radius:7px;font-size:12px;
                font-family:monospace;box-shadow:2px 2px 6px rgba(0,0,0,0.2);
                min-width:220px;pointer-events:none;">
      <div style="font-weight:700;margin-bottom:4px;">Uncertainty</div>
      <div style="width:210px;height:14px;
                  background:linear-gradient(to right,#fff5eb,#fdae6b,#a63603);
                  border-radius:3px;margin-bottom:3px;"></div>
      <div style="display:flex;justify-content:space-between;color:#333;">
        <span>{_fmt(unc_vmin)}</span>
        <span style="color:#888;font-size:10px;">q10–q90</span>
        <span>{_fmt(unc_vmax)}</span>
      </div>
    </div>
    {{% endmacro %}}
    """
    el = MacroElement()
    el._template = Template(template)
    el.add_to(m)


# ---------------------------------------------------------------------------
# On-map control panel + swipe bar (all inline JS, no CDN)
#
# The floating panel (top-right) contains:
#   • Prediction layer toggle  (checkbox)
#   • Uncertainty layer toggle (checkbox) — uncertainty always on top (zindex 4)
#   • Swipe compare toggle     (checkbox) — shows/hides the divider bar
#   • Basemap switcher         (radio: OSM / Light / Satellite / None)
#
# The swipe bar is always in the DOM; the toggle just shows/hides it.
# No external CDN scripts are loaded — works inside Gradio's sandboxed iframe.
# ---------------------------------------------------------------------------

def _add_map_controls(m: folium.Map, pred_col: str) -> None:
    map_var = m.get_name()

    js = f"""
<script>
(function() {{
  function init() {{
    var map = window['{map_var}'];
    if (!map) {{ setTimeout(init, 150); return; }}
    var container = map.getContainer();
    var pane      = map.getPanes().overlayPane;
    if (!container || !pane) {{ setTimeout(init, 150); return; }}
    if (getComputedStyle(container).position === 'static')
      container.style.position = 'relative';

    /* wait for both image overlays to exist */
    var attempts = 0;
    function waitForLayers() {{
      var imgs = pane.querySelectorAll('img.leaflet-image-layer');
      if (imgs.length < 2 && attempts++ < 40) {{ setTimeout(waitForLayers, 200); return; }}
      buildUI();
    }}

    /* basemap definitions */
    var basemaps = {{
      'OpenStreetMap': L.tileLayer(
        'https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',
        {{attribution:'&copy; OpenStreetMap contributors', maxZoom:19}}),
      'Light': L.tileLayer(
        'https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png',
        {{attribution:'&copy; CartoDB', subdomains:'abcd', maxZoom:19}}),
      'Satellite': L.tileLayer(
        'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}',
        {{attribution:'Esri World Imagery', maxZoom:19}}),
      'None': null
    }};
    var activeBasemap = basemaps['OpenStreetMap'];

    function buildUI() {{
      /* ── swipe bar ─────────────────────────────────────────────────── */
      var bar = document.createElement('div');
      bar.style.cssText =
        'position:absolute;top:0;bottom:0;left:50%;width:5px;' +
        'transform:translateX(-50%);background:rgba(255,255,255,0.92);' +
        'box-shadow:0 0 10px rgba(0,0,0,0.75);cursor:ew-resize;' +
        'z-index:1000;pointer-events:auto;';
      var handle = document.createElement('div');
      handle.innerHTML = '&#8596;';
      handle.style.cssText =
        'position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);' +
        'background:#fff;border:2px solid #444;border-radius:50%;' +
        'width:34px;height:34px;line-height:30px;text-align:center;' +
        'font-size:17px;font-weight:bold;' +
        'box-shadow:0 2px 8px rgba(0,0,0,.5);user-select:none;pointer-events:none;';
      bar.appendChild(handle);
      container.appendChild(bar);

      var swipeOn = true;
      var pct     = 50;

      function imgs() {{ return pane.querySelectorAll('img.leaflet-image-layer'); }}

      function applyClip(p) {{
        pct = p;
        var ii = imgs(); if (ii.length < 2) return;
        var c = 'inset(0 0 0 ' + p + '%)';
        ii[1].style.clipPath = c; ii[1].style.webkitClipPath = c;
      }}
      function clearClip() {{
        var ii = imgs(); if (ii.length < 2) return;
        ii[1].style.clipPath = ''; ii[1].style.webkitClipPath = '';
      }}

      var drag = false;
      function pctFromE(e) {{
        var r = container.getBoundingClientRect();
        var x = e.touches ? e.touches[0].clientX : e.clientX;
        return Math.max(0, Math.min(100, (x - r.left) / r.width * 100));
      }}
      bar.addEventListener('mousedown',  function(e){{ drag=true; e.preventDefault(); e.stopPropagation(); }});
      bar.addEventListener('touchstart', function(e){{ drag=true; e.stopPropagation(); }}, {{passive:true}});
      document.addEventListener('mouseup',   function(){{ drag=false; }});
      document.addEventListener('touchend',  function(){{ drag=false; }});
      document.addEventListener('mousemove', function(e){{
        if(!drag) return; var p=pctFromE(e); bar.style.left=p+'%'; applyClip(p);
      }});
      document.addEventListener('touchmove', function(e){{
        if(!drag) return; var p=pctFromE(e); bar.style.left=p+'%'; applyClip(p);
      }}, {{passive:true}});
      map.on('moveend zoomend layeradd', function(){{
        setTimeout(function(){{ if(swipeOn) applyClip(pct); }}, 60);
      }});
      applyClip(50);

      /* ── floating control panel ────────────────────────────────────── */
      var panel = document.createElement('div');
      panel.style.cssText =
        'position:absolute;top:12px;right:12px;z-index:1100;' +
        'background:rgba(18,20,32,0.92);color:#e8eaf6;border-radius:10px;' +
        'padding:14px 16px;font-family:Segoe UI,Arial,sans-serif;font-size:13px;' +
        'box-shadow:0 4px 20px rgba(0,0,0,0.65);min-width:195px;' +
        'backdrop-filter:blur(8px);border:1px solid rgba(255,255,255,0.08);';

      /* helper: checkbox row */
      function cbRow(label, checked, color, onChange) {{
        var d  = document.createElement('div');
        d.style.cssText = 'display:flex;align-items:center;gap:9px;margin-bottom:8px;cursor:pointer;';
        var cb = document.createElement('input');
        cb.type='checkbox'; cb.checked=checked;
        cb.style.cssText = 'width:15px;height:15px;cursor:pointer;accent-color:'+color+';flex-shrink:0;';
        cb.addEventListener('change', function(){{ onChange(cb.checked); }});
        var sp = document.createElement('span');
        sp.textContent = label; sp.style.userSelect='none';
        d.appendChild(cb); d.appendChild(sp);
        d.addEventListener('click', function(e){{
          if(e.target!==cb){{ cb.checked=!cb.checked; cb.dispatchEvent(new Event('change')); }}
        }});
        return d;
      }}

      /* Section label helper */
      function sectionLabel(text) {{
        var d = document.createElement('div');
        d.textContent = text;
        d.style.cssText = 'font-size:10px;color:#7c84a8;letter-spacing:.08em;' +
          'text-transform:uppercase;margin:6px 0 6px 0;font-weight:600;';
        return d;
      }}

      /* Separator */
      function sep() {{
        var d = document.createElement('div');
        d.style.cssText = 'border-top:1px solid rgba(255,255,255,0.1);margin:8px 0;';
        return d;
      }}

      panel.appendChild(sectionLabel('Layers'));

      /* Prediction toggle */
      panel.appendChild(cbRow('Prediction', true, '#74c476', function(on){{
        var ii = imgs(); if(!ii.length) return;
        ii[0].style.opacity = on ? '1' : '0';
      }}));

      /* Uncertainty toggle */
      panel.appendChild(cbRow('Uncertainty', true, '#fd8d3c', function(on){{
        var ii = imgs(); if(ii.length<2) return;
        ii[1].style.opacity = on ? '1' : '0';
      }}));

      panel.appendChild(sep());
      panel.appendChild(sectionLabel('Swipe'));

      /* Swipe toggle */
      panel.appendChild(cbRow('Compare mode', true, '#9fa8d5', function(on){{
        swipeOn = on;
        bar.style.display = on ? '' : 'none';
        if(on) applyClip(pct); else clearClip();
      }}));

      panel.appendChild(sep());
      panel.appendChild(sectionLabel('Basemap'));

      /* Basemap radio buttons */
      ['OpenStreetMap','Light','Satellite','None'].forEach(function(name){{
        var d  = document.createElement('div');
        d.style.cssText = 'display:flex;align-items:center;gap:9px;margin-bottom:6px;cursor:pointer;';
        var rb = document.createElement('input');
        rb.type='radio'; rb.name='bm_{map_var}'; rb.value=name;
        rb.checked = (name === 'OpenStreetMap');
        rb.style.cssText = 'cursor:pointer;accent-color:#9fa8d5;flex-shrink:0;';
        rb.addEventListener('change', function(){{
          if(activeBasemap) map.removeLayer(activeBasemap);
          activeBasemap = basemaps[name];
          if(activeBasemap) activeBasemap.addTo(map);
        }});
        var sp = document.createElement('span');
        sp.textContent = name; sp.style.userSelect='none';
        d.appendChild(rb); d.appendChild(sp);
        d.addEventListener('click', function(e){{
          if(e.target!==rb){{ rb.checked=true; rb.dispatchEvent(new Event('change')); }}
        }});
        panel.appendChild(d);
      }});

      container.appendChild(panel);
    }}

    waitForLayers();
  }}

  if (document.readyState === 'loading')
    document.addEventListener('DOMContentLoaded', init);
  else
    init();
}})();
</script>
"""
    m.get_root().html.add_child(Element(js))


# ---------------------------------------------------------------------------
# Map builder
# ---------------------------------------------------------------------------

def _build_folium_overlay_map(
    df: pd.DataFrame,
    trait_name: str,
    src_crs_override: str | None,
    scene_tif_path: str | None = None,
    output_pred_tif_path: str | None = None,
) -> str:
    if "row" not in df.columns or "col" not in df.columns:
        return "<h3>No raster indices (row/col) in output — raster inference required.</h3>"

    pred_col = _resolve_pred_col(df, trait_name=trait_name)

    pred_grid, unc_grid, bounds, has_geo = _resolve_grids_and_bounds(
        df, pred_col, src_crs_override,
        scene_tif_path=scene_tif_path,
        output_pred_tif_path=output_pred_tif_path,
    )

    pred_rgba, pred_vmin, pred_vmax = _rgba_from_grid(pred_grid, "Greens",  is_uncertainty=False)
    unc_rgba,  unc_vmin,  unc_vmax  = _rgba_from_grid(unc_grid,  "Oranges", is_uncertainty=True)

    center = [(bounds[0][0] + bounds[1][0]) / 2.0,
              (bounds[0][1] + bounds[1][1]) / 2.0]
    m = folium.Map(
        location=center,
        zoom_start=12 if has_geo else 1,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    # Prediction = bottom (zindex 3), Uncertainty always on top (zindex 4).
    # mercator_project=False: grid already reprojected by rasterio — no further
    # warping needed, and avoids the horizontal-stripe artefact caused by
    # mercator_transform interpolating across NaN scan-line gaps.
    folium.raster_layers.ImageOverlay(
        image=pred_rgba, bounds=bounds, name="Prediction",
        opacity=0.85, mercator_project=False,
        interactive=True, cross_origin=False, zindex=3,
    ).add_to(m)
    folium.raster_layers.ImageOverlay(
        image=unc_rgba, bounds=bounds, name="Uncertainty",
        opacity=0.85, mercator_project=False,
        interactive=True, cross_origin=False, zindex=4,
    ).add_to(m)

    # On-map floating panel: layer toggles + swipe bar + basemap switcher
    _add_map_controls(m, pred_col=pred_col)

    m.fit_bounds(bounds)
    _add_colorbars(
        m,
        pred_vmin=pred_vmin, pred_vmax=pred_vmax,
        unc_vmin=unc_vmin,   unc_vmax=unc_vmax,
        pred_label=f"Prediction ({pred_col})",
    )
    return m._repr_html_()


# ---------------------------------------------------------------------------
# Artifact resolution
# ---------------------------------------------------------------------------

def _resolve_uncertainty_artifacts(cfg):
    q_path = Path(str(cfg.inference.quantile_models_path))
    d_path = Path(str(cfg.inference.distance_reference_npz))
    if q_path.exists() and d_path.exists():
        return str(q_path), str(d_path)

    root = (
        q_path.parents[1]
        if len(q_path.parents) >= 2
        else Path("results/experiments/study2_uncertainty_1522")
    )
    t1 = root / "target_1"
    if (t1 / "quantile_models.pkl").exists() and (t1 / "distance_reference.npz").exists():
        return str(t1 / "quantile_models.pkl"), str(t1 / "distance_reference.npz")

    for c in sorted(root.glob("target_*")):
        q, d = c / "quantile_models.pkl", c / "distance_reference.npz"
        if q.exists() and d.exists():
            return str(q), str(d)

    raise FileNotFoundError(
        f"Could not find uncertainty artifacts under {root}. "
        "Run uncertainty stage=distance first."
    )


# ---------------------------------------------------------------------------
# Progress tracking
# ---------------------------------------------------------------------------

# Maps substrings found in inference log messages to (progress_fraction, label).
# Fractions are calibrated to real wall-clock proportions from observed runs.
_LOG_PROGRESS = [
    ("Starting scene inference",        0.02, "Initialising inference pipeline…"),
    ("Loaded model",                    0.05, "Model loaded · preparing scene…"),
    ("Loading TIFF scene",              0.07, "Loading hyperspectral image…"),
    ("TIFF prepared",                   0.12, "Scene prepared · running trait prediction…"),
    ("[predict] batch",                 None, None),   # handled per-batch below
    ("Prediction complete",             0.46, "Traits predicted · extracting spectral embeddings…"),
    ("Loaded quantile models",          0.48, "Quantile models loaded…"),
    ("Loaded distance reference",       0.50, "Computing spectral distances  ⏳  (~1–2 min)…"),
    ("[embedding] batch",               None, None),   # handled per-batch below
    ("Embedding extraction complete",   0.72, "Embeddings done · computing uncertainty distances…"),
    ("Distance features computed",      0.88, "Distances computed · predicting uncertainty…"),
    ("Uncertainty prediction complete", 0.92, "Uncertainty ready · saving outputs…"),
    ("Saved GeoTIFF",                   0.95, "GeoTIFFs saved · building map…"),
    ("Saved scene prediction",          0.97, "Results saved · rendering visualisation…"),
]

_BATCH_RE = re.compile(r"\[(predict|embedding)\] batch\s+(\d+)/(\d+)")

# Phase windows for per-batch interpolation
_BATCH_WINDOWS = {
    "predict":   (0.12, 0.46),   # 34 % of total
    "embedding": (0.50, 0.72),   # 22 % of total
}


class _LogQueueHandler(logging.Handler):
    """Forwards every log record message to a queue.Queue."""
    def __init__(self, q: queue.Queue):
        super().__init__(level=logging.DEBUG)  # accept all levels
        self.q = q

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.q.put_nowait(record.getMessage())
        except Exception:
            pass


def _parse_progress(msg: str) -> tuple[float, str] | None:
    """
    Return (fraction 0-1, label) for a recognised log message, or None.
    Batch messages are interpolated smoothly within their phase window.
    """
    m = _BATCH_RE.search(msg)
    if m:
        phase = m.group(1)
        n, total = int(m.group(2)), int(m.group(3))
        lo, hi = _BATCH_WINDOWS.get(phase, (0, 1))
        frac = lo + (n / total) * (hi - lo)
        verb = "Predicting traits" if phase == "predict" else "Extracting embeddings"
        return frac, f"{verb}  {n}/{total} batches…"

    for substring, frac, label in _LOG_PROGRESS:
        if frac is not None and substring in msg:
            return frac, label

    return None


# ---------------------------------------------------------------------------
# Core inference callback (non-generator; called from the generator wrapper)
# ---------------------------------------------------------------------------

def _build_results(
    scene_tif_file,
    sensor_bands_file,
    src_crs_override,
    trait_name,
) -> tuple:
    """Run full inference and return (overlay_html, trait_upd, pred_path, unc_path, cached)."""
    if scene_tif_file is None or sensor_bands_file is None:
        raise ValueError("Please upload both a Scene TIFF and a Sensor Bands CSV.")

    tmp_out = Path(tempfile.mkdtemp()) / "scene_predictions_uncertainty.csv"
    cfg = OmegaConf.load(
        Path(__file__).resolve().parents[3]
        / "configs" / "experiments" / "scene_infer_uncertainty.yaml"
    )

    cfg.inference.input_type = "tif"
    cfg.inference.scene_tif = scene_tif_file.name
    cfg.inference.sensor_bands_csv = sensor_bands_file.name

    cfg.inference.output_path      = str(tmp_out)
    cfg.inference.output_pred_tif  = str(tmp_out.with_name(tmp_out.stem + "_predictions_20traits.tif"))
    cfg.inference.output_unc_tif   = str(tmp_out.with_name(tmp_out.stem + "_uncertainty_20traits.tif"))

    q_art, d_art = _resolve_uncertainty_artifacts(cfg)
    cfg.inference.quantile_models_path   = q_art
    cfg.inference.distance_reference_npz = d_art
    run_inference(cfg)

    df = pd.read_csv(tmp_out)
    pred_cols     = [c for c in df.columns if c.startswith("pred_")]
    trait_choices = [c.replace("pred_", "", 1) for c in pred_cols]
    selected      = (str(trait_name) if str(trait_name) in trait_choices
                     else (trait_choices[0] if trait_choices else ""))

    crs        = str(src_crs_override).strip() or None
    _scene_tif = str(scene_tif_file.name) if scene_tif_file is not None else None
    _pred_tif  = str(tmp_out.with_name(tmp_out.stem + "_predictions_20traits.tif"))

    overlay_html = _build_folium_overlay_map(
        df,
        trait_name=selected,
        src_crs_override=crs,
        scene_tif_path=_scene_tif,
        output_pred_tif_path=_pred_tif,
    )

    pred_tif = Path(cfg.inference.output_pred_tif)
    unc_tif  = Path(cfg.inference.output_unc_tif)
    cached   = str(tmp_out) + "|" + (_scene_tif or "") + "|" + _pred_tif
    return (
        overlay_html,
        gr.update(choices=trait_choices or DEFAULT_TRAIT_CHOICES, value=selected or None),
        str(pred_tif) if pred_tif.exists() else None,
        str(unc_tif)  if unc_tif.exists()  else None,
        cached,
    )


def _refresh_map(cached_out_csv: str, src_crs_override: str, trait_name: str) -> str:
    """Re-render map for a new trait without re-running inference."""
    if not cached_out_csv:
        return ""
    parts          = str(cached_out_csv).split("|")
    csv_path       = parts[0]
    scene_tif_path = parts[1] if len(parts) > 1 and parts[1] else None
    pred_tif_path  = parts[2] if len(parts) > 2 and parts[2] else None

    out_path = Path(csv_path)
    if not out_path.exists():
        return f"<h3>Cached CSV not found: {out_path}</h3>"

    df  = pd.read_csv(out_path)
    crs = str(src_crs_override).strip() or None
    return _build_folium_overlay_map(
        df,
        trait_name=str(trait_name),
        src_crs_override=crs,
        scene_tif_path=scene_tif_path,
        output_pred_tif_path=pred_tif_path,
    )


# ---------------------------------------------------------------------------
# Gradio UI  —  Biodiversity / Earth Observation theme
# ---------------------------------------------------------------------------

_CSS = """
/* ════════════════════════════════════════════════════════════════════
   ROOT & PAGE
   ════════════════════════════════════════════════════════════════════ */
:root {
    --bio-bg:          #0b1a12;
    --bio-surface:     #112318;
    --bio-surface2:    #172e1e;
    --bio-border:      #1f4028;
    --bio-border2:     #2d5c3a;
    --bio-green:       #3a9e5f;
    --bio-green-light: #52c97a;
    --bio-green-pale:  #a8e6bc;
    --bio-amber:       #e8a94a;
    --bio-amber-pale:  #f5d48e;
    --bio-text:        #d4edda;
    --bio-text-muted:  #7aad8a;
    --bio-text-dim:    #4a7558;
    --bio-radius:      10px;
    --bio-radius-lg:   16px;
    --bio-shadow:      0 4px 24px rgba(0,0,0,0.55);
}

/* full-page background */
body, .gradio-container, gradio-app {
    background: var(--bio-bg) !important;
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif !important;
}

.gradio-container {
    max-width: 1120px !important;
    margin: 0 auto !important;
    padding: 0 16px 40px !important;
}

/* ════════════════════════════════════════════════════════════════════
   HEADER  (rendered as gr.HTML)
   ════════════════════════════════════════════════════════════════════ */
#app-header {
    background: linear-gradient(135deg, #0e2a18 0%, #163522 60%, #1a3d28 100%);
    border: 1px solid var(--bio-border2);
    border-radius: var(--bio-radius-lg);
    padding: 28px 32px 22px;
    margin-bottom: 20px;
    position: relative;
    overflow: hidden;
}
#app-header::before {               /* subtle leaf-vein texture */
    content: '';
    position: absolute;
    inset: 0;
    background:
        radial-gradient(ellipse 60% 40% at 90% 20%, rgba(58,158,95,0.08) 0%, transparent 70%),
        radial-gradient(ellipse 40% 60% at 10% 80%, rgba(58,158,95,0.06) 0%, transparent 70%);
    pointer-events: none;
}
#app-header h1 {
    color: var(--bio-green-pale) !important;
    font-size: 1.75rem !important;
    font-weight: 700 !important;
    margin: 0 0 6px !important;
    letter-spacing: -0.02em;
}
#app-header p {
    color: var(--bio-text-muted) !important;
    font-size: 0.9rem !important;
    margin: 0 !important;
    line-height: 1.5;
}
#app-header .badge {
    display: inline-block;
    background: rgba(58,158,95,0.18);
    border: 1px solid rgba(82,201,122,0.35);
    color: var(--bio-green-light);
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: .06em;
    text-transform: uppercase;
    padding: 2px 9px;
    border-radius: 20px;
    margin-bottom: 10px;
}

/* ════════════════════════════════════════════════════════════════════
   SECTION LABELS
   ════════════════════════════════════════════════════════════════════ */
.section-label {
    color: var(--bio-text-dim) !important;
    font-size: 0.7rem !important;
    font-weight: 700 !important;
    letter-spacing: .1em !important;
    text-transform: uppercase !important;
    margin: 0 0 10px !important;
    display: flex;
    align-items: center;
    gap: 7px;
}
.section-label::before {
    content: '';
    display: inline-block;
    width: 3px; height: 13px;
    background: var(--bio-green);
    border-radius: 2px;
}

/* ════════════════════════════════════════════════════════════════════
   UPLOAD CARD
   ════════════════════════════════════════════════════════════════════ */
#upload-card {
    background: var(--bio-surface) !important;
    border: 1px solid var(--bio-border) !important;
    border-radius: var(--bio-radius-lg) !important;
    padding: 22px 24px 18px !important;
    box-shadow: var(--bio-shadow) !important;
}

/* Gradio file-upload widget overrides */
#upload-card .file-preview,
#upload-card [data-testid="file-upload"] {
    background: var(--bio-surface2) !important;
    border: 1px dashed var(--bio-border2) !important;
    border-radius: var(--bio-radius) !important;
    color: var(--bio-text-muted) !important;
    transition: border-color .2s, background .2s;
}
#upload-card [data-testid="file-upload"]:hover {
    border-color: var(--bio-green) !important;
    background: #1a3521 !important;
}

/* input labels */
#upload-card label span,
#upload-card .label-wrap span {
    color: var(--bio-text-muted) !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
}

/* textbox */
#upload-card input[type=text], #upload-card textarea {
    background: var(--bio-surface2) !important;
    border: 1px solid var(--bio-border2) !important;
    border-radius: var(--bio-radius) !important;
    color: var(--bio-text) !important;
    font-size: 0.85rem !important;
}
#upload-card input[type=text]:focus, #upload-card textarea:focus {
    border-color: var(--bio-green) !important;
    box-shadow: 0 0 0 3px rgba(58,158,95,0.18) !important;
    outline: none !important;
}

/* ════════════════════════════════════════════════════════════════════
   RUN BUTTON
   ════════════════════════════════════════════════════════════════════ */
#run-btn {
    background: linear-gradient(135deg, #2d8a50 0%, #3a9e5f 100%) !important;
    color: #fff !important;
    font-size: 15px !important;
    font-weight: 700 !important;
    letter-spacing: .03em !important;
    padding: 13px 0 !important;
    border-radius: var(--bio-radius) !important;
    border: none !important;
    margin-top: 14px !important;
    box-shadow: 0 4px 14px rgba(58,158,95,0.35) !important;
    transition: all .2s !important;
}
#run-btn:hover {
    background: linear-gradient(135deg, #3aad64 0%, #4dbd74 100%) !important;
    box-shadow: 0 6px 20px rgba(58,158,95,0.50) !important;
    transform: translateY(-1px) !important;
}
#run-btn:active { transform: translateY(0) !important; }

/* ════════════════════════════════════════════════════════════════════
   RESULTS PANEL
   ════════════════════════════════════════════════════════════════════ */
#results-panel {
    margin-top: 22px;
}

/* trait row */
#trait-row {
    background: var(--bio-surface) !important;
    border: 1px solid var(--bio-border) !important;
    border-radius: var(--bio-radius-lg) !important;
    padding: 14px 18px !important;
    margin-bottom: 14px !important;
    box-shadow: var(--bio-shadow) !important;
    align-items: flex-end !important;
}
#trait-row label span,
#trait-row .label-wrap span {
    color: var(--bio-text-muted) !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: .05em;
}
#trait-row select, #trait-row .wrap {
    background: var(--bio-surface2) !important;
    border: 1px solid var(--bio-border2) !important;
    border-radius: var(--bio-radius) !important;
    color: var(--bio-text) !important;
}

/* refresh button */
#refresh-btn {
    background: var(--bio-surface2) !important;
    color: var(--bio-green-light) !important;
    border: 1px solid var(--bio-border2) !important;
    border-radius: var(--bio-radius) !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    transition: all .2s !important;
}
#refresh-btn:hover {
    background: #1f3d28 !important;
    border-color: var(--bio-green) !important;
}

/* map container */
#map-output {
    border-radius: var(--bio-radius-lg) !important;
    overflow: hidden !important;
    border: 1px solid var(--bio-border2) !important;
    box-shadow: var(--bio-shadow) !important;
}
#map-output iframe, #map-output > div {
    width: 100% !important;
    border-radius: var(--bio-radius-lg) !important;
}

/* ════════════════════════════════════════════════════════════════════
   DOWNLOAD ROW
   ════════════════════════════════════════════════════════════════════ */
#dl-row {
    margin-top: 14px !important;
}
#dl-row .file-preview,
#dl-row [data-testid="file-upload"] {
    background: var(--bio-surface) !important;
    border: 1px solid var(--bio-border) !important;
    border-radius: var(--bio-radius) !important;
    color: var(--bio-text-muted) !important;
}
#dl-row label span {
    color: var(--bio-text-muted) !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
}

/* ════════════════════════════════════════════════════════════════════
   PROGRESS BAR
   ════════════════════════════════════════════════════════════════════ */
.progress-bar-wrap .progress-bar { background: var(--bio-green) !important; }
.progress-bar-wrap { background: var(--bio-border) !important; border-radius: 4px !important; }

/* ════════════════════════════════════════════════════════════════════
   STATUS / PROGRESS BOX  (between run button and results panel)
   ════════════════════════════════════════════════════════════════════ */
#status-box {
    margin-top: 14px !important;
    padding: 14px 20px !important;
    background: var(--bio-surface) !important;
    border: 1px solid var(--bio-border2) !important;
    border-radius: var(--bio-radius) !important;
    box-shadow: var(--bio-shadow) !important;
    min-height: 58px !important;
}

/* ════════════════════════════════════════════════════════════════════
   SCROLLBAR
   ════════════════════════════════════════════════════════════════════ */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bio-bg); }
::-webkit-scrollbar-thumb { background: var(--bio-border2); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--bio-green); }
"""

_HEADER_HTML = """
<div id="app-header">
  <div class="badge">🛰 Hyperspectral · Biodiversity · Earth Observation</div>
  <h1>BioDis-UN &nbsp;·&nbsp; Plant Trait Mapping</h1>
  <p>
    Airborne hyperspectral scene inference with calibrated uncertainty estimation.
    Upload a raw HSI scene TIFF and sensor band list to retrieve spatially explicit
    plant functional traits and their prediction confidence across the landscape.
  </p>
</div>
"""


def launch() -> None:
    with gr.Blocks(
        title="BioDis-UN — Plant Trait Mapping",
        css=_CSS,
        theme=gr.themes.Base(),
    ) as demo:

        # ── App header ────────────────────────────────────────────────────
        gr.HTML(_HEADER_HTML)

        # ── STATE ─────────────────────────────────────────────────────────
        cached_out_csv = gr.State(value="")

        # ═════════════════════════════════════════════════════════════════
        # PHASE 1  —  always visible: data upload
        # ═════════════════════════════════════════════════════════════════
        gr.HTML('<p class="section-label">Scene Data Input</p>')

        with gr.Group(elem_id="upload-card"):
            with gr.Row():
                scene_tif_file = gr.File(
                    label="Scene HSI TIFF  (.tif / .tiff)",
                    file_types=[".tif", ".tiff"],
                    scale=2,
                )
                sensor_bands_file = gr.File(
                    label="Sensor Bands CSV  (wavelength list)",
                    file_types=[".csv"],
                    scale=1,
                )
            with gr.Row():
                src_crs_override = gr.Textbox(
                    label="CRS override  (optional — e.g. EPSG:32633)",
                    value="",
                    placeholder="Auto-detected from TIFF metadata",
                )

        run_btn = gr.Button(
            "▶  Run Scene Inference",
            variant="primary",
            elem_id="run-btn",
        )

        # Progress bar — visible only during inference
        status_box = gr.HTML(value="", visible=False, elem_id="status-box")

        # ═════════════════════════════════════════════════════════════════
        # PHASE 2  —  hidden until inference completes
        # ═════════════════════════════════════════════════════════════════
        with gr.Group(visible=False, elem_id="results-panel") as results_panel:

            gr.HTML('<p class="section-label" style="margin-top:8px;">Trait Visualisation</p>')

            with gr.Row(elem_id="trait-row"):
                trait_name = gr.Dropdown(
                    label="Plant Functional Trait",
                    choices=DEFAULT_TRAIT_CHOICES,
                    value="EWT_mg_cm2",
                    allow_custom_value=False,
                    scale=5,
                )
                refresh_btn = gr.Button(
                    "↻  Refresh map",
                    elem_id="refresh-btn",
                    scale=1,
                )

            # Map — layer controls, swipe bar and basemap switcher are
            # all rendered inside the Folium HTML (see _add_map_controls)
            overlay_map = gr.HTML(elem_id="map-output")

            # Downloads — revealed only when GeoTIFFs are written
            gr.HTML('<p class="section-label" style="margin-top:6px;">Export Results</p>')
            with gr.Row(elem_id="dl-row", visible=False) as dl_row:
                pred_tif_file = gr.File(
                    label="⬇  Prediction GeoTIFF  (all 20 traits)",
                    interactive=False,
                )
                unc_tif_file = gr.File(
                    label="⬇  Uncertainty GeoTIFF  (all 20 traits)",
                    interactive=False,
                )

        # ── WIRING ────────────────────────────────────────────────────────

        def _run_and_reveal(scene_tif, sensor_bands, src_crs, trait):
            """
            Generator — streams live progress into status_box (always visible)
            while _build_results() runs in a background thread.

            Output order matches run_btn.click outputs list:
              0  status_box    — progress HTML during inference, cleared after
              1  overlay_map   — map HTML, set only on final yield
              2  trait_name    — dropdown update, set only on final yield
              3  pred_tif_file — file path, set only on final yield
              4  unc_tif_file  — file path, set only on final yield
              5  cached_out_csv— state string, set only on final yield
              6  results_panel — visibility toggle
              7  dl_row        — visibility toggle

            Intermediate yields update only status_box (index 0) and show
            the results_panel as hidden so nothing jumps around.
            """
            def _status_html(frac: float, label: str) -> str:
                pct = int(frac * 100)
                shimmer = (
                    '@keyframes _bio_shimmer{'
                    'from{background-position:200% 0}'
                    'to{background-position:-200% 0}}'
                )
                return (
                    f'<style>{shimmer}</style>'
                    f'<div style="padding:14px 0 6px;font-family:Segoe UI,sans-serif;">'
                    f'<div style="display:flex;justify-content:space-between;'
                    f'font-size:13px;color:#7aad8a;margin-bottom:8px;">'
                    f'<span>{label}</span>'
                    f'<span style="color:#52c97a;font-weight:700;">{pct}%</span></div>'
                    f'<div style="height:10px;background:#1f4028;border-radius:5px;overflow:hidden;">'
                    f'<div style="height:100%;width:{pct}%;'
                    f'background:linear-gradient(90deg,#2d8a50,#52c97a);'
                    f'border-radius:5px;transition:width .5s ease;'
                    f'box-shadow:0 0 12px rgba(82,201,122,.6);'
                    f'position:relative;overflow:hidden;">'
                    f'<div style="position:absolute;inset:0;'
                    f'background:linear-gradient(90deg,transparent,rgba(255,255,255,.22),transparent);'
                    f'background-size:200% 100%;'
                    f'animation:_bio_shimmer 1.4s linear infinite;"></div>'
                    f'</div></div></div>'
                )

            # Helper: intermediate yield — only updates status_box, everything else unchanged
            def _mid(frac, label):
                return (
                    gr.update(value=_status_html(frac, label), visible=True),  # status_box
                    gr.update(),   # overlay_map     — no change
                    gr.update(),   # trait_name      — no change
                    gr.update(),   # pred_tif_file   — no change
                    gr.update(),   # unc_tif_file    — no change
                    gr.update(),   # cached_out_csv  — no change
                    gr.update(),   # results_panel   — no change
                    gr.update(),   # dl_row          — no change
                )

            # ── immediate first yield so Gradio sees the generator is live ──
            yield _mid(0.01, "Initialising inference pipeline…")

            result_box: list = [None]
            error_box:  list = [None]

            log_q: "queue.Queue[str]" = queue.Queue()
            handler = _LogQueueHandler(log_q)
            root_logger = logging.getLogger()
            # Root logger defaults to WARNING — INFO messages from run_inference
            # are silently filtered before reaching any handler. Temporarily
            # lower the level so every INFO record flows into our queue.
            _saved_level = root_logger.level
            root_logger.setLevel(logging.INFO)
            root_logger.addHandler(handler)

            def _worker():
                try:
                    result_box[0] = _build_results(scene_tif, sensor_bands, src_crs, trait)
                except Exception as exc:
                    error_box[0] = exc

            thread = threading.Thread(target=_worker, daemon=True)
            thread.start()

            last_frac  = 0.01
            last_label = "Initialising inference pipeline…"

            while thread.is_alive():
                while True:
                    try:
                        msg = log_q.get_nowait()
                        parsed = _parse_progress(msg)
                        if parsed:
                            frac, lbl = parsed
                            if frac > last_frac:
                                last_frac, last_label = frac, lbl
                    except queue.Empty:
                        break
                # Always yield to keep the stream alive
                yield _mid(last_frac, last_label)
                thread.join(timeout=0.3)

            # Drain final messages
            while True:
                try:
                    msg = log_q.get_nowait()
                    parsed = _parse_progress(msg)
                    if parsed:
                        frac, lbl = parsed
                        if frac > last_frac:
                            last_frac, last_label = frac, lbl
                except queue.Empty:
                    break

            root_logger.removeHandler(handler)
            root_logger.setLevel(_saved_level)   # restore original level

            if error_box[0] is not None:
                raise error_box[0]

            yield _mid(0.99, "Building map visualisation…")

            overlay_html, trait_upd, pred_path, unc_path, cached = result_box[0]

            # Final yield — hide status_box, reveal map + results panel
            yield (
                gr.update(value="", visible=False),   # status_box — hide
                overlay_html,                          # overlay_map
                trait_upd,                             # trait_name
                pred_path,                             # pred_tif_file
                unc_path,                              # unc_tif_file
                cached,                                # cached_out_csv
                gr.update(visible=True),               # results_panel
                gr.update(visible=bool(pred_path or unc_path)),  # dl_row
            )

        run_btn.click(
            _run_and_reveal,
            inputs=[scene_tif_file, sensor_bands_file, src_crs_override, trait_name],
            outputs=[status_box, overlay_map, trait_name, pred_tif_file, unc_tif_file,
                     cached_out_csv, results_panel, dl_row],
            queue=True,
        )

        def _do_refresh(cached, src_crs, trait):
            return _refresh_map(cached, src_crs, trait)

        refresh_btn.click(
            _do_refresh,
            inputs=[cached_out_csv, src_crs_override, trait_name],
            outputs=[overlay_map],
        )

        trait_name.change(
            _do_refresh,
            inputs=[cached_out_csv, src_crs_override, trait_name],
            outputs=[overlay_map],
        )

    demo.launch(server_name="0.0.0.0", server_port=7862)


if __name__ == "__main__":
    launch()