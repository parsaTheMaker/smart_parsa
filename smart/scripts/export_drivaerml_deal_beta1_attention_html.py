#!/usr/bin/env python3
"""Export an interactive DeAL encoder-attention map for a beta=1 DrivAerML view.

For a fixed 131,072-point beta=1 surface input, SMART's encoder uses 16,384
sampled geometry keys per block.  This script computes the exact softmax
attention mass received by every sampled key in all encoder blocks, averages
duplicate key occurrences, and uses inverse-distance interpolation only for
the input points that were not selected as keys.  The result is a full-input
diagnostic while retaining a direct-attention count for every displayed point.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial import cKDTree

SCRIPT_DIR = Path(__file__).resolve().parent
SMART_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(SMART_ROOT) not in sys.path:
    sys.path.insert(0, str(SMART_ROOT))

from data.ahmedml_dataset_v2 import AhmedMLDatasetV2  # noqa: E402
from export_drivaerml_smart_anchor_attention import (  # noqa: E402
    build_model,
    sample_condition,
    sample_model_indices,
)


DEFAULT_CHECKPOINT = Path(
    "/home/parsa/smart_parsa/checkpoints/smart-satloss7-range100-smart-satloss7-range100-from-smart-150ep-drivaerml-s42_best.pt"
)
DEFAULT_OUTPUT_DIR = Path("/home/parsa/smart_parsa/results/final/drivaerml_deal_beta1_cross_attention")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/mnt/ssdraid/parsa/drivaerml_preprocessed"))
    parser.add_argument("--run-id", type=int, default=29)
    parser.add_argument("--deal-config", default="drivaerml_satloss7_range100")
    parser.add_argument("--deal-checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--input-points", type=int, default=131072)
    parser.add_argument("--density-knn-k", type=int, default=16)
    parser.add_argument("--density-estimator", choices=("kde", "rk2", "tangent_cov"), default="kde")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attention-key-chunk-size", type=int, default=512)
    parser.add_argument("--interpolation-neighbors", type=int, default=12)
    parser.add_argument("--interpolation-workers", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def _binary_payload(values: np.ndarray, dtype: np.dtype) -> str:
    array = np.ascontiguousarray(values, dtype=dtype)
    return base64.b64encode(array.tobytes()).decode("ascii")


def _score_percentile(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.shape[0], dtype=np.float32)
    ranks[order] = np.linspace(0.0, 1.0, values.shape[0], dtype=np.float32)
    return ranks


def _idw_missing_scores(
    points: np.ndarray,
    direct_scores: np.ndarray,
    direct_count: np.ndarray,
    neighbors: int,
    workers: int,
) -> np.ndarray:
    """Preserve exact attended scores; interpolate only never-sampled input points."""
    observed = direct_count > 0
    if observed.sum() < 4:
        raise RuntimeError("Too few directly attended keys to interpolate a full input map.")
    output = np.empty(points.shape[0], dtype=np.float32)
    output[observed] = direct_scores[observed]
    missing = ~observed
    if not missing.any():
        return output
    tree = cKDTree(np.ascontiguousarray(points[observed], dtype=np.float64))
    count = min(max(4, int(neighbors)), int(observed.sum()))
    distance, indices = tree.query(np.ascontiguousarray(points[missing], dtype=np.float64), k=count, workers=int(workers))
    if count == 1:
        distance, indices = distance[:, None], indices[:, None]
    support = direct_scores[observed]
    exact = distance[:, 0] <= 1.0e-12
    weights = 1.0 / np.maximum(distance, 1.0e-12) ** 2
    values = (weights * support[indices]).sum(axis=1) / weights.sum(axis=1)
    values[exact] = support[indices[exact, 0]]
    output[missing] = values.astype(np.float32)
    return output


@torch.inference_mode()
def exact_encoder_key_attention(model, geometry: torch.Tensor, seed: int, key_chunk_size: int) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Compute mean anchor/head attention mass for actual keys in each encoder block."""
    device = next(model.parameters()).device
    geo = geometry.unsqueeze(0).to(device=device, dtype=torch.float32, non_blocking=True)
    point_count = int(geo.shape[1])
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    geo_scaled = geo * float(model.pos_scale_factor)
    latent_idx = sample_model_indices(point_count, int(model.num_geo), False, generator, device).unsqueeze(0)
    latent_pos = torch.gather(geo_scaled, 1, latent_idx.unsqueeze(-1).expand(-1, -1, 3))
    latent = model.pos_encoder(latent_pos)
    accumulated = torch.zeros(point_count, dtype=torch.float32, device=device)
    visit_count = torch.zeros(point_count, dtype=torch.float32, device=device)
    layer_mass: list[float] = []

    for layer_index, block in enumerate(model.encoder_blocks):
        sub_idx = sample_model_indices(
            point_count,
            int(model.subsampled_geometry_points),
            bool(model.subsampled_geometry_with_replacement),
            generator,
            device,
        ).unsqueeze(0)
        sub_pos = torch.gather(geo_scaled, 1, sub_idx.unsqueeze(-1).expand(-1, -1, 3))
        sub_tokens = model.pos_encoder(sub_pos)
        attention = block.geo_attn
        q = attention.q(attention.norm_q(latent))
        q = q.view(1, model.num_geo, attention.num_heads, attention.head_dim).permute(0, 2, 1, 3)
        q = attention.rope(q, latent_pos).float()

        running_max = None
        running_exp = None
        for start in range(0, sub_tokens.shape[1], int(key_chunk_size)):
            end = min(start + int(key_chunk_size), sub_tokens.shape[1])
            kv = attention.kv(attention.norm_kv(sub_tokens[:, start:end]))
            kv = kv.view(1, end - start, 2 * attention.num_heads, attention.head_dim).permute(0, 2, 1, 3)
            key = attention.rope(kv[:, : attention.num_heads], sub_pos[:, start:end]).float()
            logits = torch.matmul(q, key.transpose(-1, -2)) / math.sqrt(float(attention.head_dim))
            block_max = logits.amax(dim=-1)
            if running_max is None:
                running_max = block_max
                running_exp = torch.exp(logits - block_max.unsqueeze(-1)).sum(dim=-1)
            else:
                merged_max = torch.maximum(running_max, block_max)
                running_exp = running_exp * torch.exp(running_max - merged_max) + torch.exp(logits - merged_max.unsqueeze(-1)).sum(dim=-1)
                running_max = merged_max
        if running_max is None or running_exp is None:
            raise RuntimeError(f"Encoder block {layer_index} produced no attention chunks.")

        block_mass = torch.zeros((), dtype=torch.float32, device=device)
        for start in range(0, sub_tokens.shape[1], int(key_chunk_size)):
            end = min(start + int(key_chunk_size), sub_tokens.shape[1])
            kv = attention.kv(attention.norm_kv(sub_tokens[:, start:end]))
            kv = kv.view(1, end - start, 2 * attention.num_heads, attention.head_dim).permute(0, 2, 1, 3)
            key = attention.rope(kv[:, : attention.num_heads], sub_pos[:, start:end]).float()
            logits = torch.matmul(q, key.transpose(-1, -2)) / math.sqrt(float(attention.head_dim))
            probabilities = torch.exp(logits - running_max.unsqueeze(-1)) / running_exp.unsqueeze(-1).clamp_min(torch.finfo(torch.float32).tiny)
            # Mean of the exact softmax weights over all anchors and heads.
            key_mass = probabilities.mean(dim=(1, 2))[0]
            indices = sub_idx[0, start:end]
            accumulated.scatter_add_(0, indices, key_mass)
            visit_count.scatter_add_(0, indices, torch.ones_like(key_mass))
            block_mass += key_mass.sum()
        mass = float(block_mass.item())
        if abs(mass - 1.0) > 2.0e-4:
            raise RuntimeError(f"Encoder attention mass was not conserved in layer {layer_index}: {mass:.6f}")
        layer_mass.append(mass)
        latent, _ = block(latent, sub_tokens, None, latent_geometry_pos=latent_pos, subsampled_geometry_pos=sub_pos)

    count = visit_count.cpu().numpy().astype(np.uint8)
    total = accumulated.cpu().numpy()
    direct = np.divide(total, np.maximum(count, 1), dtype=np.float32)
    return direct.astype(np.float32), count, layer_mass


def _html_document(payload: dict[str, str | int | float], metadata: dict[str, Any]) -> str:
    plotly_js = get_plotlyjs()
    metadata_json = json.dumps(metadata, separators=(",", ":"))
    template = """<!doctype html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\">
<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
<title>DeAL beta=1 encoder cross-attention</title>
<style>
  :root {{ color-scheme: dark; --panel: #111821; --line: #344352; --text: #edf4fa; --muted: #a8b8c7; --accent: #f9cf58; }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; background: #090d12; color: var(--text); font: 15px/1.45 Georgia, 'Times New Roman', serif; }}
  main {{ max-width: 1550px; margin: 0 auto; padding: 22px 26px 30px; }}
  h1 {{ margin: 0 0 5px; font-family: 'Trebuchet MS', sans-serif; font-size: clamp(22px, 2.2vw, 33px); font-weight: 700; letter-spacing: .01em; }}
  .sub {{ margin: 0 0 16px; color: var(--muted); max-width: 1050px; }}
  #plot {{ width: 100%; height: min(76vh, 850px); min-height: 600px; border: 1px solid var(--line); border-radius: 10px; overflow: hidden; }}
  .controls {{ display: grid; grid-template-columns: minmax(270px, 450px) 1fr; gap: 16px 25px; margin: 15px 0 8px; align-items: center; }}
  .control, #readout {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 12px 14px; }}
  label {{ font-family: 'Trebuchet MS', sans-serif; font-weight: 700; }}
  input {{ width: 100%; accent-color: var(--accent); margin-top: 8px; }}
  #readout {{ color: var(--muted); min-height: 52px; }}
  #readout strong {{ color: var(--text); }}
  details {{ margin-top: 12px; color: var(--muted); }}
  summary {{ cursor: pointer; color: var(--text); font-family: 'Trebuchet MS', sans-serif; font-weight: 700; }}
  code {{ color: #b9e8ff; }}
</style>
</head>
<body>
<main>
  <h1>DeAL encoder cross-attention on a beta=1 input view</h1>
  <p class=\"sub\">Click any geometry point. Gold points have an attention-score percentile within the selected tolerance of the clicked point. The color field is the rank of the actual encoder key-attention score, which makes a narrow score distribution readable without changing its ordering.</p>
  <div id=\"plot\"></div>
  <section class=\"controls\">
    <div class=\"control\"><label for=\"tolerance\">Similarity tolerance: <span id=\"toleranceValue\"></span> percentile points</label><input id=\"tolerance\" type=\"range\" min=\"0.25\" max=\"5\" value=\"1\" step=\"0.25\"></div>
    <div id=\"readout\"><strong>Click a point</strong> to inspect its raw cross-attention mass, percentile, and whether it was directly used as an encoder key.</div>
  </section>
  <details><summary>Interpretation and provenance</summary><p>Every colored point is one of the <code>@@COUNT@@</code> points supplied to DeAL after beta=1 inverse-density sampling. In each of the six encoder blocks, DeAL samples <code>@@KEYS_PER_LAYER@@</code> keys and computes the exact anchor-to-key softmax weights. A direct score is the mean of those weights over anchors, heads, and every repeated key occurrence. Points never sampled as keys are interpolated from directly observed keys only; the click readout reports this distinction. Similarity is based solely on score percentile, not Euclidean proximity.</p></details>
</main>
<script>@@PLOTLY_JS@@</script>
<script>
const meta = @@METADATA@@;
function bytesFromBase64(encoded) {{ const raw = atob(encoded); const out = new Uint8Array(raw.length); for (let i = 0; i < raw.length; ++i) out[i] = raw.charCodeAt(i); return out; }}
function f32(encoded) {{ const bytes = bytesFromBase64(encoded); return new Float32Array(bytes.buffer); }}
function u16(encoded) {{ const bytes = bytesFromBase64(encoded); return new Uint16Array(bytes.buffer); }}
function u8(encoded) {{ const bytes = bytesFromBase64(encoded); return new Uint8Array(bytes.buffer); }}
const xyz = f32('@@XYZ@@');
const score = f32('@@SCORE@@');
const percentile = u16('@@PERCENTILE@@');
const directCount = u8('@@DIRECT_COUNT@@');
const count = @@COUNT_RAW@@;
const x = new Float32Array(count), y = new Float32Array(count), z = new Float32Array(count), rank = new Float32Array(count);
for (let i = 0; i < count; ++i) {{ x[i] = xyz[3*i]; y[i] = xyz[3*i+1]; z[i] = xyz[3*i+2]; rank[i] = percentile[i] / 65535.0; }}
const baseTrace = {{ type: 'scatter3d', mode: 'markers', x, y, z, hoverinfo: 'skip', marker: {{ size: 1.72, color: rank, cmin: 0, cmax: 1, colorscale: [[0,'#30123b'],[.25,'#33638d'],[.50,'#23a7a4'],[.75,'#f4d35e'],[1,'#d9435e']], opacity: 0.90, colorbar: {{ title: {{text:'attention<br>score percentile', side:'right'}}, thickness: 18, len: .65 }} }}, name: 'input points' }};
const similarTrace = {{ type: 'scatter3d', mode: 'markers', x: [], y: [], z: [], customdata: [], hoverinfo: 'skip', marker: {{ size: 3.35, color: '#f9cf58', opacity: .96, line: {{color:'#17130a', width: .4}} }}, name: 'similar score' }};
const selectedTrace = {{ type: 'scatter3d', mode: 'markers', x: [], y: [], z: [], hoverinfo: 'skip', marker: {{ size: 6.5, color: '#ffffff', symbol: 'circle-open', line: {{color:'#101010', width: 2}} }}, name: 'selected point' }};
const layout = {{ paper_bgcolor:'#090d12', plot_bgcolor:'#090d12', margin:{{l:0,r:5,b:0,t:0}}, showlegend:false, scene: {{ bgcolor:'#090d12', xaxis:{{visible:false}}, yaxis:{{visible:false}}, zaxis:{{visible:false}}, aspectmode:'data', camera:{{eye:{{x:1.65,y:1.45,z:0.95}}} }} }};
const config = {{ responsive:true, displaylogo:false, scrollZoom:true, modeBarButtonsToRemove:['toImage','lasso3d','select3d'] }};
const plot = document.getElementById('plot');
const slider = document.getElementById('tolerance');
const toleranceText = document.getElementById('toleranceValue');
const readout = document.getElementById('readout');
let selectedIndex = -1;
function tolerance() {{ return Number(slider.value) / 100.0; }}
function updateToleranceText() {{ toleranceText.textContent = Number(slider.value).toFixed(2); }}
function updateSelection(index) {{
  selectedIndex = index;
  const target = rank[index], tol = tolerance();
  const candidates = [];
  for (let i = 0; i < count; ++i) {{ if (Math.abs(rank[i] - target) <= tol) candidates.push(i); }}
  const sx = [], sy = [], sz = [], ids = [];
  const stride = Math.max(1, Math.ceil(candidates.length / 7000));
  for (let j = 0; j < candidates.length; j += stride) {{ const i = candidates[j]; sx.push(x[i]); sy.push(y[i]); sz.push(z[i]); ids.push(i); }}
  if (!ids.includes(index)) {{ sx.push(x[index]); sy.push(y[index]); sz.push(z[index]); ids.push(index); }}
  Plotly.restyle(plot, {{x:[sx], y:[sy], z:[sz], customdata:[ids]}}, [1]);
  Plotly.restyle(plot, {{x:[[x[index]]], y:[[y[index]]], z:[[z[index]]]}}, [2]);
  const direct = directCount[index] > 0;
  const directText = direct ? 'direct encoder key (' + directCount[index] + ' draw' + (directCount[index] === 1 ? '' : 's') + ')' : 'interpolated from directly attended keys';
  readout.innerHTML = '<strong>Selected point ' + index.toLocaleString() + '</strong> &middot; raw attention mass <strong>' + score[index].toExponential(4) + '</strong> &middot; score percentile <strong>' + (100 * rank[index]).toFixed(2) + '%</strong> &middot; ' + directText + ' &middot; highlighted: <strong>' + sx.length.toLocaleString() + '</strong>';
}}
Plotly.newPlot(plot, [baseTrace, similarTrace, selectedTrace], layout, config).then(() => {{
  updateToleranceText();
  plot.on('plotly_click', event => {{
    const point = event.points[0];
    const index = point.curveNumber === 0 ? point.pointNumber : Number(point.customdata);
    updateSelection(index);
  }});
  slider.addEventListener('input', () => {{ updateToleranceText(); if (selectedIndex >= 0) updateSelection(selectedIndex); }});
}});
</script>
</body>
</html>"""
    template = template.replace("{{", "{").replace("}}", "}")
    return (
        template.replace("@@PLOTLY_JS@@", plotly_js)
        .replace("@@METADATA@@", metadata_json)
        .replace("@@XYZ@@", str(payload["xyz"]))
        .replace("@@SCORE@@", str(payload["score"]))
        .replace("@@PERCENTILE@@", str(payload["percentile"]))
        .replace("@@DIRECT_COUNT@@", str(payload["direct_count"]))
        .replace("@@COUNT_RAW@@", str(payload["count"]))
        .replace("@@COUNT@@", f"{int(payload['count']):,}")
        .replace("@@KEYS_PER_LAYER@@", f"{int(payload['keys_per_layer']):,}")
    )


def _gpu_html_document(payload: dict[str, str | int | float], metadata: dict[str, Any]) -> str:
    """Build a self-contained WebGL attention viewer with deterministic GPU picking."""
    three_path = SMART_ROOT / "vendor" / "three.min.js"
    if not three_path.is_file():
        raise FileNotFoundError(f"Missing vendored Three.js runtime: {three_path}")
    three_js = three_path.read_text(encoding="utf-8")
    metadata_json = json.dumps(metadata, separators=(",", ":"))
    template = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DeAL beta=1 encoder cross-attention</title>
<style>
  :root {{ color-scheme: dark; --panel: #111821; --line: #344352; --text: #edf4fa; --muted: #a8b8c7; --accent: #f9cf58; }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; background: #090d12; color: var(--text); font: 15px/1.45 Georgia, 'Times New Roman', serif; }}
  main {{ max-width: 1550px; margin: 0 auto; padding: 22px 26px 30px; }}
  h1 {{ margin: 0 0 5px; font-family: 'Trebuchet MS', sans-serif; font-size: clamp(22px, 2.2vw, 33px); font-weight: 700; letter-spacing: .01em; }}
  .sub {{ margin: 0 0 16px; color: var(--muted); max-width: 1100px; }}
  #viewer {{ position: relative; width: 100%; height: min(76vh, 850px); min-height: 600px; border: 1px solid var(--line); border-radius: 10px; overflow: hidden; cursor: crosshair; }}
  #viewer canvas {{ display: block; width: 100%; height: 100%; }}
  .colorbar {{ position: absolute; right: 20px; top: 22px; display: flex; gap: 8px; align-items: stretch; color: #eef6fc; font: 12px/1.1 'Trebuchet MS', sans-serif; pointer-events: none; text-shadow: 0 1px 2px #000; }}
  .gradient {{ width: 13px; height: 150px; border: 1px solid #d5e1e8; background: linear-gradient(to top, #30123b, #33638d 25%, #23a7a4 50%, #f4d35e 75%, #d9435e); }}
  .colorlabels {{ display: flex; flex-direction: column; justify-content: space-between; padding: 1px 0; }}
  .controls {{ display: grid; grid-template-columns: minmax(270px, 450px) 1fr; gap: 16px 25px; margin: 15px 0 8px; align-items: center; }}
  .control, #readout {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 12px 14px; }}
  label {{ font-family: 'Trebuchet MS', sans-serif; font-weight: 700; }}
  input {{ width: 100%; accent-color: var(--accent); margin-top: 8px; }}
  #readout {{ color: var(--muted); min-height: 52px; }}
  #readout strong {{ color: var(--text); }}
  details {{ margin-top: 12px; color: var(--muted); }}
  summary {{ cursor: pointer; color: var(--text); font-family: 'Trebuchet MS', sans-serif; font-weight: 700; }}
  code {{ color: #b9e8ff; }}
  @media (max-width: 700px) {{ main {{ padding: 15px; }} #viewer {{ min-height: 460px; }} .controls {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<main>
  <h1>DeAL encoder cross-attention on a beta=1 input view</h1>
  <p class="sub">Click any geometry point. The dense cloud is rendered as GPU Gaussian point splats, while an independent enlarged GPU picking pass identifies the selected point. Gold splats have an attention-score percentile within the selected tolerance; drag to orbit and use the wheel to zoom.</p>
  <div id="viewer"><div class="colorbar"><div class="gradient"></div><div class="colorlabels"><span>high</span><span>attention-score<br>percentile</span><span>low</span></div></div></div>
  <section class="controls">
    <div class="control"><label for="tolerance">Similarity tolerance: <span id="toleranceValue"></span> percentile points</label><input id="tolerance" type="range" min="0.25" max="5" value="1" step="0.25"></div>
    <div id="readout"><strong>Click a point</strong> to inspect its raw cross-attention mass, percentile, and whether it was directly used as an encoder key.</div>
  </section>
  <details><summary>Interpretation and provenance</summary><p>Every colored splat is one of the <code>@@COUNT@@</code> points supplied to DeAL after beta=1 inverse-density sampling. In each of the six encoder blocks, DeAL samples <code>@@KEYS_PER_LAYER@@</code> keys and computes the exact anchor-to-key softmax weights. A direct score is the mean of those weights over anchors, heads, and repeated key occurrences. Points never sampled as keys are interpolated from directly observed keys only; the click readout reports this distinction. Similarity is based solely on score percentile, not Euclidean proximity.</p></details>
</main>
<script>@@THREE_JS@@</script>
<script>
const meta = @@METADATA@@;
function bytesFromBase64(encoded) {{ const raw = atob(encoded); const out = new Uint8Array(raw.length); for (let i = 0; i < raw.length; ++i) out[i] = raw.charCodeAt(i); return out; }}
function f32(encoded) {{ const bytes = bytesFromBase64(encoded); return new Float32Array(bytes.buffer); }}
function u16(encoded) {{ const bytes = bytesFromBase64(encoded); return new Uint16Array(bytes.buffer); }}
function u8(encoded) {{ const bytes = bytesFromBase64(encoded); return new Uint8Array(bytes.buffer); }}
const xyz = f32('@@XYZ@@');
const score = f32('@@SCORE@@');
const percentile = u16('@@PERCENTILE@@');
const directCount = u8('@@DIRECT_COUNT@@');
const count = @@COUNT_RAW@@;
const rank = new Float32Array(count);
for (let i = 0; i < count; ++i) rank[i] = percentile[i] / 65535.0;

const viewer = document.getElementById('viewer');
const slider = document.getElementById('tolerance');
const toleranceText = document.getElementById('toleranceValue');
const readout = document.getElementById('readout');
const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: false, powerPreference: 'high-performance' }});
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.setClearColor(0x090d12, 1);
renderer.outputColorSpace = THREE.SRGBColorSpace;
viewer.insertBefore(renderer.domElement, viewer.firstChild);
const scene = new THREE.Scene();
const pickScene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(42, 1, 0.0001, 1000);
const geometry = new THREE.BufferGeometry();
geometry.setAttribute('position', new THREE.BufferAttribute(xyz, 3));
geometry.setAttribute('scoreRank', new THREE.BufferAttribute(rank, 1));
const idColor = new Float32Array(count * 3);
for (let i = 0; i < count; ++i) {{ const id = i + 1; idColor[3*i] = (id & 255) / 255; idColor[3*i+1] = ((id >> 8) & 255) / 255; idColor[3*i+2] = ((id >> 16) & 255) / 255; }}
geometry.setAttribute('idColor', new THREE.BufferAttribute(idColor, 3));

const gaussianVertex = `
  attribute float scoreRank;
  varying float vRank;
  uniform float uPointScale;
  void main() {{
    vRank = scoreRank;
    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
    gl_PointSize = clamp((16.0 * uPointScale) / max(0.15, -mvPosition.z), 1.5, 8.0);
    gl_Position = projectionMatrix * mvPosition;
  }}`;
const gaussianFragment = `
  varying float vRank;
  vec3 palette(float t) {{
    if (t < 0.25) return mix(vec3(0.188, 0.071, 0.231), vec3(0.200, 0.388, 0.553), t / 0.25);
    if (t < 0.50) return mix(vec3(0.200, 0.388, 0.553), vec3(0.137, 0.655, 0.643), (t - 0.25) / 0.25);
    if (t < 0.75) return mix(vec3(0.137, 0.655, 0.643), vec3(0.957, 0.827, 0.369), (t - 0.50) / 0.25);
    return mix(vec3(0.957, 0.827, 0.369), vec3(0.851, 0.263, 0.369), (t - 0.75) / 0.25);
  }}
  void main() {{
    vec2 d = gl_PointCoord - vec2(0.5);
    float alpha = exp(-18.0 * dot(d, d));
    if (alpha < 0.025) discard;
    gl_FragColor = vec4(palette(clamp(vRank, 0.0, 1.0)), 0.90 * alpha);
  }`;
const pointMaterial = new THREE.ShaderMaterial({{ vertexShader: gaussianVertex, fragmentShader: gaussianFragment, transparent: true, depthWrite: false, uniforms: {{ uPointScale: {{ value: 1.0 }} }} }});
scene.add(new THREE.Points(geometry, pointMaterial));

const pickVertex = `
  attribute vec3 idColor;
  varying vec3 vId;
  void main() {{
    vId = idColor;
    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
    gl_PointSize = clamp(220.0 / max(0.15, -mvPosition.z), 12.0, 30.0);
    gl_Position = projectionMatrix * mvPosition;
  }}`;
const pickFragment = `
  varying vec3 vId;
  void main() {{
    if (length(gl_PointCoord - vec2(0.5)) > 0.5) discard;
    gl_FragColor = vec4(vId, 1.0);
  }`;
const pickMaterial = new THREE.ShaderMaterial({{ vertexShader: pickVertex, fragmentShader: pickFragment, depthTest: true, depthWrite: true, transparent: false, blending: THREE.NoBlending }});
pickScene.add(new THREE.Points(geometry, pickMaterial));

const maxSimilar = 7000;
const similarPositions = new Float32Array(maxSimilar * 3);
const similarGeometry = new THREE.BufferGeometry();
similarGeometry.setAttribute('position', new THREE.BufferAttribute(similarPositions, 3));
similarGeometry.setDrawRange(0, 0);
const similarMaterial = new THREE.PointsMaterial({{ color: 0xf9cf58, size: 5.0, sizeAttenuation: false, transparent: true, opacity: 0.98, depthTest: false }});
scene.add(new THREE.Points(similarGeometry, similarMaterial));
const selectedGeometry = new THREE.BufferGeometry();
selectedGeometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(3), 3));
const selectedMaterial = new THREE.PointsMaterial({{ color: 0xffffff, size: 13.0, sizeAttenuation: false, transparent: true, opacity: 1.0, depthTest: false }});
const selectedPoints = new THREE.Points(selectedGeometry, selectedMaterial);
selectedPoints.visible = false;
scene.add(selectedPoints);

const box = new THREE.Box3().setFromBufferAttribute(geometry.getAttribute('position'));
const center = box.getCenter(new THREE.Vector3());
const span = box.getSize(new THREE.Vector3());
const radius = Math.max(span.x, span.y, span.z, 1e-3) * 0.5;
let azimuth = -0.70, elevation = 0.27, distance = radius * 2.25;
let pickTarget = null, selectedIndex = -1, dragging = false, moved = false, lastX = 0, lastY = 0;
function updateCamera() {{
  const ce = Math.cos(elevation);
  camera.position.set(center.x + distance * ce * Math.cos(azimuth), center.y + distance * ce * Math.sin(azimuth), center.z + distance * Math.sin(elevation));
  camera.lookAt(center);
  camera.updateMatrixWorld();
}}
function render() {{ updateCamera(); renderer.setRenderTarget(null); renderer.render(scene, camera); }}
function resize() {{
  const rect = viewer.getBoundingClientRect();
  const width = Math.max(1, Math.floor(rect.width)), height = Math.max(1, Math.floor(rect.height));
  renderer.setSize(width, height, false);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  const pixelRatio = renderer.getPixelRatio();
  if (pickTarget) pickTarget.dispose();
  pickTarget = new THREE.WebGLRenderTarget(Math.max(1, Math.floor(width * pixelRatio)), Math.max(1, Math.floor(height * pixelRatio)), {{ depthBuffer: true }});
  render();
}}
function similarityTolerance() {{ return Number(slider.value) / 100.0; }}
function updateToleranceText() {{ toleranceText.textContent = Number(slider.value).toFixed(2); }}
function updateSelection(index) {{
  selectedIndex = index;
  const target = rank[index], tol = similarityTolerance();
  let matches = 0, written = 0, stride = 1;
  for (let i = 0; i < count; ++i) if (Math.abs(rank[i] - target) <= tol) ++matches;
  stride = Math.max(1, Math.ceil(matches / maxSimilar));
  for (let i = 0, seen = 0; i < count; ++i) {{
    if (Math.abs(rank[i] - target) > tol) continue;
    if ((seen++ % stride) !== 0 || written >= maxSimilar) continue;
    similarPositions[3*written] = xyz[3*i]; similarPositions[3*written+1] = xyz[3*i+1]; similarPositions[3*written+2] = xyz[3*i+2]; ++written;
  }}
  if (written === 0) {{ similarPositions[0] = xyz[3*index]; similarPositions[1] = xyz[3*index+1]; similarPositions[2] = xyz[3*index+2]; written = 1; }}
  similarGeometry.attributes.position.needsUpdate = true;
  similarGeometry.setDrawRange(0, written);
  const selected = selectedGeometry.attributes.position.array;
  selected[0] = xyz[3*index]; selected[1] = xyz[3*index+1]; selected[2] = xyz[3*index+2];
  selectedGeometry.attributes.position.needsUpdate = true;
  selectedPoints.visible = true;
  const direct = directCount[index] > 0;
  const directText = direct ? 'direct encoder key (' + directCount[index] + ' draw' + (directCount[index] === 1 ? '' : 's') + ')' : 'interpolated from directly attended keys';
  const displayed = written === matches ? written.toLocaleString() : written.toLocaleString() + ' of ' + matches.toLocaleString();
  readout.innerHTML = '<strong>Selected point ' + index.toLocaleString() + '</strong> &middot; raw attention mass <strong>' + score[index].toExponential(4) + '</strong> &middot; score percentile <strong>' + (100 * rank[index]).toFixed(2) + '%</strong> &middot; ' + directText + ' &middot; highlighted: <strong>' + displayed + '</strong>';
  render();
}}
function pick(event) {{
  const rect = viewer.getBoundingClientRect(), pr = renderer.getPixelRatio();
  const px = Math.max(0, Math.min(pickTarget.width - 1, Math.floor((event.clientX - rect.left) * pr)));
  const py = Math.max(0, Math.min(pickTarget.height - 1, Math.floor((rect.bottom - event.clientY) * pr)));
  updateCamera(); renderer.setRenderTarget(pickTarget); renderer.clear(); renderer.render(pickScene, camera);
  const pixel = new Uint8Array(4); renderer.readRenderTargetPixels(pickTarget, px, py, 1, 1, pixel); renderer.setRenderTarget(null);
  const id = pixel[0] + (pixel[1] << 8) + (pixel[2] << 16);
  if (id > 0 && id <= count) updateSelection(id - 1);
}}
viewer.addEventListener('pointerdown', event => {{ dragging = true; moved = false; lastX = event.clientX; lastY = event.clientY; viewer.setPointerCapture(event.pointerId); }});
viewer.addEventListener('pointermove', event => {{
  if (!dragging) return;
  const dx = event.clientX - lastX, dy = event.clientY - lastY;
  if (Math.abs(dx) + Math.abs(dy) > 2) moved = true;
  if (moved) {{ azimuth -= dx * 0.006; elevation = Math.max(-1.45, Math.min(1.45, elevation + dy * 0.006)); render(); }}
  lastX = event.clientX; lastY = event.clientY;
}});
viewer.addEventListener('pointerup', event => {{ if (!dragging) return; dragging = false; try {{ viewer.releasePointerCapture(event.pointerId); }} catch (_) {{}} if (!moved) pick(event); }});
viewer.addEventListener('wheel', event => {{ event.preventDefault(); distance = Math.max(radius * 0.25, Math.min(radius * 8.0, distance * Math.exp(event.deltaY * 0.001))); render(); }}, {{ passive: false }});
slider.addEventListener('input', () => {{ updateToleranceText(); if (selectedIndex >= 0) updateSelection(selectedIndex); }});
window.addEventListener('resize', resize);
updateToleranceText(); resize();
</script>
</body>
</html>"""
    template = template.replace("{{", "{").replace("}}", "}")
    return (
        template.replace("@@THREE_JS@@", three_js)
        .replace("@@METADATA@@", metadata_json)
        .replace("@@XYZ@@", str(payload["xyz"]))
        .replace("@@SCORE@@", str(payload["score"]))
        .replace("@@PERCENTILE@@", str(payload["percentile"]))
        .replace("@@DIRECT_COUNT@@", str(payload["direct_count"]))
        .replace("@@COUNT_RAW@@", str(payload["count"]))
        .replace("@@COUNT@@", f"{int(payload['count']):,}")
        .replace("@@KEYS_PER_LAYER@@", f"{int(payload['keys_per_layer']):,}")
    )


def main() -> None:
    args = parse_args()
    if args.input_points != 131072:
        raise ValueError("This diagnostic is fixed to the requested 131072-point encoder input budget.")
    if min(args.attention_key_chunk_size, args.interpolation_neighbors) <= 0 or args.interpolation_workers == 0:
        raise ValueError("Attention chunks, interpolation neighbors, and nonzero worker count are required.")
    if not args.deal_checkpoint.expanduser().is_file():
        raise FileNotFoundError(f"DeAL checkpoint not found: {args.deal_checkpoint}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model, config = build_model(args.deal_config, args.deal_checkpoint.expanduser().resolve(), device)
    if int(config.architecture.subsampled_geometry_points) != int(model.subsampled_geometry_points):
        raise RuntimeError("Loaded DeAL configuration and model encoder-key budget disagree.")
    dataset = AhmedMLDatasetV2(
        saved_folder=str(args.data_root.expanduser().resolve()),
        if_test=True,
        geometry_points=0,
        surface_points=1,
        volume_points=1,
        require_preprocessed=True,
        return_geometry_density=True,
        geometry_density_knn_k=int(args.density_knn_k),
        geometry_density_estimator=str(args.density_estimator),
        geometry_density_cache_dtype="float16",
        geometry_epoch_seeded_sampling=False,
    )
    if int(args.run_id) not in set(dataset.all_ids):
        raise ValueError(f"run_{args.run_id} is not available in {args.data_root}")
    run_dir = args.data_root.expanduser().resolve() / f"run_{int(args.run_id)}"
    full_raw = np.array(np.load(run_dir / "surface_coords.npy", mmap_mode="r"), dtype=np.float32, copy=True)
    if full_raw.ndim != 2 or full_raw.shape[1] != 3 or not np.isfinite(full_raw).all():
        raise RuntimeError(f"Invalid surface point cloud in {run_dir}")
    span = torch.clamp(dataset.max_pos - dataset.min_pos, min=1.0e-12)
    full_geometry = (torch.from_numpy(full_raw) - dataset.min_pos) / span
    density = dataset._load_or_compute_full_geometry_density(int(args.run_id), expected_n=int(full_geometry.shape[0])).float()
    if density.shape[0] != full_geometry.shape[0]:
        raise RuntimeError("KDE16 density cache does not align with the source surface cloud.")

    sampling_seed = int(args.seed + 100003 * int(args.run_id))
    beta_view = sample_condition("beta1", full_geometry, density, int(args.input_points), sampling_seed, None)
    if beta_view.shape != (int(args.input_points), 3) or not torch.isfinite(beta_view).all():
        raise RuntimeError("The beta=1 view is not a finite 131072-point encoder input.")
    raw_view = (beta_view.numpy() * span.numpy() + dataset.min_pos.numpy()).astype(np.float32)
    direct, direct_count, layer_mass = exact_encoder_key_attention(
        model, beta_view, sampling_seed, int(args.attention_key_chunk_size)
    )
    score = _idw_missing_scores(raw_view, direct, direct_count, int(args.interpolation_neighbors), int(args.interpolation_workers))
    if not np.isfinite(score).all() or not np.isfinite(raw_view).all():
        raise FloatingPointError("The exported point cloud or attention scores contain non-finite values.")
    if float(np.std(score)) <= 1.0e-12:
        raise RuntimeError("Cross-attention score map is numerically collapsed; refusing to export a misleading HTML.")
    percentile = _score_percentile(score)
    percentile_u16 = np.rint(percentile * np.iinfo(np.uint16).max).astype(np.uint16)
    observed = direct_count > 0
    metadata: dict[str, Any] = {
        "run_id": int(args.run_id),
        "checkpoint": str(args.deal_checkpoint.expanduser().resolve()),
        "config": str(args.deal_config),
        "condition": "beta=1 inverse-density sampling without replacement",
        "input_points": int(raw_view.shape[0]),
        "encoder_layers": int(len(model.encoder_blocks)),
        "anchors_per_layer": int(model.num_geo),
        "encoder_keys_per_layer": int(model.subsampled_geometry_points),
        "attention_score": "mean softmax encoder cross-attention mass over anchors, heads, and repeated key occurrences",
        "interpolation": "inverse-distance interpolation only for input points never sampled as encoder keys",
        "direct_key_coverage": {"count": int(observed.sum()), "fraction": float(observed.mean())},
        "layer_attention_mass": layer_mass,
        "score_statistics": {
            "min": float(score.min()),
            "max": float(score.max()),
            "mean": float(score.mean()),
            "std": float(score.std()),
            "percentiles": {str(p): float(np.percentile(score, p)) for p in (1, 5, 25, 50, 75, 95, 99)},
        },
    }
    stem = f"drivaerml_run_{int(args.run_id)}_deal_beta1_cross_attention_131k"
    np.savez_compressed(
        output_dir / f"{stem}.npz",
        input_points=raw_view,
        attention_score=score,
        attention_score_percentile=percentile,
        direct_encoder_key_draw_count=direct_count,
    )
    payload: dict[str, str | int | float] = {
        "xyz": _binary_payload(raw_view.reshape(-1), np.float32),
        "score": _binary_payload(score, np.float32),
        "percentile": _binary_payload(percentile_u16, np.uint16),
        "direct_count": _binary_payload(direct_count, np.uint8),
        "count": int(raw_view.shape[0]),
        "keys_per_layer": int(model.subsampled_geometry_points),
    }
    html_path = output_dir / f"{stem}.html"
    html_path.write_text(_gpu_html_document(payload, metadata), encoding="utf-8")
    (output_dir / f"{stem}_summary.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {html_path}")
    print(f"Direct encoder-key coverage: {observed.sum()}/{raw_view.shape[0]} ({100.0 * observed.mean():.2f}%)")
    print(f"Attention score range: [{score.min():.6e}, {score.max():.6e}], std={score.std():.6e}")


if __name__ == "__main__":
    main()
