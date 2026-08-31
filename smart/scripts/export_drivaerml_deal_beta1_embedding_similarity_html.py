#!/usr/bin/env python3
"""Export interactive DeAL point-embedding similarity viewers for a beta=1 view.

The two exports intentionally answer different questions:

* ``encoder_key_embedding`` is the final encoder block's learned key vector
  before anchor-to-geometry cross-attention.  It is pointwise and mostly
  coordinate-driven.
* ``decoder_query_embedding`` is the final surface-query token immediately
  before SMART's prediction MLP.  It has interacted with the geometry-
  conditioned anchor latents and is the meaningful candidate semantic space.

Both use cosine similarity in a PCA-compressed representation and a local,
GPU-rendered Three.js viewer with deterministic offscreen point picking.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
SMART_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(SMART_ROOT) not in sys.path:
    sys.path.insert(0, str(SMART_ROOT))

from data.ahmedml_dataset_v2 import AhmedMLDatasetV2  # noqa: E402
from export_drivaerml_deal_beta1_attention_html import _binary_payload, _score_percentile  # noqa: E402
from export_drivaerml_smart_anchor_attention import build_model, sample_condition, sample_model_indices  # noqa: E402


DEFAULT_CHECKPOINT = Path(
    "/home/parsa/smart_parsa/checkpoints/"
    "smart-satloss7-range100-smart-satloss7-range100-from-smart-150ep-drivaerml-s42_best.pt"
)
DEFAULT_OUTPUT_DIR = Path("/home/parsa/smart_parsa/results/final/drivaerml_deal_beta1_embedding_similarity")


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
    parser.add_argument("--query-chunk-size", type=int, default=2048)
    parser.add_argument("--embedding-components", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


@torch.inference_mode()
def _encode_with_seed(model: torch.nn.Module, geometry: torch.Tensor, seed: int) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Mirror SMART.encode while retaining the exact seeded encoder subsamples."""
    device = next(model.parameters()).device
    geo = geometry.unsqueeze(0).to(device=device, dtype=torch.float32, non_blocking=True)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    scaled = geo * float(model.pos_scale_factor)
    latent_idx = sample_model_indices(int(scaled.shape[1]), int(model.num_geo), False, generator, device).unsqueeze(0)
    latent_pos = torch.gather(scaled, 1, latent_idx.unsqueeze(-1).expand(-1, -1, 3))
    latent = model.pos_encoder(latent_pos)
    intermediate: list[torch.Tensor] = []
    for block in model.encoder_blocks:
        sub_idx = sample_model_indices(
            int(scaled.shape[1]),
            int(model.subsampled_geometry_points),
            bool(model.subsampled_geometry_with_replacement),
            generator,
            device,
        ).unsqueeze(0)
        sub_pos = torch.gather(scaled, 1, sub_idx.unsqueeze(-1).expand(-1, -1, 3))
        sub_tokens = model.pos_encoder(sub_pos)
        latent, cross_attended = block(
            latent,
            sub_tokens,
            None,
            latent_geometry_pos=latent_pos,
            subsampled_geometry_pos=sub_pos,
        )
        intermediate.append(cross_attended)
    return intermediate, latent_pos


@torch.inference_mode()
def _last_encoder_key_embedding(model: torch.nn.Module, geometry: torch.Tensor) -> torch.Tensor:
    """Compute the last block's exact K vectors for every input point."""
    device = next(model.parameters()).device
    scaled = geometry.unsqueeze(0).to(device=device, dtype=torch.float32, non_blocking=True) * float(model.pos_scale_factor)
    tokens = model.pos_encoder(scaled)
    attention = model.encoder_blocks[-1].geo_attn
    key_value = attention.kv(attention.norm_kv(tokens))
    key_value = key_value.view(1, tokens.shape[1], 2 * attention.num_heads, attention.head_dim).permute(0, 2, 1, 3)
    keys = attention.rope(key_value[:, : attention.num_heads], scaled)
    return keys.permute(0, 2, 1, 3).reshape(tokens.shape[1], -1).float()


@torch.inference_mode()
def _final_decoder_query_embedding(
    model: torch.nn.Module,
    geometry: torch.Tensor,
    seed: int,
    chunk_size: int,
) -> torch.Tensor:
    """Return the native final decoder tokens immediately before SMART.mlp."""
    intermediate, latent_pos = _encode_with_seed(model, geometry, seed)
    device = next(model.parameters()).device
    chunks: list[torch.Tensor] = []
    for start in range(0, int(geometry.shape[0]), int(chunk_size)):
        queries = geometry[start : start + int(chunk_size)].unsqueeze(0).to(device=device, dtype=torch.float32)
        token = model.decode_features(intermediate, latent_pos, None, queries)
        chunks.append(token[0].float().cpu())
    return torch.cat(chunks, dim=0)


def _compress_cosine_features(
    features: torch.Tensor,
    components: int,
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Use a deterministic truncated orthogonal basis for compact browser cosine search."""
    if features.ndim != 2 or not torch.isfinite(features).all():
        raise ValueError("Expected a finite [points, features] embedding matrix.")
    max_components = min(int(features.shape[0]), int(features.shape[1]))
    components = min(max(2, int(components)), max_components)
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    normalized = F.normalize(features.to(device=device, dtype=torch.float32, non_blocking=True), dim=1, eps=1.0e-8)
    _u, singular_values, basis = torch.pca_lowrank(normalized, q=components, center=False, niter=4)
    projected = F.normalize(normalized @ basis[:, :components], dim=1, eps=1.0e-8)
    retained_energy = float((singular_values.square().sum() / normalized.square().sum().clamp_min(1.0e-12)).item())

    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed) + 1901)
    sample_count = min(20000, int(normalized.shape[0]))
    left = torch.randint(int(normalized.shape[0]), (sample_count,), generator=generator, device=device)
    right = torch.randint(int(normalized.shape[0]), (sample_count,), generator=generator, device=device)
    exact_similarity = (normalized[left] * normalized[right]).sum(dim=1)
    projected_similarity = (projected[left] * projected[right]).sum(dim=1)
    exact_centered = exact_similarity - exact_similarity.mean()
    projected_centered = projected_similarity - projected_similarity.mean()
    correlation = (exact_centered * projected_centered).mean() / (
        exact_centered.square().mean().sqrt() * projected_centered.square().mean().sqrt()
    ).clamp_min(1.0e-8)
    diagnostics = {
        "components": float(components),
        "retained_energy": retained_energy,
        "sampled_cosine_mae": float((exact_similarity - projected_similarity).abs().mean().item()),
        "sampled_cosine_correlation": float(correlation.item()),
    }
    return projected.cpu().numpy().astype(np.float32), basis[:, :components].cpu().numpy().astype(np.float32), diagnostics


def _embedding_html_document(payload: dict[str, str | int], metadata: dict[str, Any]) -> str:
    three_path = SMART_ROOT / "vendor" / "three.min.js"
    if not three_path.is_file():
        raise FileNotFoundError(f"Missing vendored Three.js runtime: {three_path}")
    three_js = three_path.read_text(encoding="utf-8")
    template = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>@@TITLE@@</title>
<style>
  :root {{ color-scheme: dark; --panel:#111821; --line:#344352; --text:#edf4fa; --muted:#a8b8c7; --accent:#f9cf58; }}
  * {{ box-sizing:border-box; }} body {{ margin:0; background:#090d12; color:var(--text); font:15px/1.45 Georgia, 'Times New Roman', serif; }}
  main {{ max-width:1550px; margin:0 auto; padding:22px 26px 30px; }} h1 {{ margin:0 0 5px; font:700 clamp(22px,2.2vw,33px)/1.15 'Trebuchet MS',sans-serif; }}
  .sub {{ margin:0 0 16px; color:var(--muted); max-width:1130px; }} #viewer {{ position:relative; height:min(76vh,850px); min-height:600px; border:1px solid var(--line); border-radius:10px; overflow:hidden; cursor:crosshair; }}
  #viewer canvas {{ display:block; width:100%; height:100%; }} .colorbar {{ position:absolute; right:20px; top:22px; display:flex; gap:8px; color:#eef6fc; font:12px/1.1 'Trebuchet MS',sans-serif; pointer-events:none; text-shadow:0 1px 2px #000; }}
  .gradient {{ width:13px; height:150px; border:1px solid #d5e1e8; background:linear-gradient(to top,#30123b,#33638d 25%,#23a7a4 50%,#f4d35e 75%,#d9435e); }} .colorlabels {{ display:flex; flex-direction:column; justify-content:space-between; padding:1px 0; }}
  .controls {{ display:grid; grid-template-columns:minmax(270px,450px) 1fr; gap:16px 25px; margin:15px 0 8px; align-items:center; }} .control,#readout {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:12px 14px; }}
  label,summary {{ font-family:'Trebuchet MS',sans-serif; font-weight:700; }} input {{ width:100%; accent-color:var(--accent); margin-top:8px; }} #readout {{ color:var(--muted); min-height:52px; }} #readout strong,summary {{ color:var(--text); }} details {{ margin-top:12px; color:var(--muted); }} summary {{ cursor:pointer; }} code {{ color:#b9e8ff; }}
  @media(max-width:700px) {{ main{{padding:15px}} #viewer{{min-height:460px}} .controls{{grid-template-columns:1fr}} }}
</style>
</head>
<body><main>
  <h1>@@TITLE@@</h1><p class="sub">@@DESCRIPTION@@</p>
  <div id="viewer"><div class="colorbar"><div class="gradient"></div><div class="colorlabels"><span>high</span><span>PCA-1 rank<br>(context only)</span><span>low</span></div></div></div>
  <section class="controls"><div class="control"><label for="threshold">Cosine-similarity threshold: <span id="thresholdValue"></span></label><input id="threshold" type="range" min="0.50" max="0.99" value="0.85" step="0.01"></div><div id="readout"><strong>Click a point</strong> to highlight points whose compressed embedding has cosine similarity above the selected threshold.</div></section>
  <details><summary>Embedding and similarity details</summary><p>Each point stores a normalized <code>@@DIM@@</code>-D embedding from an original <code>@@ORIGINAL_DIM@@</code>-D model representation. The browser computes cosine similarity for every click. The compression retains <code>@@ENERGY@@</code> of normalized-feature energy; sampled exact-versus-compressed cosine correlation is <code>@@CORRELATION@@</code> with MAE <code>@@MAE@@</code>. Gold splats are not a spatial neighborhood: they are points satisfying the selected feature-similarity threshold.</p></details>
</main><script>@@THREE_JS@@</script><script>
function bytesFromBase64(encoded) {{ const raw=atob(encoded), out=new Uint8Array(raw.length); for(let i=0;i<raw.length;++i) out[i]=raw.charCodeAt(i); return out; }}
function f32(encoded) {{ const bytes=bytesFromBase64(encoded); return new Float32Array(bytes.buffer); }} function u16(encoded) {{ const bytes=bytesFromBase64(encoded); return new Uint16Array(bytes.buffer); }}
const xyz=f32('@@XYZ@@'), rank16=u16('@@RANK@@'), embedding=f32('@@EMBEDDING@@'); const count=@@COUNT@@, dim=@@DIM_RAW@@; const rank=new Float32Array(count); for(let i=0;i<count;++i) rank[i]=rank16[i]/65535.0;
const viewer=document.getElementById('viewer'), slider=document.getElementById('threshold'), thresholdText=document.getElementById('thresholdValue'), readout=document.getElementById('readout');
const renderer=new THREE.WebGLRenderer({{antialias:true,alpha:false,powerPreference:'high-performance'}}); renderer.setPixelRatio(Math.min(window.devicePixelRatio||1,2)); renderer.setClearColor(0x090d12,1); renderer.outputColorSpace=THREE.SRGBColorSpace; viewer.insertBefore(renderer.domElement,viewer.firstChild);
const scene=new THREE.Scene(), pickScene=new THREE.Scene(), camera=new THREE.PerspectiveCamera(42,1,.0001,1000), geometry=new THREE.BufferGeometry(); geometry.setAttribute('position',new THREE.BufferAttribute(xyz,3)); geometry.setAttribute('scoreRank',new THREE.BufferAttribute(rank,1));
const idColor=new Float32Array(count*3); for(let i=0;i<count;++i){{const id=i+1;idColor[3*i]=(id&255)/255;idColor[3*i+1]=((id>>8)&255)/255;idColor[3*i+2]=((id>>16)&255)/255;}} geometry.setAttribute('idColor',new THREE.BufferAttribute(idColor,3));
const vertex=`attribute float scoreRank;varying float vRank;void main(){{vRank=scoreRank;vec4 mv=modelViewMatrix*vec4(position,1.0);gl_PointSize=clamp(16.0/max(.15,-mv.z),1.5,8.0);gl_Position=projectionMatrix*mv;}}`;
const fragment=`varying float vRank;vec3 p(float t){{if(t<.25)return mix(vec3(.188,.071,.231),vec3(.2,.388,.553),t/.25);if(t<.5)return mix(vec3(.2,.388,.553),vec3(.137,.655,.643),(t-.25)/.25);if(t<.75)return mix(vec3(.137,.655,.643),vec3(.957,.827,.369),(t-.5)/.25);return mix(vec3(.957,.827,.369),vec3(.851,.263,.369),(t-.75)/.25);}}void main(){{vec2 d=gl_PointCoord-vec2(.5);float a=exp(-18.*dot(d,d));if(a<.025)discard;gl_FragColor=vec4(p(vRank),.90*a);}}`;
scene.add(new THREE.Points(geometry,new THREE.ShaderMaterial({{vertexShader:vertex,fragmentShader:fragment,transparent:true,depthWrite:false}})));
const pickVertex=`attribute vec3 idColor;varying vec3 vId;void main(){{vId=idColor;vec4 mv=modelViewMatrix*vec4(position,1.0);gl_PointSize=clamp(220.0/max(.15,-mv.z),12.0,30.0);gl_Position=projectionMatrix*mv;}}`,pickFragment=`varying vec3 vId;void main(){{if(length(gl_PointCoord-vec2(.5))>.5)discard;gl_FragColor=vec4(vId,1.0);}}`; pickScene.add(new THREE.Points(geometry,new THREE.ShaderMaterial({{vertexShader:pickVertex,fragmentShader:pickFragment,depthTest:true,depthWrite:true,blending:THREE.NoBlending}})));
const maxSimilar=10000, similarPositions=new Float32Array(maxSimilar*3), similarGeometry=new THREE.BufferGeometry();similarGeometry.setAttribute('position',new THREE.BufferAttribute(similarPositions,3));similarGeometry.setDrawRange(0,0);scene.add(new THREE.Points(similarGeometry,new THREE.PointsMaterial({{color:0xf9cf58,size:5,sizeAttenuation:false,transparent:true,opacity:.98,depthTest:false}})));
const selectedGeometry=new THREE.BufferGeometry();selectedGeometry.setAttribute('position',new THREE.BufferAttribute(new Float32Array(3),3));const selected=new THREE.Points(selectedGeometry,new THREE.PointsMaterial({{color:0xffffff,size:13,sizeAttenuation:false,depthTest:false}}));selected.visible=false;scene.add(selected);
const box=new THREE.Box3().setFromBufferAttribute(geometry.getAttribute('position')),center=box.getCenter(new THREE.Vector3()),span=box.getSize(new THREE.Vector3()),radius=Math.max(span.x,span.y,span.z,.001)*.5;let azimuth=-.70,elevation=.27,distance=radius*2.25,pickTarget=null,selectedIndex=-1,dragging=false,moved=false,lastX=0,lastY=0;
function cameraUpdate(){{const ce=Math.cos(elevation);camera.position.set(center.x+distance*ce*Math.cos(azimuth),center.y+distance*ce*Math.sin(azimuth),center.z+distance*Math.sin(elevation));camera.lookAt(center);camera.updateMatrixWorld();}}function render(){{cameraUpdate();renderer.setRenderTarget(null);renderer.render(scene,camera);}}function resize(){{const r=viewer.getBoundingClientRect(),w=Math.max(1,Math.floor(r.width)),h=Math.max(1,Math.floor(r.height));renderer.setSize(w,h,false);camera.aspect=w/h;camera.updateProjectionMatrix();if(pickTarget)pickTarget.dispose();const pr=renderer.getPixelRatio();pickTarget=new THREE.WebGLRenderTarget(Math.floor(w*pr),Math.floor(h*pr),{{depthBuffer:true}});render();}}
function updateThreshold(){{thresholdText.textContent=Number(slider.value).toFixed(2);}}function updateSelection(index){{selectedIndex=index;const threshold=Number(slider.value),base=index*dim;let matches=0;for(let i=0;i<count;++i){{let dot=0,off=i*dim;for(let j=0;j<dim;++j)dot+=embedding[base+j]*embedding[off+j];if(dot>=threshold)++matches;}}const stride=Math.max(1,Math.ceil(matches/maxSimilar));let written=0,seen=0,best=-2,bestIndex=-1;for(let i=0;i<count;++i){{let dot=0,off=i*dim;for(let j=0;j<dim;++j)dot+=embedding[base+j]*embedding[off+j];if(i!==index&&dot>best){{best=dot;bestIndex=i;}}if(dot<threshold||(seen++%stride)!==0||written>=maxSimilar)continue;similarPositions[3*written]=xyz[3*i];similarPositions[3*written+1]=xyz[3*i+1];similarPositions[3*written+2]=xyz[3*i+2];++written;}}similarGeometry.attributes.position.needsUpdate=true;similarGeometry.setDrawRange(0,written);const pos=selectedGeometry.attributes.position.array;pos[0]=xyz[3*index];pos[1]=xyz[3*index+1];pos[2]=xyz[3*index+2];selectedGeometry.attributes.position.needsUpdate=true;selected.visible=true;const shown=written===matches?written.toLocaleString():written.toLocaleString()+' of '+matches.toLocaleString();readout.innerHTML='<strong>Selected point '+index.toLocaleString()+'</strong> &middot; self cosine <strong>1.000</strong> &middot; nearest non-self cosine <strong>'+best.toFixed(4)+'</strong> &middot; highlighted <strong>'+shown+'</strong> at cosine &ge; '+threshold.toFixed(2);render();}}
function pick(event){{const rect=viewer.getBoundingClientRect(),pr=renderer.getPixelRatio(),x=Math.max(0,Math.min(pickTarget.width-1,Math.floor((event.clientX-rect.left)*pr))),y=Math.max(0,Math.min(pickTarget.height-1,Math.floor((rect.bottom-event.clientY)*pr)));cameraUpdate();renderer.setRenderTarget(pickTarget);renderer.clear();renderer.render(pickScene,camera);const pixel=new Uint8Array(4);renderer.readRenderTargetPixels(pickTarget,x,y,1,1,pixel);renderer.setRenderTarget(null);const id=pixel[0]+(pixel[1]<<8)+(pixel[2]<<16);if(id>0&&id<=count)updateSelection(id-1);}}
viewer.addEventListener('pointerdown',e=>{{dragging=true;moved=false;lastX=e.clientX;lastY=e.clientY;viewer.setPointerCapture(e.pointerId);}});viewer.addEventListener('pointermove',e=>{{if(!dragging)return;const dx=e.clientX-lastX,dy=e.clientY-lastY;if(Math.abs(dx)+Math.abs(dy)>2)moved=true;if(moved){{azimuth-=dx*.006;elevation=Math.max(-1.45,Math.min(1.45,elevation+dy*.006));render();}}lastX=e.clientX;lastY=e.clientY;}});viewer.addEventListener('pointerup',e=>{{if(!dragging)return;dragging=false;try{{viewer.releasePointerCapture(e.pointerId);}}catch(_){{}}if(!moved)pick(e);}});viewer.addEventListener('wheel',e=>{{e.preventDefault();distance=Math.max(radius*.25,Math.min(radius*8,distance*Math.exp(e.deltaY*.001)));render();}},{{passive:false}});slider.addEventListener('input',()=>{{updateThreshold();if(selectedIndex>=0)updateSelection(selectedIndex);}});window.addEventListener('resize',resize);updateThreshold();resize();
</script></body></html>"""
    template = template.replace("{{", "{").replace("}}", "}")
    diagnostics = metadata["compression"]
    return (
        template.replace("@@THREE_JS@@", three_js)
        .replace("@@TITLE@@", str(metadata["title"]))
        .replace("@@DESCRIPTION@@", str(metadata["description"]))
        .replace("@@XYZ@@", str(payload["xyz"]))
        .replace("@@RANK@@", str(payload["rank"]))
        .replace("@@EMBEDDING@@", str(payload["embedding"]))
        .replace("@@COUNT@@", str(payload["count"]))
        .replace("@@DIM_RAW@@", str(payload["dim"]))
        .replace("@@DIM@@", str(int(diagnostics["components"])))
        .replace("@@ORIGINAL_DIM@@", str(int(metadata["original_dimension"])))
        .replace("@@ENERGY@@", f"{100.0 * diagnostics['retained_energy']:.2f}%")
        .replace("@@CORRELATION@@", f"{diagnostics['sampled_cosine_correlation']:.4f}")
        .replace("@@MAE@@", f"{diagnostics['sampled_cosine_mae']:.4f}")
    )


def _presentation_embedding_html_document(payload: dict[str, str | int], metadata: dict[str, Any]) -> str:
    """A text-free GPU presentation view; similarity search runs off the render thread."""
    three_path = SMART_ROOT / "vendor" / "three.min.js"
    if not three_path.is_file():
        raise FileNotFoundError(f"Missing vendored Three.js runtime: {three_path}")
    three_js = three_path.read_text(encoding="utf-8")
    template = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>@@TITLE@@</title>
<style>html,body,#viewer{{width:100%;height:100%;margin:0;overflow:hidden;background:#fff}}#viewer{{cursor:crosshair}}canvas{{display:block;width:100%;height:100%;touch-action:none}}</style>
</head><body><div id="viewer"></div><script>@@THREE_JS@@</script><script>
function bytesFromBase64(encoded){{const raw=atob(encoded),out=new Uint8Array(raw.length);for(let i=0;i<raw.length;++i)out[i]=raw.charCodeAt(i);return out;}}function f32(encoded){{const b=bytesFromBase64(encoded);return new Float32Array(b.buffer);}}
const xyz=f32('@@XYZ@@'),embedding=f32('@@EMBEDDING@@'),count=@@COUNT@@,dim=@@DIM@@,viewer=document.getElementById('viewer');
const renderer=new THREE.WebGLRenderer({{antialias:false,alpha:false,powerPreference:'high-performance',precision:'highp'}});renderer.setPixelRatio(Math.min(window.devicePixelRatio||1,1.25));renderer.setClearColor(0xffffff,1);renderer.sortObjects=false;viewer.appendChild(renderer.domElement);
const scene=new THREE.Scene(),pickScene=new THREE.Scene(),camera=new THREE.PerspectiveCamera(40,1,.0001,1000),geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.BufferAttribute(xyz,3));const idColor=new Float32Array(count*3);for(let i=0;i<count;++i){{const id=i+1;idColor[3*i]=(id&255)/255;idColor[3*i+1]=((id>>8)&255)/255;idColor[3*i+2]=((id>>16)&255)/255;}}geometry.setAttribute('idColor',new THREE.BufferAttribute(idColor,3));
scene.add(new THREE.Points(geometry,new THREE.PointsMaterial({{color:0x080808,size:1.4,sizeAttenuation:false,depthWrite:true,depthTest:true}})));
const pickVertex=`attribute vec3 idColor;varying vec3 vId;void main(){{vId=idColor;vec4 mv=modelViewMatrix*vec4(position,1.0);gl_PointSize=clamp(260.0/max(.15,-mv.z),14.0,34.0);gl_Position=projectionMatrix*mv;}}`,pickFragment=`varying vec3 vId;void main(){{if(length(gl_PointCoord-vec2(.5))>.5)discard;gl_FragColor=vec4(vId,1.0);}}`;pickScene.add(new THREE.Points(geometry,new THREE.ShaderMaterial({{vertexShader:pickVertex,fragmentShader:pickFragment,depthTest:true,depthWrite:true,blending:THREE.NoBlending}})));
const maxMatches=5000,matchPos=new Float32Array(maxMatches*3),matchGeom=new THREE.BufferGeometry();matchGeom.setAttribute('position',new THREE.BufferAttribute(matchPos,3));matchGeom.setDrawRange(0,0);scene.add(new THREE.Points(matchGeom,new THREE.PointsMaterial({{color:0xe31919,size:4.0,sizeAttenuation:false,depthTest:false,depthWrite:false}})));
const bounds=new THREE.Box3().setFromBufferAttribute(geometry.getAttribute('position')),center=bounds.getCenter(new THREE.Vector3()),extent=bounds.getSize(new THREE.Vector3()),radius=Math.max(extent.x,extent.y,extent.z,.001)*.5;const initial={{azimuth:-.70,elevation:.27,distance:radius*2.25}};let azimuth=initial.azimuth,elevation=initial.elevation,distance=initial.distance,targetAzimuth=azimuth,targetElevation=elevation,targetDistance=distance,velocityAzimuth=0,velocityElevation=0,raf=0,pickTarget=null,dragging=false,moved=false,lastX=0,lastY=0,latestRequest=0;
function cameraUpdate(){{const ce=Math.cos(elevation);camera.position.set(center.x+distance*ce*Math.cos(azimuth),center.y+distance*ce*Math.sin(azimuth),center.z+distance*Math.sin(elevation));camera.lookAt(center);camera.updateMatrixWorld();}}function draw(){{cameraUpdate();renderer.setRenderTarget(null);renderer.render(scene,camera);}}function animate(){{raf=0;azimuth+=(targetAzimuth-azimuth)*.18+velocityAzimuth; elevation+=(targetElevation-elevation)*.18+velocityElevation; distance+=(targetDistance-distance)*.18;velocityAzimuth*=.84;velocityElevation*=.84;draw();if(Math.abs(targetAzimuth-azimuth)+Math.abs(targetElevation-elevation)+Math.abs(targetDistance-distance)/Math.max(radius,.001)+Math.abs(velocityAzimuth)+Math.abs(velocityElevation)>1e-4)requestDraw();}}function requestDraw(){{if(!raf)raf=requestAnimationFrame(animate);}}function resize(){{const rect=viewer.getBoundingClientRect(),width=Math.max(1,Math.floor(rect.width)),height=Math.max(1,Math.floor(rect.height)),pr=renderer.getPixelRatio();renderer.setSize(width,height,false);camera.aspect=width/height;camera.updateProjectionMatrix();if(pickTarget)pickTarget.dispose();pickTarget=new THREE.WebGLRenderTarget(Math.max(1,Math.floor(width*pr)),Math.max(1,Math.floor(height*pr)),{{depthBuffer:true}});draw();}}
const workerSource=`let e,count,dim;self.onmessage=event=>{{const m=event.data;if(m.type==='init'){{e=new Float32Array(m.buffer);count=m.count;dim=m.dim;return;}}if(m.type==='select'){{const base=m.index*dim,similarity=new Float32Array(count),histogram=new Uint32Array(2048);for(let i=0;i<count;++i){{let dot=0,offset=i*dim;for(let j=0;j<dim;++j)dot+=e[base+j]*e[offset+j];similarity[i]=dot;histogram[Math.max(0,Math.min(2047,Math.floor((dot+1)*1023.5)))]++;}}let accumulated=0,bin=2047;for(;bin>=0;--bin){{accumulated+=histogram[bin];if(accumulated>=m.target)break;}}const cutoff=bin/1023.5-1,indices=new Uint32Array(Math.min(m.max,accumulated));let written=0;for(let i=0;i<count&&written<indices.length;++i)if(similarity[i]>=cutoff)indices[written++]=i;self.postMessage({{type:'selection',request:m.request,indices:indices.buffer,written,cutoff}},[indices.buffer]);}}}};`;
const worker=new Worker(URL.createObjectURL(new Blob([workerSource],{{type:'application/javascript'}})));worker.postMessage({{type:'init',buffer:embedding.buffer,count,dim}},[embedding.buffer]);worker.onmessage=event=>{{const m=event.data;if(m.type!=='selection'||m.request!==latestRequest)return;const ids=new Uint32Array(m.indices);for(let i=0;i<m.written;++i){{const index=ids[i];matchPos[3*i]=xyz[3*index];matchPos[3*i+1]=xyz[3*index+1];matchPos[3*i+2]=xyz[3*index+2];}}matchGeom.attributes.position.needsUpdate=true;matchGeom.setDrawRange(0,m.written);draw();}};
function pick(event){{const rect=viewer.getBoundingClientRect(),pr=renderer.getPixelRatio(),x=Math.max(0,Math.min(pickTarget.width-1,Math.floor((event.clientX-rect.left)*pr))),y=Math.max(0,Math.min(pickTarget.height-1,Math.floor((rect.bottom-event.clientY)*pr)));cameraUpdate();renderer.setRenderTarget(pickTarget);renderer.clear();renderer.render(pickScene,camera);const pixel=new Uint8Array(4);renderer.readRenderTargetPixels(pickTarget,x,y,1,1,pixel);renderer.setRenderTarget(null);const id=pixel[0]+(pixel[1]<<8)+(pixel[2]<<16);if(id>0&&id<=count){{latestRequest++;worker.postMessage({{type:'select',index:id-1,target:2500,max:maxMatches,request:latestRequest}});}}}}
viewer.addEventListener('pointerdown',event=>{{dragging=true;moved=false;lastX=event.clientX;lastY=event.clientY;viewer.setPointerCapture(event.pointerId);}});viewer.addEventListener('pointermove',event=>{{if(!dragging)return;const dx=event.clientX-lastX,dy=event.clientY-lastY;if(Math.abs(dx)+Math.abs(dy)>2)moved=true;if(moved){{targetAzimuth-=dx*.006;targetElevation=Math.max(-1.45,Math.min(1.45,targetElevation+dy*.006));velocityAzimuth=-dx*.0008;velocityElevation=dy*.0008;requestDraw();}}lastX=event.clientX;lastY=event.clientY;}});viewer.addEventListener('pointerup',event=>{{if(!dragging)return;dragging=false;try{{viewer.releasePointerCapture(event.pointerId);}}catch(_){{}}if(!moved)pick(event);}});viewer.addEventListener('wheel',event=>{{event.preventDefault();targetDistance=Math.max(radius*.25,Math.min(radius*8,targetDistance*Math.exp(event.deltaY*.001)));requestDraw();}},{{passive:false}});viewer.addEventListener('dblclick',()=>{{targetAzimuth=initial.azimuth;targetElevation=initial.elevation;targetDistance=initial.distance;velocityAzimuth=0;velocityElevation=0;requestDraw();}});window.addEventListener('resize',resize);resize();
</script></body></html>"""
    template = template.replace("{{", "{").replace("}}", "}")
    return (
        template.replace("@@THREE_JS@@", three_js)
        .replace("@@TITLE@@", str(metadata["title"]))
        .replace("@@XYZ@@", str(payload["xyz"]))
        .replace("@@EMBEDDING@@", str(payload["embedding"]))
        .replace("@@COUNT@@", str(payload["count"]))
        .replace("@@DIM@@", str(payload["dim"]))
    )


def _performance_presentation_html_document(payload: dict[str, str | int], metadata: dict[str, Any]) -> str:
    """Minimal white-canvas viewer with adaptive LOD and parallel feature search."""
    three_path = SMART_ROOT / "vendor" / "three.min.js"
    if not three_path.is_file():
        raise FileNotFoundError(f"Missing vendored Three.js runtime: {three_path}")
    three_js = three_path.read_text(encoding="utf-8")
    template = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>@@TITLE@@</title>
<style>html,body,#viewer{{width:100%;height:100%;margin:0;overflow:hidden;background:#fff}}#viewer{{position:relative;cursor:crosshair}}canvas{{display:block;width:100%;height:100%;touch-action:none}}#similarity{{position:absolute;left:50%;bottom:18px;width:min(340px,52vw);height:5px;transform:translateX(-50%);appearance:none;background:#bbb;border-radius:5px;opacity:.82;outline:0}}#similarity::-webkit-slider-thumb{{appearance:none;width:14px;height:14px;border-radius:50%;background:#d7191c;border:1px solid #fff;box-shadow:0 0 0 1px #555}}#similarity::-moz-range-thumb{{width:13px;height:13px;border-radius:50%;background:#d7191c;border:1px solid #fff;box-shadow:0 0 0 1px #555}}</style>
</head><body><div id="viewer"><input id="similarity" aria-label="Similarity threshold" type="range" min="0.60" max="0.99" value="0.88" step="0.01"></div><script>@@THREE_JS@@</script><script>
function bytesFromBase64(encoded){{const raw=atob(encoded),out=new Uint8Array(raw.length);for(let i=0;i<raw.length;++i)out[i]=raw.charCodeAt(i);return out;}}function f32(encoded){{const bytes=bytesFromBase64(encoded);return new Float32Array(bytes.buffer);}}function i8(encoded){{const bytes=bytesFromBase64(encoded);return new Int8Array(bytes.buffer);}}
const xyz=f32('@@XYZ@@');let embedding=i8('@@EMBEDDING@@');const count=@@COUNT@@,dim=@@DIM@@,quantizationScale=127,viewer=document.getElementById('viewer'),slider=document.getElementById('similarity');
const renderer=new THREE.WebGLRenderer({{antialias:false,alpha:false,powerPreference:'high-performance',precision:'highp'}});renderer.setPixelRatio(1);renderer.setClearColor(0xffffff,1);renderer.sortObjects=false;viewer.insertBefore(renderer.domElement,slider);
const scene=new THREE.Scene(),pickScene=new THREE.Scene(),camera=new THREE.PerspectiveCamera(40,1,.0001,1000),geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.BufferAttribute(xyz,3));geometry.getAttribute('position').setUsage(THREE.StaticDrawUsage);const idColor=new Float32Array(count*3);for(let i=0;i<count;++i){{const id=i+1;idColor[3*i]=(id&255)/255;idColor[3*i+1]=((id>>8)&255)/255;idColor[3*i+2]=((id>>16)&255)/255;}}geometry.setAttribute('idColor',new THREE.BufferAttribute(idColor,3));
const baseMaterial=new THREE.PointsMaterial({{color:0x050505,size:1.25,sizeAttenuation:false,depthTest:true,depthWrite:true}}),fullPoints=new THREE.Points(geometry,baseMaterial);scene.add(fullPoints);const previewStride=4,previewCount=Math.ceil(count/previewStride),previewPositions=new Float32Array(previewCount*3);for(let i=0,j=0;i<count;i+=previewStride,++j){{previewPositions[3*j]=xyz[3*i];previewPositions[3*j+1]=xyz[3*i+1];previewPositions[3*j+2]=xyz[3*i+2];}}const previewGeometry=new THREE.BufferGeometry();previewGeometry.setAttribute('position',new THREE.BufferAttribute(previewPositions,3));previewGeometry.getAttribute('position').setUsage(THREE.StaticDrawUsage);const previewPoints=new THREE.Points(previewGeometry,baseMaterial);previewPoints.visible=false;scene.add(previewPoints);
const pickVertex=`attribute vec3 idColor;varying vec3 vId;void main(){{vId=idColor;vec4 mv=modelViewMatrix*vec4(position,1.0);gl_PointSize=clamp(260.0/max(.15,-mv.z),14.0,34.0);gl_Position=projectionMatrix*mv;}}`,pickFragment=`varying vec3 vId;void main(){{if(length(gl_PointCoord-vec2(.5))>.5)discard;gl_FragColor=vec4(vId,1.0);}}`;pickScene.add(new THREE.Points(geometry,new THREE.ShaderMaterial({{vertexShader:pickVertex,fragmentShader:pickFragment,depthTest:true,depthWrite:true,blending:THREE.NoBlending}})));
const maxMatches=12000,matchPositions=new Float32Array(maxMatches*3),matchGeometry=new THREE.BufferGeometry();matchGeometry.setAttribute('position',new THREE.BufferAttribute(matchPositions,3));matchGeometry.setDrawRange(0,0);scene.add(new THREE.Points(matchGeometry,new THREE.PointsMaterial({{color:0xdf1717,size:4.2,sizeAttenuation:false,depthTest:false,depthWrite:false}})));
const bounds=new THREE.Box3().setFromBufferAttribute(geometry.getAttribute('position')),center=bounds.getCenter(new THREE.Vector3()),extent=bounds.getSize(new THREE.Vector3()),radius=Math.max(extent.x,extent.y,extent.z,.001)*.5,initial={{azimuth:-.70,elevation:.27,distance:radius*2.25}};let azimuth=initial.azimuth,elevation=initial.elevation,distance=initial.distance,targetAzimuth=azimuth,targetElevation=elevation,targetDistance=distance,velocityAzimuth=0,velocityElevation=0,raf=0,pickTarget=null,dragging=false,moved=false,lastX=0,lastY=0,latestRequest=0,selectedIndex=-1;
function cameraUpdate(){{const ce=Math.cos(elevation);camera.position.set(center.x+distance*ce*Math.cos(azimuth),center.y+distance*ce*Math.sin(azimuth),center.z+distance*Math.sin(elevation));camera.lookAt(center);camera.updateMatrixWorld();}}function draw(){{cameraUpdate();renderer.setRenderTarget(null);renderer.render(scene,camera);}}function setMotionQuality(moving){{fullPoints.visible=!moving;previewPoints.visible=moving;}}function animate(){{raf=0;azimuth+=(targetAzimuth-azimuth)*.20+velocityAzimuth;elevation+=(targetElevation-elevation)*.20+velocityElevation;distance+=(targetDistance-distance)*.20;velocityAzimuth*=.80;velocityElevation*=.80;const active=Math.abs(targetAzimuth-azimuth)+Math.abs(targetElevation-elevation)+Math.abs(targetDistance-distance)/Math.max(radius,.001)+Math.abs(velocityAzimuth)+Math.abs(velocityElevation)>1e-4;setMotionQuality(active);draw();if(active)requestDraw();}}function requestDraw(){{if(!raf)raf=requestAnimationFrame(animate);}}function resize(){{const rect=viewer.getBoundingClientRect(),width=Math.max(1,Math.floor(rect.width)),height=Math.max(1,Math.floor(rect.height));renderer.setSize(width,height,false);camera.aspect=width/height;camera.updateProjectionMatrix();if(pickTarget)pickTarget.dispose();pickTarget=new THREE.WebGLRenderTarget(width,height,{{depthBuffer:true}});setMotionQuality(false);draw();}}
const workerSource=`let features,start,localCount,dim,scale;self.onmessage=event=>{{const m=event.data;if(m.type==='init'){{features=new Int8Array(m.buffer);start=m.start;localCount=m.count;dim=m.dim;scale=m.scale;self.postMessage({{type:'ready'}});return;}}if(m.type==='select'){{const vector=new Int8Array(m.vector),hits=new Uint32Array(localCount);let written=0;for(let i=0;i<localCount;++i){{let dot=0,offset=i*dim;for(let j=0;j<dim;++j)dot+=vector[j]*features[offset+j];if(dot/(scale*scale)>=m.threshold)hits[written++]=start+i;}}const output=hits.slice(0,written);self.postMessage({{type:'selection',request:m.request,indices:output.buffer}},[output.buffer]);}}}};`;
const workerCount=Math.max(1,Math.min(4,navigator.hardwareConcurrency||4)),workers=[],workerReady=[];let pending=0,allHits=[];for(let workerIndex=0;workerIndex<workerCount;++workerIndex){{const start=Math.floor(workerIndex*count/workerCount),end=Math.floor((workerIndex+1)*count/workerCount),piece=embedding.slice(start*dim,end*dim),worker=new Worker(URL.createObjectURL(new Blob([workerSource],{{type:'application/javascript'}})));worker.onmessage=event=>{{const m=event.data;if(m.type==='ready'){{workerReady[workerIndex]=true;return;}}if(m.type!=='selection'||m.request!==latestRequest)return;allHits.push(new Uint32Array(m.indices));if(--pending===0)applyMatches();}};worker.postMessage({{type:'init',buffer:piece.buffer,start,count:end-start,dim,scale:quantizationScale}},[piece.buffer]);workers.push(worker);}};
function applyMatches(){{let total=0;for(const hit of allHits)total+=hit.length;const stride=Math.max(1,Math.ceil(total/maxMatches));let written=0,seen=0;for(const hit of allHits)for(let j=0;j<hit.length;j++){{if((seen++%stride)!==0||written>=maxMatches)continue;const index=hit[j];matchPositions[3*written]=xyz[3*index];matchPositions[3*written+1]=xyz[3*index+1];matchPositions[3*written+2]=xyz[3*index+2];++written;}}matchGeometry.attributes.position.needsUpdate=true;matchGeometry.setDrawRange(0,written);draw();}}
function requestSelection(){{if(selectedIndex<0||workerReady.filter(Boolean).length!==workerCount)return;latestRequest++;pending=workerCount;allHits=[];const vector=new Int8Array(embedding.subarray(selectedIndex*dim,(selectedIndex+1)*dim));const threshold=Number(slider.value);for(const worker of workers)worker.postMessage({{type:'select',vector,threshold,request:latestRequest}});}}function pick(event){{const rect=viewer.getBoundingClientRect(),x=Math.max(0,Math.min(pickTarget.width-1,Math.floor(event.clientX-rect.left))),y=Math.max(0,Math.min(pickTarget.height-1,Math.floor(rect.bottom-event.clientY)));cameraUpdate();renderer.setRenderTarget(pickTarget);renderer.clear();renderer.render(pickScene,camera);const pixel=new Uint8Array(4);renderer.readRenderTargetPixels(pickTarget,x,y,1,1,pixel);renderer.setRenderTarget(null);const id=pixel[0]+(pixel[1]<<8)+(pixel[2]<<16);if(id>0&&id<=count){{selectedIndex=id-1;requestSelection();}}}}
viewer.addEventListener('pointerdown',event=>{{if(event.target===slider)return;dragging=true;moved=false;lastX=event.clientX;lastY=event.clientY;viewer.setPointerCapture(event.pointerId);}});viewer.addEventListener('pointermove',event=>{{if(!dragging)return;const dx=event.clientX-lastX,dy=event.clientY-lastY;if(Math.abs(dx)+Math.abs(dy)>2)moved=true;if(moved){{targetAzimuth-=dx*.006;targetElevation=Math.max(-1.45,Math.min(1.45,targetElevation+dy*.006));velocityAzimuth=-dx*.0007;velocityElevation=dy*.0007;requestDraw();}}lastX=event.clientX;lastY=event.clientY;}});viewer.addEventListener('pointerup',event=>{{if(!dragging)return;dragging=false;try{{viewer.releasePointerCapture(event.pointerId);}}catch(_){{}}if(!moved)pick(event);else requestDraw();}});viewer.addEventListener('wheel',event=>{{event.preventDefault();targetDistance=Math.max(radius*.25,Math.min(radius*8,targetDistance*Math.exp(event.deltaY*.001)));requestDraw();}},{{passive:false}});viewer.addEventListener('dblclick',()=>{{targetAzimuth=initial.azimuth;targetElevation=initial.elevation;targetDistance=initial.distance;velocityAzimuth=0;velocityElevation=0;requestDraw();}});slider.addEventListener('input',requestSelection);window.addEventListener('resize',resize);resize();
</script></body></html>"""
    template = template.replace("{{", "{").replace("}}", "}")
    return (
        template.replace("@@THREE_JS@@", three_js)
        .replace("@@TITLE@@", str(metadata["title"]))
        .replace("@@XYZ@@", str(payload["xyz"]))
        .replace("@@EMBEDDING@@", str(payload["embedding"]))
        .replace("@@COUNT@@", str(payload["count"]))
        .replace("@@DIM@@", str(payload["dim"]))
    )


def _vtk_presentation_html_document(payload: dict[str, str | int], metadata: dict[str, Any]) -> str:
    """VTK.js point-cloud viewer with ParaView-style trackball interaction."""
    vtk_path = SMART_ROOT / "vendor" / "vtk.js"
    if not vtk_path.is_file():
        raise FileNotFoundError(f"Missing vendored VTK.js runtime: {vtk_path}")
    vtk_js = vtk_path.read_text(encoding="utf-8")
    template = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>@@TITLE@@</title>
<style>html,body,#viewer{{width:100%;height:100%;margin:0;overflow:hidden;background:#fff}}#viewer{{position:relative;cursor:grab}}#viewer:active{{cursor:grabbing}}canvas{{display:block;width:100%;height:100%;touch-action:none}}#similarity{{position:absolute;z-index:10;left:50%;bottom:18px;width:min(340px,52vw);height:5px;transform:translateX(-50%);appearance:none;background:#bbb;border-radius:5px;opacity:.82;outline:0}}#similarity::-webkit-slider-thumb{{appearance:none;width:14px;height:14px;border-radius:50%;background:#d7191c;border:1px solid #fff;box-shadow:0 0 0 1px #555}}#similarity::-moz-range-thumb{{width:13px;height:13px;border-radius:50%;background:#d7191c;border:1px solid #fff;box-shadow:0 0 0 1px #555}}</style>
</head><body><div id="viewer"><input id="similarity" aria-label="Similarity threshold" type="range" min="0.60" max="0.99" value="0.88" step="0.01"></div><script>@@VTK_JS@@</script><script>
function bytesFromBase64(encoded){{const raw=atob(encoded),out=new Uint8Array(raw.length);for(let i=0;i<raw.length;++i)out[i]=raw.charCodeAt(i);return out;}}function f32(encoded){{const bytes=bytesFromBase64(encoded);return new Float32Array(bytes.buffer);}}function i8(encoded){{const bytes=bytesFromBase64(encoded);return new Int8Array(bytes.buffer);}}
const xyz=f32('@@XYZ@@');let embedding=i8('@@EMBEDDING@@');const count=@@COUNT@@,dim=@@DIM@@,quantizationScale=127,viewer=document.getElementById('viewer'),slider=document.getElementById('similarity');
function makePointData(positionData){{const points=vtk.Common.Core.vtkPoints.newInstance();points.setData(positionData,3);const cells=new Uint32Array((positionData.length/3)*2);for(let i=0;i<positionData.length/3;++i){{cells[2*i]=1;cells[2*i+1]=i;}}const verts=vtk.Common.Core.vtkCellArray.newInstance();verts.setData(cells);const polyData=vtk.Common.DataModel.vtkPolyData.newInstance();polyData.setPoints(points);polyData.setVerts(verts);return {{points,verts,polyData}};}}
const generic=vtk.Rendering.Misc.vtkGenericRenderWindow.newInstance({{background:[1,1,1]}});generic.setContainer(viewer);generic.resize();const renderer=generic.getRenderer(),renderWindow=generic.getRenderWindow(),interactor=generic.getInteractor();interactor.setInteractorStyle(vtk.Interaction.Style.vtkInteractorStyleTrackballCamera.newInstance());
const fullData=makePointData(xyz),fullMapper=vtk.Rendering.Core.vtkMapper.newInstance(),fullActor=vtk.Rendering.Core.vtkActor.newInstance();fullMapper.setInputData(fullData.polyData);fullActor.setMapper(fullMapper);fullActor.getProperty().setRepresentationToPoints();fullActor.getProperty().setPointSize(1.875);fullActor.getProperty().setColor(0.02,0.02,0.02);renderer.addActor(fullActor);
const previewStride=4,previewCount=Math.ceil(count/previewStride),previewXYZ=new Float32Array(previewCount*3);for(let i=0,j=0;i<count;i+=previewStride,++j){{previewXYZ[3*j]=xyz[3*i];previewXYZ[3*j+1]=xyz[3*i+1];previewXYZ[3*j+2]=xyz[3*i+2];}}const previewData=makePointData(previewXYZ),previewMapper=vtk.Rendering.Core.vtkMapper.newInstance(),previewActor=vtk.Rendering.Core.vtkActor.newInstance();previewMapper.setInputData(previewData.polyData);previewActor.setMapper(previewMapper);previewActor.getProperty().setRepresentationToPoints();previewActor.getProperty().setPointSize(1.875);previewActor.getProperty().setColor(0.02,0.02,0.02);previewActor.setVisibility(false);renderer.addActor(previewActor);
const redData=makePointData(new Float32Array(0)),redMapper=vtk.Rendering.Core.vtkMapper.newInstance(),redActor=vtk.Rendering.Core.vtkActor.newInstance();redMapper.setInputData(redData.polyData);redActor.setMapper(redMapper);redActor.getProperty().setRepresentationToPoints();redActor.getProperty().setPointSize(6.75);redActor.getProperty().setColor(0.87,0.05,0.05);renderer.addActor(redActor);
renderer.resetCamera();renderer.resetCameraClippingRange();renderWindow.render();
function setInteractiveLOD(active){{fullActor.setVisibility(!active);previewActor.setVisibility(active);renderWindow.render();}}interactor.onStartInteraction(()=>setInteractiveLOD(true));interactor.onEndInteraction(()=>setInteractiveLOD(false));
const workerSource=`let features,start,localCount,dim,scale;self.onmessage=event=>{{const m=event.data;if(m.type==='init'){{features=new Int8Array(m.buffer);start=m.start;localCount=m.count;dim=m.dim;scale=m.scale;self.postMessage({{type:'ready'}});return;}}if(m.type==='select'){{const vector=new Int8Array(m.vector),hits=new Uint32Array(localCount);let written=0;for(let i=0;i<localCount;++i){{let dot=0,offset=i*dim;for(let j=0;j<dim;++j)dot+=vector[j]*features[offset+j];if(dot/(scale*scale)>=m.threshold)hits[written++]=start+i;}}const output=hits.slice(0,written);self.postMessage({{type:'selection',request:m.request,indices:output.buffer}},[output.buffer]);}}}};`;
const workerCount=Math.max(1,Math.min(4,navigator.hardwareConcurrency||4)),workers=[],workerReady=[];let pending=0,allHits=[],latestRequest=0,selectedIndex=-1;for(let workerIndex=0;workerIndex<workerCount;++workerIndex){{const start=Math.floor(workerIndex*count/workerCount),end=Math.floor((workerIndex+1)*count/workerCount),piece=embedding.slice(start*dim,end*dim),worker=new Worker(URL.createObjectURL(new Blob([workerSource],{{type:'application/javascript'}})));worker.onmessage=event=>{{const m=event.data;if(m.type==='ready'){{workerReady[workerIndex]=true;if(selectedIndex>=0&&workerReady.filter(Boolean).length===workerCount)requestSelection();return;}}if(m.type!=='selection'||m.request!==latestRequest)return;allHits.push(new Uint32Array(m.indices));if(--pending===0)applyMatches();}};worker.postMessage({{type:'init',buffer:piece.buffer,start,count:end-start,dim,scale:quantizationScale}},[piece.buffer]);workers.push(worker);}};
function applyMatches(){{let total=0;for(const hit of allHits)total+=hit.length;const maxMatches=12000,stride=Math.max(1,Math.ceil(total/maxMatches)),positions=new Float32Array(Math.min(maxMatches,total)*3);let written=0,seen=0;for(const hit of allHits)for(let j=0;j<hit.length;j++){{if((seen++%stride)!==0||written>=maxMatches)continue;const index=hit[j];positions[3*written]=xyz[3*index];positions[3*written+1]=xyz[3*index+1];positions[3*written+2]=xyz[3*index+2];++written;}}redData.points.setData(positions.subarray(0,written*3),3);const cells=new Uint32Array(written*2);for(let i=0;i<written;++i){{cells[2*i]=1;cells[2*i+1]=i;}}redData.verts.setData(cells);redData.polyData.modified();redMapper.modified();viewer.dataset.dealMatchCount=String(written);renderWindow.render();}}
function requestSelection(){{if(selectedIndex<0||workerReady.filter(Boolean).length!==workerCount)return;latestRequest++;pending=workerCount;allHits=[];const vector=new Int8Array(embedding.subarray(selectedIndex*dim,(selectedIndex+1)*dim)),threshold=Number(slider.value);for(const worker of workers)worker.postMessage({{type:'select',vector,threshold,request:latestRequest}});}}
const canvas=viewer.querySelector('canvas');let pressX=0,pressY=0;function selectNearestVisiblePoint(clientX,clientY){{const rect=canvas.getBoundingClientRect(),targetX=2*(clientX-rect.left)/Math.max(rect.width,1)-1,targetY=1-2*(clientY-rect.top)/Math.max(rect.height,1),aspect=canvas.width/Math.max(canvas.height,1),matrix=renderer.getActiveCamera().getCompositeProjectionMatrix(aspect,-1,1),pickRadiusPx=16,pickRadiusSquared=pickRadiusPx*pickRadiusPx;let frontDepth=Infinity,hasCandidate=false;for(let index=0,offset=0;index<count;++index,offset+=3){{const x=xyz[offset],y=xyz[offset+1],z=xyz[offset+2],w=matrix[12]*x+matrix[13]*y+matrix[14]*z+matrix[15];if(w<=0)continue;const sx=(matrix[0]*x+matrix[1]*y+matrix[2]*z+matrix[3])/w,sy=(matrix[4]*x+matrix[5]*y+matrix[6]*z+matrix[7])/w,depth=(matrix[8]*x+matrix[9]*y+matrix[10]*z+matrix[11])/w,dx=(sx-targetX)*rect.width*.5,dy=(sy-targetY)*rect.height*.5;if(dx*dx+dy*dy>pickRadiusSquared)continue;hasCandidate=true;if(depth<frontDepth)frontDepth=depth;}}if(!hasCandidate)return -1;const frontDepthTolerance=.0025;let best=-1,bestDistance=Infinity;for(let index=0,offset=0;index<count;++index,offset+=3){{const x=xyz[offset],y=xyz[offset+1],z=xyz[offset+2],w=matrix[12]*x+matrix[13]*y+matrix[14]*z+matrix[15];if(w<=0)continue;const sx=(matrix[0]*x+matrix[1]*y+matrix[2]*z+matrix[3])/w,sy=(matrix[4]*x+matrix[5]*y+matrix[6]*z+matrix[7])/w,depth=(matrix[8]*x+matrix[9]*y+matrix[10]*z+matrix[11])/w,dx=(sx-targetX)*rect.width*.5,dy=(sy-targetY)*rect.height*.5,distance=dx*dx+dy*dy;if(distance>pickRadiusSquared||depth>frontDepth+frontDepthTolerance)continue;if(distance<bestDistance){{bestDistance=distance;best=index;}}}}viewer.dataset.dealPickDepth=String(frontDepth);viewer.dataset.dealPickDistancePx=String(Math.sqrt(bestDistance));return best;}}function isCanvasInteraction(event){{return event.target!==slider&&!slider.contains(event.target);}}viewer.addEventListener('pointerdown',event=>{{if(!isCanvasInteraction(event))return;pressX=event.clientX;pressY=event.clientY;}},true);viewer.addEventListener('pointerup',event=>{{if(!isCanvasInteraction(event)||Math.abs(event.clientX-pressX)+Math.abs(event.clientY-pressY)>3)return;viewer.dataset.dealPickStatus='received';try{{const id=selectNearestVisiblePoint(event.clientX,event.clientY);viewer.dataset.dealPickId=String(id);if(id>=0){{selectedIndex=id;requestSelection();}}}}catch(error){{viewer.dataset.dealPickStatus=`error:${{error.message}}`;console.error(error);}}}},true);canvas.addEventListener('dblclick',()=>{{renderer.resetCamera();renderer.resetCameraClippingRange();renderWindow.render();}});for(const eventName of ['pointerdown','pointermove','pointerup','wheel'])slider.addEventListener(eventName,event=>event.stopPropagation());slider.addEventListener('input',requestSelection);window.addEventListener('resize',()=>{{generic.resize();renderWindow.render();}});
</script></body></html>"""
    template = template.replace("{{", "{").replace("}}", "}")
    return (
        template.replace("@@VTK_JS@@", vtk_js)
        .replace("@@TITLE@@", str(metadata["title"]))
        .replace("@@XYZ@@", str(payload["xyz"]))
        .replace("@@EMBEDDING@@", str(payload["embedding"]))
        .replace("@@COUNT@@", str(payload["count"]))
        .replace("@@DIM@@", str(payload["dim"]))
    )


def _write_embedding_view(
    output_dir: Path,
    stem: str,
    title: str,
    description: str,
    raw_points: np.ndarray,
    features: torch.Tensor,
    method: str,
    components: int,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    reduced, basis, diagnostics = _compress_cosine_features(features, components, seed, device)
    color_rank = _score_percentile(reduced[:, 0])
    rank_u16 = np.rint(color_rank * np.iinfo(np.uint16).max).astype(np.uint16)
    quantization_scale = 127
    reduced_int8 = np.rint(np.clip(reduced, -1.0, 1.0) * quantization_scale).astype(np.int8)
    rng = np.random.default_rng(int(seed) + 919)
    sample_count = min(100_000, reduced.shape[0])
    left = rng.integers(0, reduced.shape[0], size=sample_count)
    right = rng.integers(0, reduced.shape[0], size=sample_count)
    exact_cosine = np.einsum("ij,ij->i", reduced[left], reduced[right])
    int8_cosine = np.einsum(
        "ij,ij->i",
        reduced_int8[left].astype(np.int32),
        reduced_int8[right].astype(np.int32),
    ) / float(quantization_scale * quantization_scale)
    diagnostics["int8_cosine_mae"] = float(np.abs(exact_cosine - int8_cosine).mean())
    diagnostics["int8_cosine_correlation"] = float(np.corrcoef(exact_cosine, int8_cosine)[0, 1])
    metadata: dict[str, Any] = {
        "title": title,
        "description": description,
        "method": method,
        "input_points": int(raw_points.shape[0]),
        "original_dimension": int(features.shape[1]),
        "compression": diagnostics,
        "similarity": "cosine on L2-normalized PCA-compressed feature vectors",
        "base_color": "rank of PCA component 1; shown only as context, not as similarity",
        "viewer_quantization": {
            "dtype": "int8",
            "scale": quantization_scale,
            "sampled_cosine_mae": diagnostics["int8_cosine_mae"],
            "sampled_cosine_correlation": diagnostics["int8_cosine_correlation"],
        },
    }
    payload: dict[str, str | int] = {
        "xyz": _binary_payload(raw_points.reshape(-1), np.float32),
        "rank": _binary_payload(rank_u16, np.uint16),
        "embedding": _binary_payload(reduced_int8.reshape(-1), np.int8),
        "count": int(raw_points.shape[0]),
        "dim": int(reduced.shape[1]),
    }
    html_path = output_dir / f"{stem}.html"
    npz_path = output_dir / f"{stem}.npz"
    html_path.write_text(_vtk_presentation_html_document(payload, metadata), encoding="utf-8")
    np.savez_compressed(
        npz_path,
        input_points=raw_points.astype(np.float32),
        compressed_l2_normalized_embedding_int8=reduced_int8,
        embedding_quantization_scale=np.int16(quantization_scale),
        pca_basis=basis,
        pca1_color_rank=color_rank.astype(np.float32),
    )
    metadata["html"] = html_path.name
    metadata["npz"] = npz_path.name
    return metadata


def main() -> None:
    args = parse_args()
    if args.input_points != 131072:
        raise ValueError("This diagnostic is fixed to the 131072-point DrivAerML encoder budget.")
    if args.query_chunk_size <= 0 or args.embedding_components < 2:
        raise ValueError("Query chunks must be positive and embedding components must be at least two.")
    checkpoint = args.deal_checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"DeAL checkpoint not found: {checkpoint}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model, config = build_model(args.deal_config, checkpoint, device)
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
    raw_surface = np.array(np.load(run_dir / "surface_coords.npy", mmap_mode="r"), dtype=np.float32, copy=True)
    if raw_surface.ndim != 2 or raw_surface.shape[1] != 3 or not np.isfinite(raw_surface).all():
        raise RuntimeError(f"Invalid surface points in {run_dir}")
    span = torch.clamp(dataset.max_pos - dataset.min_pos, min=1.0e-12)
    full_geometry = (torch.from_numpy(raw_surface) - dataset.min_pos) / span
    density = dataset._load_or_compute_full_geometry_density(int(args.run_id), expected_n=int(full_geometry.shape[0])).float()
    sampling_seed = int(args.seed + 100003 * int(args.run_id))
    beta_view = sample_condition("beta1", full_geometry, density, int(args.input_points), sampling_seed, None)
    if beta_view.shape != (int(args.input_points), 3) or not torch.isfinite(beta_view).all():
        raise RuntimeError("Beta=1 sampling did not produce a finite 131072-point view.")
    raw_view = (beta_view.numpy() * span.numpy() + dataset.min_pos.numpy()).astype(np.float32)

    print("Extracting final pre-attention encoder key embeddings...", flush=True)
    encoder_keys = _last_encoder_key_embedding(model, beta_view).cpu()
    print("Extracting final geometry-conditioned decoder query embeddings...", flush=True)
    decoder_queries = _final_decoder_query_embedding(model, beta_view, sampling_seed, int(args.query_chunk_size))
    if encoder_keys.shape != decoder_queries.shape or encoder_keys.shape[0] != raw_view.shape[0]:
        raise RuntimeError("Embedding point counts or feature dimensions are inconsistent.")
    if not torch.isfinite(encoder_keys).all() or not torch.isfinite(decoder_queries).all():
        raise FloatingPointError("Refusing to export non-finite embeddings.")

    encoder_record = _write_embedding_view(
        output_dir,
        f"drivaerml_run_{args.run_id}_deal_beta1_encoder_key_embedding_131k",
        "DeAL beta=1: pre-attention encoder key similarity",
        "This is the final encoder block's learned key vector immediately before anchor-to-geometry cross-attention. It is a valid learned representation, but it is pointwise and coordinate-driven; use this viewer as the requested pre-attention baseline rather than as a semantic-part claim.",
        raw_view,
        encoder_keys,
        "last_encoder_block.geo_attn RoPE-transformed K vector before softmax attention",
        int(args.embedding_components),
        int(args.seed) + 17,
        device,
    )
    encoder_keys = torch.empty(0)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    decoder_record = _write_embedding_view(
        output_dir,
        f"drivaerml_run_{args.run_id}_deal_beta1_decoder_query_embedding_131k",
        "DeAL beta=1: geometry-conditioned decoder-query similarity",
        "This is the final surface-query token immediately before DeAL's prediction head. It has received information from the encoded anchor latents, so it is the appropriate representation for testing whether repeated vehicle parts, such as wheels, are grouped by the trained field-prediction model.",
        raw_view,
        decoder_queries,
        "final decoder query token immediately before SMART.mlp",
        int(args.embedding_components),
        int(args.seed) + 37,
        device,
    )
    summary = {
        "run_id": int(args.run_id),
        "checkpoint": str(checkpoint),
        "config": str(args.deal_config),
        "condition": "beta=1 inverse-density sampling without replacement",
        "input_points": int(args.input_points),
        "encoder_key_view": encoder_record,
        "decoder_query_view": decoder_record,
    }
    (output_dir / "embedding_similarity_export_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote two interactive embedding viewers to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
