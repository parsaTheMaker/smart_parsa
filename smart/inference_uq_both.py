#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, random
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm.auto import tqdm
from data.datasets import get_dataset
from models.smart.cat import CAT

SURFACE_PREFERRED=["pressure","normal_x","normal_y","normal_z","wall_shear_x","wall_shear_y","wall_shear_z"]

def parse_args():
    p=argparse.ArgumentParser("CAT inference with separate surface/volume UQ")
    p.add_argument("--config-name",default="drivaerml_cat")
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--uq-params",required=True)
    p.add_argument("--split",default="test",choices=["train","test"])
    p.add_argument("--output-dir",default=None)
    p.add_argument("--gamma-surface",type=float,default=0.1)
    p.add_argument("--gamma-volume",type=float,default=0.1)
    p.add_argument("--max-runs",type=int,default=-1)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--device",default=None)
    p.add_argument("--cat-vol-chunk",type=int,default=131072)
    return p.parse_args()

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def load_config(name):
    return OmegaConf.load(Path(__file__).resolve().parent/"config"/f"{name}.yaml").experiment

def resolve_targets(fields):
    sf=list(fields.get("surface",[])); vf=list(fields.get("volume",[]))
    s_idx=[sf.index(x) for x in SURFACE_PREFERRED if x in sf] or list(range(len(sf)))
    vel=[i for i,n in enumerate(vf) if str(n).startswith("velocity_")]
    p=[vf.index("pressure")] if "pressure" in vf else []
    v_idx=(p+vel) or list(range(len(vf)))
    return s_idx,[sf[i] for i in s_idx],v_idx,[vf[i] for i in v_idx]

def norm_pos(x,mn,mx): return (x-mn)/torch.clamp(mx-mn,min=1e-12)
def denorm(x,m,s): return x*s+m

def rel_l2(gt,p,eps=1e-12): return float(np.linalg.norm(p-gt)/max(np.linalg.norm(gt),eps))

def load_case(run_dir:Path):
    sc=np.load(run_dir/"surface_coords.npy").astype(np.float32,copy=False)
    sp=np.load(run_dir/"surface_pMeanTrim.npy").astype(np.float32,copy=False).reshape(-1,1)
    sn=np.load(run_dir/"surface_normals.npy").astype(np.float32,copy=False)
    swx=np.load(run_dir/"surface_wallShearStressMeanTrim_x.npy").astype(np.float32,copy=False).reshape(-1,1)
    swy=np.load(run_dir/"surface_wallShearStressMeanTrim_y.npy").astype(np.float32,copy=False).reshape(-1,1)
    swz=np.load(run_dir/"surface_wallShearStressMeanTrim_z.npy").astype(np.float32,copy=False).reshape(-1,1)
    surf=np.concatenate([sp,sn,swx,swy,swz],1)
    vc=np.load(run_dir/"volume_coords.npy").astype(np.float32,copy=False)
    vp=np.load(run_dir/"volume_pMeanTrim.npy").astype(np.float32,copy=False).reshape(-1,1)
    vu=np.load(run_dir/"volume_UMeanTrim.npy").astype(np.float32,copy=False)
    vol=np.concatenate([vp,vu],1)
    return sc,surf,vc,vol

def sample_input_idx(n,k,rng):
    if k<=0 or k>=n: return np.arange(n,dtype=np.int64)
    return rng.integers(0,n,size=k,dtype=np.int64)

def stage2_with_features(model,si,sq,vq,chunk):
    s_pred,aux=model.forward_stage1_only(si,sq,return_aux=True)
    geom=aux["geom_latents"]; anchor=aux["anchor_pos"]; gf=aux["geom_final"]
    prev,_=model._encode_stage2(sq[..., : model.spatial_dim],s_pred,anchor,initial_latent=gf)
    new,_=model._encode_stage2(sq[..., : model.spatial_dim],s_pred,anchor,initial_latent=None)
    wc,wf=model._compute_dynamic_skip_weights(geom,prev,new,s_pred)
    fused=[]
    for m in range(model.loops):
        a=model.surface_to_volume_latent_norm(geom[m]); b=model.surface_to_volume_latent_norm(prev[m]); c=model.surface_to_volume_latent_norm(new[m])
        x=b+wc[:,m,:].unsqueeze(-1)*(a-b); y=c+wf[:,m,:].unsqueeze(-1)*(x-c); fused.append(y)
    preds=[]; qv_all=[]; zv_all=[]
    for st in range(0,vq.shape[1],chunk):
        q=vq[:,st:st+chunk,:]
        qv=model._decode(q[..., : model.spatial_dim],fused,anchor,model.volume_decoder_blocks)
        z1=model.volume_head[0](qv); z2=model.volume_head[1](z1); z3=model.volume_head[2](z2); z4=model.volume_head[3](z3)
        pv=model.volume_head[4](z4)
        preds.append(pv); qv_all.append(qv); zv_all.append(z4)
    return s_pred,torch.cat(preds,1),torch.cat(qv_all,1),torch.cat(zv_all,1)

def uq_from(q,z,params,gamma):
    mu=params["mu_train"]; inv=params["inv_sigma_train"]; S=params["Sigma_LLL"]; V=params["V_skew"]; K=float(params["K"].item() if isinstance(params["K"],torch.Tensor) else params["K"])
    qg=q.mean(1); dq=qg-mu.unsqueeze(0); md=torch.sqrt(torch.clamp(torch.einsum("bi,ij,bj->b",dq,inv,dq),min=1e-12))
    z2=z[0]; var=torch.sum((z2@S)*z2,dim=-1); var=torch.clamp(var,min=1e-12); var=var*(1.0+gamma*md[0])
    cross=z2@V; den=torch.sqrt(torch.clamp(var*K-cross*cross,min=1e-6)); alpha=cross/den
    return var,alpha,float(md[0].item())

def main():
    a=parse_args(); set_seed(a.seed)
    dev=torch.device(a.device) if a.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg=load_config(a.config_name)
    tr,te,_,sd,sc,vc,pd,fields=get_dataset(cfg)
    ds=te if a.split=="test" else tr
    ids=list(ds.data); ids=ids if a.max_runs<=0 else ids[:int(a.max_runs)]
    s_idx,s_fields,v_idx,v_fields=resolve_targets(fields)
    arch=OmegaConf.to_container(cfg.architecture,resolve=True); arch["stage2_surface_channels"]=len(s_idx)
    model=CAT(spatial_dim=sd,surface_channels=sc,volume_channels=vc,parameter_channels=pd,**arch).to(dev)
    ck=torch.load(a.checkpoint,map_location=dev); model.load_state_dict(ck.get("model_state_dict",ck),strict=False); model.eval()
    up=torch.load(a.uq_params,map_location=dev)
    surf_params=up["surface"]; vol_params=up["volume"]
    for d in (surf_params,vol_params):
        for k in ("mu_train","inv_sigma_train","Sigma_LLL","V_skew","K"):
            if isinstance(d[k],torch.Tensor): d[k]=d[k].to(dev)
    out=Path(a.output_dir) if a.output_dir else Path("results")/"uq_inference_both"/cfg.dataset/Path(a.checkpoint).stem
    runs=out/"runs"; runs.mkdir(parents=True,exist_ok=True)
    ms=ds.mean_surf_data[s_idx].float(); ss=torch.clamp(ds.std_surf_data[s_idx].float(),min=1e-12)
    mv=ds.mean_vol_data[v_idx].float(); sv=torch.clamp(ds.std_vol_data[v_idx].float(),min=1e-12)
    mn=ds.min_pos.float(); mx=ds.max_pos.float()
    met={"surface_rel_l2":[],"volume_rel_l2":[],"md_surface":[],"md_volume":[]}
    with torch.inference_mode():
        for rid in tqdm(ids,desc=f"UQ both ({a.split})",dynamic_ncols=True):
            rdir=Path(cfg.data_path)/f"run_{rid}"; sc,sg,vc_,vg=load_case(rdir); sg=sg[:,s_idx]; vg=vg[:,v_idx]
            sq=norm_pos(torch.from_numpy(sc),mn,mx); vq=norm_pos(torch.from_numpy(vc_),mn,mx)
            idx=sample_input_idx(sc.shape[0],int(getattr(cfg,"single_surface_input_points",getattr(cfg,"num_body_points",sc.shape[0]))),np.random.default_rng(a.seed+int(rid)))
            si=sq[torch.from_numpy(idx)].unsqueeze(0).to(dev); sqb=sq.unsqueeze(0).to(dev); vqb=vq.unsqueeze(0).to(dev)
            sp,vp,qv,zv=stage2_with_features(model,si,sqb,vqb,int(a.cat_vol_chunk))
            # surface q/z via direct recompute hooks-free
            s_aux=model.forward_stage1_only(si,sqb,return_aux=True)[1]
            q_s=model._decode(sqb[..., : model.spatial_dim],s_aux["geom_latents"],s_aux["anchor_pos"],model.surface_decoder_blocks)
            z_s=model.stage2_head[3](model.stage2_head[2](model.stage2_head[1](model.stage2_head[0](q_s))))
            var_s,alpha_s,md_s=uq_from(q_s,z_s,surf_params,float(a.gamma_surface))
            var_v,alpha_v,md_v=uq_from(qv,zv,vol_params,float(a.gamma_volume))
            sp_d=denorm(sp[0].cpu(),ms,ss).numpy(); vp_d=denorm(vp[0].cpu(),mv,sv).numpy()
            met["surface_rel_l2"].append(rel_l2(sg.reshape(-1),sp_d.reshape(-1))); met["volume_rel_l2"].append(rel_l2(vg.reshape(-1),vp_d.reshape(-1)))
            met["md_surface"].append(md_s); met["md_volume"].append(md_v)
            np.savez_compressed(runs/f"run_{rid}_uq_both.npz",run_id=np.array([rid]),surface_coords=sc,surface_gt=sg,surface_mu_pred=sp_d,surface_variance_final=var_s.cpu().numpy(),surface_alpha_exact=alpha_s.cpu().numpy(),volume_coords=vc_,volume_gt=vg,volume_mu_pred=vp_d,volume_variance_final=var_v.cpu().numpy(),volume_alpha_exact=alpha_v.cpu().numpy(),md_surface=np.array([md_s],dtype=np.float32),md_volume=np.array([md_v],dtype=np.float32))
    summ={"runs_processed":len(ids),"surface_rel_l2_mean":float(np.mean(met["surface_rel_l2"])) if met["surface_rel_l2"] else float("nan"),"volume_rel_l2_mean":float(np.mean(met["volume_rel_l2"])) if met["volume_rel_l2"] else float("nan"),"md_surface_mean":float(np.mean(met["md_surface"])) if met["md_surface"] else float("nan"),"md_volume_mean":float(np.mean(met["md_volume"])) if met["md_volume"] else float("nan")}
    out.mkdir(parents=True,exist_ok=True); (out/"uq_metrics.json").write_text(json.dumps(summ,indent=2))
    print(f"Saved run arrays to: {runs}")
    print(f"Saved summary metrics to: {out/'uq_metrics.json'}")
    print(json.dumps(summ,indent=2))

if __name__=="__main__":
    main()
