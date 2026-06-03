#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, random
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from data.datasets import get_dataset
from models.smart.cat import CAT

SURFACE_PREFERRED=["pressure","normal_x","normal_y","normal_z","wall_shear_x","wall_shear_y","wall_shear_z"]

def parse_args():
    p=argparse.ArgumentParser("Calibrate CAT UQ params for surface and volume")
    p.add_argument("--config-name",default="drivaerml_cat")
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--output",default=None)
    p.add_argument("--max-batches",type=int,default=80)
    p.add_argument("--probe-batches",type=int,default=1)
    p.add_argument("--batch-size",type=int,default=None)
    p.add_argument("--num-workers",type=int,default=0)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--device",default=None)
    p.add_argument("--pinv-rcond",type=float,default=1e-6)
    p.add_argument("--cov-shrink",type=float,default=0.05)
    p.add_argument("--hessian-damping",type=float,default=1e-4)
    p.add_argument("--eps",type=float,default=1e-3)
    return p.parse_args()

def set_seed(seed:int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def load_config(name:str):
    cfg_path=Path(__file__).resolve().parent/"config"/f"{name}.yaml"
    return OmegaConf.load(cfg_path).experiment

def resolve_targets(fields:Dict[str,List[str]]):
    sf=list(fields.get("surface",[])); vf=list(fields.get("volume",[]))
    s_idx=[sf.index(x) for x in SURFACE_PREFERRED if x in sf] or list(range(len(sf)))
    vel=[i for i,n in enumerate(vf) if str(n).startswith("velocity_")]
    p=[vf.index("pressure")] if "pressure" in vf else []
    v_idx=(p+vel) or list(range(len(vf)))
    return s_idx,[sf[i] for i in s_idx],v_idx,[vf[i] for i in v_idx]

def sample_indices(n,k,device,disjoint=None):
    if k<=0: return torch.empty((0,),dtype=torch.long,device=device)
    if disjoint is not None:
        mask=torch.ones((n,),dtype=torch.bool,device=device); mask[disjoint]=False
        cand=torch.where(mask)[0]
        if cand.numel()==0: return torch.randint(0,n,(k,),device=device)
        if k<=cand.numel(): return cand[torch.randperm(cand.numel(),device=device)[:k]]
        extra=cand[torch.randint(0,cand.numel(),(k-cand.numel(),),device=device)]
        return torch.cat([cand,extra],0)
    if k<=n: return torch.randperm(n,device=device)[:k]
    return torch.cat([torch.arange(n,device=device),torch.randint(0,n,(k-n,),device=device)],0)

def gather_b(x,idxs): return torch.stack([x[b,idxs[b],:] for b in range(x.shape[0])],0)

def prep_batch(batch,config,device,s_idx,v_idx):
    geo,surf,sd,vol,vd=batch; del geo
    surf=surf.to(device); sd=sd.to(device); vol=vol.to(device); vd=vd.to(device)
    b,ns,_=surf.shape; nv=vol.shape[1]
    s_in=int(getattr(config,"single_surface_input_points",getattr(config,"num_body_points",ns)))
    s_q=int(getattr(config,"single_surface_query_points",getattr(config,"num_surface_points",ns)))
    v_q=int(getattr(config,"single_volume_query_points",getattr(config,"num_volume_points",nv)))
    s_in=ns if s_in<=0 else s_in; s_q=ns if s_q<=0 else s_q; v_q=nv if v_q<=0 else v_q
    e_idx=[]; sq_idx=[]; vq_idx=[]
    for _ in range(b):
        e=sample_indices(ns,s_in,device); sq=sample_indices(ns,s_q,device,disjoint=e)
        e_idx.append(e); sq_idx.append(sq); vq_idx.append(sample_indices(nv,v_q,device))
    s_t=gather_b(sd.index_select(2,torch.tensor(s_idx,dtype=torch.long,device=device)),sq_idx)
    v_t=gather_b(vd.index_select(2,torch.tensor(v_idx,dtype=torch.long,device=device)),vq_idx)
    return gather_b(surf,e_idx),gather_b(surf,sq_idx),s_t,gather_b(vol,vq_idx),v_t

def est_inv_cov(x,pinv,csh):
    x=x.double().cpu(); mu=x.mean(0); xc=x-mu; n,d=xc.shape
    try:
        from sklearn.covariance import LedoitWolf
        cov=torch.from_numpy(LedoitWolf().fit(x.numpy()).covariance_).double()
    except Exception:
        cov=(xc.T@xc)/float(max(n-1,1)); tr=torch.trace(cov)/float(d)
        cov=(1-csh)*cov + csh*tr*torch.eye(d,dtype=torch.double)
    return mu,torch.linalg.pinv(cov,rcond=pinv)

def calibrate_block(model,mode,loader,config,device,s_idx,v_idx,args):
    cache={}
    if mode=="surface":
        h_q=model.stage2_head[0].register_forward_hook(lambda m,i,o: cache.__setitem__("q",i[0].detach()))
        h_z=model.stage2_head[3].register_forward_hook(lambda m,i,o: cache.__setitem__("z",o.detach()))
    else:
        h_q=model.volume_head[0].register_forward_hook(lambda m,i,o: cache.__setitem__("q",i[0].detach()))
        h_z=model.volume_head[3].register_forward_hook(lambda m,i,o: cache.__setitem__("z",o.detach()))
    qg=[]; H=None; probe=[]; processed=0
    try:
        for batch in loader:
            si,sq,st,vq,vt=prep_batch(batch,config,device,s_idx,v_idx)
            with torch.no_grad():
                if mode=="surface": _=model.forward_stage1_only(si,sq,return_aux=False)
                else: _=model.forward_stage2_only(si,sq,vq,return_aux=False)
            q=cache["q"].double().cpu(); z=cache["z"].double().cpu()
            qg.append(q.mean(1)); z2=z.reshape(-1,z.shape[-1]); cur=z2.T@z2; H=cur if H is None else H+cur
            if len(probe)<int(args.probe_batches): probe.append((si.detach().clone(),sq.detach().clone(),st.detach().clone(),vq.detach().clone(),vt.detach().clone()))
            processed+=1
            if processed>=int(args.max_batches): break
    finally:
        h_q.remove(); h_z.remove()
    mu,inv=est_inv_cov(torch.cat(qg,0),args.pinv_rcond,args.cov_shrink)
    Hreg=H.double()+float(args.hessian_damping)*torch.eye(H.shape[0],dtype=torch.double)
    Sigma=torch.linalg.pinv(Hreg,rcond=float(args.pinv_rcond))
    vals,vecs=torch.linalg.eigh(Sigma); principal=vecs[:,-1]; principal=principal/torch.clamp(torch.linalg.norm(principal),min=1e-12)
    head=model.stage2_head[-1] if mode=="surface" else model.volume_head[-1]
    W0=head.weight.detach().clone(); dW=principal.to(W0.device,W0.dtype).unsqueeze(0).repeat(W0.shape[0],1)
    dW=dW/torch.clamp(torch.linalg.norm(dW),min=1e-12); eps=float(args.eps)
    si,sq,st,vq,vt=probe[0]
    def f(scale):
        with torch.no_grad():
            head.weight.copy_(W0+(scale*eps)*dW)
            pred=model.forward_stage1_only(si,sq,return_aux=False) if mode=="surface" else model.forward_stage2_only(si,sq,vq,return_aux=False)
            loss=F.mse_loss(pred,st if mode=="surface" else vt)
        return float(loss.item())
    with torch.no_grad():
        fm2,fm1,fp1,fp2=f(-2),f(-1),f(1),f(2); head.weight.copy_(W0)
    d3=(fm2-2*fm1+2*fp1-fp2)/(2*(eps**3)); alpha=(d3*principal).double().cpu(); V=(alpha@Sigma).double().cpu(); K=float(1.0+(alpha@Sigma@alpha).item())
    return {"mu_train":mu.float(),"inv_sigma_train":inv.float(),"Sigma_LLL":Sigma.float(),"alpha_w":alpha.float(),"V_skew":V.float(),"K":torch.tensor(K,dtype=torch.float32),"processed_batches":processed,"skew_directional_d3":float(d3)}

def out_path(ckpt,explicit):
    if explicit: return Path(explicit)
    p=Path(ckpt); stem=p.stem
    for s in ("_best","_last"):
        if stem.endswith(s): stem=stem[:-len(s)]
    return p.parent/f"{stem}_uq_params_both.pt"

def main():
    a=parse_args(); set_seed(a.seed)
    device=torch.device(a.device) if a.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg=load_config(a.config_name)
    if a.batch_size is not None: cfg.batch_size=int(a.batch_size)
    cfg.num_workers=int(a.num_workers)
    train_data,_,_,sd,sc,vc,pd,fields=get_dataset(cfg)
    s_idx,s_fields,v_idx,v_fields=resolve_targets(fields)
    arch=OmegaConf.to_container(cfg.architecture,resolve=True); arch["stage2_surface_channels"]=len(s_idx)
    model=CAT(spatial_dim=sd,surface_channels=sc,volume_channels=vc,parameter_channels=pd,**arch).to(device)
    ckpt=torch.load(a.checkpoint,map_location=device); model.load_state_dict(ckpt.get("model_state_dict",ckpt),strict=False); model.eval()
    dl=torch.utils.data.DataLoader(train_data,batch_size=int(cfg.batch_size),shuffle=True,num_workers=int(cfg.num_workers),pin_memory=bool(getattr(cfg,"pin_memory",True)))
    surf=calibrate_block(model,"surface",dl,cfg,device,s_idx,v_idx,a)
    vol=calibrate_block(model,"volume",dl,cfg,device,s_idx,v_idx,a)
    payload={"surface":surf,"volume":vol,"surface_target_fields":s_fields,"volume_target_fields":v_fields,"calibration_meta":{"checkpoint":str(a.checkpoint),"config_name":a.config_name}}
    op=out_path(a.checkpoint,a.output); op.parent.mkdir(parents=True,exist_ok=True); torch.save(payload,op)
    print(f"Saved: {op}")
    print(json.dumps({"surface_batches":surf["processed_batches"],"volume_batches":vol["processed_batches"]},indent=2))

if __name__=="__main__":
    main()
