"""Actual GIN training, separated from exact state-screen results.

BREC: one model per pair/order; official relabel and reliability blocks.
SR: one shared encoder per order, trained with relabel contrastive loss;
    all pair distances are counted in blocks (no M-by-M storage).
"""
from dataclasses import asdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import csv, json, multiprocessing as mp
import numpy as np
import torch
import torch.nn.functional as F
from .states import cached_order, feature_matrix
from .data import BREC, CATEGORIES, category, file_sha256, load_graph6, srg_parameters
from .model import KSGL, pack, permute_graph, expand_adam_state
from .runtime import (RunLock, StopFlag, freeze_config, atomic_json, atomic_csv,
                      atomic_torch_save, seed_all, rng_state, restore_rng, torch_load_compat)

def _extract(args):
    A,m,cfg,cache=args
    return {s:cached_order(A,s,cfg,cache) for s in range(1,m+1)}

def extract_many(graphs,m,cfg,cache,workers):
    jobs=[(A,m,cfg,str(cache)) for A in graphs]
    if workers<=1: return list(map(_extract,jobs))
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as ex:
        out=[]
        for j,b in enumerate(ex.map(_extract,jobs),1):
            out.append(b)
            if j%16==0 or j==len(graphs): print(f'features {j}/{len(graphs)}',flush=True)
        return out

def get_model(i,a): return KSGL(i,a.hidden,a.layers,a.out_dim).to(a.device)
def optimizer(model,a): return torch.optim.Adam(model.parameters(),lr=a.lr,weight_decay=a.weight_decay)

def save_training(path,model,opt,epoch):
    atomic_torch_save(path,dict(model=model.state_dict(),optimizer=opt.state_dict(),epoch=epoch,
                               order=model.order,rng=rng_state()))

def initialize(i,a,last,previous=None):
    seed_all(a.seed)
    ckpt=last if last.exists() else previous if previous and previous.exists() else None
    if ckpt:
        # Load ONLY trusted local checkpoints created by this package.
        d=torch_load_compat(ckpt,map_location=a.device)
        model=get_model(d['order'],a); model.load_state_dict(d['model'])
        model.expand_order(i)
        opt=optimizer(model,a); opt.load_state_dict(d['optimizer']); expand_adam_state(opt)
        restore_rng(d['rng'])
        return model,opt,(int(d['epoch']) if ckpt==last else 0)
    model=get_model(i,a)
    return model,optimizer(model,a),0

def t2_statistics(z,ridge=1e-10):
    z=z.detach().double().cpu(); D=z[0::2]-z[1::2]; dm=D.mean(0)
    centered=D-dm; S=centered.T@centered/max(1,len(D)-1)
    rank=int(torch.linalg.matrix_rank(S).item())
    raw=float(dm@torch.linalg.pinv(S)@dm)
    scale=max(float(torch.trace(S))/len(dm),1.0)
    stabilized=float(dm@torch.linalg.solve(S+ridge*scale*torch.eye(len(dm),dtype=torch.float64),dm))
    gap=float(torch.linalg.vector_norm(dm)); norm=float(torch.linalg.vector_norm(z.mean(0)))
    return dict(t2_pinv=raw,t2_ridge=stabilized,gap=gap,mean_norm=norm,cov_rank=rank)

@torch.no_grad()
def embeddings(model,graphs,features,device,batch_size=64):
    model.eval(); result=[]
    for j in range(0,len(graphs),batch_size):
        result.append(model(*pack(graphs[j:j+batch_size],features[j:j+batch_size],device)).cpu())
    return torch.cat(result)

def _configuration(a,cfg,task):
    keys=['orders','hidden','layers','out_dim','epochs','lr','weight_decay','seed','branch','transform',
          'continuation','protocol','threshold','ridge','batch_size','relabels','steps','eval_block']
    return dict(task=task,data_sha256=file_sha256(a.data),features=asdict(cfg),
                **{k:getattr(a,k) for k in keys if hasattr(a,k)})

def _read_results(path):
    if not path.exists(): return []
    return json.loads(path.read_text())

def _save_brec(out,rows,orders):
    atomic_json(out/'results.json',rows)
    if rows: atomic_csv(out/'neural_pair_results.csv',rows,list(rows[0]))
    summary=[]
    for i in orders:
        for cat,lo,hi in CATEGORIES+[('TOTAL',0,400)]:
            r=[x for x in rows if x['order']==i and lo<=x['pair_id']<hi]
            summary.append(dict(order=i,category=cat,evaluated=len(r),separated=sum(x['separated'] for x in r),
                                reliability_failures=sum(not x['reliable'] for x in r),
                                exact_feature_pairs=sum(x['exact_features'] for x in r),official_total=hi-lo))
    atomic_csv(out/'neural_category_summary.csv',summary,list(summary[0]))

def run_brec(a,cfg):
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True); cache=Path(a.cache or out/'cache')
    dataset=BREC(a.data)
    if not dataset.official: raise ValueError('Neural BREC requires the official 51200-entry file')
    with RunLock(out/'RUN.lock'):
        freeze_config(out,_configuration(a,cfg,'neural_brec'))
        stop=StopFlag(); rows=_read_results(out/'results.json'); done={(x['pair_id'],x['order']) for x in rows}
        for pid in range(a.start,a.end):
            if all((pid,i) in done for i in a.orders): continue
            tr,rr=dataset.block(pid); graphs=tr+rr
            print(f'PAIR {pid}/399 {category(pid)}: extracting independent relabel features',flush=True)
            bs=extract_many(graphs,max(a.orders),cfg,cache,a.workers)
            previous=None
            for i in a.orders:
                folder=out/'checkpoints'/f'pair_{pid:03d}'/f'order_{i}'
                last=folder/'last.pt'
                if (pid,i) in done:
                    previous=last; continue
                xs=[feature_matrix(b,i,a.transform,a.branch) for b in bs]
                model,opt,epoch=initialize(i,a,last,previous if a.continuation else None)
                for ep in range(epoch,a.epochs):
                    model.train()
                    for begin in range(0,64,a.batch_size):
                        stop_at=min(begin+a.batch_size,64)
                        opt.zero_grad()
                        z=model(*pack(tr[begin:stop_at],xs[begin:stop_at],a.device))
                        target=torch.full((len(z)//2,),-1.0,device=a.device)
                        loss=F.cosine_embedding_loss(z[0::2],z[1::2],target,margin=0)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(),5)
                        opt.step()
                    if (ep+1)%a.save_every==0 or ep+1==a.epochs or stop.requested:
                        save_training(last,model,opt,ep+1)
                    if ep==0 or (ep+1)%20==0:
                        print(f'pair={pid} order={i} epoch={ep+1} loss={loss.item():.6g}',flush=True)
                    if stop.requested: return
                st=t2_statistics(embeddings(model,tr,xs[:64],a.device,a.batch_size),a.ridge)
                sr=t2_statistics(embeddings(model,rr,xs[64:],a.device,a.batch_size),a.ridge)
                key='t2_pinv' if a.protocol=='legacy' else 't2_ridge'
                reliable=sr[key]<a.threshold
                gapok=True if a.protocol=='legacy' else st['gap']>1e-6*(1+st['mean_norm'])
                ok=st[key]>a.threshold and reliable and gapok
                exact=all(bool(b[s]['g_exact']) and b[s]['l_exact'].all() for b in bs for s in range(1,i+1))
                row=dict(pair_id=pid,category=category(pid),order=i,separated=int(ok),reliable=bool(reliable),
                         exact_features=bool(exact),protocol=a.protocol,t2=st[key],t2_rel=sr[key],
                         t2_pinv=st['t2_pinv'],t2_rel_pinv=sr['t2_pinv'],gap=st['gap'],rel_gap=sr['gap'],
                         cov_rank=st['cov_rank'],rel_cov_rank=sr['cov_rank'])
                rows.append(row); done.add((pid,i)); _save_brec(out,rows,a.orders)
                print('[RESULT]',row,flush=True); previous=last
            if stop.requested: return
        _save_brec(out,rows,a.orders)

@torch.no_grad()
def sr_pair_count(model,graphs,xs,a):
    """All pairs, conservative empirical L2 margin over relabel variation.
    Distinct from the BREC T2 protocol; no claim of a statistical certificate.
    """
    zs=[]; rng=np.random.default_rng(a.seed+123)
    for _ in range(a.relabels):
        ag=[]; xf=[]
        for A,x in zip(graphs,xs):
            pA,px=permute_graph(A,x,rng); ag.append(pA); xf.append(px)
        zs.append(embeddings(model,ag,xf,a.device,a.batch_size).double().numpy())
    Z=np.stack(zs); mean=Z.mean(0); radius=np.linalg.norm(Z-mean,axis=2).max(0)
    norms=np.linalg.norm(mean,axis=1); n=len(graphs); sep=0; block=a.eval_block
    for start in range(0,n,block):
        U=mean[start:start+block]; end=start+len(U)
        # Stable direct subtraction avoids cancellation in squared-distance identities.
        for begin in range(start,n,block):
            V=mean[begin:begin+block]
            d=np.linalg.norm(U[:,None,:]-V[None,:,:],axis=2)
            margin=2*(radius[start:end,None]+radius[None,begin:begin+len(V)])+1e-6*(1+np.maximum(norms[start:end,None],norms[None,begin:begin+len(V)]))
            mask=d>margin
            if begin==start: mask=np.triu(mask,1)
            sep+=int(mask.sum())
        print(f'pair-distance blocks through graph {end}/{n}',flush=True)
    return sep,n*(n-1)//2,mean,radius

def run_sr(a,cfg):
    if cfg.mode!='exact': raise ValueError('Shared SR encoder requires exact features; no frozen MC noise across relabels')
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True); cache=Path(a.cache or out/'cache')
    graphs=load_graph6(a.data)
    if a.limit: graphs=graphs[:a.limit]
    if len(graphs)<2: raise ValueError('Need at least two graphs')
    if a.srg:
        want=tuple(map(int,a.srg.split(',')))
        if any(srg_parameters(A)!=want for A in graphs): raise ValueError('SRG parameter mismatch')
    with RunLock(out/'RUN.lock'):
        conf=_configuration(a,cfg,'neural_sr_shared_encoder'); conf.update(limit=a.limit,srg=a.srg)
        freeze_config(out,conf); stop=StopFlag()
        bs=extract_many(graphs,max(a.orders),cfg,cache,a.workers)
        rows=_read_results(out/'results.json'); done={x['order'] for x in rows}; previous=None
        for i in a.orders:
            last=out/'checkpoints'/f'order_{i}'/'last.pt'
            if i in done: previous=last; continue
            xs=[feature_matrix(b,i,a.transform,a.branch) for b in bs]
            model,opt,epoch=initialize(i,a,last,previous if a.continuation else None)
            # Global np RNG is checkpointed, including relabel sampling and mini-batches.
            for ep in range(epoch,a.epochs):
                model.train(); perm=np.random.permutation(len(graphs)); losses=[]
                for start in range(0,len(perm),a.batch_size):
                    ids=perm[start:start+a.batch_size]
                    if len(ids)<2: continue
                    ga=[]; xa=[]
                    for _ in range(2):
                        for j in ids:
                            A,x=permute_graph(graphs[j],xs[j],np.random); ga.append(A); xa.append(x)
                    z=F.normalize(model(*pack(ga,xa,a.device)),dim=1)
                    b=len(ids); sim=z[:b]@z[b:].T/0.2; label=torch.arange(b,device=a.device)
                    loss=(F.cross_entropy(sim,label)+F.cross_entropy(sim.T,label))/2
                    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5); opt.step(); losses.append(float(loss))
                if (ep+1)%a.save_every==0 or ep+1==a.epochs or stop.requested: save_training(last,model,opt,ep+1)
                if ep==0 or (ep+1)%20==0: print(f'order={i} epoch={ep+1} contrastive_loss={np.mean(losses):.6g}',flush=True)
                if stop.requested: return
            sep,total,mean,radius=sr_pair_count(model,graphs,xs,a)
            np.savez_compressed(out/f'embeddings_order_{i}.npz',embeddings=mean,relabel_radius=radius)
            row=dict(order=i,graphs=len(graphs),separated=sep,total=total,protocol='shared_encoder_relabel_L2')
            rows.append(row); atomic_json(out/'results.json',rows); atomic_csv(out/'neural_order_summary.csv',rows,list(row))
            print(row,flush=True); previous=last
