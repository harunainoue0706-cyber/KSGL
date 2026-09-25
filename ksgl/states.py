"""Equation (2). Exact integers and sampled estimates are kept separate.
No canonical labeling, vertex IDs, pair IDs or target labels enter features.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from functools import lru_cache
from itertools import combinations, islice
from math import comb
import hashlib, json, os, tempfile
from pathlib import Path
import numpy as np

@dataclass(frozen=True)
class FeatureConfig:
    radius: int = 1
    mode: str = 'exact'
    exact_limit: int = 2_000_000
    samples: int = 10_000
    seed: int = 2023
    chunk: int = 32768
    version: str = 'ksgl-H-v1'
    def __post_init__(self):
        if self.radius < 0 or self.mode not in {'exact','hybrid'}:
            raise ValueError('radius >= 0; mode: exact or hybrid')
        if min(self.exact_limit,self.samples,self.chunk)<1:
            raise ValueError('Budgets must be positive')

def choose(n,k): return comb(n,k) if 0<=k<=n else 0

def validate_adjacency(A):
    A=np.asarray(A)
    if A.ndim!=2 or A.shape[0]!=A.shape[1] or len(A)==0:
        raise ValueError('Expected nonempty square adjacency matrix')
    if not np.array_equal(A,A.T) or np.any(np.diag(A)) or not np.isin(A,(0,1)).all():
        raise ValueError('Equation (2) requires a simple undirected binary graph')
    return A.astype(np.uint8)

def rank_gf2(A):
    """Exact elimination using Python integer bitsets."""
    pivots={}
    for row in np.asarray(A,dtype=np.uint8):
        bits=sum(int(x&1)<<j for j,x in enumerate(row))
        while bits:
            lead=bits.bit_length()-1
            if lead in pivots: bits^=pivots[lead]
            else:
                pivots[lead]=bits
                break
    return len(pivots)

@lru_cache(None)
def nullity_lut(s):
    if s>6: raise ValueError('LUT restricted to s<=6')
    edges=list(combinations(range(s),2))
    out=np.empty(1<<len(edges),dtype=np.uint8)
    for mask in range(len(out)):
        A=np.zeros((s,s),dtype=np.uint8)
        for bit,(u,v) in enumerate(edges): A[u,v]=A[v,u]=(mask>>bit)&1
        out[mask]=s-rank_gf2(A)
    return out

def subset_nullities(A,S):
    S=np.asarray(S,dtype=np.int64); s=S.shape[1]
    if s<=6:
        masks=np.zeros(len(S),dtype=np.uint32)
        for bit,(u,v) in enumerate(combinations(range(s),2)):
            masks|=A[S[:,u],S[:,v]].astype(np.uint32)<<bit
        return nullity_lut(s)[masks].astype(np.int64)
    return np.array([s-rank_gf2(A[np.ix_(row,row)]) for row in S],dtype=np.int64)

def balls(A,radius):
    n=len(A); B=np.eye(n,dtype=bool); neighbors=[np.flatnonzero(row) for row in A]
    for u in range(n):
        seen,frontier={u},{u}
        for _ in range(radius):
            nxt={int(v) for x in frontier for v in neighbors[x]}-seen
            seen.update(nxt); frontier=nxt
            if not frontier: break
        B[u,list(seen)]=True
    return B

def exact_histogram(A,candidates,s,root=None,chunk=32768):
    k=s-int(root is not None); hist=np.zeros(s+1,dtype=np.int64)
    it=combinations([int(v) for v in candidates],k)
    while True:
        rows=list(islice(it,chunk))
        if not rows: break
        S=np.asarray(rows,dtype=np.int64).reshape(len(rows),k)
        if root is not None: S=np.column_stack([np.full(len(S),root,dtype=np.int64),S])
        hist+=np.bincount(subset_nullities(A,S),minlength=s+1)
    return hist

def sampled_histogram(A,candidates,s,samples,rng,root=None,chunk=32768):
    candidates=np.asarray(candidates,dtype=np.int64); k=s-int(root is not None)
    out=np.zeros(s+1,dtype=np.int64)
    # Independent random keys induce a uniform k-subset in every row.
    # Bound temporary memory instead of constructing samples one Python loop at a time.
    chunk=min(chunk,max(1,4_000_000//max(1,len(candidates))))
    for start in range(0,samples,chunk):
        b=min(chunk,samples-start)
        if k==0:
            S=np.empty((b,0),dtype=np.int64)
        else:
            keys=rng.random((b,len(candidates)))
            indices=np.argpartition(keys,k-1,axis=1)[:,:k]
            S=candidates[indices]
        if root is not None: S=np.column_stack([np.full(b,root,dtype=np.int64),S])
        out+=np.bincount(subset_nullities(A,S),minlength=s+1)
    return out

def _low_order(A,s,B):
    """Closed forms valid for ALL simple graphs, not an SRG shortcut."""
    n=len(A); g=np.zeros(s+1,dtype=np.int64); l=np.zeros((n,s+1),dtype=np.int64)
    d=A.sum(1).astype(np.int64)
    if s==1: g[1],l[:,1]=n,1
    elif s==2:
        g[0]=int(d.sum())//2; g[2]=choose(n,2)-g[0]
        for u in range(n):
            vertices=np.flatnonzero(B[u]); du=int(A[u,vertices].sum())
            l[u,0],l[u,2]=du,len(vertices)-1-du
    else:
        e=int(d.sum())//2; a=A.astype(np.int64)
        t=int(np.sum((a@a)*a))//6; w=sum(choose(int(x),2) for x in d)
        g[3]=choose(n,3)-e*(n-2)+w-t if n>=3 else 0; g[1]=choose(n,3)-g[3]
        for u in range(n):
            q=np.flatnonzero(B[u]&(A[u]==0)); q=q[q!=u]
            l[u,3]=choose(len(q),2)-int(A[np.ix_(q,q)].sum())//2
            l[u,1]=choose(int(B[u].sum())-1,2)-l[u,3]
    return g,l

def extract_order(A,s,config=FeatureConfig()):
    A=validate_adjacency(A)
    if not 1<=s<=20: raise ValueError('Supported orders: 1,...,20')
    n=len(A); B=balls(A,config.radius); ng=choose(n,s)
    if ng>np.iinfo(np.int64).max: raise OverflowError('Population exceeds int64')
    nl=np.array([choose(int(row.sum())-1,s-1) for row in B],dtype=np.int64)
    digest=hashlib.sha256(A.tobytes()).digest()
    rng=np.random.default_rng(np.random.SeedSequence([config.seed,s,int.from_bytes(digest[:8],'little')]))
    g=np.full(s+1,-1,dtype=np.int64); l=np.full((n,s+1),-1,dtype=np.int64)
    gh=np.full(s+1,np.nan); lh=np.full((n,s+1),np.nan); ge=False; le=np.zeros(n,dtype=bool)
    if s<=3:
        g,l=_low_order(A,s,B); gh=g.astype(float); lh=l.astype(float); ge=True; le[:]=True
    else:
        if ng<=config.exact_limit:
            g=exact_histogram(A,range(n),s,chunk=config.chunk); gh=g.astype(float); ge=True
        elif config.mode=='hybrid':
            gh=sampled_histogram(A,range(n),s,config.samples,rng,chunk=config.chunk)*(ng/config.samples)
        for u in range(n):
            cand=np.flatnonzero(B[u]); cand=cand[cand!=u]
            if int(nl[u])<=config.exact_limit:
                l[u]=exact_histogram(A,cand,s,root=u,chunk=config.chunk); lh[u]=l[u]; le[u]=True
            elif config.mode=='hybrid':
                lh[u]=sampled_histogram(A,cand,s,config.samples,rng,root=u,chunk=config.chunk)*(int(nl[u])/config.samples)
    if ge and int(g.sum())!=ng: raise AssertionError('Global mass check failed')
    if le.any() and not np.array_equal(l[le].sum(1),nl[le]): raise AssertionError('Rooted mass check failed')
    parity=np.arange(s+1)%2!=s%2
    if ge and g[parity].any(): raise AssertionError('Alternating-rank parity failed')
    return dict(g=g,l=l,g_hat=gh,l_hat=lh,g_exact=np.array(ge),l_exact=le,
                global_population=np.array(ng),local_population=nl,order=np.array(s),radius=np.array(config.radius))

def graph_digest(A):
    A=validate_adjacency(A)
    return hashlib.sha256(len(A).to_bytes(8,'little')+A.tobytes()).hexdigest()

def cached_order(A,s,config,cache):
    meta=json.dumps(dict(config=asdict(config),order=s,graph=graph_digest(A)),sort_keys=True)
    key=hashlib.sha256(meta.encode()).hexdigest(); folder=Path(cache)/key[:2]
    folder.mkdir(parents=True,exist_ok=True); path=folder/(key+'.npz')
    if path.exists():
        with np.load(path,allow_pickle=False) as z:
            if str(z['metadata'])!=meta: raise RuntimeError('Cache metadata mismatch')
            return {k:z[k] for k in z.files if k!='metadata'}
    result=extract_order(A,s,config)
    fd,tmp=tempfile.mkstemp(dir=folder,suffix='.npz'); os.close(fd)
    try:
        np.savez_compressed(tmp,metadata=np.array(meta),**result); os.replace(tmp,path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)
    return result

def feature_matrix(blocks,order,transform='log1p',branch='joint'):
    """[1,L1,G1,...,Li,Gi]. Count transforms do not change Equation (2)."""
    n=blocks[1]['l_hat'].shape[0]; parts=[np.ones((n,1))]
    for s in range(1,order+1):
        b=blocks[s]; l=np.array(b['l_hat'],copy=True); g=np.broadcast_to(b['g_hat'],(n,s+1)).copy()
        if branch in {'global','base'}: l[:]=0
        if branch in {'local','base'}: g[:]=0
        if not np.isfinite(l).all() or not np.isfinite(g).all():
            raise RuntimeError('Missing features: increase exact-limit or explicitly select hybrid')
        if transform=='log1p': l,g=np.log1p(l),np.log1p(g)
        elif transform!='counts': raise ValueError('transform: log1p or counts')
        parts.extend([l,g])
    return np.concatenate(parts,axis=1).astype(np.float32)
