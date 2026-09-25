"""Graph6 collections and the official 51200-entry BREC layout."""
from pathlib import Path
import gzip
import hashlib
import numpy as np
import networkx as nx
from .states import validate_adjacency

CATEGORIES=[('Basic',0,60),('SimpleRegular',60,110),('StronglyRegular',110,160),
            ('Extension',160,260),('CFI-3WL-easy',260,320),('CFI-3WL-hard',320,360),
            ('4-Vertex',360,380),('DistanceRegular',380,400)]
def category(pid):
    for name,lo,hi in CATEGORIES:
        if lo<=pid<hi: return name
    raise ValueError('BREC pair ID must be in [0,400)')

def decode(raw):
    if isinstance(raw,np.ndarray) and raw.ndim==0: raw=raw.item()
    if isinstance(raw,str): raw=raw.encode('ascii')
    G=nx.from_graph6_bytes(bytes(raw).strip())
    return validate_adjacency(nx.to_numpy_array(G,dtype=np.uint8))

def load_graph6(path):
    path=Path(path); op=gzip.open if path.suffix=='.gz' else open
    with op(path,'rb') as f:
        raw=[x.strip().replace(b'>>graph6<<',b'') for x in f if x.strip()]
    raw=[x for x in raw if x]
    graphs=[decode(x) for x in raw]
    if not graphs: raise ValueError('Empty graph collection')
    return graphs

class BREC:
    def __init__(self,path):
        # Use allow_pickle ONLY for a trusted local benchmark dataset.
        self.raw=np.load(path,allow_pickle=True).reshape(-1)
        if len(self.raw) not in {51200,800}:
            raise ValueError(f'BREC expects 51200 official entries or 800 representatives, got {len(self.raw)}')
        self.official=len(self.raw)==51200
    def representatives(self,pid):
        j=pid*64 if self.official else pid*2
        return decode(self.raw[j]),decode(self.raw[j+1])
    def block(self,pid):
        if not self.official:
            raise ValueError('Neural BREC needs all 51200 official relabel/control entries')
        tr=[decode(x) for x in self.raw[pid*64:(pid+1)*64]]
        rr=[decode(x) for x in self.raw[(pid+400)*64:(pid+401)*64]]
        return tr,rr

def srg_parameters(A):
    """Return verified (n,k,lambda,mu), or None. No filename inference."""
    n=len(A); d=A.sum(1)
    if not np.all(d==d[0]) or int(d[0]) in {0,n-1}: return None
    C=A.astype(np.int64)@A.astype(np.int64)
    adj=C[A.astype(bool)]; non=C[(A==0)&~np.eye(n,dtype=bool)]
    if len(set(adj.tolist()))!=1 or len(set(non.tolist()))!=1: return None
    return n,int(d[0]),int(adj[0]),int(non[0])

def file_sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
