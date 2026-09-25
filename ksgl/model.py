"""Order-indexed GIN. Pure PyTorch; torch_geometric is NOT required.
The layer equals GINConv(MLP, train_eps=True), followed by ReLU.
"""
from __future__ import annotations
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

def input_dim(order): return 1+order*(order+3)

class KSGL(nn.Module):
    def __init__(self,order=6,hidden=64,layers=4,out_dim=16):
        super().__init__()
        self.order,self.hidden,self.layers,self.out_dim=order,hidden,layers,out_dim
        self.input=nn.Sequential(nn.Linear(input_dim(order),hidden),nn.ReLU(),nn.Linear(hidden,hidden))
        self.mlps=nn.ModuleList([nn.Sequential(nn.Linear(hidden,hidden),nn.ReLU(),nn.Linear(hidden,hidden)) for _ in range(layers)])
        self.eps=nn.Parameter(torch.zeros(layers))
        self.out=nn.Sequential(nn.Linear(hidden,hidden),nn.ReLU(),nn.Linear(hidden,out_dim))
    def forward(self,x,edge_index,batch):
        h=self.input(x)
        for layer,mlp in enumerate(self.mlps):
            agg=torch.zeros_like(h)
            if edge_index.numel(): agg.index_add_(0,edge_index[1],h[edge_index[0]])
            h=F.relu(mlp((1+self.eps[layer])*h+agg))
        pooled=h.new_zeros((int(batch.max())+1,self.hidden))
        pooled.index_add_(0,batch,h)
        return self.out(pooled)
    def expand_order(self,new_order):
        """Zero-extension preserves the previous network function exactly."""
        if new_order<self.order: raise ValueError('Continuation only expands order')
        old=self.input[0]
        new=nn.Linear(input_dim(new_order),self.hidden,device=old.weight.device,dtype=old.weight.dtype)
        with torch.no_grad():
            new.weight.zero_(); new.weight[:,:old.in_features].copy_(old.weight); new.bias.copy_(old.bias)
        self.input[0]=new; self.order=new_order


def expand_adam_state(optimizer):
    """After loading old optimizer state, pad newly added input moments with zero."""
    for group in optimizer.param_groups:
        for p in group['params']:
            for key in ('exp_avg','exp_avg_sq','max_exp_avg_sq'):
                v=optimizer.state.get(p,{}).get(key)
                if v is not None and v.shape!=p.shape:
                    if v.ndim!=p.ndim or any(a>b for a,b in zip(v.shape,p.shape)):
                        raise ValueError('Incompatible optimizer expansion')
                    w=torch.zeros_like(p); w[tuple(slice(0,n) for n in v.shape)]=v
                    optimizer.state[p][key]=w

def pack(graphs,features,device='cpu'):
    xs=[]; es=[]; bs=[]; offset=0
    for j,(A,x) in enumerate(zip(graphs,features)):
        if len(A)!=len(x): raise ValueError('Feature/node alignment mismatch')
        xs.append(torch.as_tensor(x,dtype=torch.float32)); e=np.vstack(np.nonzero(A)).astype(np.int64)
        es.append(torch.from_numpy(e)+offset); bs.append(torch.full((len(A),),j,dtype=torch.long)); offset+=len(A)
    return torch.cat(xs).to(device),torch.cat(es,1).to(device),torch.cat(bs).to(device)

def permute_graph(A,x,rng):
    p=rng.permutation(len(A))
    return A[np.ix_(p,p)],x[p]
