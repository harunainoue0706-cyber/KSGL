#!/usr/bin/env python3
import argparse, gzip
from pathlib import Path
import networkx as nx
import numpy as np
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ksgl.data import load_graph6, srg_parameters

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--expect-source')
    p.add_argument('--expect-target')
    a=p.parse_args()
    graphs=load_graph6(a.data)
    if a.expect_source:
        want=tuple(map(int,a.expect_source.split(',')))
        bad=[i for i,A in enumerate(graphs) if srg_parameters(A)!=want]
        if bad: raise SystemExit(f'source parameter mismatch, first bad index={bad[0]}')
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('wb') as f:
        for A in graphs:
            B=(1-np.eye(len(A),dtype=np.uint8)-A).astype(np.uint8)
            B[B>1]=0
            if a.expect_target:
                want=tuple(map(int,a.expect_target.split(',')))
                got=srg_parameters(B)
                if got!=want: raise SystemExit(f'complement parameter mismatch: {got} != {want}')
            G=nx.from_numpy_array(B)
            f.write(nx.to_graph6_bytes(G,header=False))
    print(f'wrote {len(graphs)} complemented graphs -> {out}')
if __name__=='__main__': main()
