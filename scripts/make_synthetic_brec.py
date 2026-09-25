#!/usr/bin/env python3
"""Tiny-content, full-layout fixture. NOT the actual BREC dataset."""
import argparse
from pathlib import Path
import networkx as nx
import numpy as np
p=argparse.ArgumentParser()
p.add_argument('--out',default='validation/synthetic_brec_layout.npy')
a=p.parse_args()
A=nx.to_graph6_bytes(nx.cycle_graph(6),header=False).strip()
B=nx.to_graph6_bytes(nx.complete_graph(6),header=False).strip()
rows=np.array([A,B]*32*400+[A,A]*32*400,dtype=object)
path=Path(a.out);path.parent.mkdir(parents=True,exist_ok=True)
np.save(path,rows)
print('SYNTHETIC layout fixture:',path)
