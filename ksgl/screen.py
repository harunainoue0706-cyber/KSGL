"""Exact state discrimination, with a separate KS-initialized 1-WL ceiling.
This module does NOT train a neural network. Positive witnesses use integers.
Large SR collections are partitioned; O(M^2) pairs are never materialized.
"""
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
import json
import time
import numpy as np
from .states import cached_order, choose
from .data import BREC, load_graph6, category, CATEGORIES, file_sha256, srg_parameters
from .runtime import atomic_json, atomic_csv, freeze_config, RunLock

VIEWS=('global','local','joint','ks_refinement')

def _job(args):
    idx,A,s,cfg,cache=args
    return idx,cached_order(A,s,cfg,cache)

def node_keys(blocks,m,gorders=None,lorders=None):
    n=blocks[1]['l'].shape[0]
    if gorders is None: gorders=[s for s in range(1,m+1) if bool(blocks[s]['g_exact'])]
    if lorders is None: lorders=[s for s in range(1,m+1) if blocks[s]['l_exact'].all()]
    g=tuple(int(x) for s in gorders for x in blocks[s]['g'])
    return [tuple(int(x) for s in lorders for x in blocks[s]['l'][u])+g for u in range(n)]

def refine_keys(graphs,initial):
    """Shared exact color vocabulary. Returns stable graph color multisets."""
    def intern(rows):
        book={}; ans=[]
        for graph in rows:
            arr=[]
            for key in graph:
                if key not in book: book[key]=len(book)
                arr.append(book[key])
            ans.append(arr)
        return ans,len(book)
    colors,ncolors=intern(initial)
    neighbors=[[np.flatnonzero(row).tolist() for row in A] for A in graphs]
    for _ in range(sum(map(len,graphs))+1):
        rows=[[(c[u],tuple(sorted(c[v] for v in nb[u]))) for u in range(len(c))]
              for c,nb in zip(colors,neighbors)]
        new,count=intern(rows)
        colors=new
        if count==ncolors: break
        ncolors=count
    return [tuple(sorted(c)) for c in colors]

def split_group(ids,graphs,blocks,m,view):
    # Only coordinates EXACT for every graph in this collision class are used.
    go=[s for s in range(1,m+1) if all(bool(blocks[j][s]['g_exact']) for j in ids)]
    lo=[s for s in range(1,m+1) if all(blocks[j][s]['l_exact'].all() for j in ids)]
    if view=='global':
        keys=[tuple(int(x) for s in go for x in blocks[j][s]['g']) for j in ids]
    elif view=='local':
        keys=[tuple(sorted(node_keys(blocks[j],m,[],lo))) for j in ids]
    elif view=='joint':
        keys=[(tuple(int(x) for s in go for x in blocks[j][s]['g']),tuple(sorted(node_keys(blocks[j],m,[],lo)))) for j in ids]
    else:
        keys=refine_keys([graphs[j] for j in ids],[node_keys(blocks[j],m,go,lo) for j in ids])
    out=defaultdict(list)
    for j,key in zip(ids,keys): out[key].append(j)
    complete=(len(go)==m if view=='global' else len(lo)==m if view=='local' else len(go)==m and len(lo)==m)
    return list(out.values()),complete

def pair_stats(A,B,ba,bb,m):
    gcomplete=all(bool(ba[s]['g_exact']) and bool(bb[s]['g_exact']) for s in range(1,m+1))
    lcomplete=all(ba[s]['l_exact'].all() and bb[s]['l_exact'].all() for s in range(1,m+1))
    dg=sum(int(np.abs(ba[s]['g']-bb[s]['g']).sum()) for s in range(1,m+1)) if gcomplete else None
    dl=None
    if lcomplete:
        ca=Counter(node_keys(ba,m,[],list(range(1,m+1))))
        cb=Counter(node_keys(bb,m,[],list(range(1,m+1))))
        # Count-measure TV: zero iff rooted joint multisets agree, even at unequal n.
        dl=sum(abs(ca[k]-cb[k]) for k in ca.keys()|cb.keys())/2
    d={}; localblocks={0:ba,1:bb}
    for view in VIEWS:
        groups,complete=split_group([0,1],[A,B],localblocks,m,view)
        d[view]='separated' if len(groups)==2 else 'equal' if complete else 'unknown'
    d.update(delta_global_l1=dg,delta_local_count_tv=dl)
    return d

def run_screen(args,cfg):
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    cache=Path(args.cache or out/'cache')
    orders=args.orders; maxorder=max(orders)
    config=dict(task='exact_screen',data_sha256=file_sha256(args.data),kind=args.kind,
                orders=orders,features=asdict(cfg),srg=args.srg,limit=args.limit)
    with RunLock(out/'RUN.lock'):
        freeze_config(out,config)
        if args.kind=='brec':
            dataset=BREC(args.data); pids=list(range(min(400,args.limit or 400)))
            graphs=[g for p in pids for g in dataset.representatives(p)]
            partitions={v:[[2*p,2*p+1] for p in pids] for v in VIEWS}
        else:
            graphs=load_graph6(args.data)
            if args.limit: graphs=graphs[:args.limit]
            if args.srg:
                want=tuple(map(int,args.srg.split(',')))
                for j,A in enumerate(graphs):
                    got=srg_parameters(A)
                    if got!=want: raise ValueError(f'Graph {j}: expected SRG{want}, got {got}')
            partitions={v:[list(range(len(graphs)))] for v in VIEWS}
        M=len(graphs); total=len(graphs)//2 if args.kind=='brec' else choose(M,2)
        if M<2: raise ValueError('Need at least two graphs')
        blocks={j:{} for j in range(M)}; rows=[]; pairrows=[]
        print(f'KSGL exact screen: graphs={M}, designated/all pairs={total}, orders={orders}, R={cfg.radius}',flush=True)
        print('This is NOT neural training. Missing exact coordinates are UNKNOWN, never sampled witnesses.',flush=True)
        executor=ProcessPoolExecutor(max_workers=args.workers) if args.workers>1 else None
        try:
            for s in range(1,maxorder+1):
                active=sorted({j for v in VIEWS for group in partitions[v] if len(group)>1 for j in group})
                # For BREC, retain complete pair-level diagnostics at every requested order.
                if args.kind=='brec': active=list(range(M))
                jobs=[(j,graphs[j],s,cfg,str(cache)) for j in active]
                requested = s in orders
                t0=time.perf_counter()
                print(f'order {s}: START active_graphs={len(jobs)} requested={requested}',flush=True)
                if executor:
                    futures=[executor.submit(_job,j) for j in jobs]
                    iterator=(f.result() for f in as_completed(futures))
                else: iterator=map(_job,jobs)
                step=max(1,len(jobs)//20) if jobs else 1
                for done,(j,b) in enumerate(iterator,1):
                    blocks[j][s]=b
                    if done%step==0 or done==len(jobs):
                        elapsed=max(time.perf_counter()-t0,1e-9)
                        rate=done/elapsed
                        eta=(len(jobs)-done)/rate if rate>0 else float('inf')
                        print(f'order {s}: {done}/{len(jobs)} active graphs cached '
                              f'elapsed={elapsed:.1f}s eta={eta:.1f}s',flush=True)

                # IMPORTANT: refine the collision partitions at EVERY intermediate order.
                # The representation is cumulative, so this is mathematically equivalent to
                # waiting until the next requested checkpoint, but it prunes graphs that were
                # already separated at an omitted order (e.g. K=5 when requesting 1,2,3,4,6).
                partition_dump={}
                order_rows=[]
                for view in VIEWS:
                    new=[]; unknown=0
                    for group in partitions[view]:
                        if len(group)==1: new.append(group); continue
                        children,complete=split_group(group,graphs,blocks,s,view)
                        new.extend(children)
                        if not complete: unknown+=sum(choose(len(c),2) for c in children)
                    partitions[view]=new
                    collisions=sum(choose(len(c),2) for c in new)
                    if requested:
                        row=dict(order=s,view=view,total=total,separated=total-collisions,
                                 unresolved_exact=collisions-unknown,unknown=unknown)
                        rows.append(row); order_rows.append(row); print(row,flush=True)
                        partition_dump[view]=new

                if not requested:
                    print(f'order {s}: intermediate cumulative block applied for pruning; '
                          'summary not requested',flush=True)
                    continue

                atomic_json(out/f'partitions_order_{s}.json',partition_dump)
                if args.kind=='brec':
                    for p in pids:
                        stats=pair_stats(graphs[2*p],graphs[2*p+1],blocks[2*p],blocks[2*p+1],s)
                        pairrows.append(dict(order=s,pair_id=p,category=category(p),**stats))
                    atomic_csv(out/'pair_results.csv',pairrows,list(pairrows[0]))
                    cr=[]
                    for name,lo,hi in CATEGORIES:
                        for v in VIEWS:
                            rr=[x for x in pairrows if x['order']==s and x['category']==name]
                            cr.append(dict(order=s,category=name,view=v,separated=sum(x[v]=='separated' for x in rr),
                                           equal=sum(x[v]=='equal' for x in rr),unknown=sum(x[v]=='unknown' for x in rr),evaluated=len(rr),official_total=hi-lo))
                    old=[]
                    if (out/'category_history.json').exists(): old=json.loads((out/'category_history.json').read_text())
                    old=[r for r in old if r['order']!=s]+cr
                    atomic_json(out/'category_history.json',old); atomic_csv(out/'category_summary.csv',old,list(cr[0]))
                atomic_csv(out/'order_summary.csv',rows,list(rows[0]))
        finally:
            if executor: executor.shutdown()
        return rows
