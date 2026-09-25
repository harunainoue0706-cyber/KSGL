#!/usr/bin/env python3
"""Aggregate completed screen/neural outputs without inventing missing results."""
import argparse,csv,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_NAMES=['SR16','SR25','SR26','SR28','SR29','SR40','SRG35_16','SRG35_18',
               'SRG36_14','SRG36_15','SRG37_18','SRG45_12','SRG50_21','SRG64_18','SRG65_32','BREC']

def read_csv(p):
    if not p.is_file(): return []
    with p.open(newline='') as f: return list(csv.DictReader(f))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--manifest',default=str(ROOT/'configs/datasets.local.json'))
    ap.add_argument('--results',default=str(ROOT/'results/paper_table'))
    ap.add_argument('--neural-results',default='')
    ap.add_argument('--radius',type=int,default=1)
    ap.add_argument('--orders',default='1,2,3,4,5,6')
    ap.add_argument('--out',default=str(ROOT/'results/order_sweep_summary.csv'))
    a=ap.parse_args()
    cfg=json.loads(Path(a.manifest).read_text())
    orders=[int(x) for x in a.orders.split(',')]
    rows=[]
    for name in DEFAULT_NAMES:
        if name not in cfg: continue
        c=cfg[name]; base=Path(a.results)/name/f'screen_R{a.radius}'
        screen=read_csv(base/'order_summary.csv')
        neural=[]
        nr=Path(a.neural_results) if a.neural_results else Path(a.results)
        nbase=nr/name/f'neural_R{a.radius}'
        if c['kind']=='sr': neural=read_csv(nbase/'neural_order_summary.csv')
        for order in orders:
            row={'benchmark':name,'order':order,'graphs':c.get('graphs',''),'pairs':c.get('pairs',400 if name=='BREC' else '')}
            for view in ['global','local','joint','ks_refinement']:
                q=[r for r in screen if int(r['order'])==order and r['view']==view]
                row[f'{view}_separated']=q[0]['separated'] if q else ''
                row[f'{view}_unknown']=q[0]['unknown'] if q else ''
            q=[r for r in neural if int(r['order'])==order]
            row['neural_separated']=q[0]['separated'] if q else ''
            rows.append(row)
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True)
    fields=list(rows[0]) if rows else ['benchmark','order']
    with out.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    print(out)
if __name__=='__main__':main()
