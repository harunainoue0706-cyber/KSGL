#!/usr/bin/env python3
"""Run the complete SR/BREC benchmark suite registered in a manifest.

The paper-table SR collections are:
SR16, SR25, SR26, SR28, SR29, SR40,
SRG(35,16,6,8), SRG(35,18,9,9), SRG(36,14,4,6), SRG(36,15,6,6),
SRG(37,18,8,9), SRG(45,12,3,3), SRG(50,21,8,9), SRG(64,18,2,6),
SRG(65,32,15,16), plus BREC.

SRG35_18 is derived by graph complementation from the 3,854-graph SRG35_16
collection when reproducing the 3,854-row convention in the paper table.
"""
import argparse, json, os, subprocess, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PAPER_TABLE_NAMES=[
    'SR16','SR25','SR26','SR28','SR29','SR40',
    'SRG35_16','SRG35_18','SRG36_14','SRG36_15','SRG37_18',
    'SRG45_12','SRG50_21','SRG64_18','SRG65_32','BREC'
]

def _manifest_path(p):
    if not p:
        return None
    p=Path(p).expanduser()
    if not p.is_absolute():
        p=ROOT/p
    return p.resolve()

def resolve_path(name,c,cfg,out,dry=False):
    p=_manifest_path(c.get('path',''))
    if p and p.is_file(): return str(p)
    d=c.get('derive')
    if not d: return ''
    if d.get('op')!='complement': raise ValueError(f'Unsupported derive op for {name}: {d}')
    src_name=d['source']; src=cfg.get(src_name,{})
    src_path=_manifest_path(src.get('path','') or c.get('source_path',''))
    if not src_path or not src_path.is_file(): return ''
    src_path=str(src_path)
    derived=Path(out)/'_derived'/f'{name}.g6'
    if not derived.is_file() and not dry:
        cmd=[sys.executable,str(ROOT/'scripts/make_complement_collection.py'),
             '--data',src_path,'--out',str(derived),
             '--expect-source',','.join(map(str,src['srg'])),
             '--expect-target',','.join(map(str,c['srg']))]
        print('DERIVE:',repr(cmd),flush=True); subprocess.check_call(cmd,cwd=ROOT)
    return str(derived)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',default=str(ROOT/'configs/datasets.local.json'))
    p.add_argument('--names',default='paper-table',help='comma list or paper-table')
    p.add_argument('--stage',choices=['screen','neural'],default='screen')
    p.add_argument('--out',default=str(ROOT/'results'))
    p.add_argument('--orders',default='1,2,3,4,5,6')
    p.add_argument('--radius',type=int,default=1)
    p.add_argument('--workers',type=int,default=1)
    p.add_argument('--exact-limit',type=int,default=100_000_000)
    p.add_argument('--brec-mode',choices=['exact','hybrid'],default='hybrid')
    p.add_argument('--samples',type=int,default=10000)
    p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--device',default='auto')
    p.add_argument('--continuation',action='store_true')
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    mp=Path(a.manifest)
    if not mp.is_file(): p.error(f'Manifest not found: {mp}. Run scripts/setup_data.py (download) or scripts/locate_data.py (existing files) first.')
    cfg=json.loads(mp.read_text())
    names=PAPER_TABLE_NAMES if a.names=='paper-table' else [x for x in a.names.split(',') if x]
    unknown=[x for x in names if x not in cfg]
    if unknown: p.error('Unknown dataset names: '+','.join(unknown))

    # Resolve raw and derived paths before launching expensive experiments.
    resolved={}
    for name in names:
        path=resolve_path(name,cfg[name],cfg,a.out,a.dry_run)
        if not path or (not a.dry_run and not Path(path).is_file()):
            p.error(f'{name} is unresolved. Run scripts/setup_data.py, or re-run scripts/locate_data.py and inspect its candidates.')
        resolved[name]=path

    env=os.environ.copy()
    env.setdefault('OMP_NUM_THREADS','1'); env.setdefault('OPENBLAS_NUM_THREADS','1')
    env.setdefault('MKL_NUM_THREADS','1'); env.setdefault('NUMEXPR_NUM_THREADS','1')
    for name in names:
        c=cfg[name]; brec=c['kind']=='brec'; data=resolved[name]
        task='screen' if a.stage=='screen' else ('train-brec' if brec else 'train-sr')
        out=Path(a.out)/name/f'{a.stage}_R{a.radius}'
        cmd=[sys.executable,'-u','-m','ksgl',task,'--data',data,'--kind',c['kind'],
             '--out',str(out),'--cache',str(Path(a.out)/name/f'cache_R{a.radius}'),
             '--orders',a.orders,'--radius',str(a.radius),'--workers',str(a.workers),
             '--exact-limit',str(a.exact_limit)]
        if c.get('srg'): cmd+=['--srg',','.join(map(str,c['srg']))]
        if a.stage=='neural':
            cmd+=['--epochs',str(a.epochs),'--device',a.device,'--batch-size','64' if brec else '32']
            if a.continuation: cmd+=['--continuation']
            if brec: cmd+=['--mode',a.brec_mode,'--samples',str(a.samples)]
            else: cmd+=['--mode','exact']
        print('\n'+'='*100,flush=True)
        print(f'{name}: expected graphs={c.get("graphs","-")} pairs={c.get("pairs","-")}',flush=True)
        print('COMMAND:',repr(cmd),flush=True)
        if a.dry_run: continue
        out.mkdir(parents=True,exist_ok=True)
        with (out/'console.log').open('a',buffering=1) as log:
            log.write('\nCOMMAND '+repr(cmd)+'\n')
            process=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT,text=True,bufsize=1)
            for line in process.stdout:
                print(line,end='',flush=True); log.write(line)
            rc=process.wait()
        if rc: raise SystemExit(f'{name} exited with code {rc}; see {out / "console.log"}')
if __name__=='__main__': main()
