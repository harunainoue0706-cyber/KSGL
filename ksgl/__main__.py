from argparse import ArgumentParser
from .states import FeatureConfig

def main():
    p=ArgumentParser(description='KSGL order-controlled state extraction and genuine GIN training')
    p.add_argument('task',choices=['screen','train-brec','train-sr'])
    p.add_argument('--data',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--cache')
    p.add_argument('--kind',choices=['sr','brec'],default='sr')
    p.add_argument('--orders',default='1,2,3,4,5,6',help='CUMULATIVE maxima; include intermediate orders (e.g. 5) to expose progress and maximize pruning')
    p.add_argument('--radius',type=int,default=1)
    p.add_argument('--srg',help='Verify n,k,lambda,mu; do not infer from a short dataset name')
    p.add_argument('--mode',choices=['exact','hybrid'],default='exact')
    p.add_argument('--exact-limit',type=int,default=2_000_000)
    p.add_argument('--samples',type=int,default=10000)
    p.add_argument('--workers',type=int,default=1)
    p.add_argument('--limit',type=int,default=0)
    p.add_argument('--seed',type=int,default=2023)
    p.add_argument('--start',type=int,default=0); p.add_argument('--end',type=int,default=400)
    p.add_argument('--hidden',type=int,default=64); p.add_argument('--layers',type=int,default=4)
    p.add_argument('--out-dim',type=int,default=16); p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--lr',type=float,default=0.001); p.add_argument('--weight-decay',type=float,default=0.0001)
    p.add_argument('--device',default='auto'); p.add_argument('--threads',type=int,default=4)
    p.add_argument('--branch',choices=['joint','local','global','base'],default='joint')
    p.add_argument('--transform',choices=['counts','log1p'],default='log1p')
    p.add_argument('--continuation',action='store_true',help='Warm-start next order by zero-extension; separate experiment')
    p.add_argument('--save-every',type=int,default=1)
    p.add_argument('--protocol',choices=['legacy','stabilized'],default='stabilized')
    p.add_argument('--threshold',type=float,default=20.0); p.add_argument('--ridge',type=float,default=1e-10)
    p.add_argument('--batch-size',type=int,default=32); p.add_argument('--relabels',type=int,default=4)
    p.add_argument('--eval-block',type=int,default=256)
    a=p.parse_args(); a.orders=sorted(set(map(int,a.orders.split(','))))
    if not a.orders or min(a.orders)<1 or max(a.orders)>20: p.error('orders must be in 1,...,20')
    if a.workers<1 or not 0<=a.start<a.end<=400: p.error('Invalid workers or BREC range')
    if min(a.epochs,a.batch_size,a.save_every,a.relabels,a.eval_block,a.threads)<1 or a.ridge<=0: p.error('Positive budgets required')
    if a.task=='train-brec' and (a.batch_size<2 or a.batch_size%2): p.error('BREC requires an even batch-size >= 2')
    if a.task=='train-sr' and a.batch_size<2: p.error('SR contrastive training requires batch-size >= 2')
    cfg=FeatureConfig(radius=a.radius,mode=a.mode,exact_limit=a.exact_limit,samples=a.samples,seed=a.seed)
    if a.task=='screen':
        if a.mode!='exact': p.error('The exact screen never compares MC samples as exact graph invariants')
        from .screen import run_screen
        run_screen(a,cfg)
    else:
        import torch
        torch.set_num_threads(a.threads)
        if a.device=='auto': a.device='cuda' if torch.cuda.is_available() else 'cpu'
        from .train import run_brec,run_sr
        (run_brec if a.task=='train-brec' else run_sr)(a,cfg)

if __name__=='__main__': main()
