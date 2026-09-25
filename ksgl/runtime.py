"""Atomic writes, run locks, signal-safe stopping and full RNG snapshots."""
from pathlib import Path
import csv, json, os, random, signal, tempfile
import numpy as np

class RunLock:
    def __init__(self,path): self.path=Path(path)
    def __enter__(self):
        import fcntl
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.handle=self.path.open('a+')
        try: fcntl.flock(self.handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError(f'Another process owns {self.path}')
        self.handle.seek(0); self.handle.truncate(); self.handle.write(str(os.getpid())); self.handle.flush()
        return self
    def __exit__(self,*_): self.handle.close()

class StopFlag:
    def __init__(self):
        self.requested=False
        for sig in (signal.SIGTERM,signal.SIGINT): signal.signal(sig,self._handler)
    def _handler(self,*_):
        self.requested=True
        print('STOP requested: finish current epoch and save checkpoint.',flush=True)

def atomic_json(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(dir=path.parent); os.close(fd)
    try:
        Path(tmp).write_text(json.dumps(obj,indent=2,sort_keys=True))
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)

def atomic_csv(path,rows,fields):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(dir=path.parent); os.close(fd)
    try:
        with open(tmp,'w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)

def freeze_config(out,config):
    path=Path(out)/'config.json'
    # JSON round-trip gives stable tuple/list handling.
    config=json.loads(json.dumps(config,sort_keys=True))
    if path.exists() and json.loads(path.read_text())!=config:
        raise RuntimeError('Output contains a different configuration. Use a new --out; do not delete old results.')
    if not path.exists(): atomic_json(path,config)

def seed_all(seed):
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def rng_state():
    import torch
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)

def restore_rng(state):
    import torch
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None and torch.cuda.is_available(): torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])

def atomic_torch_save(path,obj):
    import torch
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(dir=path.parent); os.close(fd)
    try:
        torch.save(obj,tmp)
        with open(tmp,'rb') as f: os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def torch_load_compat(path, map_location=None):
    """Load a trusted local checkpoint across old and new PyTorch versions.

    PyTorch releases before ``weights_only`` was added route unknown keyword
    arguments into pickle.Unpickler, producing:
      TypeError: 'weights_only' is an invalid keyword argument for Unpickler
    Newer releases accept the keyword.  We first try the modern safe/explicit
    call and fall back only for that compatibility failure.  Checkpoints loaded
    here are created locally by this package and include optimizer/RNG state, so
    ``weights_only=True`` is not applicable.
    """
    import torch
    kwargs={} if map_location is None else {'map_location': map_location}
    try:
        return torch.load(path, weights_only=False, **kwargs)
    except TypeError as exc:
        msg=str(exc)
        if 'weights_only' not in msg and 'Unpickler' not in msg:
            raise
        return torch.load(path, **kwargs)
