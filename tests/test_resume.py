"""Full optimizer/checkpoint restart must match uninterrupted CPU optimization."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from unittest.mock import patch
from ksgl.model import KSGL, pack
from ksgl.runtime import seed_all, freeze_config, torch_load_compat
from ksgl.train import initialize, save_training

class ResumeTests(unittest.TestCase):
    def test_checkpoint_next_step(self):
        torch.set_num_threads(1)
        args=SimpleNamespace(seed=123,hidden=8,layers=2,out_dim=4,lr=.001,
                             weight_decay=.0001,device='cpu')
        A=np.ones((4,4),dtype=np.uint8)-np.eye(4,dtype=np.uint8)
        x=np.arange(4*5,dtype=np.float32).reshape(4,5)/10
        seed_all(args.seed)
        model=KSGL(1,8,2,4); opt=torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
        def step(m,o):
            noise=torch.randn(4,5)*.01
            batch=pack([A],[x+noise.numpy()])
            o.zero_grad(); z=m(*batch); loss=(z-.1).square().sum();loss.backward();o.step()
        step(model,opt)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'last.pt';save_training(path,model,opt,1)
            step(model,opt)
            resumed,resumed_opt,epoch=initialize(1,args,path)
            self.assertEqual(epoch,1)
            step(resumed,resumed_opt)
            for a,b in zip(model.parameters(),resumed.parameters()):
                self.assertTrue(torch.equal(a,b))

    def test_legacy_torch_load_compat(self):
        """Simulate a pre-weights_only torch.load and require fallback."""
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'x.pt'
            torch.save({'x': torch.tensor([1,2,3])},path)
            real_load=torch.load
            calls=[]
            def legacy_load(*args,**kwargs):
                calls.append(dict(kwargs))
                if 'weights_only' in kwargs:
                    raise TypeError("'weights_only' is an invalid keyword argument for Unpickler()")
                return real_load(*args,**kwargs)
            with patch('torch.load',side_effect=legacy_load):
                obj=torch_load_compat(path,map_location='cpu')
            self.assertTrue(torch.equal(obj['x'],torch.tensor([1,2,3])))
            self.assertEqual(len(calls),2)
            self.assertIn('weights_only',calls[0])
            self.assertNotIn('weights_only',calls[1])

    def test_mixed_configuration_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            freeze_config(Path(d),{'radius':1})
            freeze_config(Path(d),{'radius':1})
            with self.assertRaises(Exception): freeze_config(Path(d),{'radius':2})
if __name__=='__main__': unittest.main()
