import itertools
import tempfile
from pathlib import Path
import unittest
import networkx as nx
import numpy as np
import torch
from ksgl.states import FeatureConfig, extract_order, exact_histogram, rank_gf2, cached_order, feature_matrix, balls
from ksgl.model import KSGL, pack, input_dim, expand_adam_state
from ksgl.screen import pair_stats
from ksgl.data import srg_parameters, BREC
from ksgl.runtime import seed_all, rng_state, restore_rng, atomic_torch_save

def adj(G): return nx.to_numpy_array(G,dtype=np.uint8)
def srg16():
    vertices=list(itertools.product(range(4),repeat=2)); rook=nx.Graph(); sh=nx.Graph()
    rook.add_nodes_from(vertices); sh.add_nodes_from(vertices)
    for u,v in itertools.combinations(vertices,2):
        if u[0]==v[0] or u[1]==v[1]: rook.add_edge(u,v)
    for a,b in vertices:
        for dx,dy in [(1,0),(3,0),(0,1),(0,3),(1,1),(3,3)]: sh.add_edge((a,b),((a+dx)%4,(b+dy)%4))
    return adj(rook),adj(sh)

class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)
    def test_rank_examples(self):
        for G,nu in [(nx.cycle_graph(4),2),(nx.complete_graph(4),0),(nx.star_graph(3),2),(nx.path_graph(4),0)]:
            self.assertEqual(len(G)-rank_gf2(adj(G)),nu)
    def test_closed_forms(self):
        for G in nx.graph_atlas_g():
            if not 1<=len(G)<=5: continue
            A=adj(G)
            for R in [0,1,2]:
                B=balls(A,R)
                for s in [1,2,3]:
                    b=extract_order(A,s,FeatureConfig(radius=R))
                    np.testing.assert_array_equal(b['g'],exact_histogram(A,range(len(A)),s))
                    for u in range(len(A)):
                        c=np.flatnonzero(B[u]); c=c[c!=u]
                        np.testing.assert_array_equal(b['l'][u],exact_histogram(A,c,s,root=u))
    def test_order20_exact_fallback(self):
        # Exercises the exact GF(2) elimination path beyond the s<=6 LUT.
        A=adj(nx.cycle_graph(20))
        b=extract_order(A,20,FeatureConfig(radius=15,exact_limit=100))
        self.assertTrue(bool(b['g_exact']))
        self.assertEqual(int(b['g'].sum()),1)
        self.assertTrue(b['l_exact'].all())
        self.assertEqual(int(b['l'].sum()),20)

    def test_four_vertex_witness(self):
        for mask in range(64):
            A=np.zeros((4,4),dtype=np.uint8)
            for bit,(u,v) in enumerate(itertools.combinations(range(4),2)): A[u,v]=A[v,u]=(mask>>bit)&1
            b=extract_order(A,4)
            pm=(int(A[0,1])*int(A[2,3])+int(A[0,2])*int(A[1,3])+int(A[0,3])*int(A[1,2]))%2
            self.assertEqual(int(b['g'][0]),pm)
            self.assertEqual(int(b['g'][4]),int(not A.any()))
    def test_srg_low_collapse(self):
        A,B=srg16()
        self.assertEqual(srg_parameters(A),(16,6,2,2)); self.assertEqual(srg_parameters(B),(16,6,2,2))
        for R in [1,2,3]:
            for s in [1,2,3]:
                a=extract_order(A,s,FeatureConfig(radius=R)); b=extract_order(B,s,FeatureConfig(radius=R))
                np.testing.assert_array_equal(a['g'],b['g']); np.testing.assert_array_equal(a['l'],b['l'])
    def test_equivariance(self):
        A=adj(nx.gnp_random_graph(8,.4,seed=4)); p=np.random.default_rng(0).permutation(8)
        for s in [1,2,3,4,6]:
            a=extract_order(A,s); b=extract_order(A[np.ix_(p,p)],s)
            np.testing.assert_array_equal(a['g'],b['g']); np.testing.assert_array_equal(a['l'][p],b['l'])
    def test_root_sum(self):
        A=adj(nx.path_graph(7))
        for s in [1,2,3,4,6]:
            b=extract_order(A,s,FeatureConfig(radius=10))
            np.testing.assert_array_equal(b['l'].sum(0),s*b['g'])
    def test_cache(self):
        A=adj(nx.cycle_graph(6))
        with tempfile.TemporaryDirectory() as d:
            a=cached_order(A,4,FeatureConfig(),d); b=cached_order(A,4,FeatureConfig(),d)
            np.testing.assert_array_equal(a['g'],b['g'])
            self.assertEqual(len(list(Path(d).rglob('*.npz'))),1)
    def test_unknown_and_mc(self):
        A=adj(nx.path_graph(9))
        b=extract_order(A,4,FeatureConfig(exact_limit=1))
        self.assertFalse(bool(b['g_exact'])); self.assertTrue(np.isnan(b['g_hat']).all()); self.assertTrue((b['g']==-1).all())
        c=extract_order(A,4,FeatureConfig(mode='hybrid',exact_limit=1,samples=100))
        self.assertFalse(bool(c['g_exact'])); self.assertTrue(np.isfinite(c['g_hat']).all())
    def test_neural_equivariance(self):
        seed_all(7); A=adj(nx.gnp_random_graph(9,.4,seed=5)); b={s:extract_order(A,s) for s in range(1,5)}
        x=feature_matrix(b,4); p=np.random.default_rng(1).permutation(9); model=KSGL(4)
        with torch.no_grad():
            z=model(*pack([A,A[np.ix_(p,p)]],[x,x[p]]))
        torch.testing.assert_close(z[0],z[1],rtol=1e-5,atol=1e-5)
    def test_expansion_preserves_function(self):
        seed_all(1); A=adj(nx.cycle_graph(6)); model=KSGL(2)
        x=np.random.default_rng(1).random((6,input_dim(2))).astype(np.float32)
        with torch.no_grad(): z=model(*pack([A],[x]))
        model.expand_order(6); xx=np.column_stack([x,np.ones((6,input_dim(6)-input_dim(2)),dtype=np.float32)])
        with torch.no_grad(): zz=model(*pack([A],[xx]))
        torch.testing.assert_close(z,zz,rtol=0,atol=0)
    def test_optimizer_expansion(self):
        m=KSGL(2); o=torch.optim.Adam(m.parameters()); A=adj(nx.cycle_graph(4)); x=np.ones((4,input_dim(2)),dtype=np.float32)
        m(*pack([A],[x])).sum().backward(); o.step(); state=o.state_dict()
        m.expand_order(4); o2=torch.optim.Adam(m.parameters()); o2.load_state_dict(state); expand_adam_state(o2)
        self.assertEqual(o2.state[m.input[0].weight]['exp_avg'].shape,m.input[0].weight.shape)
        self.assertTrue((o2.state[m.input[0].weight]['exp_avg'][:,input_dim(2):]==0).all())
    def test_rng_checkpoint(self):
        seed_all(8); state=rng_state(); a=torch.randn(10); b=np.random.rand(10)
        restore_rng(state); torch.testing.assert_close(a,torch.randn(10)); np.testing.assert_array_equal(b,np.random.rand(10))
    def test_brec_layout(self):
        a=nx.to_graph6_bytes(nx.cycle_graph(6),header=False).strip(); b=nx.to_graph6_bytes(nx.complete_graph(6),header=False).strip()
        arr=np.array([a,b]*12800+[a,a]*12800)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'b.npy'; np.save(p,arr); ds=BREC(p); tr,rr=ds.block(1)
            self.assertEqual(len(tr),64); self.assertEqual(len(rr),64); np.testing.assert_array_equal(rr[0],rr[1]); self.assertFalse(np.array_equal(tr[0],tr[1]))
    def test_exact_gnn_low_collision(self):
        A,B=srg16(); ba={s:extract_order(A,s) for s in range(1,4)}; bb={s:extract_order(B,s) for s in range(1,4)}
        st=pair_stats(A,B,ba,bb,3); self.assertEqual(st['ks_refinement'],'equal')

if __name__=='__main__': unittest.main()
