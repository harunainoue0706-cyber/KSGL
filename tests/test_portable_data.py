import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class PortableDataTests(unittest.TestCase):
    def test_setup_offline_sr16_sr25(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            data = td / 'data'
            manifest = td / 'datasets.local.json'
            cmd = [
                sys.executable, str(ROOT / 'scripts/setup_data.py'),
                '--data-dir', str(data),
                '--manifest', str(manifest),
                '--only', 'SR16,SR25',
                '--no-download',
            ]
            subprocess.check_call(cmd, cwd=ROOT)
            cfg = json.loads(manifest.read_text())
            self.assertTrue(cfg['SR16']['path'])
            self.assertTrue(cfg['SR25']['path'])

    def test_locator_finds_renamed_graph6_by_content(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            renamed = td / 'not_the_registry_name.g6'
            shutil.copy2(ROOT / 'validation/sr251256.g6', renamed)
            manifest = td / 'located.json'
            cmd = [
                sys.executable, str(ROOT / 'scripts/locate_data.py'),
                '--root', str(td),
                '--out', str(manifest),
            ]
            subprocess.check_call(cmd, cwd=ROOT)
            cfg = json.loads(manifest.read_text())
            self.assertEqual(Path(cfg['SR25']['path']).resolve(), renamed.resolve())

    def test_srg_matrix_converter_roundtrip(self):
        import importlib.util
        import networkx as nx
        import numpy as np

        spec = importlib.util.spec_from_file_location("ksgl_setup_data", ROOT / "scripts/setup_data.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        def rook4():
            V=[(a,b) for a in range(4) for b in range(4)]
            G=nx.Graph(); G.add_nodes_from(V)
            for i,u in enumerate(V):
                for v in V[i+1:]:
                    if u[0]==v[0] or u[1]==v[1]: G.add_edge(u,v)
            return nx.convert_node_labels_to_integers(G, ordering="sorted")

        def srg16_alt():
            V=[(a,b) for a in range(4) for b in range(4)]
            S={(1,0),(3,0),(0,1),(0,3),(1,1),(3,3)}
            G=nx.Graph(); G.add_nodes_from(V)
            for a,b in V:
                for da,db in S:
                    G.add_edge((a,b),((a+da)%4,(b+db)%4))
            return nx.convert_node_labels_to_integers(G, ordering="sorted")

        rows=[b"dim = 16, degree = 6, lambda = 2, mu = 2"]
        for G in (rook4(), srg16_alt()):
            A=nx.to_numpy_array(G,dtype=np.uint8)
            rows.extend("".join(str(int(x)) for x in row).encode("ascii") for row in A)
        payload=b"\n".join(rows)+b"\n"

        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/"sr16622.g6"
            mod.convert_srg_matrix(payload,out,{"srg":[16,6,2,2],"graphs":2})
            ok,meta=mod.inspect_graph6(out,{"srg":[16,6,2,2],"graphs":2})
            self.assertTrue(ok,meta)


if __name__ == '__main__':
    unittest.main()
