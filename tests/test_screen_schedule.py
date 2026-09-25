import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ScreenScheduleTests(unittest.TestCase):
    def _run_sr25(self, td: Path, orders: str, name: str):
        manifest = td / 'manifest.json'
        if not manifest.exists():
            subprocess.check_call([
                sys.executable, str(ROOT / 'scripts/setup_data.py'),
                '--data-dir', str(td / 'data'),
                '--manifest', str(manifest),
                '--only', 'SR25', '--no-download',
            ], cwd=ROOT)
        out = td / name
        proc = subprocess.run([
            sys.executable, '-u', str(ROOT / 'scripts/run_suite.py'),
            '--manifest', str(manifest), '--names', 'SR25', '--stage', 'screen',
            '--orders', orders, '--radius', '1', '--workers', '1',
            '--exact-limit', '100000000', '--out', str(out),
        ], cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)
        summary = out / 'SR25' / 'screen_R1' / 'order_summary.csv'
        with summary.open() as fh:
            rows = list(csv.DictReader(fh))
        k6 = [r for r in rows if r['order'] == '6']
        return proc.stdout, k6

    def test_skipped_intermediate_order_is_applied_and_equivalent(self):
        with tempfile.TemporaryDirectory() as tmp:
            td = Path(tmp)
            full_log, full_k6 = self._run_sr25(td, '1,2,3,4,5,6', 'full')
            skip_log, skip_k6 = self._run_sr25(td, '1,2,3,4,6', 'skip')
            self.assertEqual(full_k6, skip_k6)
            self.assertIn('order 5: START', skip_log)
            self.assertIn('order 5: intermediate cumulative block applied for pruning; summary not requested', skip_log)


if __name__ == '__main__':
    unittest.main()
