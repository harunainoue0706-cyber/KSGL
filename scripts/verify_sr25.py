#!/usr/bin/env python3
"""Verify a new exact SR25 screen against the recorded reference (not used by extraction)."""
import argparse
import csv
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('summary',type=Path)
a=p.parse_args()
rows=list(csv.DictReader(a.summary.open()))
expected={1:0,2:0,3:0,4:94,6:105}
for order,value in expected.items():
    got=[r for r in rows if int(r['order'])==order and r['view']=='global']
    assert len(got)==1, f'Missing global order {order}'
    r=got[0]
    assert int(r['total'])==105 and int(r['unknown'])==0
    assert int(r['separated'])==value,(order,r,value)
print('SR25 exact GLOBAL reference verified. No assertion is imposed on trained neural scores.')
