#!/usr/bin/env python3
"""Locate already-existing KSGL benchmark files without machine-specific paths.

The locator first checks known basenames, then performs a content-based scan of
*.g6 / *.graph6 files.  A renamed graph6 collection can therefore be recognized
by its graph count and verified SRG parameters.  This script never downloads or
modifies benchmark data; use scripts/setup_data.py for that.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ksgl.data import decode, srg_parameters

REGISTRY = json.loads((ROOT / 'configs/benchmark_registry.json').read_text())
SKIP_DIRS = {'.git', '__pycache__', '.cache', 'site-packages', 'node_modules', '.venv', 'venv'}


def open_graph6(path: Path):
    return gzip.open(path, 'rb') if path.suffix == '.gz' else path.open('rb')


def graph6_metadata(path: Path):
    count = 0
    first = None
    try:
        with open_graph6(path) as f:
            for line in f:
                raw = line.strip().replace(b'>>graph6<<', b'')
                if not raw:
                    continue
                count += 1
                if first is None:
                    first = srg_parameters(decode(raw))
        return count, first
    except Exception:
        return None, None


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def brec_entries(path: Path):
    try:
        return int(np.load(path, allow_pickle=True, mmap_mode=None).reshape(-1).shape[0])
    except Exception:
        return None


def portable(path: Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', action='append', default=[], help='root to search; repeatable. Default: repository root')
    ap.add_argument('--out', default=str(ROOT / 'configs/datasets.local.json'))
    ap.add_argument('--include-direct-227', action='store_true')
    ap.add_argument('--no-content-scan', action='store_true', help='only match registry basenames')
    args = ap.parse_args()

    roots = [Path(x).expanduser().resolve() for x in args.root] or [ROOT.resolve()]
    for p in roots:
        if not p.is_dir():
            ap.error(f'Root is not a directory: {p}')

    wanted = {k: v for k, v in REGISTRY.items() if args.include_direct_227 or k != 'SRG35_18_DIRECT227'}
    alias_to_keys = {}
    for key, spec in wanted.items():
        for name in spec.get('filenames', []):
            alias_to_keys.setdefault(name, set()).add(key)

    files = set()
    for search in roots:
        for folder, dirs, names in os.walk(search):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            base = Path(folder)
            for name in names:
                p = (base / name).resolve()
                if name in alias_to_keys:
                    files.add(p)
                    continue
                if not args.no_content_scan:
                    low = name.lower()
                    if low.endswith(('.g6', '.graph6', '.g6.gz', '.graph6.gz')) or low == 'brec_v3.npy':
                        files.add(p)

    by_signature = {}
    brec_files = []
    for p in sorted(files):
        if p.name.lower() == 'brec_v3.npy':
            n = brec_entries(p)
            if n in {51200, 800}:
                brec_files.append(p)
            continue
        n, params = graph6_metadata(p)
        if n is not None and params is not None:
            by_signature.setdefault((n, tuple(params)), []).append(p)

    manifest = {}
    print('=' * 96)
    print('KSGL PORTABLE BENCHMARK LOCATOR')
    print('roots:')
    for p in roots:
        print('  ', p)
    print('=' * 96)

    for key, spec in wanted.items():
        entry = {k: v for k, v in spec.items() if k not in {'filenames', 'sources'}}
        entry['path'] = ''
        entry['candidates'] = []

        if 'derive' in spec:
            manifest[key] = entry
            print(f'{key:22s} DERIVED {spec["derive"]}')
            continue

        if spec['kind'] == 'brec':
            good = list(brec_files)
        else:
            sig = (int(spec['graphs']), tuple(spec['srg']))
            good = list(by_signature.get(sig, []))

        # Prefer exact registry basenames, then repository-local files, then shortest path.
        aliases = set(spec.get('filenames', []))
        good.sort(key=lambda p: (0 if p.name in aliases else 1, 0 if ROOT.resolve() in p.parents else 1, len(str(p)), str(p)))
        entry['candidates'] = [portable(p) for p in good]

        if not good:
            print(f'{key:22s} NOT FOUND')
        else:
            # If multiple byte-identical copies exist, selecting one is safe.  If they differ,
            # still choose a deterministic preferred candidate but make the ambiguity explicit.
            chosen = good[0]
            entry['path'] = portable(chosen)
            if len(good) == 1:
                print(f'{key:22s} OK  {entry["path"]}')
            else:
                hashes = []
                for p in good:
                    try:
                        hashes.append(sha256(p))
                    except Exception:
                        hashes.append(None)
                status = 'DUPLICATE COPIES' if len(set(hashes)) == 1 else 'MULTIPLE VALID COLLECTIONS'
                print(f'{key:22s} {status}; selected {entry["path"]}')
                for p in good:
                    print(' ' * 25 + portable(p))
        manifest[key] = entry

    for key, entry in manifest.items():
        d = entry.get('derive')
        if d:
            entry['source_path'] = manifest.get(d['source'], {}).get('path', '')
            if not entry['source_path']:
                print(f'WARNING: {key} needs {d["source"]}, but its source is unresolved.')

    out = Path(args.out).expanduser()
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2) + '\n')
    print('-' * 96)
    print('Manifest:', out)
    print('No data were downloaded or modified. Use scripts/setup_data.py to fetch public datasets.')


if __name__ == '__main__':
    main()
