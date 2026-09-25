#!/usr/bin/env python3
"""Portable KSGL benchmark bootstrap.

Downloads/generates the public SR/BREC benchmark files into this repository,
validates them against configs/benchmark_registry.json, and writes a relocatable
configs/datasets.local.json.  No development-machine paths are used.

The script is intentionally separate from locate_data.py:
  * setup_data.py may download/convert public data.
  * locate_data.py only discovers data that already exists on disk.
"""
from __future__ import annotations

import argparse
import bz2
import gzip
import io
import json
import os
import shutil
import sys
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ksgl.data import decode, srg_parameters

REGISTRY = json.loads((ROOT / 'configs/benchmark_registry.json').read_text())


def _repo_path(path: Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _open_graph6(path: Path):
    return gzip.open(path, 'rb') if path.suffix == '.gz' else path.open('rb')


def inspect_graph6(path: Path, spec: dict) -> tuple[bool, dict]:
    count = 0
    first = None
    try:
        with _open_graph6(path) as f:
            for line in f:
                raw = line.strip().replace(b'>>graph6<<', b'')
                if not raw:
                    continue
                count += 1
                if first is None:
                    first = srg_parameters(decode(raw))
    except Exception as exc:
        return False, {'reason': f'cannot read graph6: {exc}'}
    want = tuple(spec['srg'])
    ok = count == int(spec['graphs']) and tuple(first or ()) == want
    return ok, {'graphs': count, 'expected_graphs': spec['graphs'], 'srg': first, 'expected_srg': list(want)}


def inspect_brec(path: Path, spec: dict) -> tuple[bool, dict]:
    try:
        n = int(np.load(path, allow_pickle=True, mmap_mode=None).reshape(-1).shape[0])
    except Exception as exc:
        return False, {'reason': f'cannot load npy: {exc}'}
    expected = int(spec.get('entries', 51200))
    return n in {expected, 800}, {'entries': n, 'expected_entries': expected}


def inspect_dataset(path: Path, spec: dict) -> tuple[bool, dict]:
    return inspect_brec(path, spec) if spec['kind'] == 'brec' else inspect_graph6(path, spec)


def download(url: str, destination: Path, retries: int = 4) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size:
        return
    headers = {'User-Agent': 'KSGL-portable-data-bootstrap/1.2.3'}
    last = None
    for attempt in range(1, retries + 1):
        fd, tmp_name = tempfile.mkstemp(prefix=destination.name + '.', suffix='.part', dir=destination.parent)
        os.close(fd)
        tmp = Path(tmp_name)
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as src, tmp.open('wb') as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
            if not tmp.stat().st_size:
                raise RuntimeError('empty download')
            os.replace(tmp, destination)
            return
        except Exception as exc:
            last = exc
            try:
                tmp.unlink()
            except FileNotFoundError:
                pass
            if attempt < retries:
                time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f'failed to download {url}: {last}')


def generate_sr16(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        return

    def rook4():
        vertices = [(a, b) for a in range(4) for b in range(4)]
        G = nx.Graph()
        G.add_nodes_from(vertices)
        for i, u in enumerate(vertices):
            for v in vertices[i + 1:]:
                if u[0] == v[0] or u[1] == v[1]:
                    G.add_edge(u, v)
        return nx.convert_node_labels_to_integers(G, ordering='sorted')

    def srg16_alt():
        vertices = [(a, b) for a in range(4) for b in range(4)]
        steps = {(1, 0), (3, 0), (0, 1), (0, 3), (1, 1), (3, 3)}
        G = nx.Graph()
        G.add_nodes_from(vertices)
        for a, b in vertices:
            for da, db in steps:
                G.add_edge((a, b), ((a + da) % 4, (b + db) % 4))
        return nx.convert_node_labels_to_integers(G, ordering='sorted')

    with destination.open('wb') as f:
        for G in (rook4(), srg16_alt()):
            f.write(nx.to_graph6_bytes(G, header=False))


def _matrix_lines(payload: bytes, n: int) -> list[bytes]:
    rows = []
    for line in payload.splitlines():
        s = line.strip()
        if len(s) == n and set(s) <= {48, 49}:  # ASCII 0/1 only
            rows.append(s)
    return rows


def convert_srg_matrix(payload: bytes, destination: Path, spec: dict) -> None:
    n = int(spec['srg'][0])
    rows = _matrix_lines(payload, n)
    expected_rows = int(spec['graphs']) * n
    if len(rows) != expected_rows:
        raise RuntimeError(
            f'{destination.name}: expected {expected_rows} binary matrix rows '
            f'({spec["graphs"]} graphs x {n}), found {len(rows)}'
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=destination.name + '.', suffix='.part', dir=destination.parent)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        with tmp.open('wb') as out:
            for offset in range(0, len(rows), n):
                block = rows[offset:offset + n]
                A = np.array([[c == 49 for c in row] for row in block], dtype=np.uint8)
                if A.shape != (n, n) or np.any(np.diag(A)) or not np.array_equal(A, A.T):
                    raise RuntimeError(f'invalid adjacency matrix at graph {offset // n}')
                got = srg_parameters(A)
                want = tuple(spec['srg'])
                if got != want:
                    raise RuntimeError(f'SRG mismatch at graph {offset // n}: got {got}, expected {want}')
                G = nx.from_numpy_array(A)
                out.write(nx.to_graph6_bytes(G, header=False))
        os.replace(tmp, destination)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def fetch_source(key: str, spec: dict, source: dict, data_dir: Path, force: bool) -> Path:
    typ = source['type']
    aliases = spec.get('filenames', [])
    default_name = next((x for x in aliases if x.endswith('.g6')), None) or next((x for x in aliases if x.endswith('.g6.gz')), None)

    if typ == 'packaged':
        p = ROOT / source['path']
        if not p.is_file():
            raise FileNotFoundError(p)
        return p

    if typ == 'generate':
        destination = data_dir / (default_name or f'{key}.g6')
        if force:
            destination.unlink(missing_ok=True)
        if source.get('generator') == 'srg16_pair':
            generate_sr16(destination)
            return destination
        raise ValueError(f'unknown generator: {source}')

    if typ in {'graph6', 'graph6_gz'}:
        suffix = '.g6.gz' if typ == 'graph6_gz' else '.g6'
        name = next((x for x in aliases if x.endswith(suffix)), None) or f'{key}{suffix}'
        destination = data_dir / name
        if destination.is_file() and not force:
            ok, _ = inspect_graph6(destination, spec)
            if ok:
                return destination
            destination.unlink()
        elif force:
            destination.unlink(missing_ok=True)
        print(f'  FETCH {key}: {source["url"]}', flush=True)
        download(source['url'], destination)
        return destination

    if typ in {'srg_matrix', 'srg_matrix_bz2'}:
        destination = data_dir / (default_name or f'{key}.g6')
        if destination.is_file() and not force:
            ok, _ = inspect_graph6(destination, spec)
            if ok:
                return destination
            destination.unlink()
        raw_name = f'.source_{key}.txt' + ('.bz2' if typ.endswith('_bz2') else '')
        raw = data_dir / raw_name
        if force:
            raw.unlink(missing_ok=True)
        print(f'  FETCH {key}: {source["url"]}', flush=True)
        download(source['url'], raw)
        payload = raw.read_bytes()
        if typ.endswith('_bz2'):
            payload = bz2.decompress(payload)
        convert_srg_matrix(payload, destination, spec)
        return destination

    if typ == 'brec_zip':
        destination = data_dir / 'brec_v3.npy'
        if destination.is_file() and not force:
            ok, _ = inspect_brec(destination, spec)
            if ok:
                return destination
            destination.unlink()
        archive = data_dir / '.BREC_data_all.zip'
        if force:
            archive.unlink(missing_ok=True)
        print(f'  FETCH {key}: {source["url"]}', flush=True)
        download(source['url'], archive)
        with zipfile.ZipFile(archive) as zf:
            member = source.get('member', 'brec_v3.npy')
            matches = [name for name in zf.namelist() if Path(name).name == member]
            if not matches:
                raise RuntimeError(f'{member} not found in {archive}')
            with zf.open(matches[0]) as src, destination.open('wb') as dst:
                shutil.copyfileobj(src, dst)
        return destination

    raise ValueError(f'unsupported source type {typ!r} for {key}')


def candidate_existing(key: str, spec: dict, data_dir: Path) -> list[Path]:
    out = []
    if key == 'SR25':
        out.append(ROOT / 'validation/sr251256.g6')
    for name in spec.get('filenames', []):
        p = data_dir / name
        if p.is_file():
            out.append(p)
    if spec['kind'] == 'brec':
        p = data_dir / 'brec_v3.npy'
        if p.is_file():
            out.append(p)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description='Download/generate and register portable KSGL benchmark data.')
    ap.add_argument('--data-dir', default=str(ROOT / 'data/benchmarks'))
    ap.add_argument('--manifest', default=str(ROOT / 'configs/datasets.local.json'))
    ap.add_argument('--only', default='', help='comma-separated dataset keys; default: full paper table')
    ap.add_argument('--include-direct-227', action='store_true')
    ap.add_argument('--no-download', action='store_true', help='use only packaged/already existing files')
    ap.add_argument('--skip-brec', action='store_true')
    ap.add_argument('--force', action='store_true', help='redownload/regenerate selected raw files')
    args = ap.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    selected = [x for x in args.only.split(',') if x] if args.only else [
        'SR16', 'SR25', 'SR26', 'SR28', 'SR29', 'SR40',
        'SRG35_16', 'SRG35_18', 'SRG36_14', 'SRG36_15', 'SRG37_18',
        'SRG45_12', 'SRG50_21', 'SRG64_18', 'SRG65_32', 'BREC'
    ]
    if args.include_direct_227 and 'SRG35_18_DIRECT227' not in selected:
        selected.append('SRG35_18_DIRECT227')
    if args.skip_brec:
        selected = [x for x in selected if x != 'BREC']

    unknown = [x for x in selected if x not in REGISTRY]
    if unknown:
        ap.error('unknown dataset keys: ' + ', '.join(unknown))

    manifest = {}
    failures = []
    print('=' * 96)
    print('KSGL PORTABLE DATA SETUP')
    print('repository :', ROOT)
    print('data dir   :', data_dir)
    print('=' * 96)

    for key in selected:
        spec = REGISTRY[key]
        entry = {k: v for k, v in spec.items() if k not in {'filenames', 'sources'}}
        entry['path'] = ''

        if 'derive' in spec:
            manifest[key] = entry
            print(f'{key:22s} DERIVED {spec["derive"]}')
            continue

        chosen = None
        meta = None
        for p in candidate_existing(key, spec, data_dir):
            ok, current_meta = inspect_dataset(p, spec)
            if ok:
                chosen, meta = p, current_meta
                break

        if chosen is None:
            errors = []
            for source in spec.get('sources', []):
                if args.no_download and source.get('type') not in {'packaged', 'generate'}:
                    continue
                try:
                    p = fetch_source(key, spec, source, data_dir, args.force)
                    ok, current_meta = inspect_dataset(p, spec)
                    if not ok:
                        raise RuntimeError(f'validation failed: {current_meta}')
                    chosen, meta = p, current_meta
                    break
                except Exception as exc:
                    errors.append(f'{source.get("url", source.get("type"))}: {exc}')
            if chosen is None:
                if args.no_download and not errors:
                    errors = ['not present locally and --no-download was used']
                failures.append((key, errors))

        if chosen is not None:
            entry['path'] = _repo_path(chosen)
            entry['source'] = 'portable-setup'
            entry['validation'] = meta
            print(f'{key:22s} OK  {entry["path"]}')
        else:
            print(f'{key:22s} FAILED')
        manifest[key] = entry

    # Keep derived source metadata coherent.
    for key, entry in manifest.items():
        d = entry.get('derive')
        if d:
            src = manifest.get(d['source'], {})
            entry['source_path'] = src.get('path', '')
            if not entry['source_path']:
                failures.append((key, [f'derived source {d["source"]} is unresolved']))

    out = Path(args.manifest).expanduser()
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2) + '\n')
    print('-' * 96)
    print('Manifest:', out)

    if failures:
        print('\nSome datasets are unresolved:')
        for key, errors in failures:
            print('  ' + key)
            for e in errors:
                print('    - ' + e)
        print('\nThe manifest was written with unresolved paths left empty.')
        raise SystemExit(2)

    print('All selected datasets were validated and registered.')
    print('Next:')
    print('  python -u scripts/run_suite.py --manifest configs/datasets.local.json --names paper-table --stage screen --orders 1,2,3,4,5,6 --radius 1 --workers 12 --exact-limit 100000000 --out results/SR_ORDER_SWEEP')


if __name__ == '__main__':
    main()
