"""Re-encode this study directly from public source workbooks, never old results."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import openpyxl
from scipy import sparse

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'data/processed/plant_counterinference/source_tables_v1'
CACHE = ROOT / 'data/processed/plant_resampling_v1'
OUT = ROOT / 'results/plant_resampling_v1'
TRAITS = ['Germination Rate', 'Albino Plant', 'Tiller Number', 'Days to Heading', 'Seed number',
          'Seed Yield', 'Filled Grain Number', 'Panicle Weight', 'Plant Height']
NORMAL = {'Normal', 'Normal productivity'}
OTHER = {'Low', 'Very Low', 'High', 'Later', 'Short', 'High productivity', 'Low productivity',
         'Very high productivity', 'Very low productivity', 'Sterile seeds', 'Albino and Normal'}
SOURCES = {
    'rice': ('kitbase2026_supplement/Supplementary-Table-4.xlsx',
             '4e8ca39186aad4f593e5b988d64f5c2ab38cbf2e730983c52eb8013d3d58ca54'),
    'alga': ('chlamy_conditions2022_pmc_supplement/41588_2022_1052_MOESM4_ESM.xlsx',
             'ac78244c0501ac2b2b6455c3b389767d8d32ac6eb9a10fd7053eb51105d209ad'),
}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def read_rice(path):
    book = openpyxl.load_workbook(path, read_only=True, data_only=True)
    observations = defaultdict(list)
    exclusions = []
    for sheet in book:
        name = sheet.title.strip()
        if name not in TRAITS:
            continue
        rows = list(sheet.iter_rows(values_only=True))
        header_id = next(i for i, row in enumerate(rows[:22]) if 'Mutant ID' in row)
        h = rows[header_id]
        id_col, cls_col = h.index('Mutant ID'), h.index('Classification')
        for index, row in enumerate(rows[header_id + 1:], header_id + 2):
            identifier = str(row[id_col] or '').strip().upper()
            cls = str(row[cls_col] or '').strip()
            if not identifier.startswith('FN'):
                if any(value is not None for value in row):
                    exclusions.append(dict(sheet=name, row=index, reason='non_FN_record'))
                continue
            if not cls:
                observations[(identifier, name)].append(-1)
            elif cls in NORMAL:
                observations[(identifier, name)].append(0)
            elif cls in OTHER:
                observations[(identifier, name)].append(1)
            else:
                raise ValueError(f'Unregistered rice source classification: {name}/{index}/{cls}')
    book.close()
    entities = sorted({key[0] for key in observations})
    row_id = {value: index for index, value in enumerate(entities)}
    matrix = np.full((len(entities), len(TRAITS)), -1, dtype=np.int8)
    conflicts = []
    duplicates = []
    for (entity, trait), values in observations.items():
        known = {v for v in values if v >= 0}
        if len(known) == 1:
            matrix[row_id[entity], TRAITS.index(trait)] = next(iter(known))
        elif len(known) > 1:
            conflicts.append(dict(entity=entity, trait=trait, source_values=values))
        if len(values) > 1:
            duplicates.append(dict(entity=entity, trait=trait, records=len(values)))
    groups = [re.match(r'^FN-?(\d+)', v).group(1) if re.match(r'^FN-?(\d+)', v) else v for v in entities]
    info = [dict(target=t, primary=True, target_index=i, control_id='', treatment_id='') for i, t in enumerate(TRAITS)]
    return matrix, entities, groups, info, dict(exclusions=exclusions, conflicts=conflicts, duplicates=duplicates)


def read_alga(path):
    book = openpyxl.load_workbook(path, read_only=True, data_only=True)
    metadata = {}
    iterator = book['Supplementary Table 2-Screens'].iter_rows(values_only=True)
    for _ in range(9):
        next(iterator)
    for row in iterator:
        if len(row) < 11 or not isinstance(row[9], str) or not row[9].startswith('c_'):
            continue
        record = dict(treatment_category=str(row[1]), treatment_id=str(row[2]),
                      control_category=str(row[5]), control_id=str(row[6]), screen_type=str(row[10]))
        if row[9] in metadata and metadata[row[9]] != record:
            raise ValueError('Conflicting literal screen metadata')
        metadata[row[9]] = record
    iterator = book['Supplementary Table 6'].iter_rows(values_only=True)
    for _ in range(7):
        next(iterator)
    header = next(iterator)
    columns = [(i, v) for i, v in enumerate(header) if isinstance(v, str) and v.startswith('c_')]
    info = []
    for index, (_, name) in enumerate(columns):
        meta = metadata.get(name, {})
        technical = bool(meta and meta['treatment_category'] == meta['control_category'] and
                         'control' in meta['treatment_category'].lower())
        info.append(dict(target=name, target_index=index, primary=bool(meta and not technical),
                         exclusion='unmatched_metadata' if not meta else 'technical_control' if technical else '', **meta))
    values, entities, excluded = [], [], []
    for row_index, row in enumerate(iterator, 9):
        if not isinstance(row[0], str) or not row[0].startswith('Cre'):
            if any(v is not None for v in row):
                excluded.append(row_index)
            continue
        entities.append(row[0])
        current = []
        for col, _ in columns:
            value = row[col] if col < len(row) else None
            if value is None or str(value).strip().lower() in {'', 'na', 'nan', 'n/a', '#n/a'}:
                current.append(-1)
            else:
                fdr = float(value)
                if not np.isfinite(fdr) or not 0 <= fdr <= 1:
                    raise ValueError('Invalid finite source FDR')
                current.append(int(fdr <= 0.05))
        values.append(current)
    book.close()
    if len(entities) != len(set(entities)):
        raise ValueError('Duplicated algal gene ID')
    return np.asarray(values, dtype=np.int8), entities, entities, info, dict(excluded_non_gene_rows=excluded)


def prepare(out=OUT, cache=CACHE):
    if (out / 'data_manifest.json').exists():
        existing = json.loads((out / 'data_manifest.json').read_text())
        for record in existing['sources']:
            if sha(ROOT / record['source']) != record['sha256']:
                raise ValueError('Source identity changed after registration')
            if sha(cache / (record['name'] + '.npz')) != record['matrix_archive_sha256']:
                raise ValueError('Prepared study input differs from manifest')
        return existing
    cache.mkdir(parents=True, exist_ok=True)
    records = []
    for name, (relative, expected) in SOURCES.items():
        path = SOURCE / relative
        if sha(path) != expected:
            raise ValueError('Primary source SHA256 mismatch')
        matrix, entities, groups, info, audit = (read_rice if name == 'rice' else read_alga)(path)
        for row in info:
            y = matrix[:, row['target_index']]
            row.update(positive=int((y == 1).sum()), source_zero=int((y == 0).sum()), unknown=int((y < 0).sum()))
            row['eligible_global'] = row['primary'] and row['positive'] >= 20 and row['source_zero'] >= 20
            if row['primary'] and not row['eligible_global']:
                row['exclusion'] = 'fewer_than_20_observed_cells_in_a_class'
        np.savez_compressed(cache / f'{name}.npz', y=matrix, entities=np.asarray(entities), groups=np.asarray(groups))
        sparse.save_npz(cache / f'{name}_Y_positive.npz', sparse.csr_matrix(matrix == 1))
        sparse.save_npz(cache / f'{name}_M_observed.npz', sparse.csr_matrix(matrix >= 0))
        save_json(out / f'{name}_targets.json', info)
        save_json(out / f'{name}_source_audit.json', audit)
        records.append(dict(name=name, source=str(path.relative_to(ROOT)), sha256=expected,
                            matrix_archive_sha256=sha(cache / f'{name}.npz'), entities=len(entities),
                            entity_groups=len(set(groups)), released_targets=matrix.shape[1],
                            primary_targets=sum(r['primary'] for r in info),
                            eligible_targets=sum(r['eligible_global'] for r in info),
                            measured_cells=int((matrix >= 0).sum()), source_positive_cells=int((matrix == 1).sum()),
                            source_zero_cells=int((matrix == 0).sum()), unknown_cells=int((matrix < 0).sum()),
                            matrix_shape=list(matrix.shape), unknown_is_negative=False))
    manifest = dict(completed_utc=datetime.now(timezone.utc).isoformat(), sources=records,
                    new_source_encoding=True, previous_result_tables_read=False,
                    previous_models_read=False, source_code_sha256=sha(Path(__file__)))
    save_json(out / 'data_manifest.json', manifest)
    return manifest


if __name__ == '__main__':
    print(json.dumps(prepare(), indent=2))
