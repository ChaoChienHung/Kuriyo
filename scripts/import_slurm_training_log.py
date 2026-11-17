"""Extract observed TIGER/SID measurements without inventing missing history."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re


NUMBER = r'[-+\d.eE]+'
STEP = re.compile(rf'^step (\d+)/(\d+) loss=({NUMBER}) lr=({NUMBER})$')
EPOCH = re.compile(rf'^epoch (\d+)/(\d+) loss=({NUMBER})$')


def parse_log(source):
    raw = source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    provenance = dict(origin='slurm_import', source_file=source.name,
                      source_sha256=digest, session_id='slurm-import-' + digest[:16],
                      timestamp=None, elapsed_seconds=None)
    records = {stage: {'train': [], 'events': []} for stage in ('tiger', 'sid')}
    metadata = {}
    for line_number, line in enumerate(raw.decode('utf-8').split('\n'), 1):
        line = line.strip()
        base = dict(provenance, source_line=line_number)
        if line.startswith('job='):
            metadata = dict(part.split('=', 1) for part in line.split() if '=' in part)
            for stage in records:
                records[stage]['events'].append(dict(base, event='source_job', metadata=metadata))
        match = STEP.fullmatch(line)
        if match:
            step, total, loss, lr = match.groups()
            records['tiger']['train'].append(dict(base, step=int(step), total_steps=int(total),
                                                  loss=float(loss), learning_rate=float(lr)))
            continue
        match = EPOCH.fullmatch(line)
        if match:
            epoch, total, loss = match.groups()
            records['sid']['train'].append(dict(base, epoch=int(epoch), total_epochs=int(total),
                                                loss=float(loss), learning_rate=None))
            continue
        if line.startswith('T5 parameters='):
            records['tiger']['events'].append(dict(base, event='training_started', message=line))
        elif line.startswith('Saved '):
            records['tiger']['events'].append(dict(base, event='training_saved', reported_path=line[6:]))
        elif line.startswith('{'):
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                if 'validation_examples' in value and 'ndcg@10' in value:
                    records['tiger']['events'].append(dict(base, event='validation', metrics=value))
                elif 'test_examples' in value and 'ndcg@10' in value:
                    records['tiger']['events'].append(dict(base, event='test_completed', metrics=value))
                elif 'codebook_usage' in value:
                    records['sid']['events'].append(dict(base, event='sid_summary', metrics=value))
                elif 'counts' in value and 'sid_version' in value:
                    records['tiger']['events'].append(dict(base, event='materialized_splits', metrics=value))
    return records, dict(provenance, job=metadata)


def import_log(source, output):
    records, provenance = parse_log(source)
    if not any(stage['train'] for stage in records.values()):
        raise ValueError('No recognized training measurements in source')
    # A fresh archive avoids mixing inferred records with native logger output.
    output.mkdir(parents=True, exist_ok=False)
    counts = {}
    for stage, streams in records.items():
        if not streams['train']:
            continue
        directory = output / stage
        directory.mkdir()
        for stream, rows in streams.items():
            name = 'train.jsonl' if stream == 'train' else 'events.log'
            (directory / name).write_text(''.join(json.dumps(row, allow_nan=False) + '\n' for row in rows))
        counts[stage] = {key: len(rows) for key, rows in streams.items()}
    manifest = dict(provenance, imported_at=datetime.now(timezone.utc).isoformat(), counts=counts,
                    limitations=['Only measurements printed in the source are available.',
                                 'Original timestamps and elapsed times are unknown (null).',
                                 'SID learning rate is not printed (null).',
                                 'No checkpoint, job success, or missing event is inferred.'])
    (output / 'import_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(import_log(args.source, args.output_dir), indent=2))


if __name__ == '__main__':
    main()
