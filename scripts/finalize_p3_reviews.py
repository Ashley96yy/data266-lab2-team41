"""Validate submitted human adjudication and aggregate saved runs; no inference."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.p3_review_forms import MODES, merge, read, sha, write
from scripts.export_p3_public_traces import PRIVATE_PATTERNS
from src.p3_evaluation import summarize


def resolution_key(item):
    return item['mode'], item['case_id'], item['kind'], item.get('index')


def finalize(run_dir, member_a, member_b, adjudication, private_output, public_output):
    run_dir, private_output, public_output = map(Path, (run_dir, private_output, public_output))
    if private_output.exists() or public_output.exists():
        raise FileExistsError('Use new output directories to preserve previous records.')
    a, b, joint = map(read, (member_a, member_b, adjudication))
    hashes = {mode: sha(run_dir / f'{mode}.json') for mode in MODES}
    names = [a['reviewer_name'], b['reviewer_name']]
    if joint.get('schema_version') != 1 or joint.get('raw_execution_hashes') != hashes:
        raise ValueError('Adjudication belongs to another run or schema.')
    if joint.get('reviewer_names') != names or joint.get('joint_human_adjudication') is not True:
        raise ValueError('Named joint human adjudication is required.')
    resolutions = joint['resolutions']
    keys = [resolution_key(r) for r in resolutions]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate adjudication decisions.')
    for r in resolutions:
        if type(r.get('value')) is not bool or not isinstance(r.get('resolution_note'), str) or not r['resolution_note'].strip():
            raise ValueError('Each resolution needs a boolean and a human-written reason.')
    # The existing importer verifies reviewer identities, cases, attempts and hashes.
    merge(run_dir, member_a, member_b, private_output)
    expected = {resolution_key(r) for r in read(private_output / 'adjudication_needed.json')}
    if set(keys) != expected:
        raise ValueError('Adjudication must resolve exactly the submitted disagreements.')
    runs = {mode: read(private_output / f'{mode}.json') for mode in MODES}
    lookup = {(mode, r['case_id']): r for mode, run in runs.items() for r in run['records']}
    for r in resolutions:
        record = lookup[r['mode'], r['case_id']]
        if r['kind'] == 'case':
            record.update(adjudicated_pass=r['value'], resolution_note=r['resolution_note'])
        elif r['kind'] == 'attempt':
            attempt = next(x for x in record['trace']['attempts'] if x['index'] == r['index'])
            attempt.update(human_accept=r['value'], human_accept_resolution_note=r['resolution_note'])
        else:
            raise ValueError('Unknown adjudication kind.')
    metrics = {mode: summarize(run) for mode, run in runs.items()}
    for mode, result in metrics.items():
        if result['critic_reviewed_attempts'] != result['critic_agreement_denominator']:
            raise ValueError(f'Incomplete critic review: {mode}')
    safe_runs = {}
    for mode, run in runs.items():
        run.update(scored=True)
        write(private_output / f'{mode}.json', run)
        safe = {k: run[k] for k in ('set_id', 'critique_enabled', 'execution_complete', 'reviewer_a_name', 'reviewer_b_name', 'scored')}
        safe.update(raw_execution_sha256=hashes[mode], records=[])
        for r in run['records']:
            record = {k: deepcopy(r[k]) for k in ('case_id', 'reviewer_a_pass', 'reviewer_b_pass', 'reviewer_a_note', 'reviewer_b_note', 'adjudicated_pass', 'resolution_note')}
            trace = r['trace']
            record['trace'] = dict(status=trace['status'], elapsed_seconds=trace['elapsed_seconds'], attempts=[])
            for attempt in trace['attempts']:
                if 'critique' not in attempt:
                    raise ValueError('Generated attempt lacks a recorded critic decision.')
                item = {k: deepcopy(attempt[k]) for k in ('index', 'reviewer_a_accept', 'reviewer_b_accept', 'human_accept', 'human_accept_resolution_note')}
                item['critique'] = {'decision': attempt['critique']['decision']}
                record['trace']['attempts'].append(item)
            safe['records'].append(record)
        assert summarize(safe) == metrics[mode]
        safe_runs[mode] = safe
    paired = []
    for x, y in zip(runs['with']['records'], runs['without']['records']):
        assert x['case_id'] == y['case_id']
        if not x['trace']['attempts']:
            continue
        image_hashes = []
        for record in (x, y):
            trace = record['trace']
            image = run_dir / 'requests' / trace['run_id'] / Path(trace['attempts'][-1]['generation']['image_path']).name
            image_hashes.append(sha(image))
        paired.append(dict(case_id=x['case_id'], with_image_sha256=image_hashes[0], without_image_sha256=image_hashes[1], identical=image_hashes[0] == image_hashes[1], with_case_pass=x['adjudicated_pass'], without_case_pass=y['adjudicated_pass']))
    summary = dict(source_commit=read(run_dir / 'manifest.json')['source_commit'], source_run=run_dir.name, gpu=read(run_dir / 'manifest.json')['gpu'], seed=read(run_dir / 'manifest.json')['seed'], inference_rerun=False, reviewers=names, human_review_complete=True, modes=metrics, resolved_case_disagreements=sum(r['kind'] == 'case' for r in resolutions), resolved_attempt_disagreements=sum(r['kind'] == 'attempt' for r in resolutions), paired_generated_images=paired)
    provenance = dict(raw_execution_hashes=hashes, review_file_hashes={Path(p).name: sha(Path(p)) for p in (member_a, member_b, adjudication)}, review_declarations='The supplied files declare independent human review and joint adjudication; importing files does not independently establish the review process.', metadata_note='The submitted B file has member_label A. Distinct reviewer_name fields identify the reviewers; original metadata and labels are retained.', privacy='Public copies contain human labels, notes, status, timing, critic decisions, and hashes; no input images, generated images, local paths, or model contexts.')
    payloads = {f'{mode}.reviewed.json': run for mode, run in safe_runs.items()}
    payloads.update({'metrics.json': summary, 'provenance.json': provenance})
    for path in (member_a, member_b, adjudication):
        payloads[Path(path).name] = read(path)
    for name, value in payloads.items():
        if any(p.search(json.dumps(value)) for p in PRIVATE_PATTERNS):
            raise ValueError(f'Private-looking content in public export: {name}')
    public_output.mkdir(parents=True)
    for name, value in payloads.items():
        write(public_output / name, value)
    # Retain byte-identical originals, including their original metadata.
    for path in (member_a, member_b, adjudication):
        shutil.copy2(path, public_output / Path(path).name)
    manifest = read(private_output / 'manifest.json')
    manifest['review_derivation'].update(adjudication_sha256=sha(Path(adjudication)), unresolved_disagreements=0)
    write(private_output / 'manifest.json', manifest)
    write(private_output / 'adjudication_needed.json', [])
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('run-dir', 'member-a', 'member-b', 'adjudication', 'private-output', 'public-output'):
        p.add_argument('--' + name, required=True, type=Path)
    print(json.dumps(finalize(**vars(p.parse_args())), indent=2))
