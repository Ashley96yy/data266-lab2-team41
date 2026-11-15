"""Validate and aggregate supplied Part 1 content reviews without model inference."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from statistics import mean
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from p1_content_review_forms import contract, digest, read, validate_submission
from src.p1_evaluation import load_specification, numeric_targets_match, summarize


def aggregate(results, spec_path, submitted, correction_log, groundedness):
    expected = contract(results, spec_path)
    validate_submission(submitted, expected)
    effective = deepcopy(submitted)
    if correction_log['source_review_sha256'] != correction_log['verified_source_review_sha256']:
        raise ValueError('Correction authority refers to another original file.')
    applied = set()
    for correction in correction_log['corrections']:
        key = (correction['mode'], correction['question_id'], correction['field'])
        if key in applied:
            raise ValueError('Duplicate correction.')
        applied.add(key)
        if correction['field'] != 'answer_correct' or type(correction['new_value']) is not bool:
            raise ValueError('Only explicitly confirmed correctness corrections are supported.')
        if not correction.get('user_confirmation') or not correction.get('reason'):
            raise ValueError('A correction requires documented user authority and reason.')
        rows = effective['modes'][correction['mode']]['records']
        row = next(r for r in rows if r['question_id'] == correction['question_id'])
        if row[correction['field']] is not correction['old_value']:
            raise ValueError('Original label does not match the correction log.')
        row[correction['field']] = correction['new_value']
        row['content_review_note'] = (row['content_review_note'] + '\n' + correction['reason']).strip()
    spec = load_specification(spec_path)
    if groundedness['source_run_sha256'] != expected['modes']['final']['run_sha256']:
        raise ValueError('Groundedness belongs to another final run.')
    if [r['question_id'] for r in groundedness['records']] != [c['question_id'] for c in spec['cases']]:
        raise ValueError('Groundedness question identity mismatch.')
    names = [groundedness[f'reviewer_{x}_name'] for x in ('a', 'b')]
    if not all(names) or names[0].strip().casefold() == names[1].strip().casefold():
        raise ValueError('Groundedness needs two named distinct members.')
    ratings = [(r['reviewer_a_groundedness'], r['reviewer_b_groundedness']) for r in groundedness['records']]
    if any(type(v) is not int or not 1 <= v <= 5 for pair in ratings for v in pair):
        raise ValueError('Invalid groundedness score.')
    predictions = read(Path(results) / 'golden_predictions.json')['records']
    metrics, audit = [], []
    for mode in ('dense_only', 'final'):
        rows = effective['modes'][mode]['records']
        saved = [r for r in predictions if r['mode'] == mode]
        # Preserve recorded ranking/timing in a disclosed projection, without
        # claiming to restore a byte-identical original full inference record.
        projection = dict(mode=mode, execution_complete=True, records=[dict(
            question=case, run_status='completed', turn={'answer': record['prediction']},
            top_five=record['top_five'], elapsed_seconds=record['elapsed_seconds'])
            for case, record in zip(spec['cases'], saved)])
        m = summarize(projection, spec)
        correct = supported = citations = 0
        for row in rows:
            status_matches = row['candidate']['status'] == row['expected_status']
            # Human source-content review includes numerical values, units and
            # sign/direction. The older token-presence heuristic is diagnostic:
            # e.g. a target 1.6 for a decrease can be correctly expressed as -1.6.
            token_match = numeric_targets_match(row['candidate']['answer'], row['numeric_targets'])
            counted = row['answer_correct'] and status_matches
            correct += counted
            supported += sum(row['citation_support'].values())
            citations += len(row['citation_support'])
            audit.append(dict(mode=mode, question_id=row['question_id'],
                human_answer_correct=row['answer_correct'], status_matches=status_matches,
                counted_correct=counted, legacy_numeric_token_check=token_match,
                numeric_token_discrepancy_requires_context=row['answer_correct'] and not token_match,
                supported_citations=sum(row['citation_support'].values()),
                emitted_citations=len(row['citation_support'])))
        m.update(answer_correct_out_of_25=correct, answer_correctness=correct / 25,
            citation_accuracy=supported / citations if citations else None,
            supported_citations=supported, citation_denominator=citations,
            content_review_complete=True, citation_review_complete=True,
            content_reviewer=submitted['content_reviewer'],
            content_review_method=submitted['content_review_method'],
            numeric_validation='Supplied human source review against frozen numerical targets, units and direction; token-only heuristic retained separately as a diagnostic.')
        if mode == 'final':
            m.update(groundedness_mean=mean(v for pair in ratings for v in pair),
                groundedness_agree_within_one=mean(abs(a-b) <= 1 for a,b in ratings),
                groundedness_exact_agreement=mean(a == b for a,b in ratings),
                two_member_review_complete=True)
        metrics.append(m)
    return dict(schema_version=1, source_round=2, source_gpu='Tesla T4',
        inference_rerun=False, content_reviewer=submitted['content_reviewer'],
        groundedness_reviewers=names, metrics=metrics, records=audit,
        numerical_review_boundary='Numeric tolerance and frozen references are unchanged. Content labels judge full numerical meaning; legacy token matching is not the official score when sign/direction is expressed differently.',
        review_process='Submitted files and user confirmation declare human judgments; import verifies identity/structure, not independence or review behavior.'), effective


def finalize(review_path, corrections_path, results, spec_path, groundedness_path, output):
    corrections = read(corrections_path)
    corrections['verified_source_review_sha256'] = digest(review_path)
    outcome, effective = aggregate(results, spec_path, read(review_path), corrections, read(groundedness_path))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    # Original bytes stay intact; effective labels are a separate derived file.
    (output / 'part1-content-review.original.json').write_bytes(Path(review_path).read_bytes())
    (output / 'corrections.json').write_bytes(Path(corrections_path).read_bytes())
    def write(name, value):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    write('effective_review.json', effective)
    write('metrics.json', outcome)
    write('provenance.json', dict(original_review_sha256=digest(review_path),
        corrections_sha256=digest(corrections_path), groundedness_sha256=digest(groundedness_path),
        prediction_sha256=digest(Path(results) / 'golden_predictions.json'),
        source_summary_sha256=digest(Path(results) / 'summary.json'),
        specification_sha256=digest(spec_path),
        original_scores_modified=False, frozen_references_modified=False,
        no_new_inference=True, source_projection='Saved compact predictions, rankings and timing; not restored original full run bytes.'))
    print(json.dumps(outcome['metrics'], indent=2))
    return outcome


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--review', type=Path, required=True)
    p.add_argument('--corrections', type=Path, required=True)
    p.add_argument('--results', type=Path, default=ROOT / 'eval/yuyao_ding/part1/results/round2_colab_20261114T032206Z')
    p.add_argument('--spec', type=Path, default=ROOT / 'eval/yuyao_ding/p1_golden_v2.json')
    p.add_argument('--groundedness', type=Path, default=ROOT / 'eval/yuyao_ding/part1/results/round2_colab_20261114T032206Z/human_review_20261114/ratings_and_metrics.json')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    finalize(a.review, a.corrections, a.results, a.spec, a.groundedness, a.output)
