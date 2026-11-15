"""Prepare local answer-correctness/citation-support forms from saved predictions.

This does not create labels or run models. Source-page PNGs are private runtime
artifacts, never repository deliverables. Original groundedness files stay intact.
"""
import argparse
import hashlib
import html
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.yuyao_ding.p1_evaluation import load_specification, digest as semantic_digest

DEFAULT_RESULTS = ROOT / 'eval/yuyao_ding/part1/results/round2_colab_20261114T032206Z'
PDF_NAMES = {'DOC1': 'Document.pdf',
             'DOC2': 'Document2_Supplement_A_Outturns_2011-2013.pdf',
             'DOC3': 'Document3_Briefing_B_Labour_Prices_Policy.pdf'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def contract(results, spec_path):
    results, spec_path = Path(results), Path(spec_path)
    spec = load_specification(spec_path)
    summary = read(results / 'summary.json')
    if semantic_digest(spec) != summary['questions_specification_sha256']:
        raise ValueError('Frozen specification does not match this recorded run.')
    predictions = read(results / 'golden_predictions.json')['records']
    cases = spec['cases']
    modes = {}
    for mode in ('dense_only', 'final'):
        rows = [r for r in predictions if r['mode'] == mode]
        if [r['question_id'] for r in rows] != [c['question_id'] for c in cases]:
            raise ValueError('Expected all 25 questions once and in frozen order.')
        modes[mode] = dict(run_sha256=summary['golden'][mode]['run_sha256'], records=[])
        for row, case in zip(rows, cases):
            if str(row['turn']) != str(case['turn']) or row['conversation_id'] != case['conversation_id']:
                raise ValueError('Conversation identity mismatch.')
            answer = row['prediction']
            if len(set(answer['citations'])) != len(answer['citations']):
                raise ValueError('Duplicate candidate citations require explicit handling.')
            modes[mode]['records'].append(dict(question_id=case['question_id'],
                question=case['question'], candidate=answer,
                reference=case['reference_answer'], expected_status=case['status'],
                numeric_targets=case.get('numeric_targets', []),
                scoring_notes=case['scoring_notes'], reference_citations=case['citations'],
                conversation_id=case['conversation_id'], turn=case['turn'],
                answer_correct=None, citation_support={c: None for c in answer['citations']},
                content_review_note=''))
    return dict(schema_version=1, review_kind='answer_correctness_and_citation_support',
        predictions_sha256=digest(results / 'golden_predictions.json'),
        summary_sha256=digest(results / 'summary.json'), specification_sha256=digest(spec_path),
        run_identity_note='Projection of saved predictions; original run hashes from summary, not reconstructed original runs.',
        groundedness_policy='Existing independent groundedness ratings remain unchanged; this form does not collect or infer them.',
        content_reviewer=None, content_review_method=None, modes=modes)


def validate_submission(value, expected):
    for key in ('schema_version', 'review_kind', 'predictions_sha256', 'summary_sha256',
                'specification_sha256'):
        if value.get(key) != expected[key]:
            raise ValueError('Review belongs to different saved evidence.')
    if not isinstance(value.get('content_reviewer'), str) or not value['content_reviewer'].strip():
        raise ValueError('Name the actual content reviewer.')
    if value.get('content_review_method') != 'independent_human_source_review':
        raise ValueError('Declare the actual source-review method.')
    if set(value.get('modes', {})) != set(expected['modes']):
        raise ValueError('Both baseline and final are required.')
    for mode, wanted in expected['modes'].items():
        supplied = value['modes'][mode]
        if supplied.get('run_sha256') != wanted['run_sha256']:
            raise ValueError('Run identity mismatch.')
        rows = supplied['records']
        if [r['question_id'] for r in rows] != [r['question_id'] for r in wanted['records']]:
            raise ValueError('Missing, duplicated or reordered review rows.')
        for row, original in zip(rows, wanted['records']):
            # Check displayed questions/answers/references as well as run identity.
            for key, item in original.items():
                if key not in ('answer_correct', 'citation_support', 'content_review_note') and row.get(key) != item:
                    raise ValueError('Displayed evidence was modified.')
            if type(row.get('answer_correct')) is not bool:
                raise ValueError('Answer correctness requires a boolean for every answer.')
            support = row.get('citation_support', {})
            if set(support) != set(original['citation_support']) or any(type(v) is not bool for v in support.values()):
                raise ValueError('Review exactly the citations actually emitted.')
    return value


def export(results, spec_path, output, input_dir):
    value = contract(results, spec_path)
    import fitz
    output = Path(output)
    # Do not overwrite an earlier form or its local browser draft.
    output.mkdir(parents=True, exist_ok=False)
    assets = output / 'source_pages'
    assets.mkdir()
    wanted = set()
    for data in value['modes'].values():
        for r in data['records']:
            wanted.update(r['reference_citations'])
            wanted.update(r['candidate']['citations'])
    for citation in sorted(wanted):
        m = re.fullmatch(r'(DOC[123]):p(\d+)', citation)
        if not m:
            raise ValueError(f'Unsupported citation: {citation}')
        with fitz.open(Path(input_dir) / PDF_NAMES[m[1]]) as pdf:
            pdf[int(m[2]) - 1].get_pixmap(matrix=fitz.Matrix(1.25, 1.25)).save(assets / f'{m[1]}_p{m[2]}.png')
    esc = html.escape
    options = '<option value="">Choose</option><option value="true">Yes</option><option value="false">No</option>'
    sections = []
    for mode, data in value['modes'].items():
        for i, row in enumerate(data['records']):
            a = row['candidate']
            preceding = [r for r in data['records'][:i] if r['conversation_id'] == row['conversation_id']]
            context = ''.join(f'<p><b>Earlier turn {esc(str(r["turn"]))}:</b> {esc(r["question"])}<br><b>Earlier system answer:</b> {esc(r["candidate"]["answer"])}</p>' for r in preceding)
            sources = []
            for c in sorted(set(a['citations'] + row['reference_citations'])):
                doc, page = c.split(':p')
                warning = '<b>LOW TRUST: Appendix C is not admissible evidence.</b>' if c == 'DOC3:p7' else ''
                sources.append(f'<details><summary>{esc(c)}</summary>{warning}<img loading="lazy" src="source_pages/{doc}_p{page}.png" alt="{esc(c)}"></details>')
            citations = ''.join(f'<label>Does {esc(c)} actually support a factual claim in this answer? <select class="citation" data-citation="{esc(c)}" required>{options}</select></label>' for c in a['citations']) or '<p>No emitted citations: no citation-support labels are requested. Judge answer correctness separately.</p>'
            sections.append(f'''<section data-mode="{mode}" data-index="{i}"><h2>{mode} · {row['question_id']}</h2>{context}<p><b>Question:</b> {esc(row['question'])}</p><blockquote>{esc(a['answer'])}</blockquote><p><b>Returned status:</b> {esc(a['status'])}<br><b>Figures:</b> {esc(', '.join(a['figure_references']) or 'NONE')}</p><details><summary>Frozen reference and original source pages</summary><p>{esc(row['reference'])}</p><p><b>Expected status:</b> {esc(row['expected_status'])}</p><p><b>Numeric targets (±1%):</b> {esc(str(row['numeric_targets']))}</p><p>{esc(row['scoring_notes'])}</p>{''.join(sources)}</details><label>Is the complete answer correct, with the required facts, numbers and status? <select class="correct" required>{options}</select></label>{citations}<label>Notes (recommended for incorrect, unsupported or ambiguous cases)<textarea rows="2"></textarea></label></section>''')
    template = Path(__file__).with_name('p1_content_review_template.html').read_text(encoding='utf-8')
    payload = json.dumps(value, ensure_ascii=False).replace('<', '\\u003c')
    page = template.replace('__SECTIONS__', ''.join(sections)).replace('__PAYLOAD__', payload)
    (output / 'content_review.html').write_text(page, encoding='utf-8')
    (output / 'review_contract.json').write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    print('Review page:', (output / 'content_review.html').resolve())
    print('Answers:', sum(len(m['records']) for m in value['modes'].values()))
    print('Emitted citations:', sum(len(r['citation_support']) for m in value['modes'].values() for r in m['records']))
    print('No judgments prefilled. Existing groundedness ratings are unchanged. No model inference.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=DEFAULT_RESULTS)
    parser.add_argument('--spec', type=Path, default=ROOT / 'eval/yuyao_ding/p1_golden_v2.json')
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    export(args.results, args.spec, args.output, args.input_dir)
