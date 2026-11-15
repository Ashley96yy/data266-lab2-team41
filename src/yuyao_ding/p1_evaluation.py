"""Part 1 evaluation: frozen questions, dense-only comparison and explicit reviews.

Reference answers are used only AFTER inference. Citation presence or overlap
with a reference is never represented as factual citation accuracy.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import mean, median
import time

COVERAGE = {'facts_tables': 5, 'figures': 6, 'two_sources': 3,
            'multi_turn': 5, 'conflicts_revisions': 3, 'safety_absence': 3}
QUESTION_FIELDS = ('id', 'conversation_id', 'turn', 'question_id', 'question')
BLOCKED_CITATION = 'DOC3:p7'


@dataclass(frozen=True)
class EvaluationQuestion:
    id: str
    conversation_id: str
    turn: str
    question_id: str
    question: str


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode('utf-8')).hexdigest()


def load_specification(path):
    result = json.loads(Path(path).read_text(encoding='utf-8'))
    validate_coverage(result['cases'])
    return result


def validate_coverage(cases):
    if len(cases) != 25 or len({c['question_id'] for c in cases}) != 25:
        raise ValueError('Exactly 25 unique golden questions are required.')
    if len({c['id'] for c in cases}) != 25:
        raise ValueError('Golden row IDs must be unique.')
    counts = Counter(c['category'] for c in cases)
    if counts != COVERAGE:
        raise ValueError(f'Incorrect requirement coverage: {dict(counts)}; expected {COVERAGE}.')
    conversations = {}
    for case in cases:
        conversations.setdefault(case['conversation_id'], []).append(case)
        if case['status'] not in {'answered', 'conflict', 'not_found', 'false_premise', 'session_override'}:
            raise ValueError('Invalid reference status.')
        if BLOCKED_CITATION in case.get('citations', []) or 'Figure C-1' in case.get('figure_references', []):
            raise ValueError('An untrusted appendix cannot support a reference answer.')
        if not case.get('reference_answer') or not case.get('scoring_notes'):
            raise ValueError('Every case needs a source-checked reference and scoring notes.')
    multiple = []
    for conversation, turns in conversations.items():
        if [int(c['turn']) for c in turns] != list(range(1, len(turns)+1)):
            raise ValueError(f'Nonconsecutive or reordered turns in {conversation}.')
        if len(turns) > 1:
            multiple.extend(turns)
            if any(c['category'] != 'multi_turn' for c in turns):
                raise ValueError('All turns in the five-question conversation subset must be categorized together.')
    if len(multiple) != 5 or len({c['conversation_id'] for c in multiple}) < 2:
        raise ValueError('Five multi-turn questions in at least two conversations are required.')
    if not any(c['status'] == 'session_override' for c in multiple):
        raise ValueError('The conversation subset must include a user-supplied value.')
    return dict(counts)


def inference_questions(specification):
    """An immutable whitelist prevents answer/rubric leakage into the runner."""
    return [EvaluationQuestion(**{k: str(case[k]) for k in QUESTION_FIELDS})
            for case in specification['cases']]


def make_dense_retriever(namespace):
    """Dense BGE + dense CLIP only: no BM25, label routing or table expansion."""
    def retrieve(query, text_k=5, visual_k=2, **_kwargs):
        text = []
        for rank, hit in enumerate(namespace['search_dense_text'](query, top_k=text_k), 1):
            chunk = namespace['rag_chunk_by_id'][hit['chunk_id']]
            text.append({**hit, 'text': chunk['text'], 'dense_rank': rank,
                         'bm25_rank': None, 'rrf_score': 0.0})
        return {'query': query, 'requested_doc_ids': [], 'text_results': text,
                'visual_route': {'mode': 'dense_clip_only', 'blocked_labels': [],
                    'results': namespace['search_dense_visual'](query, top_k=visual_k)}}
    return retrieve


def ranked_top_five(retrieval):
    """Fixed T1,V1,T2,V2,... pooling, de-duplicated, for both systems.

    The evidence unit is a page for text or an exact numbered visual for images.
    A larger generation context is never called a top-five ranking.
    """
    retrieval = retrieval or {}
    text = retrieval.get('text_results', [])
    visuals = retrieval.get('visual_route', {}).get('results', [])
    result, seen = [], set()
    for index in range(max(len(text), len(visuals))):
        for items, kind in [(text, 'page'), (visuals, 'figure')]:
            if index >= len(items):
                continue
            item = items[index]
            key = item.get('figure_id') if kind == 'figure' else item['citation']
            if key in seen:
                continue
            seen.add(key)
            result.append({'kind': kind, 'id': key, 'citation': item['citation']})
            if len(result) == 5:
                return result
    return result


def recall_at_five(ranking, required_groups):
    """Mean fraction of required source groups present; alternatives use OR.

    No-evidence cases are excluded instead of being counted as retrieval failures.
    """
    if not required_groups:
        return None
    available = {item['id'] for item in ranking[:5]}
    return mean(bool(available.intersection(group)) for group in required_groups)


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


def execute_questions(namespace, questions, *, mode, processor, model,
                      output_path, provenance):
    if mode not in ('dense_only', 'final'):
        raise ValueError('Unknown evaluation mode.')
    if not all(type(q) is EvaluationQuestion for q in questions):
        raise TypeError('Inference accepts question-only objects, never reference dictionaries.')
    questions_hash = digest([asdict(q) for q in questions])
    result = {'schema_version': 1, 'mode': mode, 'questions_sha256': questions_hash,
              'provenance': provenance, 'execution_complete': False,
              'scored': False, 'records': []}
    memory = namespace['ConversationMemoryStore']()
    retriever = make_dense_retriever(namespace) if mode == 'dense_only' else namespace['retrieve_multimodal']
    for question in questions:
        start = time.perf_counter()
        record = {'question': asdict(question), 'run_status': 'running'}
        result['records'].append(record)
        try:
            turn = namespace['run_conversation_turn'](
                asdict(question), memory, processor, model, retriever=retriever,
                enable_structured_resolution=(mode == 'final'))
            record.update(run_status='completed', turn=turn,
                          top_five=ranked_top_five(turn.get('retrieval_initial', turn.get('retrieval'))))
            answer = turn['answer']
            if BLOCKED_CITATION in answer['citations'] or 'Figure C-1' in answer['figure_references']:
                raise ValueError('Untrusted appendix used as evidence.')
        except Exception as exc:
            record.update(run_status='error', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            record['elapsed_seconds'] = time.perf_counter() - start
            _write_json(output_path, result)
        print(f"{mode}: {question.question_id} {record['run_status']} {record['elapsed_seconds']:.2f}s", flush=True)
    result['execution_complete'] = True
    result['run_sha256'] = digest(result['records'])
    _write_json(output_path, result)
    return result


def make_review_template(run, specification):
    references = {c['question_id']: c for c in specification['cases']}
    rows = []
    for record in run['records']:
        question = record['question']
        case = references[question['question_id']]
        answer = record['turn']['answer']
        rows.append({'question_id': question['question_id'], 'question': question['question'],
                     'candidate': answer, 'reference': case['reference_answer'],
                     'scoring_notes': case['scoring_notes'],
                     'answer_correct': None,
                     'citation_support': {c: None for c in answer['citations']},
                     'content_review_note': '', 'reviewer_a_groundedness': None,
                     'reviewer_b_groundedness': None})
    return {'run_sha256': run['run_sha256'], 'mode': run['mode'],
            'content_reviewer': None, 'content_review_method': None,
            'reviewer_a_name': None, 'reviewer_b_name': None, 'records': rows}


def numeric_targets_match(answer, targets):
    values = [float(value.replace('−', '-')) for value in
              re.findall(r'(?<![\w.])[−-]?\d+(?:\.\d+)?', answer)]
    return all(any(math.isclose(value, float(target), rel_tol=0.01, abs_tol=1e-9)
                   for value in values) for target in targets)


def summarize(run, specification, review=None):
    cases = specification['cases']
    if not run.get('execution_complete') or len(run['records']) != 25:
        raise ValueError('A complete 25-question execution is required.')
    expected = [c['question_id'] for c in cases]
    if [r['question']['question_id'] for r in run['records']] != expected:
        raise ValueError('Wrong question identities or ordering.')
    if any(r['run_status'] != 'completed' for r in run['records']):
        raise ValueError('Resolve execution errors before aggregation.')
    recalls = [recall_at_five(r['top_five'], c['required_retrieval'])
               for r, c in zip(run['records'], cases)]
    eligible = [r for r in recalls if r is not None]
    metrics = {'mode': run['mode'], 'denominator': 25,
               'answer_correct_out_of_25': None, 'answer_correctness': None,
               'retrieval_recall_at_5': mean(eligible) if eligible else None,
               'retrieval_denominator': len(eligible), 'citation_accuracy': None,
               'citation_denominator': sum(len(r['turn']['answer']['citations']) for r in run['records']),
               'groundedness_mean': None, 'groundedness_agree_within_one': None,
               'median_seconds_per_question': median(r['elapsed_seconds'] for r in run['records']),
               'content_review_complete': False, 'two_member_review_complete': False}
    if review is None:
        return metrics
    if review.get('run_sha256') != run['run_sha256'] or review.get('mode') != run['mode']:
        raise ValueError('Reviews belong to a different run.')
    by_id = {r['question_id']: r for r in review['records']}
    if set(by_id) != set(expected) or len(review['records']) != 25:
        raise ValueError('Missing or duplicate reviews.')
    reviewed = [by_id[i] for i in expected]
    if all(type(r.get('answer_correct')) is bool for r in reviewed):
        if not review.get('content_reviewer') or not review.get('content_review_method'):
            raise ValueError('Name the content reviewer and distinguish human/AI-assisted review.')
        correct = sum(r['answer_correct'] and a['turn']['answer']['status'] == c['status']
                      and numeric_targets_match(a['turn']['answer']['answer'], c.get('numeric_targets', []))
                      for r, a, c in zip(reviewed, run['records'], cases))
        metrics.update(answer_correct_out_of_25=correct, answer_correctness=correct/25,
                       content_review_complete=True, content_review_method=review['content_review_method'])
    labels = []
    for row, record in zip(reviewed, run['records']):
        actual = set(record['turn']['answer']['citations'])
        support = row.get('citation_support', {})
        if set(support) != actual:
            raise ValueError('Citation reviews must cover exactly the emitted citations.')
        labels.extend(support.values())
    if labels and all(type(v) is bool for v in labels):
        if not review.get('content_reviewer') or not review.get('content_review_method'):
            raise ValueError('Citation accuracy requires a documented source-content review.')
        metrics['citation_accuracy'] = mean(labels)
    ratings = [(r.get('reviewer_a_groundedness'), r.get('reviewer_b_groundedness')) for r in reviewed]
    if all(a is not None and b is not None for a, b in ratings):
        names = [review.get('reviewer_a_name'), review.get('reviewer_b_name')]
        if not all(names) or names[0].strip().casefold() == names[1].strip().casefold():
            raise ValueError('Two distinct team members must independently rate the answers.')
        if any(type(v) is not int or not 1 <= v <= 5 for pair in ratings for v in pair):
            raise ValueError('Groundedness ratings must be integers from 1 to 5.')
        metrics.update(groundedness_mean=mean(v for pair in ratings for v in pair),
                       groundedness_agree_within_one=mean(abs(a-b)<=1 for a,b in ratings),
                       two_member_review_complete=True)
    return metrics
