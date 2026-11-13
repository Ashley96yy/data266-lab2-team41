import json
from pathlib import Path
import tempfile
import unittest
from src.p1_evaluation import (COVERAGE, EvaluationQuestion, inference_questions,
    validate_coverage, ranked_top_five, recall_at_five, make_dense_retriever,
    execute_questions, make_review_template, summarize)


def fixture_spec():
    cases=[]
    for category,count in COVERAGE.items():
        for index in range(count):
            number=len(cases)+1
            conversation=f'c{number}'
            turn=1
            if category=='multi_turn':
                conversation='multi-a' if index<3 else 'multi-b'
                turn=index+1 if index<3 else index-2
            cases.append(dict(id=str(number),question_id=f't{number}',
                conversation_id=conversation,turn=str(turn),question='A diagnostic question?',
                category=category,reference_answer='A diagnostic reference, 10 percent.',
                scoring_notes='Check the actual evidence.',status='session_override' if category=='multi_turn' else 'answered',
                citations=['DOC1:p1'],figure_references=[],required_retrieval=[['DOC1:p1']],numeric_targets=[10]))
    return {'set_id':'synthetic_test_only','cases':cases}


def fixture_run(spec):
    records=[]
    for case in spec['cases']:
        records.append(dict(question={k:case[k] for k in ('id','conversation_id','turn','question_id','question')},
            run_status='completed',top_five=[dict(id='DOC1:p1',citation='DOC1:p1',kind='page')],
            elapsed_seconds=1,turn={'answer':dict(answer='The measured value is 10%.',status=case['status'],
                                                 citations=['DOC1:p1'],figure_references=[])}))
    return dict(mode='final',execution_complete=True,run_sha256='diagnostic',records=records)


class EvaluationTest(unittest.TestCase):
    def test_coverage_and_reference_leak_boundary(self):
        spec=fixture_spec()
        self.assertEqual(validate_coverage(spec['cases']),COVERAGE)
        questions=inference_questions(spec)
        self.assertEqual(set(questions[0].__dict__),{'id','conversation_id','turn','question_id','question'})
        spec['cases'][-1]['citations']=['DOC3:p7']
        with self.assertRaisesRegex(ValueError,'untrusted'):validate_coverage(spec['cases'])

    def test_top_five_is_not_the_full_generation_context(self):
        retrieval={'text_results':[{'citation':f'DOC1:p{i}'} for i in range(1,7)],
                   'visual_route':{'results':[{'citation':'DOC2:p4','figure_id':'Figure A-2'}]}}
        ranking=ranked_top_five(retrieval)
        self.assertEqual(len(ranking),5)
        self.assertEqual(recall_at_five(ranking,[['DOC1:p6']]),0)
        self.assertEqual(recall_at_five(ranking,[['Figure A-2'],['DOC1:p6']]),0.5)
        self.assertIsNone(recall_at_five(ranking,[]))

    def test_dense_baseline_has_no_hidden_lexical_or_exact_router(self):
        hit=dict(chunk_id='x',citation='DOC1:p1',doc_id='DOC1',page=1,trust_level='high')
        calls=[]
        def dense_text(q,top_k):calls.append(('text',q));return [hit]
        def dense_image(q,top_k):calls.append(('image',q));return []
        result=make_dense_retriever(dict(search_dense_text=dense_text,search_dense_visual=dense_image,
                                          rag_chunk_by_id={'x':{'text':'actual source'}}))('Figure A-2')
        self.assertEqual(calls,[('text','Figure A-2'),('image','Figure A-2')])
        self.assertIsNone(result['text_results'][0]['bm25_rank'])
        self.assertEqual(result['visual_route']['mode'],'dense_clip_only')

    def test_missing_reviews_cannot_become_accuracy_or_human_scores(self):
        spec=fixture_spec();run=fixture_run(spec)
        review=make_review_template(run,spec)
        result=summarize(run,spec,review)
        self.assertEqual(result['retrieval_recall_at_5'],1)
        for key in ['answer_correctness','citation_accuracy','groundedness_mean']:
            self.assertIsNone(result[key])
        review['run_sha256']='another run'
        with self.assertRaisesRegex(ValueError,'different run'):summarize(run,spec,review)

    def test_status_numeric_tolerance_and_independent_reviewers(self):
        spec=fixture_spec();run=fixture_run(spec)
        review=make_review_template(run,spec)
        review.update(content_reviewer='Diagnostic reviewer',content_review_method='synthetic test labels',
                      reviewer_a_name='Member A',reviewer_b_name='Member B')
        for row in review['records']:
            row.update(answer_correct=True,citation_support={'DOC1:p1':True},
                       reviewer_a_groundedness=4,reviewer_b_groundedness=5)
        run['records'][0]['turn']['answer']['status']='not_found'
        run['records'][1]['turn']['answer']['answer']='The measured value is 10.2%.'
        run['records'][2]['turn']['answer']['answer']='The measured value is 10.05%.'
        result=summarize(run,spec,review)
        self.assertEqual(result['answer_correct_out_of_25'],23)
        self.assertEqual(result['groundedness_mean'],4.5)
        review['reviewer_b_name']='Member A'
        with self.assertRaisesRegex(ValueError,'distinct'):summarize(run,spec,review)

    def test_interrupted_batch_keeps_error_and_never_scores_partial_run(self):
        class Memory:pass
        calls=[]
        def runner(row,*args,**kwargs):
            calls.append(kwargs['enable_structured_resolution'])
            if len(calls)==2:raise RuntimeError('diagnostic interruption')
            return {'answer':{'answer':'Unknown.','status':'not_found','citations':[],
                              'figure_references':[]},'retrieval':None}
        namespace=dict(ConversationMemoryStore=Memory,run_conversation_turn=runner,
                       search_dense_text=lambda *a,**k:[],search_dense_visual=lambda *a,**k:[],rag_chunk_by_id={})
        questions=[EvaluationQuestion(str(i),'c','1',f't{i}','Describe the source.') for i in range(2)]
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'run.json'
            with self.assertRaises(RuntimeError):
                execute_questions(namespace,questions,mode='dense_only',processor=None,model=None,
                                  output_path=path,provenance={'test_only':True})
            saved=json.loads(path.read_text(encoding='utf-8'))
            self.assertFalse(saved['execution_complete'])
            self.assertEqual(saved['records'][1]['run_status'],'error')
            self.assertEqual(calls,[False,False])


if __name__=='__main__':unittest.main()
