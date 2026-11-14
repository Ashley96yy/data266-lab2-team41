import unittest
from src.p3_evaluation import summarize
class EvaluationTest(unittest.TestCase):
    def test_unreviewed_and_missing_cases_cannot_be_scored(self):
        with self.assertRaises(ValueError): summarize(dict(set_id='p3_primary_15',records=[]))
    def test_denominators_and_human_review(self):
        records=[dict(case_id=f'P3-RT{i:02d}',reviewer_a_pass=True,reviewer_b_pass=True,
                      adjudicated_pass=True,trace=dict(status='refused',attempts=[],elapsed_seconds=i)) for i in range(1,11)]
        reviewed=dict(set_id='p3_red_team_10',records=records,reviewer_a_name='Diagnostic A',reviewer_b_name='Diagnostic B')
        result=summarize(reviewed)
        self.assertEqual(result['attack_successes'],0); self.assertEqual(result['denominator'],10)
        records[0]['reviewer_b_pass']=False
        with self.assertRaises(ValueError): summarize(reviewed)

    def test_partial_critic_labels_cannot_report_perfect_agreement(self):
        records=[dict(case_id=f'P3-RT{i:02d}',reviewer_a_pass=True,reviewer_b_pass=True,
            adjudicated_pass=True,trace=dict(status='completed',elapsed_seconds=1,attempts=[
                dict(human_accept=True,critique=dict(decision='accept')),
                dict(human_accept=None,critique=dict(decision='revise'))])) for i in range(1,11)]
        reviewed=dict(set_id='p3_red_team_10',records=records,reviewer_a_name='Synthetic A',reviewer_b_name='Synthetic B')
        result=summarize(reviewed)
        self.assertIsNone(result['critic_agreement'])
        self.assertEqual(result['critic_reviewed_attempts'],10)
        self.assertEqual(result['critic_agreement_denominator'],20)
        reviewed['reviewer_b_name']='Synthetic A'
        with self.assertRaises(ValueError):summarize(reviewed)

    def test_interrupted_gpu_batch_preserves_unscored_checkpoint(self):
        import json
        from pathlib import Path
        import tempfile
        from unittest.mock import Mock, patch
        from src.p3_evaluation import run_cases
        specification=dict(set_id='diagnostic',seed=266,cases=[
            dict(id=f'smoke-{i}',request='Describe this image.',fixture='new.png',pass_if=['Review visually.'])
            for i in range(2)])
        pipeline=Mock()
        pipeline.run.side_effect=[dict(status='answered',elapsed_seconds=1,attempts=[]),
                                  RuntimeError('interrupted batch')]
        with tempfile.TemporaryDirectory() as directory:
            target=Path(directory)/'checkpoint.json'
            with patch('subprocess.run'), self.assertRaises(RuntimeError):
                run_cases(pipeline,specification,root=directory,checkpoint_path=target)
            saved=json.loads(target.read_text(encoding='utf-8'))
            self.assertEqual([r['case_id'] for r in saved['records']],['smoke-0'])
            self.assertFalse(saved['execution_complete'])
            self.assertFalse(saved['scored'])
            self.assertIsNone(saved['records'][0]['human_pass'])
