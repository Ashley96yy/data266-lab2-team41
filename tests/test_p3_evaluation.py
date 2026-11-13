import unittest
from src.p3_evaluation import summarize
class EvaluationTest(unittest.TestCase):
    def test_unreviewed_and_missing_cases_cannot_be_scored(self):
        with self.assertRaises(ValueError): summarize(dict(set_id='p3_primary_15',records=[]))
    def test_denominators_and_human_review(self):
        records=[dict(case_id=f'P3-RT{i:02d}',reviewer_a_pass=True,reviewer_b_pass=True,
                      adjudicated_pass=True,trace=dict(status='refused',attempts=[],elapsed_seconds=i)) for i in range(1,11)]
        result=summarize(dict(set_id='p3_red_team_10',records=records))
        self.assertEqual(result['attack_successes'],0); self.assertEqual(result['denominator'],10)
        records[0]['reviewer_b_pass']=False
        with self.assertRaises(ValueError): summarize(dict(set_id='p3_red_team_10',records=records))
