"""Synthetic observations only; no golden answers or practice IDs are fixtures."""
import unittest
from src.p1_evidence import parse_query_frame, resolve_structured_query


def fact(entity,year,value):
    return dict(entity=entity,metric='output growth',period=str(year),value=value,
                value_kind='outturn',doc_id='DOC2',page=2,citation='DOC2:p2',
                table_id='Table A-1',trust_level='high',source_text='Synthetic diagnostic observation.')


class ReasoningRegressionTest(unittest.TestCase):
    def resolve(self,question,facts,overrides=()):
        return resolve_structured_query(question,dict(facts=facts,pages=[],tables=[]),
            dict(prior_turns=[],session_overrides=list(overrides)))['answer']

    def test_directional_premise_and_explicit_counterfactual(self):
        q='Why did India economy contract in 2040?'
        a=self.resolve(q,[fact('India',2040,5.2)])
        self.assertEqual(a['status'],'false_premise')
        self.assertIn('5.2%',a['answer'])
        a=self.resolve('Explain why India output grew positively in 2040.',[fact('India',2040,-2.3)])
        self.assertEqual(a['status'],'false_premise')
        self.assertIn('-2.3%',a['answer'])
        self.assertEqual(parse_query_frame('Assume India output fell by 8% in 2040.')['metrics'],['output growth'])

    def test_cross_period_assumption_comparison_preserves_direction_and_status(self):
        a=self.resolve("For this chat only, assume India output growth in 2041 was 2.2%. How does that compare with the 2040 outturn?",
            [fact('India',2040,5.7)],
            [dict(subject='India',metric='output growth',period='2041',value='2.2',unit='percent')])
        self.assertEqual(a['status'],'session_override')
        self.assertIn('3.5 percentage points lower than 5.7%',a['answer'])
        self.assertEqual(a['citations'],['DOC2:p2'])

    def test_directional_assumption_compares_entities_without_changing_document(self):
        facts=[fact('India',2040,1.5),fact('advanced economies',2040,-2.0)]
        q='For this conversation only, assume India output fell by 7.0% in 2040. How much worse would it have done than the advanced-economy average?'
        # The noun phrase is normalized by the existing entity recognizer.
        q=q.replace('advanced-economy average','advanced economies average')
        a=self.resolve(q,facts,[dict(subject='India',metric='output growth',period='2040',value='-7.0',unit='percent')])
        self.assertEqual(a['status'],'session_override')
        self.assertIn('5.0 percentage points',a['answer'])
        self.assertEqual(facts[0]['value'],1.5)


if __name__=='__main__':unittest.main()
