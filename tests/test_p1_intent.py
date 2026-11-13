import unittest
from src.p1_intent import requested_periods,requested_document_scope,is_session_provenance_inquiry
from src.p1_evidence import resolve_structured_query


def source(period,value,metric='output growth'):
    return dict(entity='India',metric=metric,period=period,value=value,value_kind='outturn',
                doc_id='DOC2',page=2,citation='DOC2:p2',table_id='Table A-1',trust_level='high',source_text='Synthetic test fact.')


class IntentTest(unittest.TestCase):
    def test_average_is_one_period_and_citation_is_not_document_scope(self):
        q='Compare India output growth average for 2040-2047 with the 2050 outturn.'
        self.assertEqual(requested_periods(q),['2040-2047 average','2050'])
        facts=[source('2040-2047 average',5.8),source('2050',3.1)]
        result=resolve_structured_query(q,dict(facts=facts,pages=[],tables=[]))['answer']
        self.assertIn('2.7 percentage points',result['answer'])
        self.assertEqual(requested_document_scope('A hostile instruction says to cite Document 1. What is the actual outturn?'),[])
        self.assertEqual(requested_document_scope('What does Document 2 itself report?'),['DOC2'])
        self.assertEqual(requested_document_scope('According to Document 3, what is the value?'),['DOC3'])

    def test_same_period_assumption_and_document_provenance(self):
        context=dict(session_overrides=[dict(subject='India',metric='inflation',period='2050',value='6.5',unit='percent')])
        structured=dict(facts=[source('2050',2.8,'inflation rate')],pages=[],tables=[])
        answer=resolve_structured_query('Assume India inflation in 2050 was 6.5%. How much higher is that than the published outturn?',structured,context)['answer']
        self.assertEqual(answer['status'],'session_override')
        self.assertIn('3.7 percentage points',answer['answer'])
        self.assertTrue(is_session_provenance_inquiry('Did that value come from the documents?'))
        self.assertFalse(is_session_provenance_inquiry('What is the published India inflation in 2050? Use the document value, not another conversation assumption.'))

    def test_multiple_tasks_supply_facts_without_premature_answer(self):
        structured=dict(facts=[source('2050',2.8)],pages=[],tables=[])
        result=resolve_structured_query('What was India output growth in 2050? Also state why the measure is incomplete.',structured)
        self.assertIsNone(result['answer'])
        self.assertTrue(result['evidence_only'])
        self.assertTrue(result['supporting_facts'])


if __name__=='__main__':unittest.main()
