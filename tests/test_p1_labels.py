import unittest
from src.p1_labels import find_explicit_visual_labels
from src.p1_evidence import resolve_structured_query


class LabelTest(unittest.TestCase):
    def test_plural_lists_dash_variants_and_corrected_distinction(self):
        self.assertEqual(find_explicit_visual_labels('Read Figures A–8 and A-9, then Tables B-3 and B-4.'),
                         ['Figure A-8','Figure A-9','Table B-3','Table B-4'])
        self.assertEqual(find_explicit_visual_labels('Compare Figure B-9 with Figure B-9 (corrected), then Table A-8.'),
                         ['Figure B-9','Figure B-9 (corrected)','Table A-8'])
        self.assertEqual(find_explicit_visual_labels('Use figure a - 5.'),['Figure A-5'])
        self.assertEqual(find_explicit_visual_labels('The 2007-2010 average was 1.2.'),[])
        self.assertEqual(find_explicit_visual_labels('Use Figure C-1.'),['Figure C-1'])

    def test_named_chart_cannot_be_replaced_by_a_table_lookup(self):
        diagnostic=dict(entity='India',metric='output growth',period='2040',value=4,
            value_kind='outturn',doc_id='DOC2',page=2,citation='DOC2:p2',
            table_id='Table A-1',trust_level='high',source_text='Synthetic observation.')
        a=resolve_structured_query('According to Figure A-9, what was India output growth in 2040?',
                                   dict(facts=[diagnostic],pages=[],tables=[]))
        self.assertIsNone(a)


if __name__=='__main__':unittest.main()
