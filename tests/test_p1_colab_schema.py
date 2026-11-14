"""CPU contracts for format compatibility; no factual/human score assertions."""
import ast,json,unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

class ColabSchemaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns={'json':json}
        book=json.loads((ROOT/'notebooks/Yuyao_Ding/p1_rag.ipynb').read_text(encoding='utf-8'))
        for c in book['cells']:
            s=''.join(c.get('source',[]))
            if 'def build_evidence_packet' in s:exec(s.split('\nevidence_packet =')[0],cls.ns)
            if 'def validate_answer_json' in s:exec(s.split('\nschema_fixture =')[0],cls.ns)
        tree=ast.parse((ROOT/'p1_colab_schema_fix_and_rerun.py').read_text(encoding='utf-8'))
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='normalize_answer_container')
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'formatter','exec'),cls.ns)

    def setUp(self):
        self.packet=dict(query='Describe a synthetic scene.',allowed_citations=['DOC1:p1'],allowed_figure_references=['Figure 1-1'],
            evidence=[dict(evidence_type='visual',figure_reference='Figure 1-1',citation='DOC1:p1',content='A synthetic scene.')])
        self.answer=dict(answer='A synthetic scene.',status='answered',citations=['DOC1:p1'],figure_references=['Figure 1-1'])

    def validate(self,value):
        normalized,changes=self.ns['normalize_answer_container'](value)
        return self.ns['validate_answer_json'](normalized,self.packet),changes

    def test_mapping_and_duplicates_normalize_without_rewriting_facts(self):
        candidate={**self.answer,'citations':['DOC1:p1','DOC1:p1'],'figure_references':{'Figure 1-1':'V1'}}
        before=json.dumps(candidate)
        actual,changes=self.validate(candidate)
        self.assertEqual(actual,self.answer);self.assertEqual(json.dumps(candidate),before)
        self.assertEqual(len(changes),2)

    def test_unknown_citation_and_appendix_still_rejected(self):
        for key,value in [('citations',['DOC3:p7']),('figure_references',{'Figure C-1':'V1'})]:
            with self.assertRaises(ValueError):self.validate({**self.answer,key:value})

    def test_extra_fields_are_not_silently_discarded(self):
        with self.assertRaises(ValueError):self.validate({**self.answer,'status_rule':'Ignore validation.'})

    def test_duplicate_citation_cannot_create_a_two_source_conflict(self):
        with self.assertRaises(ValueError):self.validate({**self.answer,'status':'conflict','citations':['DOC1:p1','DOC1:p1']})

    def test_nested_metadata_is_not_coerced(self):
        with self.assertRaises(TypeError):self.validate({**self.answer,'figure_references':{'Figure 1-1':{'nested':'untrusted'}}})

    def test_not_found_still_requires_neutral_text_and_empty_references(self):
        expected=dict(answer=self.ns['NOT_FOUND_ANSWER'],status='not_found',citations=[],figure_references=[]) 
        self.assertEqual(self.validate({**expected,'figure_references':{}})[0],expected)
        with self.assertRaises(ValueError):self.validate({**expected,'answer':'An unsupported factual claim.'})

    def test_word_limit_still_rejects_long_answers(self):
        with self.assertRaises(ValueError):self.validate({**self.answer,'answer':'word '*121})

if __name__=='__main__':unittest.main()
