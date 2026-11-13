import json
import unittest
from src.p3_critique import parse_critique
class CritiqueTest(unittest.TestCase):
    def test_accept_and_revision(self):
        for decision, instruction in [('accept',''),('revise','Restore the second figure.')]:
            result=parse_critique(json.dumps(dict(decision=decision,safe=True,reason='Two figures visible.',revision_instruction=instruction)))
            self.assertEqual(result['decision'],decision)
    def test_unsafe_accept_and_empty_revision_rejected(self):
        for decision, safe in [('accept',False),('revise',True)]:
            with self.assertRaises(ValueError):
                parse_critique(json.dumps(dict(decision=decision,safe=safe,reason='Evidence',revision_instruction='')))
