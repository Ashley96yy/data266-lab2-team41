import json
from pathlib import Path
import tempfile
import unittest
from PIL import Image
from src.p3_orchestration import AgentOrchestrator
class PipelineTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.image=Path(self.tmp.name)/'input.png'; Image.new('RGB',(32,32)).save(self.image)
        self.calls=[]
    def vision(self,path,questions):
        self.calls.append('vision')
        return dict(caption='One mug.',main_objects=['mug'],visible_text=[],answers=[dict(question=q,answer='A mug.') for q in questions])
    def generate(self,path,draft,target,settings):
        self.calls.append('generate'); Image.new('RGB',(32,32)).save(target)
        return dict(image_path=str(target),safe=True,safety_checked=True)
    def critique(self,*args):
        self.calls.append('critic'); return dict(decision='revise',safe=True,reason='Mug lost.',revision_instruction='Restore mug.')
    def pipeline(self): return AgentOrchestrator(self.vision,self,self,Path(self.tmp.name)/'runs')
    def test_refusal_and_bad_input_skip_models(self):
        for request,path in [('Create graphic gore.',self.image),('Caption it.',None)]:
            trace=self.pipeline().run(request,path)
            self.assertTrue(Path(trace['trace_path']).is_file())
        self.assertEqual(self.calls,[])
    def test_caption_skips_generation(self):
        trace=self.pipeline().run('Describe this image.',self.image)
        self.assertEqual(trace['status'],'answered'); self.assertEqual(self.calls,['vision'])
    def test_two_revision_limit_and_original_source(self):
        trace=self.pipeline().run('Make a watercolor painting.',self.image)
        self.assertEqual(trace['status'],'revision_limit'); self.assertEqual(len(trace['attempts']),3)
        self.assertEqual(json.loads(Path(trace['trace_path']).read_text())['status'],'revision_limit')
    def test_ablation_retains_safety_and_only_one_generation(self):
        trace=self.pipeline().run('Make a watercolor painting.',self.image,critique_enabled=False)
        self.assertEqual(trace['status'],'completed'); self.assertEqual(self.calls,['vision','generate','critic'])
    def test_model_error_saved(self):
        def fail(*args): raise RuntimeError('Model unavailable')
        pipeline=self.pipeline(); pipeline.vision=fail
        trace=pipeline.run('Describe this image.',self.image)
        self.assertEqual(trace['status'],'error'); self.assertTrue(Path(trace['trace_path']).exists())
