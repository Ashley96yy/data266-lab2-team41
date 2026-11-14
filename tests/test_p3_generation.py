import unittest
from src.p3_generation import GenerationSettings, DiffusionGenerator
class GenerationTest(unittest.TestCase):
    def test_settings_reject_invalid_values(self):
        for settings in ({'steps':0}, {'strength':0}, {'size':64}, {'seed':-1}):
            with self.assertRaises(ValueError): GenerationSettings(**settings)
    def test_lazy_loading(self):
        self.assertIsNone(DiffusionGenerator().pipe)

class GenerationSafetyTest(unittest.TestCase):
    def test_unsafe_or_unchecked_output_is_never_saved(self):
        from unittest.mock import Mock, patch
        from types import SimpleNamespace
        import tempfile
        from pathlib import Path
        from PIL import Image
        from src.p3_prompt import build_generation_prompt
        fake_torch=Mock()
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'source.png'; Image.new('RGB',(40,32)).save(source)
            draft=build_generation_prompt('Make a watercolor painting.','style_change_img2img',{})
            for flags in (None,[True],[False]):
                generator=DiffusionGenerator(); generator._load=Mock()
                generator.revisions={'base':'fixed-sha'}
                generator.pipe=Mock(return_value=SimpleNamespace(nsfw_content_detected=flags,images=[Image.new('RGB',(32,32))]))
                generator.pipe.tokenizer=Mock(return_value=SimpleNamespace(input_ids=[1,2]))
                generator.pipe.tokenizer.model_max_length=77
                target=Path(directory)/f'{flags}.png'
                with patch.dict('sys.modules',torch=fake_torch):
                    result=generator.generate(source,draft,target)
                self.assertEqual(target.exists(),flags==[False])
                self.assertEqual(result['safe'],flags==[False])
                self.assertEqual(result['model_revision'],'fixed-sha')
