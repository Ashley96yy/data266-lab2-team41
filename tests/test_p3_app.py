import unittest
from src.p3_app import build_app
class AppTest(unittest.TestCase):
    def test_app_builds_without_loading_models(self):
        app=build_app(object())
        self.assertTrue(app.config['components'])
        self.assertEqual(len(app.config['dependencies']),1)
        app.close()
