"""Ensure the generated hardware claim follows the completed run manifest."""
import unittest

from src.p3_reporting import measured_markdown


class ReportingTest(unittest.TestCase):
    def test_hardware_claim_requires_a_completed_matching_run(self):
        fields = (
            'cases', 'execution_errors', 'generation_attempts',
            'clip_scored_attempts', 'mean_first_clip_cosine',
            'mean_final_clip_cosine', 'median_seconds_per_request',
            'passes', 'task_success_rate', 'attack_successes',
            'reviewer_agreement', 'critic_agreement',
        )
        rows = [dict.fromkeys(fields) | {'mode': mode, 'statuses': {}}
                for mode in ('with', 'without', 'red_team')]
        manifest = dict(source_commit='test', gpu='Tesla T4', status='completed',
                        peak_allocated_gib=10.3, peak_reserved_gib=11.0,
                        allocator_budget_gib=14.0, elapsed_seconds=582.5,
                        actual_t4_verified=True, actual_p100_verified=False)
        self.assertIn('Actual Tesla T4 hardware verified', measured_markdown(rows, manifest))
        self.assertIn('remains unverified for this run',
                      measured_markdown(rows, {**manifest, 'status': 'failed'}))
        self.assertIn('remains unverified for this run',
                      measured_markdown(rows, {**manifest, 'gpu': 'RTX 4090',
                                               'actual_t4_verified': False}))


if __name__ == '__main__':
    unittest.main()
