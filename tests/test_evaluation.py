"""Tests for reproducible agent evaluation and benchmark summaries."""
import unittest

from evaluate import confidence_interval, evaluate_policy, run_evaluation


class EvaluationTests(unittest.TestCase):
    def test_confidence_interval_handles_single_observation(self):
        result = confidence_interval([4])
        self.assertEqual(result["mean"], 4.0)
        self.assertEqual(result["std"], 0.0)
        self.assertEqual(result["ci95"], [4.0, 4.0])

    def test_random_policy_is_reproducible_for_fixed_seed(self):
        first = evaluate_policy("random", [23], max_pieces=25)
        second = evaluate_policy("random", [23], max_pieces=25)
        self.assertEqual(first, second)

    def test_benchmark_uses_shared_seeds_and_baselines_without_checkpoint(self):
        report = run_evaluation(
            episodes=1, seed=9, policies=["random", "heuristic"],
            device_name="cpu", max_pieces=25,
        )
        self.assertEqual(report["config"]["seeds"], [9])
        self.assertEqual(
            [row["policy"] for row in report["results"]], ["random", "heuristic"]
        )
        for row in report["results"]:
            self.assertEqual(row["episodes"], 1)
            self.assertIn("ci95", row["lines"])
            self.assertLessEqual(row["lines"]["ci95"][0], row["lines"]["mean"])
            self.assertGreaterEqual(row["lines"]["ci95"][1], row["lines"]["mean"])

    def test_ppo_requires_checkpoint(self):
        with self.assertRaisesRegex(ValueError, "requires --checkpoint"):
            run_evaluation(episodes=1, policies=["ppo"], device_name="cpu")


if __name__ == "__main__":
    unittest.main()
