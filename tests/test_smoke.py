"""Lightweight end-to-end smoke test for a complete PPO update."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class TrainingSmokeTests(unittest.TestCase):
    def test_one_update_writes_checkpoint_metrics_config_and_replay(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "smoke-run"
            env = os.environ.copy()
            env["OMP_NUM_THREADS"] = "1"
            env["MKL_NUM_THREADS"] = "1"
            result = subprocess.run(
                [
                    sys.executable, str(ROOT / "train.py"),
                    "--updates", "1",
                    "--envs", "1",
                    "--rollout", "8",
                    "--epochs", "1",
                    "--minibatch-size", "8",
                    "--save-every", "1",
                    "--seed", "17",
                    "--device", "cpu",
                    "--run-dir", str(run_dir),
                ],
                cwd=ROOT, env=env, capture_output=True, text=True, timeout=120,
            )
            self.assertEqual(
                result.returncode, 0,
                msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
            )
            self.assertTrue((run_dir / "checkpoint.pt").is_file())
            self.assertTrue((run_dir / "config.json").is_file())
            self.assertTrue((run_dir / "metrics.jsonl").is_file())
            self.assertTrue((run_dir / "moves.jsonl").is_file())
            self.assertTrue((run_dir / "moves.idx").is_file())

            config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["args"]["updates"], 1)
            with (run_dir / "metrics.jsonl").open(encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle if line.strip()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["update"], 1)
            self.assertEqual(rows[0]["env_steps"], 8)


if __name__ == "__main__":
    unittest.main()
