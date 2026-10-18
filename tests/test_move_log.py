import json
import tempfile
import unittest
from pathlib import Path

from move_log import append_move, move_count, open_move_log, read_move_at


class MoveLogTests(unittest.TestCase):
    def test_appends_and_reads_every_move_by_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            moves_path = root / "moves.jsonl"
            index_path = root / "moves.idx"
            moves, index = open_move_log(moves_path, index_path)
            try:
                expected = [
                    {"move": 0, "board_before": [[0, 1]], "board_after": [[1, 1]]},
                    {"move": 1, "board_before": [[1, 1]], "board_after": [[1, 0]]},
                ]
                for row in expected:
                    append_move(moves, index, row)
            finally:
                moves.close()
                index.close()

            self.assertEqual(move_count(index_path), 2)
            self.assertEqual(read_move_at(moves_path, index_path, 0), expected[0])
            self.assertEqual(read_move_at(moves_path, index_path, 1), expected[1])
            self.assertIsNone(read_move_at(moves_path, index_path, 2))

    def test_resume_rebuilds_index_if_last_write_was_interrupted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            moves_path = root / "moves.jsonl"
            index_path = root / "moves.idx"
            moves_path.write_text(json.dumps({"move": 0}) + "\n", encoding="utf-8")
            index_path.write_text("", encoding="ascii")

            moves, index = open_move_log(moves_path, index_path, append=True)
            try:
                self.assertEqual(move_count(index_path), 1)
                append_move(moves, index, {"move": 1})
            finally:
                moves.close()
                index.close()
            self.assertEqual(read_move_at(moves_path, index_path, 1), {"move": 1})


if __name__ == "__main__":
    unittest.main()
