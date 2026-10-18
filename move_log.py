"""Append-only move replay log with a fixed-width offset index.

The index lets the dashboard seek to one move without loading every board into RAM.
"""
import json
from pathlib import Path

INDEX_RECORD_BYTES = 21  # 20 decimal offset digits plus newline


def rebuild_index(moves_path, index_path):
    moves_path = Path(moves_path)
    index_path = Path(index_path)
    with moves_path.open("rb") as source, index_path.open("wb") as index:
        while True:
            offset = source.tell()
            line = source.readline()
            if not line:
                break
            index.write(f"{offset:020d}\n".encode("ascii"))


def open_move_log(moves_path, index_path, append=False):
    """Open the move log and index, rebuilding a missing/malformed index on resume."""
    moves_path, index_path = Path(moves_path), Path(index_path)
    moves_path.parent.mkdir(parents=True, exist_ok=True)
    if append and moves_path.exists():
        # Rebuild once on resume to recover cleanly if the previous run was
        # interrupted between writing a move and writing its index entry.
        rebuild_index(moves_path, index_path)
    else:
        index_path.write_bytes(b"")
    return (
        moves_path.open("a" if append else "w", encoding="utf-8"),
        index_path.open("a" if append else "w", encoding="ascii"),
    )


def append_move(moves_file, index_file, record):
    """Append one JSON move and its byte offset, flushing so the dashboard can see it."""
    offset = moves_file.tell()
    moves_file.write(json.dumps(record, separators=(",", ":")) + "\n")
    moves_file.flush()
    index_file.write(f"{offset:020d}\n")
    index_file.flush()


def move_count(index_path):
    path = Path(index_path)
    try:
        return path.stat().st_size // INDEX_RECORD_BYTES
    except OSError:
        return 0


def read_move_at(moves_path, index_path, index):
    """Read one indexed move. Returns None if it is not available yet."""
    index = int(index)
    if index < 0:
        return None
    try:
        with Path(index_path).open("rb") as offsets:
            offsets.seek(index * INDEX_RECORD_BYTES)
            raw_offset = offsets.read(INDEX_RECORD_BYTES)
        if len(raw_offset) != INDEX_RECORD_BYTES or not raw_offset.endswith(b"\n"):
            return None
        offset = int(raw_offset[:-1])
        with Path(moves_path).open("rb") as moves:
            moves.seek(offset)
            line = moves.readline()
        return json.loads(line.decode("utf-8")) if line else None
    except (OSError, ValueError, json.JSONDecodeError, UnicodeDecodeError):
        return None
