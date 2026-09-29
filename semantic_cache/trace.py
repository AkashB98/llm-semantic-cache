"""JSONL trace writer: one record per cache request.

Records are appended atomically (one open/append/close per write) and the
writer is lock-guarded so concurrent chat calls can't interleave lines.
"""

import json
import threading


class TraceWriter:
    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()

    def write(self, record: dict):
        line = json.dumps(record, sort_keys=True)
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    def read_all(self):
        records = []
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        except FileNotFoundError:
            pass
        return records
