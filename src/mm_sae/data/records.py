"""Shared sample IDs and image-level split policy across benchmark adapters."""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Record:
    id: str
    group: str
    side: str
    value: str
    split: str
    labels: tuple[str, ...] = ()
    edit: dict = field(default_factory=dict)


def fold(group, seed):
    bucket = int.from_bytes(hashlib.sha256(f"{seed}/{group}".encode()).digest()[:8], "big") % 10
    return "train" if bucket < 6 else "val" if bucket < 8 else "test"


def jsonlines(path):
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def json_array(path):
    """Stream a top-level JSON array of per-image VG graphs, one object at a time."""
    decoder = json.JSONDecoder()
    with Path(path).open() as stream:
        buf = ""
        eof = False

        def refill():
            nonlocal buf, eof
            chunk = stream.read(1 << 20)
            buf += chunk
            eof = not chunk

        def skipspace():
            nonlocal buf
            buf = buf.lstrip()
            while not buf and not eof:
                refill()
                buf = buf.lstrip()

        skipspace()
        if not buf.startswith("["):
            raise ValueError(f"Expected JSON array in {path}")
        buf = buf[1:]
        skipspace()
        if buf.startswith("]"):
            return
        while True:
            while True:
                try:
                    obj, end = decoder.raw_decode(buf)
                    break
                except json.JSONDecodeError:
                    if eof:
                        raise
                    refill()
            yield obj
            buf = buf[end:]
            skipspace()
            if buf.startswith("]"):
                return
            if not buf.startswith(","):
                raise ValueError(f"Malformed JSON array in {path}")
            buf = buf[1:]
            skipspace()
