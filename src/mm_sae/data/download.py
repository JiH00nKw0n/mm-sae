"""Public archive downloads. Small real smoke runs read only requested ZIP members."""
from __future__ import annotations

import hashlib
import io
import logging
import os
import urllib.request
import zipfile
from pathlib import Path
from ..io import file_lock

LOG = logging.getLogger(__name__)


def fetch(url: str, path: Path):
    with file_lock(path.with_suffix(path.suffix + ".lock")):
        _fetch(url, path)


def _fetch(url: str, path: Path):
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    LOG.info("Download %s", url)
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as response:
        append = offset > 0 and response.status == 206
        if append and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
            raise IOError("Download resume range does not match the partial file")
        received = 0
        expected = response.headers.get("Content-Length")
        with partial.open("ab" if append else "wb") as out:
            while block := response.read(4 << 20):
                out.write(block)
                received += len(block)
        if expected is not None and received != int(expected):
            raise IOError("Incomplete download; the next run will resume it")
    os.replace(partial, path)


class RangeFile(io.RawIOBase):
    def __init__(self, url: str, cache: Path):
        self.url, self.pos = url, 0
        self.cache = cache / hashlib.sha256(url.encode()).hexdigest()[:16]
        self.cache.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as response:
            self.size = int(response.headers["Content-Length"])

    def readable(self):
        return True

    def seekable(self):
        return True

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        return self.pos

    def tell(self):
        return self.pos

    def read(self, n=-1):
        n = min(n if n >= 0 else self.size - self.pos, self.size - self.pos)
        if n <= 0:
            return b""
        start, end = self.pos, self.pos + n
        lo, hi = (start // 65536) * 65536, min(self.size, ((end + 65535) // 65536) * 65536)
        dest = self.cache / f"{lo}-{hi}.bin"
        with file_lock(dest.with_suffix(".lock")):
            if not dest.exists():
                separator = "&" if "?" in self.url else "?"
                url = self.url + f"{separator}range_start={lo}&range_end={hi - 1}"
                req = urllib.request.Request(url, headers={"Range": f"bytes={lo}-{hi - 1}"})
                with urllib.request.urlopen(req, timeout=120) as response:
                    if response.status != 206:
                        raise RuntimeError("Server ignored Range request; refusing a full archive download in sample mode")
                    if not response.headers.get("Content-Range", "").startswith(f"bytes {lo}-{hi - 1}/"):
                        raise RuntimeError("Range response does not match requested bytes")
                    data = response.read()
                if len(data) != hi - lo:
                    raise IOError("Truncated archive range")
                tmp = dest.with_suffix(".tmp")
                tmp.write_bytes(data)
                os.replace(tmp, dest)
        data = dest.read_bytes()[start-lo:end-lo]
        self.pos = end
        return data


def extract_member(archive, member: str, destination: Path):
    with file_lock(destination.with_suffix(destination.suffix + ".lock")):
        _extract_member(archive, member, destination)


def _extract_member(archive, member: str, destination: Path):
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    # The caller supplies an explicit destination; archive paths cannot escape it.
    with archive.open(member) as src, tmp.open("wb") as dst:
        while block := src.read(4 << 20):
            dst.write(block)
    os.replace(tmp, destination)


def annotation_archive(config, filename: str):
    root = config.data.root / "downloads"
    local = root / filename
    if config.data.image_limits and not local.exists():
        return zipfile.ZipFile(RangeFile(config.data.downloads[filename], root / "ranges"))
    fetch(config.data.downloads[filename], local)
    return zipfile.ZipFile(local)
