"""Docker supplies PyTorch separately (CPU wheel or official CUDA base). Lock all other packages."""

import subprocess
import tomllib
from pathlib import Path

lock = tomllib.loads(Path("uv.lock").read_text())
exclude = {"torch", "triton"} | {p["name"] for p in lock["package"] if p["name"].startswith("nvidia-")}
command = [
    "uv",
    "export",
    "--frozen",
    "--no-dev",
    "--no-hashes",
    "--no-emit-project",
    "--output-file",
    "requirements-runtime.lock",
]
for name in sorted(exclude):
    command += ["--no-emit-package", name]
subprocess.run(command, check=True)
