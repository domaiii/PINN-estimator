"""Generic experiment records: JSON, input hashes, source and environment."""

import hashlib
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n")


def save_provenance(output, project_root, input_files, source_files,
                    packages=("torch", "numpy", "scipy", "pandas", "matplotlib", "PyYAML", "Pillow")):
    """Record named input groups and explicitly selected source files."""
    output, project_root = Path(output), Path(project_root)
    manifest = []
    for name, files in input_files.items():
        manifest.append({"directory": name, "files": {
            label: {"path": str(Path(path).resolve()),
                    "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
            for label, path in files.items()
        }})
    write_json(output / "input_manifest.json", manifest)
    hashes = {}
    for source in source_files:
        source = Path(source).resolve()
        relative = source.relative_to(project_root)
        target = output / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[str(relative)] = hashlib.sha256(source.read_bytes()).hexdigest()

    def git(*arguments):
        try:
            return subprocess.check_output(
                ["git", *arguments], cwd=project_root,
                stderr=subprocess.DEVNULL, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    versions = {}
    for package in packages:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    write_json(output / "provenance.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version, "platform": platform.platform(),
        "packages": versions, "git_commit": git("rev-parse", "HEAD"),
        "git_status": git("status", "--short"), "source_sha256": hashes,
    })
