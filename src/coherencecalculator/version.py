"""Build/version fingerprint identifying the exact running build."""

import os
import subprocess
import tomllib
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _pyproject() -> dict:
    """Return pyproject.toml parsed as a dict, or {} if it cannot be read."""
    try:
        with open(_PROJECT_ROOT / "pyproject.toml", "rb") as f:
            return tomllib.load(f)
    except OSError:
        return {}


def _git_short_hash() -> str:
    """Return the short HEAD commit hash, or "unknown" when unavailable."""
    baked_sha = os.environ.get("GIT_SHA")
    if baked_sha:
        return baked_sha
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


_pyproject_data = _pyproject()

# "service-version tag-git hash", e.g. "whisper-asr-webservice-0.1.0-decce53".
SYSTEM_FINGERPRINT = "{service}-{version}-{hash}".format(
    service=_pyproject_data.get("project", {}).get("name", "unknown"),
    version=_pyproject_data.get("project", {}).get("version", "unknown"),
    hash=_git_short_hash(),
)
