"""Resolve Hugging Face credentials without persisting or displaying secrets."""

from __future__ import annotations

import os
from pathlib import Path


_TOKEN_ENV_NAMES = (
    "HF_TOKEN",
    "HUGGINGFACE_HUB_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "HUGGINGFACEHUB_API_TOKEN",
)


def _dotenv_value(path: Path) -> str | None:
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        candidate = line.strip()
        if not candidate or candidate.startswith("#"):
            continue
        if candidate.startswith("export "):
            candidate = candidate[len("export "):].lstrip()
        name, separator, value = candidate.partition("=")
        if not separator or name.strip() not in _TOKEN_ENV_NAMES:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        else:
            value = value.split(" #", 1)[0].strip()
        if value:
            return value
    return None


def get_hf_token(repo_root: str | Path | None = None) -> str | None:
    """Get the HF token from process env, repo `.env`, or `hf auth login` cache.

    A token found in `.env` is exported as HF_TOKEN for child processes such as
    vLLM workers. The value is returned only to the caller and is never logged
    or added to model/run identity records.
    """
    for name in _TOKEN_ENV_NAMES:
        value = os.environ.get(name, "").strip()
        if value:
            os.environ["HF_TOKEN"] = value
            return value

    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    token = _dotenv_value(root / ".env")
    if token:
        os.environ["HF_TOKEN"] = token
        return token

    try:
        from huggingface_hub import get_token
    except ImportError:
        return None
    token = get_token()
    if token:
        os.environ["HF_TOKEN"] = token
    return token


def resolve_hf_revision(repo_id: str, token: str | None = None) -> str:
    """Resolve the current Hub revision to an immutable commit SHA."""
    from huggingface_hub import HfApi

    info = HfApi().model_info(repo_id, revision="main", token=token)
    revision = getattr(info, "sha", None)
    if not isinstance(revision, str) or len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
        raise ValueError("MODEL_REVISION_UNRESOLVED: Hugging Face Hub did not return an immutable commit SHA")
    return revision
