from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


def compute_stable_hash(*values: object) -> str:
    payload = json.dumps(values, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def compute_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_stable_id(prefix: str, *values: object, length: int = 24) -> str:
    return f"{prefix}-{compute_stable_hash(*values)[:length]}"


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
