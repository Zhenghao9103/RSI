"""Paths for project artifacts: resolve symlinks before allowing writes."""
from pathlib import Path
import hashlib
import json
import os
import tempfile

ROOT = Path('/mnt/shared_data/skill-evolution')


def data_disk_path(path: str | Path) -> Path:
    p = Path(path).expanduser().resolve()
    if not p.is_relative_to(Path('/mnt')):
        raise ValueError(f'Output must be on the /mnt data disk: {p}')
    parent = p
    while not parent.exists():
        parent = parent.parent
    if os.stat(parent).st_dev != os.stat('/mnt').st_dev:
        raise ValueError(f'Output resolves to a different filesystem: {p}')
    return p


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value) -> None:
    path = data_disk_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, suffix='.json.tmp')
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)
