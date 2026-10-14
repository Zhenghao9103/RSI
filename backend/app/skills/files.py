"""Immutable skill snapshots and a recoverable, human-readable active cache."""

import json
import os
import re
import tempfile
from pathlib import Path

import yaml


class FileStore:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, value: str | Path) -> Path:
        relative = Path(value)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("path must stay within the skill-pool root")
        candidate = self.root / relative
        for part in (candidate, *candidate.parents):
            if part == self.root:
                break
            if part.is_symlink():
                raise ValueError("symlinks are not allowed in managed paths")
        resolved = candidate.resolve()
        if not resolved.is_relative_to(self.root):
            raise ValueError("path must stay within the skill-pool root")
        return candidate

    @staticmethod
    def _name(value: str) -> str:
        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", value) or len(value) > 64:
            raise ValueError("invalid managed skill/reference name")
        return value

    @staticmethod
    def _content(kind: str, entity: dict) -> str:
        body = entity.get("body", "").rstrip() + "\n"
        if kind in {"skills", "candidates"}:
            header = yaml.safe_dump(
                {"name": entity["name"], "description": entity["description"]},
                allow_unicode=True,
                sort_keys=False,
            )
            return "---\n" + header + "---\n\n" + body
        return body

    def prepare(self, state: dict) -> None:
        for kind in ("skills", "references", "candidates"):
            for entity in state.get(kind, {}).values():
                if not entity.get("file_path"):
                    continue
                target = self._path(entity["file_path"])
                content = self._content(kind, entity)
                if target.exists():
                    if target.read_text(encoding="utf-8") != content:
                        raise ValueError("immutable snapshot already exists with different content")
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(
                    mode="w", encoding="utf-8", newline="\n", dir=target.parent, delete=False
                ) as stream:
                    temporary = Path(stream.name)
                    stream.write(content)
                    stream.flush()
                    os.fsync(stream.fileno())
                try:
                    # Exclusive atomic publication; a crash cannot leave a partial snapshot.
                    os.link(temporary, target)
                except FileExistsError:
                    if target.read_text(encoding="utf-8") != content:
                        raise ValueError(
                            "immutable snapshot already exists with different content"
                        ) from None
                finally:
                    temporary.unlink(missing_ok=True)

    def _atomic_write(self, target: Path, content: str) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=target.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(content)
        try:
            os.replace(temporary, target)
        finally:
            if temporary.exists():
                temporary.unlink()

    def materialize(self, state: dict) -> None:
        desired = {}
        for skill in state["skills"].values():
            if skill.get("status") != "active":
                continue
            name = self._name(skill["name"])
            desired[f"skills/{name}/SKILL.md"] = self.load(skill["file_path"])
            for reference in state["references"].values():
                if reference.get("skill_id") == skill["id"] and reference.get("status") in (
                    "active",
                    "promoted",
                ):
                    ref_name = self._name(reference["name"])
                    desired[f"skills/{name}/references/{ref_name}.md"] = self.load(
                        reference["file_path"]
                    )
        manifest = self._path(".managed-files.json")
        previous = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else []
        if not isinstance(previous, list) or any(not isinstance(item, str) for item in previous):
            raise ValueError("invalid managed-file manifest")
        previous_targets = {}
        for relative in previous:
            # Validate the complete manifest before changing any cache file.
            if Path(relative).as_posix() != relative:
                raise ValueError("invalid managed-file manifest")
            parts = Path(relative).parts
            allowed = len(parts) == 3 and parts[0] == "skills" and parts[2] == "SKILL.md"
            allowed |= (
                len(parts) == 4
                and parts[0] == "skills"
                and parts[2] == "references"
                and parts[3].endswith(".md")
            )
            if not allowed:
                raise ValueError("invalid managed-file manifest")
            self._name(parts[1])
            if len(parts) == 4:
                self._name(Path(parts[3]).stem)
            previous_targets[relative] = self._path(relative)
        desired_targets = {relative: self._path(relative) for relative in desired}
        for relative, content in desired.items():
            target = desired_targets[relative]
            if not target.exists() or target.read_bytes() != content.encode("utf-8"):
                self._atomic_write(target, content)
        for relative in set(previous) - desired.keys():
            target = previous_targets[relative]
            if target.exists():
                target.unlink()
        if previous != sorted(desired):
            self._atomic_write(manifest, json.dumps(sorted(desired), ensure_ascii=False))

    def load(self, path: str | Path) -> str:
        return self._path(path).read_text(encoding="utf-8")
