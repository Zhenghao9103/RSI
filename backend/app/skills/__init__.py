"""Standalone skill management. Importing this package starts no service."""

from .config import SkillPoolSettings

__all__ = ["SkillManager", "SkillPoolSettings"]


def __getattr__(name):
    if name == "SkillManager":
        from .manager import SkillManager

        return SkillManager
    raise AttributeError(name)
