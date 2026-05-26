# src/cortexgit/__init__.py

from cortexgit.core.memory import CortexGit
from cortexgit.core.event_log import EventLog
from cortexgit.core.entity_registry import EntityRegistry


__version__ = "0.1.0"
__author__ = "Antigravity"

__all__ = [
    "CortexGit",
    "EventLog",
    "EntityRegistry",
]
