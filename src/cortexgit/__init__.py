# src/cortexgit/__init__.py

from cortexgit.core.memory import CortexGit
from cortexgit.core.event_log import EventLog
from cortexgit.core.entity_registry import EntityRegistryHandler


__version__ = "0.1.0"
__author__ = "CortexGit Contributors"

__all__ = [
    "CortexGit",
    "EventLog",
    "EntityRegistryHandler",
]
