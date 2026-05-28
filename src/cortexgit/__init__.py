# src/cortexgit/__init__.py

from cortexgit.core.memory import CortexGit
from cortexgit.core.event_log import EventLog
from cortexgit.core.entity_registry import EntityRegistryHandler

# Graph & REG module direct access exports
from cortexgit.graph.graph_repository import GraphRepository
from cortexgit.graph.importance import (
    calculate_importance,
    rank_nodes_by_importance,
    get_top_k_important_nodes,
    get_high_importance_nodes,
)
from cortexgit.graph.injection import inject_high_mass_nodes, inject_high_importance_nodes
from cortexgit.graph.expiration import expire_old_nodes


__version__ = "0.2.1"
__author__ = "CortexGit Contributors"

__all__ = [
    "CortexGit",
    "EventLog",
    "EntityRegistryHandler",
    "GraphRepository",
    "calculate_importance",
    "rank_nodes_by_importance",
    "get_top_k_important_nodes",
    "get_high_importance_nodes",
    "inject_high_mass_nodes",
    "inject_high_importance_nodes",
    "expire_old_nodes",
]
