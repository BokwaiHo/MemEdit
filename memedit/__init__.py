"""MemEdit: Fine-grained CRUD operations over parametric agent memory."""

from memedit.core.config import MemEditConfig, MemoryModuleConfig, MoMEConfig
from memedit.core.editor import MemEditor
from memedit.models.mlp_memory import MLPMemory, InterpolatedMemoryLM
from memedit.data.trace import MemoryTrace, EditOperation, OperationType
from memedit.mome.shard_manager import MoMEShardManager
from memedit.operations import (
    query_memory,
    insert_memory,
    modify_memory,
    delete_memory,
)

__version__ = "0.1.0"

__all__ = [
    "MemEditConfig",
    "MemoryModuleConfig",
    "MoMEConfig",
    "MemEditor",
    "MLPMemory",
    "InterpolatedMemoryLM",
    "MemoryTrace",
    "EditOperation",
    "OperationType",
    "MoMEShardManager",
    "query_memory",
    "insert_memory",
    "modify_memory",
    "delete_memory",
]
