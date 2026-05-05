"""Load a YAML config file into nested dataclasses."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import yaml

from memedit.core.config import (
    AttributionConfig,
    DeleteConfig,
    InsertConfig,
    MemEditConfig,
    MemoryModuleConfig,
    ModifyConfig,
    MoMEConfig,
)


def load_config(path: Union[str, Path]) -> MemEditConfig:
    """Load a YAML file and build a MemEditConfig from it."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    return _dict_to_config(raw)


def _dict_to_config(d: dict) -> MemEditConfig:
    d = dict(d)
    memory = MemoryModuleConfig(**d.pop("memory", {}))
    attribution = AttributionConfig(**d.pop("attribution", {}))
    insert = InsertConfig(**d.pop("insert", {}))
    modify = ModifyConfig(**d.pop("modify", {}))
    delete = DeleteConfig(**d.pop("delete", {}))
    mome = MoMEConfig(**d.pop("mome", {}))
    return MemEditConfig(
        memory=memory,
        attribution=attribution,
        insert=insert,
        modify=modify,
        delete=delete,
        mome=mome,
        **d,
    )
