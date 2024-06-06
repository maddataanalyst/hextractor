"""A module contains utility classes and tools, needed to work with the hextractor package.
"""

from typing import Tuple
from pydantic import BaseModel


class NodeTypeParams(BaseModel):
    """Node type specs, used during the extraction process"""
    node_type_name: str
    source_name: str
    multivalue_source: bool = False
    attributes: Tuple[str] = tuple()


class EdgeTypeParams(BaseModel):
    """Edge type specs, used during the extraction process"""
    edge_type_name: str
    source_name: str
    target_name: str
    multivalue_source: bool = False
    attributes: Tuple[str] = tuple()
