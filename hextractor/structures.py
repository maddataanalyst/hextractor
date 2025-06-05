"""A module contains data structures and helper models used across the HexTractor package."""

from typing import Tuple, Literal, Dict, Any
from pydantic import BaseModel, model_validator

import torch as th


class NodeTypeParams(BaseModel):
    """Node type specs, used during the extraction process"""

    node_type_name: str
    id_col: str = None
    label_col: str = None
    id_as_attr: bool = False
    multivalue_source: bool = False
    attributes: Tuple[str, ...] = tuple()
    attr_type: Literal["float", "long"] = "float"

    @model_validator(mode="after")
    def check_multivalue_source(self):
        if self.multivalue_source:
            if not self.id_col:
                raise ValueError(
                    "Multivalue source requires the id column to be specified"
                )
            if len(self.attributes) > 0:
                raise ValueError("Multivalue source does not support attributes")
            if self.label_col:
                raise ValueError("Multivalue source does not support target column")
        return self


class EdgeTypeParams(BaseModel):
    """Edge type specs, used during the extraction process"""

    edge_type_name: str
    source_name: str
    source_id_col: str
    target_name: str
    target_id_col: str
    label_col: str = None
    multivalue_source: bool = False
    multivalue_target: bool = False
    attributes: Tuple[str, ...] = tuple()
    attr_type: Literal["float", "long", "int"] = "float"

    @model_validator(mode="after")
    def check_multivalue(self):
        if self.multivalue_source and self.multivalue_target:
            raise ValueError("Multivalue source and target are not supported")
        return self


class NodeData:
    """Node data, extracted from the source"""

    def __init__(
        self, node_type_name: str, node_data: th.Tensor, label_data: th.Tensor = None
    ):
        self.node_type_name = node_type_name
        self.node_data = node_data
        self.label_data = label_data

    def has_target(self) -> bool:
        return self.label_data is not None


class NodesData:
    """Dictionary of multiple node data"""

    def __init__(self, nodes_data: Dict[str, NodeData]):
        self.nodes_data = nodes_data

    def has_node(self, node_type) -> bool:
        return node_type in self.nodes_data

    def get_node(self, node_type) -> NodeData:
        return self.nodes_data[node_type]


class EdgeData:
    """Edge data, extracted from the source"""

    def __init__(
        self,
        source_name: str,
        edge_type_name: str,
        target_name: str,
        edge_index: th.Tensor,
        edge_attr: th.Tensor = None,
        label_data: th.Tensor = None,
    ):
        self.source_name = source_name
        self.edge_type_name = edge_type_name
        self.target_name = target_name
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.label_data = label_data

    @property
    def edge_name(self) -> Tuple[str, str, str]:
        return self.source_name, self.edge_type_name, self.target_name

    def has_target(self) -> bool:
        return self.label_data is not None

    def has_edge_attr(self) -> bool:
        return self.edge_attr is not None


class EdgesData:
    """Dictionary of multiple edge data"""

    def __init__(self, edges_data: Dict[Tuple[str, str, str], EdgeData]):
        self.edges_data = edges_data

    def has_edge(self, edge_type: Tuple[str, str, str]) -> bool:
        return edge_type in self.edges_data

    def get_edge(self, edge_type: Tuple[str, str, str]) -> EdgeData:
        return self.edges_data[edge_type]


class VisualizationConfig(BaseModel):
    """Config for visualization of the heterogeneous graph"""

    node_types: Tuple[str, ...] = tuple()
    node_types_to_colors: Dict[str, str] = {}
    node_type_label_attr_name: Dict[str, str] = {}
    node_type_label_attr_idx: Dict[str, int] = {}
    edge_types: Tuple[Tuple[str, str, str], ...] = tuple()
    edge_type_to_colors: Dict[Tuple[str, str, str], str] = {}
    edge_type_weight_attr_name: Dict[Tuple[str, str, str], str] = {}
    edge_weights_attr_idx: Dict[Tuple[str, str, str], int] = {}
    default_node_color: str = "blue"
    default_edge_color: str = "black"
    default_edge_weight: int = 1
    default_edge_weight_attr: str = None
    notebook_visualization: bool = False
    select_menu: bool = True
    filter_menu: bool = True
    width: str = "1500px"
    height: str = "1500px"
    buttons: Tuple[Literal["physics", "layout", "interaction", "selection"], ...] = (
        "layout",
        "physics",
        "selection",
    )
    pyvis_additional_kwargs: Dict[str, Any] = {}

    @property
    def all_node_types(self):
        return set(list(self.node_types) + list(self.node_types_to_colors.keys()))

    @property
    def all_edge_types(self):
        return set(
            list(self.edge_types)
            + list(self.edge_type_to_colors.keys())
            + list(self.edge_weights_attr_idx.keys())
        )

    def get_node_color(self, node_type: str) -> str:
        return self.node_types_to_colors.get(node_type, self.default_node_color)

    def get_node_label_attr_name(self, node_type: str) -> str:
        return self.node_type_label_attr_name.get(node_type, None)

    def get_node_label_attr_idx(self, node_type: str) -> int:
        return self.node_type_label_attr_idx.get(node_type, None)

    def get_edge_weight_attr_idx(self, edge_type: Tuple[str, str, str]) -> int:
        return self.edge_weights_attr_idx.get(edge_type, self.default_edge_weight_attr)

    def get_edge_weight_attr_name(self, edge_type: Tuple[str, str, str]) -> str:
        return self.edge_type_weight_attr_name.get(
            edge_type, self.default_edge_weight_attr
        )

    def get_edge_color(self, edge_type: Tuple[str, str, str]) -> str:
        return self.edge_type_to_colors.get(edge_type, self.default_node_color)

    @model_validator(mode="after")
    def check_consistency(self):
        """Validates the consistency of the setup - all node and edge types should be selected,
        additionally: edge source/targets should match the node types.

        Raises
        ------
        ValueError
            If edge/node inconsistency is detected.
        """
        for edge_type in self.all_edge_types:
            source, _, target = edge_type
            if source not in self.all_node_types:
                raise ValueError(f"Node type {source} is not selected")
            elif target not in self.all_node_types:
                raise ValueError(f"Node type {target} is not selected")
        return self
