"""A module contains utility classes and tools, needed to work with the hextractor package."""

from abc import ABC
from typing import Tuple, Literal, Dict
from pydantic import BaseModel

import numpy as np
import torch as th
import pandas as pd
import torch_geometric.data as pyg_data


TYPE_NAME_2_TORCH = {"float": th.float, "long": th.long, "int": th.int}


class NodeTypeParams(BaseModel):
    """Node type specs, used during the extraction process"""

    node_type_name: str
    id_col: str = None
    target_col: str = None
    multivalue_source: bool = False
    attributes: Tuple[str, ...] = tuple()
    attr_type: Literal["float", "long"] = "float"


class EdgeTypeParams(BaseModel):
    """Edge type specs, used during the extraction process"""

    edge_type_name: str
    source_name: str
    target_name: str
    target_col: str = None
    multivalue_source: bool = False
    attributes: Tuple[str, ...] = tuple()
    attr_type: Literal["float", "long", "int"] = "float"


class NodeData:
    """Node data, extracted from the source"""

    def __init__(
        self, node_type_name: str, node_data: th.Tensor, target_data: th.Tensor = None
    ):
        self.node_type_name = node_type_name
        self.node_data = node_data
        self.target_data = target_data

    def has_target(self) -> bool:
        return self.target_data is not None


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
        target_data: th.Tensor = None,
    ):
        self.source_name = source_name
        self.edge_type_name = edge_type_name
        self.target_name = target_name
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.target_data = target_data

    @property
    def edge_name(self) -> Tuple[str, str, str]:
        return self.source_name, self.edge_type_name, self.target_name

    def has_target(self) -> bool:
        return self.target_data is not None

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


class DataSource(ABC):
    def __init__(
        self,
        name: str,
        node_params: Tuple[NodeTypeParams],
        edge_params: Tuple[EdgeTypeParams],
        **kwargs,
    ):
        self.name = name
        self.node_params = node_params
        self.edge_params = edge_params
        self.validate()
        self._build_helper_lookups()

    def validate(self):
        node_names = set([node_param.node_type_name for node_param in self.node_params])
        for edge in self.edge_params:
            if edge.source_name not in node_names:
                raise ValueError(
                    f"Node type {edge.source_name} not found in the node types"
                )
            if edge.target_name not in node_names:
                raise ValueError(
                    f"Node type {edge.target_name} not found in the node types"
                )

    def _build_helper_lookups(self):
        self.nodetype2id = {}
        self.id2nodetype = {}
        for node_param in self.node_params:
            if node_param.id_col:
                self.nodetype2id[node_param.node_type_name] = node_param.id_col
                self.id2nodetype[node_param.id_col] = node_param.node_type_name

    def extract(self):
        raise NotImplementedError("Method 'extract' must be implemented in a subclass")

    def extract_nodes_data(self) -> NodesData:
        raise NotImplementedError(
            "Method 'extract_node_data' must be implemented in a subclass"
        )

    def extract_edges_data(self) -> EdgesData:
        raise NotImplementedError(
            "Method 'extract_edge_data' must be implemented in a subclass"
        )


class DataFrameSource(DataSource):
    def __init__(
        self,
        name: str,
        node_params: Tuple[NodeTypeParams],
        edge_params: Tuple[EdgeTypeParams],
        data_frame: pd.DataFrame,
    ):
        super().__init__(name, node_params, edge_params)
        self.data_frame = data_frame

    def process_node_param(
        self, node_param: NodeTypeParams
    ) -> Tuple[th.Tensor, th.Tensor]:
        # Step 1: Get the node-specific data, de-duplicate and sort by id
        # Dim: (Batch size, Number of attributes + id + target(if specified))
        target_col_lst = [node_param.target_col] if node_param.target_col else []
        node_df = (
            self.data_frame[
                [node_param.id_col, *node_param.attributes] + target_col_lst
            ]
            .drop_duplicates()
            .sort_values(by=node_param.id_col)
        )
        node_attr_df = (
            node_df.drop(columns=node_param.target_col)
            if node_param.target_col
            else node_df
        )

        # Check if all attributes are numeric
        node_df_attr_subset = node_attr_df[list(node_param.attributes)]
        if (
            node_df_attr_subset.select_dtypes(include=np.number).shape[1]
            != node_df_attr_subset.shape[1]
        ):
            raise ValueError("Not all attributes are numeric")

        # Step 2: Allocate the tensor for all nodes and their attributes. Number of nodes = max id + 1
        node_ids = node_attr_df[node_param.id_col].values
        max_node_id = node_ids.max()

        id_counts = pd.value_counts(node_ids).max()
        if id_counts > 1:
            if not node_param.multivalue_source:
                nodes_with_duplicates = node_attr_df[
                    node_attr_df[node_param.id_col].duplicated(keep=False)
                ]
                raise ValueError(
                    f"Node IDs are not unique after de-duplication. Same node ID appears with different attrs. Duplicated nodes: {nodes_with_duplicates}"
                )

        # 2a: Allocate empty tensor for node attrs

        # Dim: (max node id + 1, Number of attributes)
        all_node_attrs_tensor = th.zeros(
            (max_node_id + 1, len(node_param.attributes))
        ).to(TYPE_NAME_2_TORCH[node_param.attr_type])

        # 2b: turn dataframe to tensor with node attributes

        # Dim: (Batch size, Number of attributes)
        node_attrs_tensor = th.tensor(
            node_attr_df.drop(columns=node_param.id_col).values,
            dtype=TYPE_NAME_2_TORCH[node_param.attr_type],
        )

        # 2c: expand indices and fill the tensor for all nodes with values. Important: as node ids are sorted and de-duplicated
        # the 'add' aggergatioion can be used - as each id will appear only once.

        # Dim: (Batch size)
        node_indices_observed = th.tensor(node_ids, dtype=th.long)

        # Dim: (max node id + 1, Number of attributes) -> only rows with observed indices will be filled with proper index
        indices_expanded = node_indices_observed.unsqueeze(1).expand_as(
            node_attrs_tensor
        )
        all_node_attrs_tensor = all_node_attrs_tensor.scatter_add_(
            0, indices_expanded, node_attrs_tensor
        )

        # Dim: (max node id + 1)
        node_targets = th.zeros((max_node_id + 1)).to(th.long)

        # Step 3: If target column is specified, extract the target values
        if node_param.target_col:
            node_targets.scatter_add_(
                0,
                node_indices_observed,
                th.tensor(node_df[node_param.target_col].values, dtype=th.long),
            )

        return all_node_attrs_tensor, node_targets

    def extract_nodes_data(self) -> NodesData:
        node_results = {}
        for node_param in self.node_params:
            node_type_data, node_type_targets = self.process_node_param(node_param)
            node_type_targets = None
            if node_param.target_col:
                node_type_targets = node_type_targets
            node_results[node_param.node_type_name] = NodeData(
                node_param.node_type_name, node_type_data, node_type_targets
            )
        return NodesData(node_results)

    def extract_edges_data(self) -> EdgesData:
        edges_results = {}
        for edge_info in self.edge_params:
            source_id_col = self.nodetype2id[edge_info.source_name]
            target_id_col = self.nodetype2id[edge_info.target_name]
            source_target_df = (
                self.data_frame[[source_id_col, target_id_col, *edge_info.attributes]]
                .drop_duplicates()
                .sort_values(by=[source_id_col, target_id_col])
            )
            edge_index = th.tensor(
                source_target_df[[source_id_col, target_id_col]].values, dtype=th.long
            ).t()

            key = (
                edge_info.source_name,
                edge_info.edge_type_name,
                edge_info.target_name,
            )
            attrs = None
            targets = None
            if edge_info.attributes:
                attrs = th.tensor(
                    source_target_df[list(edge_info.attributes)].values,
                    dtype=TYPE_NAME_2_TORCH[edge_info.attr_type],
                )
            if edge_info.target_col:
                targets = th.tensor(
                    source_target_df[edge_info.target_col].values, dtype=th.long
                )
            edges_results[key] = EdgeData(
                edge_info.source_name,
                edge_info.edge_type_name,
                edge_info.target_name,
                edge_index,
                attrs,
                targets,
            )
        return EdgesData(edges_results)

    def extract_using_id(self) -> pyg_data.HeteroData:
        node_data = {}
        node_targets = {}
        edges = {}
        edge_attrs = {}
        for node_param in self.node_params:
            node_type_data, node_type_targets = self.process_node_param(node_param)
            node_data[node_param.node_type_name] = node_type_data
            if node_param.target_col:
                node_targets[node_param.node_type_name] = node_type_targets

        # TODO: extract edge info to a separate method + add validation of max values
        for edge_info in self.edge_params:
            source_id_col = self.nodetype2id[edge_info.source_name]
            target_id_col = self.nodetype2id[edge_info.target_name]
            source_target_df = (
                self.data_frame[[source_id_col, target_id_col, *edge_info.attributes]]
                .drop_duplicates()
                .sort_values(by=[source_id_col, target_id_col])
            )
            edges[
                (edge_info.source_name, edge_info.edge_type_name, edge_info.target_name)
            ] = th.tensor(
                source_target_df[[source_id_col, target_id_col]].values, dtype=th.long
            ).t()

            if edge_info.attributes:
                edge_attrs[
                    (
                        edge_info.source_name,
                        edge_info.edge_type_name,
                        edge_info.target_name,
                    )
                ] = th.tensor(
                    source_target_df[list(edge_info.attributes)].values,
                    dtype=TYPE_NAME_2_TORCH[edge_info.attr_type],
                )

        hetero_data = pyg_data.HeteroData()
        for node_type, node_attrs in node_data.items():
            hetero_data[node_type].x = node_attrs
            if node_type in node_targets:
                hetero_data[node_type].y = node_targets[node_type]

        for edge, edge_tensor in edges.items():
            hetero_data[edge].edge_index = edge_tensor

            if edge in edge_attrs:
                hetero_data[edge].edge_attr = edge_attrs[edge]

        return hetero_data
