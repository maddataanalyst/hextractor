"""A module contains utility classes and tools, needed to work with the hextractor package."""

from abc import ABC
from typing import Tuple, Literal, Dict
from pydantic import BaseModel, model_validator

import numpy as np
import torch as th
import pandas as pd


TYPE_NAME_2_TORCH = {"float": th.float, "long": th.long, "int": th.int}


class NodeTypeParams(BaseModel):
    """Node type specs, used during the extraction process"""

    node_type_name: str
    id_col: str = None
    target_col: str = None
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
            if self.target_col:
                raise ValueError("Multivalue source does not support target column")


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
        self.nodetype2multival = {}
        for node_param in self.node_params:
            if node_param.id_col:
                self.nodetype2id[node_param.node_type_name] = node_param.id_col
                self.id2nodetype[node_param.id_col] = node_param.node_type_name
                self.nodetype2multival[node_param.node_type_name] = (
                    node_param.multivalue_source
                )

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

    def _process_multivalue_node(
        self, node_param: NodeTypeParams
    ) -> Tuple[th.Tensor, th.Tensor]:
        source_column = self.data_frame[node_param.id_col]
        all_possible_values = source_column.explode().unique().astype(int)
        if all_possible_values.dtype == "object":
            raise ValueError("Multi-value source must have a numeric unique values!")
        max_val = all_possible_values.max()
        unique_ids_tensor = th.arange(max_val + 1)
        return unique_ids_tensor, None

    def _process_standard_node_with_attributes(
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

        node_targets = None

        # Step 3: If target column is specified, extract the target values
        if node_param.target_col:
            # Dim: (max node id + 1)
            node_targets = th.zeros((max_node_id + 1)).to(th.long)
            node_targets.scatter_add_(
                0,
                node_indices_observed,
                th.tensor(node_df[node_param.target_col].values, dtype=th.long),
            )

        return all_node_attrs_tensor, node_targets

    def process_node_param(
        self, node_param: NodeTypeParams
    ) -> Tuple[th.Tensor, th.Tensor]:
        if node_param.multivalue_source:
            return self._process_multivalue_node(node_param)
        return self._process_standard_node_with_attributes(node_param)

    def extract_nodes_data(self) -> NodesData:
        node_results = {}
        for node_param in self.node_params:
            node_type_data, node_type_targets = self.process_node_param(node_param)
            node_results[node_param.node_type_name] = NodeData(
                node_param.node_type_name, node_type_data, node_type_targets
            )
        return NodesData(node_results)

    def _extract_multivalue_edge_index(
        self, edge_info: EdgeTypeParams, multivalue_col: str
    ) -> th.Tensor:
        """Extracts the edge index for the multivalue source. Only one
        of the source or target can be multivalue.

        Parameters
        ----------
        edge_info : EdgeTypeParams
            Specification of the edge type.

        multivalue_col: str
            Name of the multivalue column.

        Returns
        -------
        th.Tensor
            Edge index tensor: dim: (2, Number of edges)
        """
        source_id_col = self.nodetype2id[edge_info.source_name]
        target_id_col = self.nodetype2id[edge_info.target_name]

        source_to_target_df = (
            self.data_frame[[source_id_col, target_id_col]]
            .explode(multivalue_col)
            .drop_duplicates()
            .dropna()
        ).sort_values(by=[source_id_col, target_id_col])
        edge_index = th.tensor(
            source_to_target_df.values.astype(int), dtype=th.long
        ).t()

        return edge_index

    def _extract_edge_index(
        self,
        edge_info: EdgeTypeParams,
        source_id_col: str,
        target_id_col: str,
    ) -> Tuple[th.Tensor, pd.DataFrame]:
        """Extracts edge index from either:
        1. single-value column to single-value column
        2. multi-value columns to/from single-value column

        Parameters
        ----------
        edge_info : EdgeTypeParams
            Specification of the edge type.
        source_id_col : str
            Name of the source column.
        target_id_col : str
            Name of the target column.

        Returns
        -------
        Tuple[th.Tensor, pd.DataFrame]
            Tuple with:
            1. Edge index tensor. Dim: (2, Number of edges)
            2. Edge attributes DataFrame.
        """
        multival_col = (
            source_id_col
            if self.nodetype2multival[edge_info.source_name]
            else target_id_col
        )
        source_target_df = (
            self.data_frame[[source_id_col, target_id_col, *edge_info.attributes]]
            .explode(multival_col)
            .drop_duplicates()
            .sort_values(by=[source_id_col, target_id_col])
        )
        if (
            self.nodetype2multival[edge_info.source_name]
            or self.nodetype2multival[edge_info.target_name]
        ):
            edge_index = self._extract_multivalue_edge_index(edge_info, multival_col)
        else:
            edge_index = th.tensor(
                source_target_df[[source_id_col, target_id_col]].values,
                dtype=th.long,
            ).t()
        return edge_index, source_target_df

    def extract_edges_data(self) -> EdgesData:
        edges_results = {}
        for edge_info in self.edge_params:
            source_id_col = self.nodetype2id[edge_info.source_name]
            target_id_col = self.nodetype2id[edge_info.target_name]

            key = (
                edge_info.source_name,
                edge_info.edge_type_name,
                edge_info.target_name,
            )
            if (
                self.nodetype2multival[edge_info.source_name]
                and self.nodetype2multival[edge_info.target_name]
            ):
                raise NotImplementedError(
                    "Multivalue source for both source and target is not allowed"
                )

            edge_index, source_target_df = self._extract_edge_index(
                edge_info, source_id_col, target_id_col
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
