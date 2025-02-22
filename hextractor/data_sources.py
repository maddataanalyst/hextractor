"""# Data sources module
A module contains utility classes and tools, needed to work with the hextractor package."""

from abc import ABC
from typing import Tuple

import numpy as np
import torch as th
import pandas as pd

import hextractor.structures as structures

TYPE_NAME_2_TORCH = {"float": th.float, "long": th.long, "int": th.int}


class DataSourceSpecs(ABC):
    """Base class for the data source specifications. It specifies how the data should be extracted from a single source."""

    def __init__(
        self,
        name: str,
        node_params: Tuple[structures.NodeTypeParams],
        edge_params: Tuple[structures.EdgeTypeParams],
        squeeze_single_dims: bool = True,
        **kwargs,
    ):
        self.name = name
        self.node_params = node_params
        self.edge_params = edge_params
        self.squeeze_single_dims = squeeze_single_dims
        self.validate()
        self._build_helper_lookups()

    def validate(self):
        if not self.node_params and not self.edge_params:
            raise ValueError("At least one node or edge parameter must be specified")

    def _build_helper_lookups(self):
        self.nodetype2id = {}
        self.id2nodetype = {}
        self.nodetype2multival = {}
        for node_param in self.node_params:
            if node_param.id_col:
                self.nodetype2id[node_param.node_type_name] = node_param.id_col
                self.id2nodetype[node_param.id_col] = node_param.node_type_name
                self.nodetype2multival[
                    node_param.node_type_name
                ] = node_param.multivalue_source

    def extract(self):
        raise NotImplementedError("Method 'extract' must be implemented in a subclass")

    def extract_nodes_data(self) -> structures.NodesData:
        raise NotImplementedError(
            "Method 'extract_node_data' must be implemented in a subclass"
        )

    def extract_edges_data(self) -> structures.EdgesData:
        raise NotImplementedError(
            "Method 'extract_edge_data' must be implemented in a subclass"
        )


class GraphSpecs:
    """Main class for the graph specs. It specifies how a SINGLE heterogeneous graph should be constructed from one or more sources."""

    def __init__(self, data_sources: Tuple[DataSourceSpecs, ...]):
        self.data_sources = data_sources
        self.validate_consistency()

    def validate_consistency(self):
        """Checks if the constructed graph is consistent. Namely:
        1. No duplicated node types.
        2. No duplicated edges.
        3. All node types are present in the edge types.

        Raises
        ------
        ValueError
            If the graph nodes or edges are duplicated or some node type is missing (it doesn't have attributes).
        """
        unique_node_types = set()
        unique_edge_types = set()
        for data_source in self.data_sources:
            for node_param in data_source.node_params:
                if node_param.node_type_name in unique_node_types:
                    raise ValueError(
                        f"Node type {node_param.node_type_name} is duplicated"
                    )
                unique_node_types.add(node_param.node_type_name)
            for edge_param in data_source.edge_params:
                edge_key = (
                    edge_param.source_name,
                    edge_param.edge_type_name,
                    edge_param.target_name,
                )
                if edge_key in unique_edge_types:
                    raise ValueError(f"Edge {edge_key} is duplicated")
                unique_edge_types.add(edge_key)

        for source_node, _, target_node in unique_edge_types:
            if source_node not in unique_node_types:
                raise ValueError(f"Node type {source_node} is missing.")
            if target_node not in unique_node_types:
                raise ValueError(f"Node type {target_node} is missing.")


class DataFrameSpecs(DataSourceSpecs):
    """Concrete implementation of the DataSourceSpecs for the pandas DataFrame."""

    def __init__(
        self,
        name: str,
        node_params: Tuple[structures.NodeTypeParams] = tuple(),
        edge_params: Tuple[structures.EdgeTypeParams] = tuple(),
        data_frame: pd.DataFrame = pd.DataFrame(),
        squeeze_single_dims: bool = True,
    ):
        super().__init__(name, node_params, edge_params, squeeze_single_dims)
        self.data_frame = data_frame

    def _process_multivalue_node(
        self, node_param: structures.NodeTypeParams
    ) -> Tuple[th.Tensor, th.Tensor]:
        """This function processes nodes that are described as multi-0valued vectors: e.g. list of tags.
        If in a single cell or row there are multiple values, they are exploded and the unique values are extracted.
        For example:
        Job offer A has tags X, Y, Z, Q --> then separate nodes for tags X, Y, Z ,Q will be created and unique IDs
        will be assigned to nodes e.g.: X -> 0, Y -> 1, Z -> 2, Q -> 3

        Parameters
        ----------
        node_param : structures.NodeTypeParams
            Node type parameters.

        Returns
        -------
        Tuple[th.Tensor, th.Tensor]
            Unique node ids tensor and None (no labels for multivalue nodes).

        Raises
        ------
        ValueError
            Raises error if node values are not numeric.
        """
        source_column = self.data_frame[node_param.id_col]
        all_possible_values = source_column.explode().unique().astype(int)
        if all_possible_values.dtype == "object":
            raise ValueError("Multi-value source must have a numeric unique values!")
        max_val = all_possible_values.max()
        unique_ids_tensor = th.arange(max_val + 1)
        if not self.squeeze_single_dims:
            unique_ids_tensor = th.atleast_2d(unique_ids_tensor)
        return unique_ids_tensor, None

    def _process_standard_node_with_attributes(
        self, node_param: structures.NodeTypeParams
    ) -> Tuple[th.Tensor, th.Tensor]:
        """A function that processes a standard node, described by a matrix of attribute values.

        Parameters
        ----------
        node_param : structures.NodeTypeParams
            Node parameter specification.

        Returns
        -------
        Tuple[th.Tensor, th.Tensor]
            A tuple of:
            1. All node attributes tensor. Dimensionality: (max node id + 1, Number of attributes)
            2. Node labels tensor. Dimensionality: (max node id + 1) or None if no labels are specified.

        Raises
        ------
        ValueError
            Raises error if not all attributes are numeric or node ids are duplicated and have different attributes
            in different instances. E.g. node with id 1 is repeated twice, first time: it has a set of attributes X,
            second time: X'.
        """

        # Step 1: Get the node-specific data, de-duplicate and sort by id
        # Dim: (Batch size, Number of attributes + id + target(if specified))
        target_col_lst = [node_param.label_col] if node_param.label_col else []
        node_df = (
            self.data_frame[
                [node_param.id_col, *node_param.attributes] + target_col_lst
            ]
            .drop_duplicates()
            .sort_values(by=node_param.id_col)
        )
        node_attr_df = (
            node_df.drop(columns=node_param.label_col)
            if node_param.label_col
            else node_df
        )

        # Check if all attributes are numeric
        attr_cols_to_use = list(node_param.attributes)
        if node_param.id_as_attr:
            attr_cols_to_use.append(node_param.id_col)
        node_df_attr_subset = node_attr_df[attr_cols_to_use]
        if (
            node_df_attr_subset.select_dtypes(include=np.number).shape[1]
            != node_df_attr_subset.shape[1]
        ):
            raise ValueError("Not all attributes are numeric")

        # Step 2: Allocate the tensor for all nodes and their attributes. Number of nodes = max id + 1
        node_ids = node_attr_df[node_param.id_col].values
        max_node_id = node_ids.max()

        id_counts = pd.Series(node_ids).value_counts().max()
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
        all_node_attrs_tensor = th.zeros((max_node_id + 1, len(attr_cols_to_use))).to(
            TYPE_NAME_2_TORCH[node_param.attr_type]
        )

        # 2b: turn dataframe to tensor with node attributes

        # Dim: (Batch size, Number of attributes)
        node_attrs_tensor = th.tensor(
            node_df_attr_subset.values,
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

        node_labels = None

        # Step 3: If target column is specified, extract the target values
        if node_param.label_col:
            # Dim: (max node id + 1)
            node_labels = th.zeros((max_node_id + 1)).to(th.long)
            node_labels.scatter_add_(
                0,
                node_indices_observed,
                th.tensor(node_df[node_param.label_col].values, dtype=th.long),
            )

        if self.squeeze_single_dims:
            all_node_attrs_tensor = th.squeeze(all_node_attrs_tensor)
            if node_labels is not None:
                node_labels = th.squeeze(node_labels)
        else:
            all_node_attrs_tensor = th.atleast_2d(all_node_attrs_tensor)
            if node_labels is not None:
                node_labels = th.atleast_2d(node_labels)
        return all_node_attrs_tensor, node_labels

    def process_node_param(
        self, node_param: structures.NodeTypeParams
    ) -> Tuple[th.Tensor, th.Tensor]:
        """Processes a single node param, according to its type: either as a standard node with attributes or as a multi-value node.

        Parameters
        ----------
        node_param : structures.NodeTypeParams
            Node type parameters.

        Returns
        -------
        Tuple[th.Tensor, th.Tensor]
            A tuple with two tensors:
            1. Node attributes tensor.
            2. Node labels tensor or None if no labels are specified.
        """
        if node_param.multivalue_source:
            return self._process_multivalue_node(node_param)
        return self._process_standard_node_with_attributes(node_param)

    def extract_nodes_data(self) -> structures.NodesData:
        node_results = {}
        for node_param in self.node_params:
            node_type_data, node_type_labels = self.process_node_param(node_param)
            node_results[node_param.node_type_name] = structures.NodeData(
                node_param.node_type_name, node_type_data, node_type_labels
            )
        return structures.NodesData(node_results)

    def _extract_multivalue_edge_index(
        self, edge_info: structures.EdgeTypeParams, multivalue_col: str
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
        source_to_target_df = (
            self.data_frame[[edge_info.source_id_col, edge_info.target_id_col]]
            .explode(multivalue_col)
            .drop_duplicates()
            .dropna()
        ).sort_values(by=[edge_info.source_id_col, edge_info.target_id_col])
        edge_index = th.tensor(
            source_to_target_df.values.astype(int), dtype=th.long
        ).t()

        return edge_index

    def _extract_edge_index(
        self, edge_info: structures.EdgeTypeParams
    ) -> Tuple[th.Tensor, pd.DataFrame]:
        """Extracts edge index from either:
        1. single-value column to single-value column
        2. multi-value columns to/from single-value column

        Parameters
        ----------
        edge_info : EdgeTypeParams
            Specification of the edge type.

        Returns
        -------
        Tuple[th.Tensor, pd.DataFrame]
            Tuple with:
            1. Edge index tensor. Dim: (2, Number of edges)
            2. Edge attributes DataFrame.
        """
        multival_col = (
            edge_info.source_id_col
            if edge_info.multivalue_source
            else edge_info.target_id_col
        )
        source_target_df = (
            self.data_frame[
                [
                    edge_info.source_id_col,
                    edge_info.target_id_col,
                    *edge_info.attributes,
                ]
            ]
            .explode(multival_col)
            .drop_duplicates()
            .sort_values(by=[edge_info.source_id_col, edge_info.target_id_col])
        )
        if edge_info.multivalue_source or edge_info.multivalue_target:
            edge_index = self._extract_multivalue_edge_index(edge_info, multival_col)
        else:
            edge_index = th.tensor(
                source_target_df[
                    [edge_info.source_id_col, edge_info.target_id_col]
                ].values,
                dtype=th.long,
            ).t()
        return edge_index, source_target_df

    def extract_edges_data(self) -> structures.EdgesData:
        """Extracts edges data from the data frame. Edge data has format specific for PyTorch Geometric hetero graphs:
        a dictionary with keys = edge types (eg. company-has-employee), and values are tensors with shape: (2 x Number of edges) for edge index.
        Edges are formatted as the adjacecy list: first dimension = 2, describes at index 0 - sender/source, at index 1 - receiver/target.
        So, for example if node 0 --sends to-->1, 0 -- sends to --> 2, 1 -- sends to --> 3, the edge index tensor will be:
        [[0, 0, 1], [1, 2, 3]].

        Returns
        -------
        structures.EdgesData
            Helper structure, that describes edges data.
        """
        edges_results = {}
        for edge_info in self.edge_params:
            key = (
                edge_info.source_name,
                edge_info.edge_type_name,
                edge_info.target_name,
            )
            edge_index, source_target_df = self._extract_edge_index(edge_info)

            attrs = None
            targets = None
            if edge_info.attributes:
                attrs = th.tensor(
                    source_target_df[list(edge_info.attributes)].values,
                    dtype=TYPE_NAME_2_TORCH[edge_info.attr_type],
                )
            if edge_info.label_col:
                targets = th.tensor(
                    source_target_df[edge_info.label_col].values, dtype=th.long
                )
            edges_results[key] = structures.EdgeData(
                edge_info.source_name,
                edge_info.edge_type_name,
                edge_info.target_name,
                edge_index,
                attrs,
                targets,
            )
        return structures.EdgesData(edges_results)
