"""A module contains utility classes and tools, needed to work with the hextractor package.
"""

from abc import ABC
from typing import Tuple, Literal
from pydantic import BaseModel

import numpy as np
import torch as th
import pandas as pd
import torch_geometric.data as pyg_data


TYPE_NAME_2_TORCH = {
    'float': th.float,
    'long': th.long,
    'int': th.int
}

class NodeTypeParams(BaseModel):
    """Node type specs, used during the extraction process"""

    node_type_name: str
    source_name: str
    id_col: str = None
    target_col: str = None
    multivalue_source: bool = False
    attributes: Tuple[str, ...] = tuple()
    attr_type: Literal['float', 'long'] = 'float'


class EdgeTypeParams(BaseModel):
    """Edge type specs, used during the extraction process"""

    edge_type_name: str
    source_name: str
    target_name: str
    multivalue_source: bool = False
    attributes: Tuple[str, ...] = tuple()
    attr_type: Literal['float', 'long', 'int'] = 'float'


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

    def extract_using_id(self) -> pyg_data.HeteroData:
        node_data = {}
        node_targets = {}
        edges = {}
        edge_attrs = {}
        for node_param in self.node_params:
            
            node_df = (
                self.data_frame[[node_param.id_col, *node_param.attributes]]
                .drop_duplicates()
                .sort_values(by=node_param.id_col)
            )

            # Check if all attributes are numeric
            node_df_attr_subset = node_df[list(node_param.attributes)]
            if (
                node_df_attr_subset.select_dtypes(include=np.number).shape[1]
                != node_df_attr_subset.shape[1]
            ):
                raise ValueError("Not all attributes are numeric")

            # TODO: later on add various types per each column: some might be required to be floats, some others: long (e.g. for embeddings)
            node_data[node_param.node_type_name] = th.tensor(
                node_df.drop(columns=node_param.id_col).values, dtype=TYPE_NAME_2_TORCH[node_param.attr_type]
            )

            if node_param.target_col:
                node_targets[node_param.node_type_name] = th.tensor(
                    node_df[node_param.target_col].values, dtype=th.long
                )

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
                    (edge_info.source_name, edge_info.edge_type_name, edge_info.target_name)
                ] = th.tensor(
                    source_target_df[list(edge_info.attributes)].values,
                    dtype=TYPE_NAME_2_TORCH[edge_info.attr_type]
                )

        hetero_data = pyg_data.HeteroData()
        for node_type in node_data.keys():
            hetero_data[node_type].x = node_data[node_type]
            if node_type in node_targets:
                hetero_data[node_type].y = node_targets[node_type]

        for edge, edge_tensor in edges.items():
            hetero_data[edge].edge_index = edge_tensor

            if edge in edge_attrs:
                hetero_data[edge].edge_attr = edge_attrs[edge]

        return hetero_data
