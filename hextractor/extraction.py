"""Main part of the tool, containing the extraction functions."""

from typing import Tuple
import pandas as pd
import torch_geometric.data as pyg_data
import hextractor.utils as hx_utils


def extract_nodes_wide(
    data: pd.DataFrame,
    node_type_params: Tuple[hx_utils.NodeTypeParams],
    edge_type_params: Tuple[hx_utils.EdgeTypeParams],
) -> pyg_data.HeteroData:
    """Extracts heterogeneous data from the input DataFrame in a wide format: where each row contains
    all the information about multiple nodes and edges.

    Parameters
    ----------
    data : pd.DataFrame
        Tabular data - source of the extracted information.
    node_type_params : Tuple[hx_utils.NodeTypeParams]
        Specs for the node extraction.
    edge_type_params : Tuple[hx_utils.EdgeTypeParams]
        Specs for the edge extraction.

    Returns
    -------
    pyg_data.HeteroData
        Constructed heterogeneous graph data.
    """
    pass


def extract_data_from_sources(
    data_sources: Tuple[hx_utils.DataFrameSource, ...],
) -> pyg_data.HeteroData:
    """Extracts heterogeneous data from the input DataFrameSources.

    Parameters
    ----------
    data_sources : Tuple[hx_utils.DataFrameSource]
        DataFrameSources containing the data and the extraction specs.

    Returns
    -------
    pyg_data.HeteroData
        Constructed heterogeneous graph data.
    """
    all_nodes_data = {}
    all_edges = {}
    for source in data_sources:
        nodes = source.extract_nodes_data()
        all_nodes_data = all_nodes_data | nodes.nodes_data
        edges = source.extract_edges_data()
        all_edges = all_edges | edges.edges_data
    hetero_data = pyg_data.HeteroData()
    for node_type, node_info in all_nodes_data.items():
        hetero_data[node_type].x = node_info.node_data
        if node_info.has_target:
            hetero_data[node_type].y = node_info.target_data

    for edge, edge_info in all_edges.items():
        hetero_data[edge].edge_index = edge_info.edge_index

        if edge_info.has_edge_attr:
            hetero_data[edge].edge_attr = edge_info.edge_attr

        if edge_info.has_target:
            hetero_data[edge].y = edge_info.target_data

    return hetero_data
