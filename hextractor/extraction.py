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
