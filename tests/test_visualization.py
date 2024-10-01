import pytest
from hextractor.visualization import VisualizationBuilder
import hextractor.structures as structures
import torch_geometric.data as pyg_data
import torch as th


def test_check_config_with_graph_valid():
    vis_config = structures.VisualizationConfig(
        node_types=["type1", "type2"],
        edge_types=[("type1", "rel", "type2")],
        edge_weights_attr_idx={("type1", "rel", "type2"): 0},
    )
    hetero_g = pyg_data.HeteroData()
    hetero_g["type1"].x = []
    hetero_g["type2"].x = []
    hetero_g[("type1", "rel", "type2")].edge_index = th.tensor(
        [[0, 0, 0, 1], [1, 2, 3, 2]]
    )
    hetero_g[("type1", "rel", "type2")].edge_attr = th.tensor([1.0] * 4).unsqueeze(-1)

    # This should not raise any exceptions
    VisualizationBuilder.check_config_with_graph(
        vis_config, hetero_g, node_labels_mapping={}
    )


def test_check_config_with_graph_missing_node_type():
    vis_config = structures.VisualizationConfig(
        node_types=["type1", "type2"],
        edge_types=[("type1", "rel", "type2")],
        edge_weights_attr_idx={("type1", "rel", "type2"): 0},
    )
    hetero_g = pyg_data.HeteroData()
    hetero_g["type1"].x = []

    with pytest.raises(ValueError, match="Node type type2 is missing."):
        VisualizationBuilder.check_config_with_graph(
            vis_config, hetero_g, node_labels_mapping={}
        )


def test_check_config_with_graph_missing_edge_type():
    vis_config = structures.VisualizationConfig(
        node_types=["type1", "type2"],
        edge_types=[("type1", "rel", "type2")],
        edge_weights_attr_idx={("type1", "rel", "type2"): 0},
    )
    hetero_g = pyg_data.HeteroData()
    hetero_g["type1"].x = []
    hetero_g["type2"].x = []

    with pytest.raises(
        ValueError, match="Edge type \('type1', 'rel', 'type2'\) is missing."
    ):
        VisualizationBuilder.check_config_with_graph(
            vis_config, hetero_g, node_labels_mapping={}
        )


def test_check_config_with_graph_missing_edge_attr():
    vis_config = structures.VisualizationConfig(
        node_types=["type1", "type2"],
        edge_types=[("type1", "rel", "type2")],
        edge_weights_attr_idx={("type1", "rel", "type2"): 1},
    )
    hetero_g = pyg_data.HeteroData()
    hetero_g["type1"].x = []
    hetero_g["type2"].x = []
    hetero_g[("type1", "rel", "type2")].edge_index = th.tensor(
        [[0, 0, 0, 1], [1, 2, 3, 2]]
    )
    hetero_g[("type1", "rel", "type2")].edge_attr = th.tensor([1.0] * 4).unsqueeze(-1)

    with pytest.raises(
        ValueError,
        match="Edge attribute 1 is missing for \('type1', 'rel', 'type2'\).",
    ):
        VisualizationBuilder.check_config_with_graph(
            vis_config, hetero_g, node_labels_mapping={}
        )
