import pytest
from hextractor.visualization import VisualizationBuilder
import hextractor.structures as structures
import torch_geometric.data as pyg_data
import torch as th


def test_check_config_with_graph_valid_no_label_mapping():
    # given
    vis_config = structures.VisualizationConfig(
        node_types=["type1", "type2"],
        edge_types=[("type1", "rel", "type2")],
        edge_type_weight_attr_name={("type1", "rel", "type2"): "edge_attr"},
        edge_weights_attr_idx={("type1", "rel", "type2"): 0},
    )
    hetero_g = pyg_data.HeteroData()
    hetero_g["type1"].x = []
    hetero_g["type2"].x = []
    hetero_g[("type1", "rel", "type2")].edge_index = th.tensor(
        [[0, 0, 0, 1], [1, 2, 3, 2]]
    )
    hetero_g[("type1", "rel", "type2")].edge_attr = th.tensor([1.0] * 4).unsqueeze(-1)

    # then
    try:
        VisualizationBuilder.check_config_with_graph(vis_config, hetero_g)
    except Exception:
        pytest.fail("check_config_with_graph() raised ValueError unexpectedly!")


def test_check_config_with_graph_valid_label_mapping():
    # given
    vis_config = structures.VisualizationConfig(
        node_types=["type1", "type2"],
        edge_types=[("type1", "rel", "type2")],
        edge_type_weight_attr_name={("type1", "rel", "type2"): "edge_attr"},
        edge_weights_attr_idx={("type1", "rel", "type2"): 0},
    )
    hetero_g = pyg_data.HeteroData()
    hetero_g["type1"].x = []
    hetero_g["type2"].x = []
    hetero_g[("type1", "rel", "type2")].edge_index = th.tensor(
        [[0, 0, 0, 1], [1, 2, 3, 2]]
    )
    hetero_g[("type1", "rel", "type2")].edge_attr = th.tensor([1.0] * 4).unsqueeze(-1)

    # then
    try:
        VisualizationBuilder.check_config_with_graph(vis_config, hetero_g)
    except Exception:
        pytest.fail("check_config_with_graph() raised ValueError unexpectedly!")


def test_check_config_with_graph_missing_node_type_no_label_mapping():
    # given
    vis_config = structures.VisualizationConfig(
        node_types=["type1", "type2"],
        edge_types=[("type1", "rel", "type2")],
        edge_weights_attr_idx={("type1", "rel", "type2"): 0},
    )
    hetero_g = pyg_data.HeteroData()
    hetero_g["type1"].x = []

    # then
    with pytest.raises(ValueError, match="Node type type2 is missing."):
        VisualizationBuilder.check_config_with_graph(vis_config, hetero_g)


def test_check_config_with_graph_missing_edge_type_no_label_mapping():
    # given
    vis_config = structures.VisualizationConfig(
        node_types=["type1", "type2"],
        edge_types=[("type1", "rel", "type2")],
        edge_weights_attr_idx={("type1", "rel", "type2"): 0},
    )
    hetero_g = pyg_data.HeteroData()
    hetero_g["type1"].x = []
    hetero_g["type2"].x = []

    # then
    with pytest.raises(
        ValueError, match="Edge type \('type1', 'rel', 'type2'\) is missing."
    ):
        VisualizationBuilder.check_config_with_graph(vis_config, hetero_g)


def test_check_config_with_graph_missing_edge_attr_no_label_mapping():
    # given
    vis_config = structures.VisualizationConfig(
        node_types=["type1", "type2"],
        edge_types=[("type1", "rel", "type2")],
        edge_type_weight_attr_name={("type1", "rel", "type2"): "edge_attr"},
        edge_weights_attr_idx={("type1", "rel", "type2"): 1},
    )
    hetero_g = pyg_data.HeteroData()
    hetero_g["type1"].x = []
    hetero_g["type2"].x = []
    hetero_g[("type1", "rel", "type2")].edge_index = th.tensor(
        [[0, 0, 0, 1], [1, 2, 3, 2]]
    )
    hetero_g[("type1", "rel", "type2")].edge_attr = th.tensor([1.0] * 4).unsqueeze(-1)

    # then
    with pytest.raises(
        ValueError,
        match="Edge attribute edge_attr is missing index: 1 for \('type1', 'rel', 'type2'\).",
    ):
        VisualizationBuilder.check_config_with_graph(vis_config, hetero_g)


def test_check_config_with_graph_correct_label_mapping():
    # given
    vis_config = structures.VisualizationConfig(
        node_types=["type1", "type2"],
        edge_types=[("type1", "rel", "type2")],
        edge_weights_attr_idx={("type1", "rel", "type2"): 0},
        edge_type_weight_attr_name={("type1", "rel", "type2"): "weight"},
        node_type_label_attr_name={"type1": "label", "type2": "label2"},
        node_type_label_attr_idx={"type1": 0, "type2": 0},
    )
    hetero_g = pyg_data.HeteroData()
    hetero_g["type1"].x = []
    hetero_g["type1"].label = th.tensor([[0], [1], [2]])
    hetero_g["type2"].x = []
    hetero_g["type2"].label2 = th.tensor([[0], [1], [2]])
    hetero_g[("type1", "rel", "type2")].edge_index = th.tensor(
        [[0, 0, 0, 1], [1, 2, 3, 2]]
    )
    hetero_g[("type1", "rel", "type2")].weight = th.tensor([1.0] * 4).unsqueeze(-1)

    # then
    try:
        VisualizationBuilder.check_config_with_graph(vis_config, hetero_g)
    except Exception:
        pytest.fail("check_config_with_graph() raised ValueError unexpectedly!")


def test_check_config_with_graph_invalid_label_mapping_no_attribute():
    # given
    vis_config = structures.VisualizationConfig(
        node_types=["type1", "type2"],
        edge_types=[("type1", "rel", "type2")],
        edge_weights_attr_idx={("type1", "rel", "type2"): 0},
        edge_type_weight_attr_name={("type1", "rel", "type2"): "weight"},
        node_type_label_attr_name={"type1": "label", "type2": "nonexisting_label"},
        node_type_label_attr_idx={"type1": 0, "type2": 0},
    )
    hetero_g = pyg_data.HeteroData()
    hetero_g["type1"].x = []
    hetero_g["type1"].label = th.tensor([[0], [1], [2]])
    hetero_g["type2"].x = []
    hetero_g["type2"].label2 = th.tensor([[0], [1], [2]])
    hetero_g[("type1", "rel", "type2")].edge_index = th.tensor(
        [[0, 0, 0, 1], [1, 2, 3, 2]]
    )
    hetero_g[("type1", "rel", "type2")].weight = th.tensor([1.0] * 4).unsqueeze(-1)

    # then
    with pytest.raises(
        ValueError,
        match="Node label attribute nonexisting_label is missing for node type type2.",
    ):
        VisualizationBuilder.check_config_with_graph(vis_config, hetero_g)


def test_check_config_with_graph_invalid_label_mapping_bad_index():
    # given
    vis_config = structures.VisualizationConfig(
        node_types=["type1", "type2"],
        edge_types=[("type1", "rel", "type2")],
        edge_weights_attr_idx={("type1", "rel", "type2"): 0},
        edge_type_weight_attr_name={("type1", "rel", "type2"): "weight"},
        node_type_label_attr_name={"type1": "label", "type2": "label2"},
        node_type_label_attr_idx={"type1": 0, "type2": 1},
    )
    hetero_g = pyg_data.HeteroData()
    hetero_g["type1"].x = []
    hetero_g["type1"].label = th.tensor([[0], [1], [2]])
    hetero_g["type2"].x = []
    hetero_g["type2"].label2 = th.tensor([[0], [1], [2]])
    hetero_g[("type1", "rel", "type2")].edge_index = th.tensor(
        [[0, 0, 0, 1], [1, 2, 3, 2]]
    )
    hetero_g[("type1", "rel", "type2")].weight = th.tensor([1.0] * 4).unsqueeze(-1)

    # then
    with pytest.raises(
        ValueError,
        match="Label attribute index 1 is invalid for node type type2.",
    ):
        VisualizationBuilder.check_config_with_graph(vis_config, hetero_g)


def test_build_node_label_correct_label_mapping():
    # given
    vis_config = structures.VisualizationConfig(
        node_types=["company", "tool"],
        edge_types=[("company", "has", "tool")],
        edge_weights_attr_idx={("company", "has", "tool"): 0},
        edge_type_weight_attr_name={("company", "has", "tool"): "weight"},
        node_type_label_attr_name={"company": "label", "tool": "label2"},
        node_type_label_attr_idx={"company": 0, "tool": 0},
    )
    hetero_g = pyg_data.HeteroData()
    hetero_g["company"].x = []
    hetero_g["company"].label = th.tensor([[0], [1], [2]])
    hetero_g["tool"].x = []
    hetero_g["tool"].label2 = th.tensor([[0], [1], [2]])
    hetero_g[("company", "has", "tool")].edge_index = th.tensor(
        [[0, 0, 0, 1], [0, 1, 2, 2]]
    )
    hetero_g[("company", "has", "tool")].weight = th.tensor([1.0] * 4).unsqueeze(-1)

    label_mapping = {
        "company": {0: "Microsoft", 1: "Google", 2: "Apple"},
        "tool": {0: "Spark", "1": "Tensorflow", 2: "PyTorch"},
    }

    expected_label = "Microsoft"

    # when
    node_label = VisualizationBuilder.build_node_label(
        "company", 0, hetero_g, vis_config, label_mapping
    )

    # then
    assert node_label == expected_label


def test_build_visualization_no_errors():
    # given
    vis_config = structures.VisualizationConfig(
        node_types=["company", "tool"],
        edge_types=[("company", "has", "tool")],
        edge_weights_attr_idx={("company", "has", "tool"): 0},
        edge_type_weight_attr_name={("company", "has", "tool"): "weight"},
        node_type_label_attr_name={"company": "label", "tool": "label2"},
        node_type_label_attr_idx={"company": 0, "tool": 0},
    )
    hetero_g = pyg_data.HeteroData()
    hetero_g["company"].x = []
    hetero_g["company"].label = th.tensor([[0], [1], [2]])
    hetero_g["tool"].x = []
    hetero_g["tool"].label2 = th.tensor([[0], [1], [2]])
    hetero_g[("company", "has", "tool")].edge_index = th.tensor(
        [[0, 0, 0, 1], [0, 1, 2, 2]]
    )
    hetero_g[("company", "has", "tool")].weight = th.tensor([1.0] * 4).unsqueeze(-1)

    label_mapping = {
        "company": {0: "Microsoft", 1: "Google", 2: "Apple"},
        "tool": {0: "Spark", "1": "Tensorflow", 2: "PyTorch"},
    }

    # then
    try:
        _ = VisualizationBuilder.build_visualization(
            vis_config, hetero_g, label_mapping
        )
    except Exception:
        pytest.fail("build_visualization() raised ValueError unexpectedly!")
