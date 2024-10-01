import pytest
from hextractor.structures import VisualizationConfig


def test_check_consistency_valid_config_minimal_setup():
    config = dict(
        node_types=("node1", "node2"),
        edge_types=(("node1", "edge", "node2"),),
    )
    try:
        VisualizationConfig(**config)
    except ValueError:
        pytest.fail("check_consistency() raised ValueError unexpectedly!")


def test_check_consistency_missing_source_node():
    config = dict(
        node_types=("node2",),
        edge_types=(("node1", "edge", "node2"),),
    )
    with pytest.raises(ValueError, match="Node type node1 is not selected"):
        VisualizationConfig(**config)


def test_check_consistency_missing_target_node():
    config = dict(
        node_types=("node1",),
        edge_types=(("node1", "edge", "node2"),),
    )
    with pytest.raises(ValueError, match="Node type node2 is not selected"):
        VisualizationConfig(**config)


def test_check_consistency_with_colors_and_weights():
    config = dict(
        node_types=("node1", "node2"),
        node_types_to_colors={"node1": "red", "node2": "blue"},
        edge_types=(("node1", "edge", "node2"),),
        edge_type_to_colors={("node1", "edge", "node2"): "green"},
        edge_weights_attr_idx={("node1", "edge", "node2"): 1},
    )
    try:
        VisualizationConfig(**config)
    except ValueError:
        pytest.fail("check_consistency() raised ValueError unexpectedly!")


def test_check_consistency_with_extra_node_colors():
    config = dict(
        node_types=("node1",),
        node_types_to_colors={"node1": "red", "node2": "blue"},
    )
    try:
        VisualizationConfig(**config)
    except ValueError:
        pytest.fail("check_consistency() raised ValueError unexpectedly!")


def test_check_consistency_with_extra_edge_colors():
    config = dict(
        node_types=("node1", "node2"),
        edge_types=(("node1", "edge", "node2"),),
        edge_type_to_colors={
            ("node1", "edge", "node2"): "green",
            ("node2", "edge", "node1"): "blue",
        },
    )
    try:
        VisualizationConfig(**config)
    except ValueError:
        pytest.fail("check_consistency() raised ValueError unexpectedly!")


def test_check_consistency_with_extra_edge_weights():
    config = dict(
        node_types=("node1", "node2"),
        edge_types=(("node1", "edge", "node2"),),
        edge_weights_attr_idx={
            ("node1", "edge", "node2"): 1,
            ("node2", "edge", "node1"): 2,
        },
    )
    try:
        VisualizationConfig(**config)
    except ValueError:
        pytest.fail("check_consistency() raised ValueError unexpectedly!")


def test_check_consistency_with_missing_node_from_edge_color():
    config = dict(
        node_types=("node2",),
        edge_types=(),
        edge_type_to_colors={("node1", "edge", "node2"): "green"},
    )
    with pytest.raises(ValueError, match="Node type node1 is not selected"):
        VisualizationConfig(**config)


def test_check_consistency_with_missing_node_for_edge_weight():
    config = dict(
        node_types=("node2",),
        edge_types=(),
        edge_weights_attr_idx={("node1", "edge", "node2"): 0},
    )
    with pytest.raises(ValueError, match="Node type node1 is not selected"):
        VisualizationConfig(**config)


def test_get_node_color_with_default_color():
    config = VisualizationConfig(
        node_types=("node1", "node2"),
        node_types_to_colors={"node1": "red"},
        default_node_color="green",
    )
    assert config.get_node_color("node2") == "green"
