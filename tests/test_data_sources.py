import pytest
import torch as th
import pandas as pd
import numpy as np
import torch_geometric.data as pyg_data
import hextractor.data_sources as data_sources
import hextractor.structures as structures
import hextractor.extraction as extr
from typing import Tuple


@pytest.fixture
def company_has_employee_df():
    df = pd.DataFrame(
        [
            (1, 100, 1000, 0, 0, 25, 0, [1, 2, 3]),
            (1, 100, 1000, 1, 1, 35, 1, [1, 2]),
            (1, 100, 1000, 3, 3, 45, 0, [3, 4]),
            (2, 5000, 100000, 4, 1, 18, 1, [1, 4]),
            (2, 5000, 100000, 5, 1, 20, 1, [1, 1]),
            (2, 5000, 100000, 6, 4, 31, 0, [1, 2]),
        ],
        columns=[
            "company_id",
            "company_employees",
            "company_revenue",
            "employee_id",
            "employee_occupation",
            "employee_age",
            "employee_promotion",
            "tags",
        ],
    )

    return df


@pytest.fixture
def company_has_employee_multi_df():
    df = pd.DataFrame(
        [
            (1, 100, 1000, 0, 0, 25, 0, [1, 2, 3]),
            (1, 100, 1000, 1, 1, 35, 1, [1, 2]),
            (1, 100, 1000, 3, 3, 45, 0, [3, 4]),
            (2, 5000, 100000, 4, 1, 18, 1, [1, 4]),
            (2, 5000, 100000, 5, 1, 20, 1, [1, 1]),
            (2, 5000, 100000, 6, 4, 31, 0, [1, 2]),
        ],
        columns=[
            "company_id",
            "company_employees",
            "company_revenue",
            "employee_id",
            "employee_occupation",
            "employee_age",
            "employee_promotion",
            "tags",
        ],
    )

    company_df = df[
        ["company_id", "company_employees", "company_revenue"]
    ].drop_duplicates()
    employee_df = df[
        ["employee_id", "employee_occupation", "employee_age", "employee_promotion"]
    ].drop_duplicates()
    tags_df = df.explode("tags")[["tags"]].drop_duplicates().astype(int)
    company_has_employee_df = df[["company_id", "employee_id"]].drop_duplicates()
    company_has_tag_df = (
        df[["company_id", "tags"]].explode("tags").drop_duplicates().astype(int)
    )

    return company_df, employee_df, tags_df, company_has_employee_df, company_has_tag_df


def test_inconsistent_graph_specs_missing_tag_specs(company_has_employee_df):
    # given
    company_node_params = structures.NodeTypeParams(
        node_type_name="company",
        id_col="company_id",
        attributes=("company_employees", "company_revenue"),
        attr_type="float",
    )

    employee_node_params = structures.NodeTypeParams(
        node_type_name="employee",
        id_col="employee_id",
        attributes=("employee_occupation", "employee_age"),
        label_col="employee_promotion",
        attr_type="long",
    )

    company_has_emp_edge_params = structures.EdgeTypeParams(
        edge_type_name="has",
        source_name="company",
        target_name="employee",
        source_id_col="company_id",
        target_id_col="employee_id",
    )

    company_has_tag_edge_params = structures.EdgeTypeParams(
        edge_type_name="has",
        source_name="company",
        target_name="tag",
        source_id_col="company_id",
        target_id_col="tags",
        multivalue_target=True,
    )

    df_source_specs = data_sources.DataFrameSpecs(
        name="df1",
        node_params=(
            company_node_params,
            employee_node_params,
        ),
        edge_params=(company_has_emp_edge_params, company_has_tag_edge_params),
        data_frame=company_has_employee_df,
    )

    with pytest.raises(ValueError) as e:
        data_sources.GraphSpecs(data_sources=(df_source_specs,))


def test_correct_hetero_graph_construction_single_df(
    company_has_employee_df: pd.DataFrame,
):
    # Given
    company_node_params = structures.NodeTypeParams(
        node_type_name="company",
        id_col="company_id",
        attributes=("company_employees", "company_revenue"),
        attr_type="float",
    )

    company_tags_node_params = structures.NodeTypeParams(
        node_type_name="tag",
        multivalue_source=True,
        id_col="tags",
    )

    employee_node_params = structures.NodeTypeParams(
        node_type_name="employee",
        id_col="employee_id",
        attributes=("employee_occupation", "employee_age"),
        label_col="employee_promotion",
        attr_type="long",
    )

    company_has_emp_edge_params = structures.EdgeTypeParams(
        edge_type_name="has",
        source_name="company",
        target_name="employee",
        source_id_col="company_id",
        target_id_col="employee_id",
    )

    company_has_tag_edge_params = structures.EdgeTypeParams(
        edge_type_name="has",
        source_name="company",
        target_name="tag",
        source_id_col="company_id",
        target_id_col="tags",
        multivalue_target=True,
    )

    df_source_specs = data_sources.DataFrameSpecs(
        name="df1",
        node_params=(
            company_node_params,
            employee_node_params,
            company_tags_node_params,
        ),
        edge_params=(company_has_emp_edge_params, company_has_tag_edge_params),
        data_frame=company_has_employee_df,
    )
    graph_specs = data_sources.GraphSpecs(data_sources=(df_source_specs,))

    expected_hetero_g = pyg_data.HeteroData()
    expected_hetero_g["company"].x = th.tensor(
        np.array([[0, 0], [100, 1000], [5000, 100000]])
    )
    expected_hetero_g["employee"].x = th.tensor(
        np.array([[0, 25], [1, 35], [0, 0], [3, 45], [1, 18], [1, 20], [4, 31]])
    )
    expected_hetero_g["tag"].x = th.arange(5)
    expected_hetero_g["employee"].y = th.tensor(np.array([0, 1, 0, 0, 1, 1, 0]))
    expected_hetero_g["company", "has", "employee"].edge_index = th.tensor(
        [[1, 1, 1, 2, 2, 2], [0, 1, 3, 4, 5, 6]]
    )

    expected_hetero_g["company", "has", "tag"].edge_index = th.tensor(
        [[1, 1, 1, 1, 2, 2, 2], [1, 2, 3, 4, 1, 2, 4]]
    )

    # when
    hetero_g = extr.extract_data(graph_specs)

    # then
    assert expected_hetero_g.node_types == hetero_g.node_types
    assert expected_hetero_g.edge_types == hetero_g.edge_types

    assert th.all(expected_hetero_g["employee"].y == hetero_g["employee"].y)
    for node_type in hetero_g.node_types:
        assert th.all(expected_hetero_g[node_type].x == hetero_g[node_type].x)

    for edge_type in hetero_g.edge_types:
        assert th.all(
            expected_hetero_g[edge_type].edge_index == hetero_g[edge_type].edge_index
        )


def test_correct_hetero_graph_construction_multi_df(
    company_has_employee_multi_df: Tuple[pd.DataFrame, ...],
):
    # Given

    """Split each node into different data frame sources. E.g.: employees are in a separate data frame (table).
    Use linking tables to create edges and connect entities (e.g. company-has-tag)
    """
    company_df, employee_df, tags_df, company_has_employee_df, company_has_tag_df = (
        company_has_employee_multi_df
    )
    company_node_params = structures.NodeTypeParams(
        node_type_name="company",
        id_col="company_id",
        attributes=("company_employees", "company_revenue"),
        attr_type="float",
    )
    company_df_source = data_sources.DataFrameSpecs(
        name="df1", node_params=(company_node_params,), data_frame=company_df
    )

    company_tags_node_params = structures.NodeTypeParams(
        node_type_name="tag",
        multivalue_source=False,
        id_col="tags",
        id_as_attr=True,
    )
    tag_df_source = data_sources.DataFrameSpecs(
        name="df2", node_params=(company_tags_node_params,), data_frame=tags_df
    )

    employee_node_params = structures.NodeTypeParams(
        node_type_name="employee",
        id_col="employee_id",
        attributes=("employee_occupation", "employee_age"),
        label_col="employee_promotion",
        attr_type="long",
    )
    employee_df_source = data_sources.DataFrameSpecs(
        name="df3", node_params=(employee_node_params,), data_frame=employee_df
    )

    company_has_emp_edge_params = structures.EdgeTypeParams(
        edge_type_name="has",
        source_name="company",
        target_name="employee",
        source_id_col="company_id",
        target_id_col="employee_id",
    )
    company_has_emp_edge_df_source = data_sources.DataFrameSpecs(
        name="df4",
        edge_params=(company_has_emp_edge_params,),
        data_frame=company_has_employee_df,
    )

    company_has_tag_edge_params = structures.EdgeTypeParams(
        edge_type_name="has",
        source_name="company",
        target_name="tag",
        source_id_col="company_id",
        target_id_col="tags",
    )
    company_has_tag_edge_df_source = data_sources.DataFrameSpecs(
        name="df5",
        edge_params=(company_has_tag_edge_params,),
        data_frame=company_has_tag_df,
    )

    graph_specs = data_sources.GraphSpecs(
        data_sources=(
            company_df_source,
            employee_df_source,
            tag_df_source,
            company_has_emp_edge_df_source,
            company_has_tag_edge_df_source,
        )
    )

    expected_hetero_g = pyg_data.HeteroData()
    expected_hetero_g["company"].x = th.tensor(
        np.array([[0, 0], [100, 1000], [5000, 100000]])
    )
    expected_hetero_g["employee"].x = th.tensor(
        np.array([[0, 25], [1, 35], [0, 0], [3, 45], [1, 18], [1, 20], [4, 31]])
    )
    expected_hetero_g["tag"].x = th.arange(5).unsqueeze(-1)
    expected_hetero_g["employee"].y = th.tensor(np.array([0, 1, 0, 0, 1, 1, 0]))
    expected_hetero_g["company", "has", "employee"].edge_index = th.tensor(
        [[1, 1, 1, 2, 2, 2], [0, 1, 3, 4, 5, 6]]
    )

    expected_hetero_g["company", "has", "tag"].edge_index = th.tensor(
        [[1, 1, 1, 1, 2, 2, 2], [1, 2, 3, 4, 1, 2, 4]]
    )

    # when
    hetero_g = extr.extract_data(graph_specs)

    # then
    assert expected_hetero_g.node_types == hetero_g.node_types
    assert expected_hetero_g.edge_types == hetero_g.edge_types

    assert th.all(expected_hetero_g["employee"].y == hetero_g["employee"].y)
    for node_type in hetero_g.node_types:
        assert th.all(expected_hetero_g[node_type].x == hetero_g[node_type].x)

    for edge_type in hetero_g.edge_types:
        assert th.all(
            expected_hetero_g[edge_type].edge_index == hetero_g[edge_type].edge_index
        )


def test_all_attributes_should_be_numeric(company_has_employee_df: pd.DataFrame):
    # given
    fake_df = company_has_employee_df.copy()
    fake_df["employee_occupation"] = ["develpoer"] * 6

    company_node_params = structures.NodeTypeParams(
        node_type_name="company",
        id_col="company_id",
        attributes=("company_employees", "company_revenue"),
        attr_type="float",
    )

    employee_node_params = structures.NodeTypeParams(
        node_type_name="employee",
        id_col="employee_id",
        attributes=("employee_occupation", "employee_age"),
        label_col="employee_promotion",
        attr_type="long",
    )

    company_has_emp_edge_params = structures.EdgeTypeParams(
        edge_type_name="has",
        source_name="company",
        target_name="employee",
        source_id_col="company_id",
        target_id_col="employee_id",
    )

    df_source_specs = data_sources.DataFrameSpecs(
        name="df1",
        node_params=(company_node_params, employee_node_params),
        edge_params=(company_has_emp_edge_params,),
        data_frame=fake_df,
    )

    graph_specs = data_sources.GraphSpecs(data_sources=(df_source_specs,))

    # when
    with pytest.raises(ValueError) as e:
        extr.extract_data(graph_specs)

    # then
    assert "Not all attributes are numeric" in str(e.value)


def test_too_few_receiver_nodes_error():
    # given
    hg = pyg_data.HeteroData()
    hg["node_a"].x = th.tensor(
        [
            [1, 2],
            [3, 4],
        ]
    )
    hg["node_b"].x = th.tensor(
        [
            [1, 2],
        ]
    )
    hg["node_a", "sends", "node_b"].edge_index = th.tensor([[0, 1, 1], [0, 0, 1]])

    # when
    with pytest.raises(ValueError) as e:
        extr.validate_consistency(hg)

    # then
    assert "Node type node_b has too few nodes" in str(e.value)


def test_too_few_sender_nodes_error():
    # given
    hg = pyg_data.HeteroData()
    hg["node_a"].x = th.tensor(
        [
            [1, 2],
            [3, 4],
        ]
    )
    hg["node_b"].x = th.tensor(
        [
            [1, 2],
        ]
    )
    hg["node_a", "sends", "node_b"].edge_index = th.tensor([[0, 1, 2], [0, 0, 0]])

    # when
    with pytest.raises(ValueError) as e:
        extr.validate_consistency(hg)

    # then
    assert "Node type node_a has too few nodes" in str(e.value)


def test_missing_nodetypes():
    # given
    hg = pyg_data.HeteroData()
    hg["node_a"].x = th.tensor(
        [
            [1, 2],
            [3, 4],
        ]
    )
    hg["node_b"].x = th.tensor(
        [
            [1, 2],
        ]
    )
    hg["node_a", "sends", "node_b"].edge_index = th.tensor([[0, 1], [0, 0]])
    hg["node_x", "sends", "node_a"].edge_index = th.tensor([[0, 1], [0, 0]])

    # when
    with pytest.raises(ValueError) as e:
        extr.validate_consistency(hg)

    # then
    assert "Node type node_x is missing" in str(e.value)


def test_nodetype_params():
    # Given
    node_type_params = structures.NodeTypeParams(
        node_type_name="company",
        id_col="company_id",
        attributes=("company_employees", "company_revenue"),
        attr_type="float",
    )

    # Then
    assert node_type_params.node_type_name == "company"
    assert node_type_params.id_col == "company_id"
    assert node_type_params.attributes == ("company_employees", "company_revenue")
    assert node_type_params.attr_type == "float"


def test_nodetype_params_multivalue_source_ok():
    # Given
    node_type_params = structures.NodeTypeParams(
        node_type_name="tag",
        multivalue_source=True,
        id_col="tags",
    )

    # Then
    assert node_type_params.node_type_name == "tag"
    assert node_type_params.id_col == "tags"


def test_nodetype_params_multivalue_source_target_col_not_allowed():
    # When

    with pytest.raises(ValueError) as e:
        structures.NodeTypeParams(
            node_type_name="tag",
            id_col="tags",
            multivalue_source=True,
            label_col="target",
        )
