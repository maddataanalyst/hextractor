import pytest
import torch as th
import pandas as pd
import numpy as np
import torch_geometric.data as pyg_data
import hextractor.utils as utils


@pytest.fixture
def company_has_employee_df():
    df = pd.DataFrame(
        [
            (1, 100, 1000, 1, 0, 25, 0),
            (1, 100, 1000, 2, 1, 35, 1),
            (1, 100, 1000, 3, 3, 45, 0),
            (2, 5000, 100000, 4, 1, 18, 1),
            (2, 5000, 100000, 5, 1, 20, 1),
            (2, 5000, 100000, 6, 4, 31, 0),
        ],
        columns=[
            "company_id",
            "company_employees",
            "company_revenue",
            "employee_id",
            "employee_occupation",
            "employee_age",
            "employee_promotion",
        ],
    )
    return df


def test_correct_hetero_graph_construction_from_id(
    company_has_employee_df: pd.DataFrame,
):
    # Given
    company_node_params = utils.NodeTypeParams(
        node_type_name="company",
        source_name="df",
        id_col="company_id",
        attributes=("company_employees", "company_revenue"),
        attr_type='float'

    )

    employee_node_params = utils.NodeTypeParams(
        node_type_name="employee",
        source_name="df",
        id_col="employee_id",
        attributes=("employee_occupation", "employee_age"),
        target_name="employee_promotion",
        attr_type='long'
    )

    company_has_emp_edge_params = utils.EdgeTypeParams(
        edge_type_name="has", source_name="company", target_name="employee"
    )

    df_source_specs = utils.DataFrameSource(
        name="df1",
        node_params=(company_node_params, employee_node_params),
        edge_params=(company_has_emp_edge_params,),
        data_frame=company_has_employee_df,
    )

    expected_hetero_g = pyg_data.HeteroData()
    expected_hetero_g['company'].x = th.tensor(np.array([[100, 1000], [5000, 100000]]))
    expected_hetero_g['employee'].x = th.tensor(np.array([[0, 25], [1, 35], [3, 45], [1, 18], [1, 20], [4, 31]]))
    expected_hetero_g['company', 'has', 'employee'].edge_index = th.tensor([[1, 1, 1, 2, 2, 2], [1, 2, 3, 4, 5, 6]])

    # when
    hetero_g = df_source_specs.extract_using_id()

    # then
    expected_hetero_g.node_types == hetero_g.node_types
    expected_hetero_g.edge_types == hetero_g.edge_types

    for node_type in hetero_g.node_types:
        assert th.all(expected_hetero_g[node_type].x == hetero_g[node_type].x)

    for edge_type in hetero_g.edge_types:
        assert th.all(expected_hetero_g[edge_type].edge_index == hetero_g[edge_type].edge_index)
