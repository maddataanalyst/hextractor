"""Example datasets for demonstrating HexTractor functionality.

This module provides sample datasets in both single-table and multi-table formats
that demonstrate common patterns in heterogeneous graph extraction.

The data represents a simple company-employee-tag relationship graph where:
- Companies have employees and tags
- Companies have attributes (employee count, revenue)
- Employees have attributes (occupation, age) and a label (promotion)
- Tags are simple identifiers

The same data is provided in two formats:
1. Single denormalized table with all relationships
2. Multiple normalized tables (companies, employees, tags, relationships)
"""

import pandas as pd
from typing import Dict


def get_single_table_data() -> pd.DataFrame:
    """Generate example data in single denormalized table format.

    The table contains company data duplicated across rows, one row per
    company-employee relationship. Companies can have multiple tags stored
    as lists in the tags column.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - company_id (int): Unique company identifier
        - company_employees (int): Number of employees
        - company_revenue (int): Company revenue
        - employee_id (int): Unique employee identifier
        - employee_occupation (int): Employee occupation code
        - employee_age (int): Employee age
        - employee_promotion (int): Binary promotion label
        - tags (List[int]): List of tag IDs for the company
    """
    return pd.DataFrame(
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


def get_multi_table_data() -> Dict[str, pd.DataFrame]:
    """Generate example data split across multiple normalized tables.

    Returns
    -------
    dict of {str: pd.DataFrame}
        Dictionary containing DataFrames:
        - companies: Company information (id, employees, revenue)
        - employees: Employee information (id, occupation, age, promotion)
        - tags: Tag IDs
        - company_employees: Company-employee relationships
        - company_tags: Company-tag relationships
    """
    companies = pd.DataFrame(
        {
            "company_id": [1, 2],
            "company_employees": [100, 5000],
            "company_revenue": [1000, 100000],
        }
    )

    employees = pd.DataFrame(
        {
            "employee_id": [0, 1, 3, 4, 5, 6],
            "employee_occupation": [0, 1, 3, 1, 1, 4],
            "employee_age": [25, 35, 45, 18, 20, 31],
            "employee_promotion": [0, 1, 0, 1, 1, 0],
        }
    )

    tags = pd.DataFrame({"tag": [1, 2, 3, 4]})

    company_employees = pd.DataFrame(
        {
            "company_id": [1, 1, 1, 2, 2, 2],
            "employee_id": [0, 1, 3, 4, 5, 6],
        }
    )

    company_tags = pd.DataFrame(
        {
            "company_id": [1, 1, 1, 2, 2, 2],
            "tags": [[1, 2, 3], [1, 2], [3, 4], [1, 4], [1, 1], [1, 2]],
        }
    )

    return {
        "companies": companies,
        "employees": employees,
        "tags": tags,
        "company_employees": company_employees,
        "company_tags": company_tags,
    }
