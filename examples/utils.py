"""Utility functions for creating graph specifications.

This module provides helper functions for creating node and edge parameters
used in both single-table and multi-table examples. These utilities help reduce
code duplication and standardize parameter creation.
"""

from typing import Tuple, Optional, List
import hextractor.structures as structures
import hextractor.data_sources as data_sources
import pandas as pd

def create_company_node_params(
    id_col: str = "company_id",
    employees_col: str = "company_employees",
    revenue_col: str = "company_revenue"
) -> structures.NodeTypeParams:
    """Create node parameters for company entities.
    
    Args:
        id_col: Column name for company ID
        employees_col: Column name for employee count
        revenue_col: Column name for company revenue
        
    Returns:
        NodeTypeParams configured for company nodes
    """
    return structures.NodeTypeParams(
        node_type_name="company",
        id_col=id_col,
        attributes=(employees_col, revenue_col),
        attr_type="float",
    )

def create_employee_node_params(
    id_col: str = "employee_id",
    occupation_col: str = "employee_occupation",
    age_col: str = "employee_age",
    promotion_col: str = "employee_promotion"
) -> structures.NodeTypeParams:
    """Create node parameters for employee entities.
    
    Args:
        id_col: Column name for employee ID
        occupation_col: Column name for occupation code
        age_col: Column name for employee age
        promotion_col: Column name for promotion label
        
    Returns:
        NodeTypeParams configured for employee nodes
    """
    return structures.NodeTypeParams(
        node_type_name="employee",
        id_col=id_col,
        attributes=(occupation_col, age_col),
        label_col=promotion_col,
        attr_type="long",
    )

def create_tag_node_params(
    id_col: str = "tags",
    multivalue: bool = True
) -> structures.NodeTypeParams:
    """Create node parameters for tag entities.
    
    Args:
        id_col: Column name containing tag IDs
        multivalue: Whether tags are stored as lists of values
        
    Returns:
        NodeTypeParams configured for tag nodes
    """
    return structures.NodeTypeParams(
        node_type_name="tag",
        id_col=id_col,
        multivalue_source=multivalue,
    )

def create_company_employee_edge_params(
    company_id_col: str = "company_id",
    employee_id_col: str = "employee_id"
) -> structures.EdgeTypeParams:
    """Create edge parameters for company-employee relationships.
    
    Args:
        company_id_col: Column name for company IDs
        employee_id_col: Column name for employee IDs
        
    Returns:
        EdgeTypeParams configured for company-employee edges
    """
    return structures.EdgeTypeParams(
        edge_type_name="has",
        source_name="company",
        target_name="employee",
        source_id_col=company_id_col,
        target_id_col=employee_id_col,
    )

def create_company_tag_edge_params(
    company_id_col: str = "company_id",
    tag_id_col: str = "tags",
    multivalue: bool = True
) -> structures.EdgeTypeParams:
    """Create edge parameters for company-tag relationships.
    
    Args:
        company_id_col: Column name for company IDs
        tag_id_col: Column name for tag IDs
        multivalue: Whether tags are stored as lists of values
        
    Returns:
        EdgeTypeParams configured for company-tag edges
    """
    return structures.EdgeTypeParams(
        edge_type_name="has",
        source_name="company",
        target_name="tag",
        source_id_col=company_id_col,
        target_id_col=tag_id_col,
        multivalue_target=multivalue
    )

def create_dataframe_specs(
    name: str,
    df: pd.DataFrame,
    node_params: Optional[Tuple[structures.NodeTypeParams, ...]] = None,
    edge_params: Optional[Tuple[structures.EdgeTypeParams, ...]] = None
) -> data_sources.DataFrameSpecs:
    """Create DataFrame specifications for a data source.
    
    Args:
        name: Name identifier for the data source
        df: Source DataFrame
        node_params: Tuple of NodeTypeParams for entities in the DataFrame
        edge_params: Tuple of EdgeTypeParams for relationships in the DataFrame
        
    Returns:
        DataFrameSpecs configured with the provided parameters
    """
    return data_sources.DataFrameSpecs(
        name=name,
        node_params=node_params or tuple(),
        edge_params=edge_params or tuple(),
        data_frame=df
    )