"""Multi table data processing example.

This module demonstrates how to use HexTractor to extract a heterogeneous graph
from multiple normalized tables. This represents a typical relational database
scenario where:
- Each entity type has its own table (companies, employees, tags)
- Relationships are stored in separate junction tables
- Data is normalized to avoid duplication
"""

from typing import Optional, Dict
import pandas as pd
import hextractor.extraction as hextract
import hextractor.data_sources as data_sources

from examples.data import get_multi_table_data
from examples.utils import (
    create_company_node_params,
    create_employee_node_params,
    create_tag_node_params,
    create_company_employee_edge_params,
    create_company_tag_edge_params,
    create_dataframe_specs
)

def create_multi_table_specs(
    tables: Optional[Dict[str, pd.DataFrame]] = None
) -> data_sources.GraphSpecs:
    """Create graph specifications for multi-table processing.
    
    Args:
        tables: Dictionary containing DataFrames:
            - companies: Company information
            - employees: Employee information
            - tags: Tag information
            - company_employees: Company-employee relationships
            - company_tags: Company-tag relationships
            If None, uses example data from get_multi_table_data().
            
    Returns:
        GraphSpecs configured for multi-table processing
    
    Example:
        >>> from examples.multi_table import create_multi_table_specs
        >>> specs = create_multi_table_specs()
        >>> # Or with custom data:
        >>> tables = {
        ...     'companies': companies_df,
        ...     'employees': employees_df,
        ...     'tags': tags_df,
        ...     'company_employees': company_employees_df,
        ...     'company_tags': company_tags_df
        ... }
        >>> specs = create_multi_table_specs(tables)
    """
    if tables is None:
        tables = get_multi_table_data()
        
    # Create node parameters with appropriate column names
    company_params = create_company_node_params()
    employee_params = create_employee_node_params()
    tag_params = create_tag_node_params(
        id_col="tag",  # Different from single table
        multivalue=False  # Tags are in their own table
    )
    
    # Create edge parameters
    company_employee_edges = create_company_employee_edge_params()
    company_tag_edges = create_company_tag_edge_params()
    
    # Create DataFrame specifications for each table
    company_specs = create_dataframe_specs(
        name="companies",
        df=tables['companies'],
        node_params=(company_params,)
    )
    
    employee_specs = create_dataframe_specs(
        name="employees",
        df=tables['employees'],
        node_params=(employee_params,)
    )
    
    tag_specs = create_dataframe_specs(
        name="tags",
        df=tables['tags'],
        node_params=(tag_params,)
    )
    
    company_employee_specs = create_dataframe_specs(
        name="company_employees",
        df=tables['company_employees'],
        edge_params=(company_employee_edges,)
    )
    
    company_tag_specs = create_dataframe_specs(
        name="company_tags",
        df=tables['company_tags'],
        edge_params=(company_tag_edges,)
    )
    
    # Create and return graph specifications
    return data_sources.GraphSpecs(
        data_sources=(
            company_specs,
            employee_specs,
            tag_specs,
            company_employee_specs,
            company_tag_specs
        )
    )

def create_multi_table_graph(
    tables: Optional[Dict[str, pd.DataFrame]] = None
):
    """Extract a heterogeneous graph from multiple normalized tables.
    
    This function demonstrates the complete workflow of:
    1. Creating node type parameters for each entity table
    2. Creating edge type parameters for each relationship table
    3. Creating DataFrame specifications for each table
    4. Creating graph specifications combining all tables
    5. Extracting the final graph
    
    Args:
        tables: Dictionary of DataFrames containing entities and relationships.
               If None, uses example data from get_multi_table_data().
            
    Returns:
        Extracted heterogeneous graph
        
    Example:
        >>> from examples.multi_table import create_multi_table_graph
        >>> graph = create_multi_table_graph()
        >>> # Or with custom data:
        >>> tables = {
        ...     'companies': companies_df,
        ...     'employees': employees_df,
        ...     'tags': tags_df,
        ...     'company_employees': company_employees_df,
        ...     'company_tags': company_tags_df
        ... }
        >>> graph = create_multi_table_graph(tables)
    """
    specs = create_multi_table_specs(tables)
    return hextract.extract_data(specs)