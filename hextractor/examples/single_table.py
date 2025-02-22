"""Single table data processing example.

This module demonstrates how to use HexTractor to extract a heterogeneous graph
from a single denormalized table containing all entities and relationships.
The example shows how to handle:
- Multiple entity types in one table
- Entity de-duplication
- Multi-value columns (tags)
"""

from typing import Optional
import pandas as pd
import hextractor.extraction as hextract
import hextractor.data_sources as data_sources

from hextractor.examples.data import get_single_table_data
from hextractor.examples.utils import (
    create_company_node_params,
    create_employee_node_params,
    create_tag_node_params,
    create_company_employee_edge_params,
    create_company_tag_edge_params,
    create_dataframe_specs
)

def create_single_table_specs(
    df: Optional[pd.DataFrame] = None
) -> data_sources.GraphSpecs:
    """Create graph specifications for single table processing.
    
    Parameters
    ----------
    df : pd.DataFrame, optional
        DataFrame containing all entities and relationships.
        If None, uses example data from get_single_table_data().
            
    Returns
    -------
    data_sources.GraphSpecs
        GraphSpecs configured for single table processing
    
    Examples
    --------
    Basic usage:
    ```python
    from hextractor.examples.single_table import create_single_table_specs
    specs = create_single_table_specs()
    ```
    
    With custom data:
    ```python
    import pandas as pd
    df = pd.DataFrame({...})  # Your data
    specs = create_single_table_specs(df)
    ```
    """
    if df is None:
        df = get_single_table_data()
        
    # Create node parameters
    company_params = create_company_node_params()
    employee_params = create_employee_node_params()
    tag_params = create_tag_node_params()
    
    # Create edge parameters
    company_employee_edges = create_company_employee_edge_params()
    company_tag_edges = create_company_tag_edge_params()
    
    # Create DataFrame specifications
    df_specs = create_dataframe_specs(
        name="single_table",
        df=df,
        node_params=(
            company_params,
            employee_params,
            tag_params
        ),
        edge_params=(
            company_employee_edges,
            company_tag_edges
        )
    )
    
    # Create and return graph specifications
    return data_sources.GraphSpecs(
        data_sources=(df_specs,)
    )

def create_single_table_graph(
    df: Optional[pd.DataFrame] = None
):
    """Extract a heterogeneous graph from a single denormalized table.
    
    This function demonstrates the complete workflow of:
    1. Creating node type parameters
    2. Creating edge type parameters
    3. Creating DataFrame specifications
    4. Creating graph specifications
    5. Extracting the final graph
    
    Parameters
    ----------
    df : pd.DataFrame, optional
        DataFrame containing all entities and relationships.
        If None, uses example data from get_single_table_data().
            
    Returns
    -------
    HeterogeneousGraph
        Extracted heterogeneous graph
        
    Examples
    --------
    Basic usage:
    ```python
    from hextractor.examples.single_table import create_single_table_graph
    graph = create_single_table_graph()
    ```
    
    With custom data:
    ```python
    import pandas as pd
    df = pd.DataFrame({...})  # Your data
    graph = create_single_table_graph(df)
    ```
    """
    specs = create_single_table_specs(df)
    return hextract.extract_data(specs)