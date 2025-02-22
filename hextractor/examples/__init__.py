"""HexTractor examples module for tabular data processing.

This module provides complete examples showing how to use HexTractor to transform 
tabular data into heterogeneous graphs. Two main cases are demonstrated:

1. Single-table data processing where all data resides in one denormalized table
2. Multi-table data processing where data is split across normalized tables

The examples show common patterns like:
- Creating node and edge type parameters
- Handling multi-value columns
- De-duplicating entities
- Joining data across tables
- Building graph specifications

Example Usage:
```python
from hextractor.examples.single_table import create_single_table_graph
graph = create_single_table_graph()

from hextractor.examples.multi_table import create_multi_table_graph  
graph = create_multi_table_graph()
```
See the individual modules for more detailed examples and documentation.
"""