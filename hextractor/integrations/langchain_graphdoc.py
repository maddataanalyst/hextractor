import torch
from torch_geometric.data import HeteroData
from collections import defaultdict
from typing import Dict, List, Tuple, Set


def group_nodes_by_type(graph_doc) -> Dict[str, List[str]]:
    """
    Group nodes by their type.
    
    Parameters
    ----------
    graph_doc : GraphDocument
        The graph document containing nodes and relationships.
        
    Returns
    -------
    Dict[str, List[str]]
        A dictionary mapping node types to lists of node IDs.
    
    Notes
    -----
    This function creates a mapping from node types to lists of node IDs,
    which is useful for further processing of nodes by their type.
    """
    nodes_by_type = defaultdict(list)
    for node in graph_doc.nodes:
        nodes_by_type[node.type].append(node.id)
    return nodes_by_type


def create_node_id_mapping(nodes_by_type: Dict[str, List[str]]) -> Dict[Tuple[str, str], int]:
    """
    Create a mapping from string node IDs to numerical indices per node type.
    
    Parameters
    ----------
    nodes_by_type : Dict[str, List[str]]
        A dictionary mapping node types to lists of node IDs.
        
    Returns
    -------
    Dict[Tuple[str, str], int]
        A dictionary mapping (node_type, original_id) tuples to type-specific indices.
    
    Notes
    -----
    This function ensures that node IDs within each type start from 0,
    which is required for PyTorch Geometric's heterogeneous graph format.
    """
    node_id_mapping = {}
    for node_type, node_ids in nodes_by_type.items():
        for idx, node_id in enumerate(node_ids):
            # Store mapping as (node_type, original_id) -> type_specific_idx
            node_id_mapping[(node_type, node_id)] = idx
    return node_id_mapping


def create_node_features(data: HeteroData, nodes_by_type: Dict[str, List[str]]) -> None:
    """
    Create feature matrices for each node type in the heterogeneous graph.
    
    Parameters
    ----------
    data : HeteroData
        The PyTorch Geometric HeteroData object to populate.
    nodes_by_type : Dict[str, List[str]]
        A dictionary mapping node types to lists of node IDs.
    
    Returns
    -------
    None
        This function modifies the data object in-place.
    
    Notes
    -----
    This implementation creates simple feature matrices where each node's feature
    is just its index. In a real application, you would use actual node features
    extracted from the graph_doc's properties.
    """
    for node_type, node_ids in nodes_by_type.items():
        num_nodes = len(node_ids)
        # Create a simple feature matrix (just node indices as features)
        data[node_type].x = torch.arange(num_nodes).view(-1, 1).float()


def extract_edge_types(graph_doc) -> Set[Tuple[str, str, str]]:
    """
    Extract all unique edge types from the graph document.
    
    Parameters
    ----------
    graph_doc : GraphDocument
        The graph document containing nodes and relationships.
        
    Returns
    -------
    Set[Tuple[str, str, str]]
        A set of (source_type, relation_type, target_type) tuples representing
        all unique edge types in the graph.
    
    Notes
    -----
    Edge types in PyTorch Geometric are defined as tuples of
    (source_node_type, edge_type, target_node_type).
    """
    edge_types = set()
    for rel in graph_doc.relationships:
        source_type = rel.source.type
        target_type = rel.target.type
        rel_type = rel.type
        edge_types.add((source_type, rel_type, target_type))
    return edge_types


def create_edge_indices(
    data: HeteroData,
    graph_doc,
    edge_types: Set[Tuple[str, str, str]],
    node_id_mapping: Dict[Tuple[str, str], int]
) -> None:
    """
    Create edge indices for each edge type in the heterogeneous graph.
    
    Parameters
    ----------
    data : HeteroData
        The PyTorch Geometric HeteroData object to populate.
    graph_doc : GraphDocument
        The graph document containing nodes and relationships.
    edge_types : Set[Tuple[str, str, str]]
        A set of (source_type, relation_type, target_type) tuples representing
        all unique edge types in the graph.
    node_id_mapping : Dict[Tuple[str, str], int]
        A dictionary mapping (node_type, original_id) tuples to type-specific indices.
    
    Returns
    -------
    None
        This function modifies the data object in-place.
    
    Raises
    ------
    ValueError
        If an edge references a node that doesn't exist in the graph.
    
    Notes
    -----
    This function creates edge indices for each edge type in the format required by
    PyTorch Geometric: a tensor of shape [2, num_edges] where the first row contains
    source node indices and the second row contains target node indices.
    """
    for source_type, rel_type, target_type in edge_types:
        # Collect all edges of this type
        edge_indices = []
        
        for rel in graph_doc.relationships:
            if (rel.source.type == source_type and 
                rel.target.type == target_type and 
                rel.type == rel_type):
                
                # Validate source node exists
                source_key = (source_type, rel.source.id)
                if source_key not in node_id_mapping:
                    raise ValueError(f"Unknown source node: {rel.source.id} of type {source_type}")
                
                # Validate target node exists
                target_key = (target_type, rel.target.id)
                if target_key not in node_id_mapping:
                    raise ValueError(f"Unknown target node: {rel.target.id} of type {target_type}")
                
                # Get type-specific source and target indices
                source_idx = node_id_mapping[source_key]
                target_idx = node_id_mapping[target_key]
                edge_indices.append([source_idx, target_idx])
        
        if edge_indices:
            # Convert to tensor with shape [2, num_edges]
            edge_index = torch.tensor(edge_indices).t().contiguous()
            data[source_type, rel_type, target_type].edge_index = edge_index


def convert_graph_document_to_hetero_data(graph_doc) -> HeteroData:
    """
    Convert a GraphDocument to a PyTorch Geometric heterogeneous graph.
    
    Parameters
    ----------
    graph_doc : GraphDocument
        The graph document containing nodes and relationships.
        
    Returns
    -------
    HeteroData
        A PyTorch Geometric heterogeneous graph object.
    
    Notes
    -----
    This function performs the following steps:
    1. Groups nodes by their type
    2. Creates a mapping from string node IDs to numerical indices per node type
    3. Creates feature matrices for each node type
    4. Extracts all unique edge types
    5. Creates edge indices for each edge type
    
    The resulting HeteroData object follows PyTorch Geometric's format for
    heterogeneous graphs, where node IDs within each type start from 0.
    """
    # Create HeteroData object
    data = HeteroData()
    
    # Step 1: Group nodes by type
    nodes_by_type = group_nodes_by_type(graph_doc)
    
    # Step 2: Create mapping from string IDs to numerical indices (per node type)
    node_id_mapping = create_node_id_mapping(nodes_by_type)
    
    # Step 3: Create feature matrices for each node type
    create_node_features(data, nodes_by_type)
    
    # Step 4: Extract all unique edge types
    edge_types = extract_edge_types(graph_doc)
    
    # Step 5: Create edge indices for each edge type
    create_edge_indices(data, graph_doc, edge_types, node_id_mapping)
    
    return data
