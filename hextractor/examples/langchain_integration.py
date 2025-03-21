"""
Langchain GraphDocument Integration Example.

This module demonstrates how to integrate Langchain's GraphDocument with a simple example.
It shows how to create nodes and relationships to form a heterogeneous knowledge graph
from a given text. The example includes creating nodes for persons, a library, and a graph,
and establishing relationships between them.

Functions
---------
get_text() -> str
    Returns a sample text describing the developers of HeXtractor and its purpose.

get_example_langchain_graphdocument()
    Creates an example GraphDocument using Langchain, with nodes and relationships
    based on the sample text.
"""

import autoroot  # noqa
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
from langchain_core.documents import Document


def get_text() -> str:
    """
    Get sample text.

    This function returns a sample text that describes the developers of HeXtractor
    and its purpose. The text is used to create nodes and relationships in the graph.

    Returns
    -------
    str
        Sample text describing the developers and purpose of HeXtractor.
    """
    return """Filip Wójcik and Marcin Malczewski are data scientists, who developed HeXtractor. It is a library
that helps in extracting heterogeneous knowledge graphs from various data source.
Heterogeneous knowledge graphs are graphs that contain different types of nodes and edges."""


def get_example_langchain_graphdocument():
    """
    Create an example Langchain GraphDocument.

    This function creates an example GraphDocument using Langchain. It defines nodes
    for persons (Filip Wójcik and Marcin Malczewski), a library (HeXtractor), and a graph
    (Heterogeneous knowledge graph). It also establishes relationships between these nodes
    to form a heterogeneous knowledge graph.

    Returns
    -------
    list of GraphDocument
        A list containing a single GraphDocument with the defined nodes and relationships.
    """
    doc = Document(page_content=get_text())
    fw_node = Node(type="Person", id="Filip Wójcik")
    mm_node = Node(type="Person", id="Marcin Malczewski")
    hx_node = Node(type="Library", id="HeXtractor")
    kg_node = Node(type="Graph", id="Heterogeneous knowledge graph")

    fw_developed_hx = Relationship(source=fw_node, target=hx_node, type="Developed")
    mm_developer_hx = Relationship(source=mm_node, target=hx_node, type="Developed")
    hx_extracts_kg = Relationship(source=hx_node, target=kg_node, type="Extracts")

    data = [
        GraphDocument(
            nodes=[fw_node, mm_node, hx_node, kg_node],
            relationships=[fw_developed_hx, mm_developer_hx, hx_extracts_kg],
            source=doc,
        )
    ]
    return data
