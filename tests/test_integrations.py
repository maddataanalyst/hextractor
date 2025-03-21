import pytest
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
from langchain_core.documents import Document
from hextractor.integrations.langchain_graphdoc import (
    convert_graph_document_to_hetero_data,
)


def test_valid_graph_extraction():
    text = """Filip Wójcik and Marcin Malczewski are data scientists, who developed HeXtractor. It is a library
    that helps in extracting heterogeneous knowledge graphs from various data source.
    Heterogeneous knowledge graphs are graphs that contain different types of nodes and edges."""

    docs = [Document(page_content=text)]

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
            source=docs[0],
        )
    ]

    graph_doc = data[0]
    hetero_data, node_mapping = convert_graph_document_to_hetero_data(graph_doc)

    assert hetero_data is not None
    assert sorted(hetero_data.node_types) == ["Graph", "Library", "Person"]
    assert sorted(hetero_data.edge_types) == [
        ("Library", "Extracts", "Graph"),
        ("Person", "Developed", "Library"),
    ]
    assert hetero_data["Person"].x.shape == (2, 1)
    assert hetero_data["Library"].x.shape == (1, 1)
    assert hetero_data["Graph"].x.shape == (1, 1)

    assert node_mapping == {
        ("Person", "Filip Wójcik"): 0,
        ("Person", "Marcin Malczewski"): 1,
        ("Library", "HeXtractor"): 0,
        ("Graph", "Heterogeneous knowledge graph"): 0,
    }


def test_invalid_graph_extraction():
    doc = Document(page_content="Michael Scott knows Pam and Elon Musk.")
    node_michael = Node(id="Michael Scott", type="Person")
    node_pam = Node(id="Pam", type="Person")
    node_elon_musk = Node(id="Elon Musk", type="Person")

    invalid_gd = GraphDocument(
        nodes=[
            node_michael,
            node_pam,
        ],
        relationships=[
            Relationship(source=node_michael, target=node_pam, type="knows"),
            Relationship(source=node_michael, target=node_elon_musk, type="knows"),
        ],
        source=doc,
    )

    with pytest.raises(ValueError, match="Unknown target node: Elon Musk of type Person"):
        convert_graph_document_to_hetero_data(invalid_gd)
