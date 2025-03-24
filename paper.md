---
title: 'HeXtractor: A Tool for Building Heterogeneous Graphs from Structured and Textual Data for Graph Neural Networks'
tags:
  - Python
  - graph neural networks
  - heterogeneous graphs
  - tabular data
  - knowledge graphs
  - data extraction
  - PyTorch Geometric
authors:
  - name: Filip Wójcik
    orcid: 0000-0001-5938-7260
    affiliation: 1
  - name: Marcin Malczewski
affiliations:
 - name: Wroclaw University of Economics and Business, Wrocław, Poland
   index: 1
date: 24 March 2025
bibliography: paper.bib
---

# Summary

**HeXtractor** is an open-source Python library designed to transform structured tabular data and unstructured textual content into heterogeneous graph representations suitable for use in Graph Neural Networks (GNNs). Fully compatible with the PyTorch Geometric (PyG) framework [@Fey_Fast_Graph_Representation_2019], HeXtractor provides a streamlined, high-level interface for defining entities (nodes), relationships (edges), and associated metadata across diverse data modalities.

GNNs have become increasingly prominent with the rise of the Message Passing Neural Network (MPNN) paradigm [@gilmer2017neural]. In particular, **heterogeneous graphs**, which support multiple node and edge types, are gaining traction in domains such as recommendation systems, fraud detection, and knowledge representation [@Yang2020; @Shi2022]. Advanced architectures—including Heterogeneous Graph Transformers [@Hu2020] and Heterogeneous Graph Attention Networks [@Wang2019]—are specifically designed to leverage the rich semantics of these graph types.

A key application of heterogeneous graphs is in **knowledge graph construction**, which models complex, real-world relationships. Such graphs are used in a variety of industries, including job-market matching [@noy2019industry; @chen2018linkedin] and credit risk analysis [@mitra2024knowledge].

Despite their utility, constructing heterogeneous graphs remains a labor-intensive and error-prone task. HeXtractor addresses this challenge by offering a standardized, automated tool to convert structured and unstructured data into GNN-compatible formats, with optional support for large language models (LLMs) for extracting structure from text.

# Statement of Need

A **heterogeneous graph** is formally defined as a tuple $G = (V, E)$, where $V$ and $E$ represent sets of nodes and edges, respectively. Each node $v \in V$ and edge $e \in E$ is associated with a type mapping: $\phi(v): V \rightarrow A$ and $\Phi(e): E \rightarrow R$, where $A$ and $R$ denote sets of node and edge types [@Shi2022]. These graphs capture both the structural and semantic heterogeneity inherent in many real-world datasets.

While libraries such as PyG [@Fey_Fast_Graph_Representation_2019] and DGL [@wang2019dgl] provide robust learning capabilities for such graphs, they offer limited tooling for graph construction—particularly when data is distributed across multiple heterogeneous sources. As a result, researchers often rely on custom-built scripts, introducing variability and undermining reproducibility.

**HeXtractor** addresses this gap by providing:

- A declarative interface for defining node and edge schemas;
- Integration with LLMs for extracting graph structure from natural language using LangChain-compatible `GraphDocument` objects;
- Schema validation and consistency checks;
- Interactive graph visualization capabilities;
- Seamless export to PyG’s `HeteroData` format.

Initially developed as part of the **HexGIN** project [@Wojcik2024], which focused on analyzing financial transaction data, HeXtractor has evolved into a domain-agnostic tool for heterogeneous graph extraction.

# Features and Usage

HeXtractor enables graph construction from both structured tabular datasets and unstructured textual content. It also supports visualization and full interoperability with the PyTorch Geometric framework.

## Structured Data Extraction

HeXtractor supports both single-table and multi-table data processing. In single-table mode, each row encodes a relationship among column-defined entities. Users specify:

1. Node types and their attributes;
2. Edge definitions among the entities.

This yields a PyG-compatible `HeteroData` object ready for downstream modeling:

```python
HeteroData(
  company={ x=[3, 2] },
  employee={ x=[7, 2], y=[7] },
  tag={ x=[5] },
  (company, has, employee)={ edge_index=[2, 6] },
  (company, has, tag)={ edge_index=[2, 7] }
)
```

Interactive visualization is supported, with customizable labels and color schemes to aid interpretability.

![Graph extracted from structured data. \label{fig:single_tab_company}](paper_figures/company_diagram.png)

In multi-table mode, HeXtractor utilizes user-defined GraphSpecs to combine entity and relationship tables into a unified heterogeneous graph.

![Entity relationship diagram. \label{fig:er_diagram}](paper_figures/er_diagram.png)

## Text-Based Graph Extraction

Through integration with **LangChain**, HeXtractor supports automated extraction of semantic graph structures from natural language. The process is as follows:

1. Input text is processed by an LLM.
2. The model outputs a `GraphDocument` containing nodes and relationships.
3. HeXtractor converts the result into a `HeteroData` object.

For instance, the following input:

> Marcin Malczewski and Filip Wójcik are data scientists  
> who developed HeXtractor. It helps extract heterogeneous  
> knowledge graphs from various data sources.

is transformed into the following heterogeneous graph:

![Graph extracted from text. \label{fig:llm_diagram}](paper_figures/llm_diagram.png)

This functionality is particularly valuable for **knowledge graph creation** and **automated document analysis**.

## Visualization

HeXtractor leverages **NetworkX** and **PyVis** to provide rich, interactive graph visualizations. Users can configure node types, edge labels, and layout styles, facilitating both interpretability and validation prior to training.

# Example Use Cases

HeXtractor is designed to be domain-agnostic and scalable, accommodating datasets of varying size and complexity. Its capabilities are broadly applicable across numerous research and industrial contexts, including:

- **Banking and fraud detection** [@Johannessen2023; @Wojcik2024]  
- **Recommendation systems** [@Deng2022; @Wu2022]  
- **Biomedical knowledge graphs** [@Jumper2021; @maclean2021knowledge]  

In each of these areas, HeXtractor enables the integration of multiple data sources—both structured and unstructured—into cohesive, semantically enriched graph representations. In the absence of a tool like HeXtractor, this process would typically require significant manual engineering and carry risks of inconsistency.

# Documentation

Comprehensive documentation, including usage examples and full API reference, is available at:  
[https://hextractor.readthedocs.io/en/latest/](https://hextractor.readthedocs.io/en/latest/)

# Acknowledgements

We gratefully acknowledge the maintainers of **PyTorch Geometric**, **NetworkX**, **LangChain**, and **pandas** for their foundational contributions. This project received no direct financial support.

# References
