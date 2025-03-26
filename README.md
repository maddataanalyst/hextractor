![Logo](./docs/assets/logo.png)

# Overview
`HeXtractor` is a tool designed to automatically convert selected data in tabular format into a PyTorch Geometric heterogeneous graph. As research into graph neural networks (GNNs) expands, the importance of heterogeneous graphs grows. However, data often comes in tabular form, and manually transforming this data into graph format can be tedious and error-prone. `HeXtractor` aims to streamline this process, providing researchers and practitioners with a more efficient workflow.

# Features
1. Automatic Conversion: Converts tabular data into heterogeneous graphs suitable for GNNs.
2. Support for Multiple Formats: Handles various tabular data formats with ease.
3. Integration with PyTorch Geometric: Directly creates graphs that can be used with PyTorch Geometric.
4. isualization: Utilizes NetworkX and PyVis for graph visualization.

# Why HeXtractor?
Heterogeneous graphs are crucial in many applications of graph neural networks, yet creating them from tabular data manually is often cumbersome. `HeXtractor` automates this process, allowing researchers to focus on developing and training their models instead of data preprocessing.

# Technologies
1. `Python`: The primary programming language used for HeXtractor.
2. `pandas`: Utilized for data manipulation and handling tabular data.
3. `PyTorch` Geometric: Framework for creating and working with graph neural networks.
4. `NetworkX`: Used for creating and managing complex graph structures.
5. `PyVis`: Enables interactive visualization of graphs.

# Installation

## From PyPI

To install the latest version from PyPI run:

```bash
pip install hextractor
```

## Manual from source code

1. Make sure, that you have Anaconda or Miniconda installed.
2.Then, create new conda env from the provided environment.yml file:
```bash
conda env create -f environment.yml
```
3. Activate environment:
```bash
conda activate hextractor
```
4. Install [**poetry**](https://python-poetry.org/docs/) - main package manager used by this project
```bash
pip install poetry
```
5. Install the package with all dependencies:
```bash
poetry install --with dev --with research
```

To use package, remember to activate the environment.

# Documentation

You can find an official, detailed documentation [here](https://hextractor.readthedocs.io/en/latest/).