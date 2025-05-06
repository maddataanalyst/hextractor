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

## From Source Code

You can install HeXtractor from source using either conda or any standard Python virtual environment. We use Poetry as our primary dependency manager because it provides robust dependency resolution, reproducible builds, and better package management.

### Option 1: Using Conda

1. If you prefer Conda for environment management:
```bash
# Create a new conda environment from the provided file
conda env create -f environment.yml


# Activate the environment
conda activate hextractor

# Install poetry inside the conda environment
pip install poetry

# Install the package with all dependencies
poetry install --with dev --with research
```

### Option 2: Using Standard Python Virtual Environment

1. Create and activate a virtual environment using your preferred method:
```bash
# Using venv (Python 3.3+)
python -m venv hextractor-env
source hextractor-env/bin/activate  # On Windows: hextractor-env\Scripts\activate

# Or using virtualenv
virtualenv hextractor-env
source hextractor-env/bin/activate  # On Windows: hextractor-env\Scripts\activate
```

2. Install Poetry and the package:
```bash
# Install poetry
pip install poetry

# Install the package with all dependencies
poetry install --with dev --with research
```

Remember to activate your environment (conda or virtual environment) whenever you want to use HeXtractor.

# Documentation

You can find an official, detailed documentation [here](https://hextractor.readthedocs.io/en/latest/).