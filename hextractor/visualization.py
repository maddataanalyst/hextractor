import hextractor.structures as structures
import torch_geometric.data as pyg_data
from pyvis.network import Network
from typing import Dict


class VisualizationBuilder:
    """A visualization builder, that constructs PyVis network from the heterogeneous graph data
    with the config provided."""

    @staticmethod
    def check_config_with_graph(
        vis_config: structures.VisualizationConfig, hetero_g: pyg_data.HeteroData
    ):
        """Checks if the visualization config is consistent with the graph data.

        Parameters
        ----------
        vis_config : structures.VisualizationConfig
            Visualization configuration to check.

        hetero_g : pyg_data.HeteroData
            Constructed heterogeneous graph data.
        """
        for node_type in vis_config.all_node_types:
            if node_type not in hetero_g.node_types:
                raise ValueError(f"Node type {node_type} is missing.")

        for edge_type in vis_config.all_edge_types:
            if edge_type not in hetero_g.edge_types:
                raise ValueError(f"Edge type {edge_type} is missing.")

        for edge, attr_idx in vis_config.edge_weights_attr_idx.items():
            edge_attr_name = vis_config.get_edge_weight_attr_name(edge)
            if edge_attr_name is None or edge_attr_name not in hetero_g[edge]:
                raise ValueError(
                    f"Edge attribute {edge_attr_name} is missing for {edge}."
                )
            edge_attrs_shape = hetero_g[edge][edge_attr_name].shape
            if attr_idx >= edge_attrs_shape[1]:
                raise ValueError(
                    f"Edge attribute {edge_attr_name} is missing index: {attr_idx} for {edge}."
                )
        for (
            node_type,
            node_label_attr_name,
        ) in vis_config.node_type_label_attr_name.items():
            if node_type not in hetero_g.node_types:
                raise ValueError(f"Node type {node_type} is missing.")
            if node_label_attr_name not in hetero_g[node_type]:
                raise ValueError(
                    f"Node label attribute {node_label_attr_name} is missing for node type {node_type}."
                )
            attr_name_idx = vis_config.get_node_label_attr_idx(node_type)
            if (
                attr_name_idx is None
                or attr_name_idx >= hetero_g[node_type][node_label_attr_name].shape[1]
            ):
                raise ValueError(
                    f"Label attribute index {attr_name_idx} is invalid for node type {node_type}."
                )

    @staticmethod
    def build_node_label(
        node_type: str,
        node_id: int,
        hetero_g: pyg_data.HeteroData,
        vis_config: structures.VisualizationConfig,
        label_mapping: Dict[str, Dict[int, str]],
    ) -> str:
        """Builds a node label for the visualization.

        Parameters
        ----------
        node_type : str
            Node type for which the label is built.
        node_id : int
            Particular node id
        hetero_g : pyg_data.HeteroData
            Constructed heterogeneous graph data.
        vis_config : structures.VisualizationConfig
            Configuration for visualization of the heterogeneous graph.
        label_mapping : Dict[str, Dict[int, str]]
            Mapping of node type to node id to label.

        Returns
        -------
        str
            Node label for the visualization.
        """
        if node_type not in label_mapping:
            return f"{node_type} {node_id}"
        node_type_mappings = label_mapping[node_type]
        attr_values_matrix = hetero_g[node_type][
            vis_config.get_node_label_attr_name(node_type)
        ]
        attr_val = attr_values_matrix[
            node_id, vis_config.get_node_label_attr_idx(node_type)
        ].item()
        if attr_val not in node_type_mappings:
            return f"{node_type} {node_id}"
        return node_type_mappings[attr_val]

    @staticmethod
    def build_visualization(
        vis_config: structures.VisualizationConfig,
        hetero_g: pyg_data.HeteroData,
        node_type_label_mappings: Dict[str, Dict[int, str]] = {},
    ) -> Network:
        """Builds a network visualization from the heterogeneous graph data, from the
        provided configuration and (optional) node label mappings.

        Parameters
        ----------
        vis_config : structures.VisualizationConfig
            Configuration for the visualization of the heterogeneous graph.
        hetero_g : pyg_data.HeteroData
            Heterogeneous graph structure.
        node_type_label_mappings : Dict[str, Dict[int, str]], optional
            Node label mapping - from ids to names., by default {}

        Returns
        -------
        Network
            PyVis interactive network visualization.
        """
        VisualizationBuilder.check_config_with_graph(vis_config, hetero_g)
        network = Network(
            notebook=vis_config.notebook_visualization,
            select_menu=vis_config.select_menu,
            filter_menu=vis_config.filter_menu,
            width=vis_config.width,
            height=vis_config.height,
        )
        for edge in vis_config.all_edge_types:
            src_node_type, rel, trg_node_type = edge
            src_node_color = vis_config.get_node_color(src_node_type)
            trg_node_color = vis_config.get_node_color(trg_node_type)
            edge_weight_attr = vis_config.get_edge_weight_attr_idx(edge)
            edge_weight_attr_name = vis_config.get_edge_weight_attr_name(edge)
            edge_color_attr = vis_config.get_edge_color(edge)
            for edge_idx, (s_idx, t_idx) in enumerate(hetero_g[edge].edge_index.T):
                s_idx = s_idx.item()
                t_idx = t_idx.item()
                s_label = VisualizationBuilder.build_node_label(
                    src_node_type,
                    s_idx,
                    hetero_g,
                    vis_config,
                    node_type_label_mappings,
                )
                t_label = VisualizationBuilder.build_node_label(
                    trg_node_type,
                    t_idx,
                    hetero_g,
                    vis_config,
                    node_type_label_mappings,
                )
                s_id = f"{src_node_type}_{s_idx}"
                t_id = f"{trg_node_type}_{t_idx}"
                network.add_node(s_id, label=s_label, color=src_node_color)
                network.add_node(t_id, label=t_label, color=trg_node_color)
                edge_weight = (
                    vis_config.default_edge_weight
                    if edge_weight_attr is None
                    else hetero_g[edge][edge_weight_attr_name][
                        edge_idx, edge_weight_attr
                    ].item()
                )
                network.add_edge(
                    s_id, t_id, color=edge_color_attr, label=rel, width=edge_weight
                )
        network.show_buttons(vis_config.buttons)
        return network
