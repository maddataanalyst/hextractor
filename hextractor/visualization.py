import hextractor.structures as structures
import torch_geometric.data as pyg_data
from pyvis.network import Network
from typing import Dict


class VisualizationBuilder:
    """A visualization builder, that constructs PyVis network from the heterogeneous graph data
    with the config provided."""

    @staticmethod
    def check_config_with_graph(
        vis_config: structures.VisualizationConfig,
        hetero_g: pyg_data.HeteroData,
        node_labels_mapping: Dict[int, str],
    ):
        """Checks if the visualization config is consistent with the graph data.

        Parameters
        ----------
        vis_config : structures.VisualizationConfig
            Visualization configuration to check.

        hetero_g : pyg_data.HeteroData
            Constructed heterogeneous graph data.

        node_labels_mapping : Dict[int, str]
            Mapping of node index to node label.
        """
        for node_type in vis_config.all_node_types:
            if node_type not in hetero_g.node_types:
                raise ValueError(f"Node type {node_type} is missing.")

        for edge_type in vis_config.all_edge_types:
            if edge_type not in hetero_g.edge_types:
                raise ValueError(f"Edge type {edge_type} is missing.")

        for edge, attr_idx in vis_config.edge_weights_attr_idx.items():
            edge_attrs_shape = hetero_g[edge].edge_attr.shape
            if attr_idx >= edge_attrs_shape[1]:
                raise ValueError(f"Edge attribute {attr_idx} is missing for {edge}.")
        for (
            node_type,
            node_label_attr_name,
        ) in vis_config.node_type_label_attr_name.items():
            if node_type not in hetero_g.node_types:
                raise ValueError(f"Node type {node_type} is missing.")
            if node_label_attr_name not in hetero_g[node_type]:
                raise ValueError(
                    f"Node label attribute {node_label_attr_name} is missing for {node_type}."
                )
            attr_name_idx = vis_config.get_node_label_attr_idx(node_type)
            if (
                attr_name_idx is None
                or attr_name_idx >= hetero_g[node_type][node_label_attr_name].shape[1]
            ):
                raise ValueError(
                    f"Label attribute index {attr_name_idx} is invalid for {node_type}."
                )

    def build_visualization(
        vis_config: structures.VisualizationConfig,
        hetero_g: pyg_data.HeteroData,
    ) -> Network:
        VisualizationBuilder.check_config_with_graph(vis_config, hetero_g)

        network = Network(
            notebook=vis_config.notebook_visualization
        )  # TODO: later on add more options in visualization config
        for edge in vis_config.all_edge_types:
            src_node_type, rel, trg_node_type = edge
            src_node_color = vis_config.get_node_color(src_node_type)
            trg_node_color = vis_config.get_node_color(trg_node_type)
            edge_weight_attr = vis_config.get_edge_weight_attr_idx(edge)
            edge_color_attr = vis_config.get_edge_color(edge)
            edge_index = hetero_g[edge].edge_index.T.numpy()
            for edge_idx, (s_idx, t_idx) in enumerate(hetero_g[edge].edge_index.T):
                # TODO: later add support for node labels
                s_idx = f"{src_node_type} {s_idx.item()}"
                t_idx = f"{trg_node_type} {t_idx.item()}"
                network.add_node(s_idx, label=s_idx, color=src_node_color)
                network.add_node(t_idx, label=t_idx, color=trg_node_color)
                edge_weight = (
                    vis_config.default_edge_weight
                    if edge_weight_attr is None
                    else hetero_g[edge][edge_idx, edge_weight_attr].item()
                )
                network.add_edge(
                    s_idx, t_idx, color=edge_color_attr, label=rel, width=edge_weight
                )
        return network
