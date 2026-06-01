import torch
import torch.nn as nn
from omegaconf import OmegaConf
from anemoi.models.models.encoder_processor_decoder import AnemoiModelEncProcDec
from torch_geometric.data import HeteroData

class SPEARGNNModel(nn.Module):
    def __init__(self, input_dim, output_dim, config):
        super().__init__()

        graph_path = config.gnn['graph_path']
        raw_graph = torch.load(graph_path, weights_only=False)

        new_graph = HeteroData()

        # 1. Manually move Nodes and force the 'x' attribute
        for node_type in ['data', 'hidden']:
            coords = raw_graph[node_type].x if 'x' in raw_graph[node_type] else raw_graph[node_type]['x']
            new_graph[node_type].x = coords.float()

        # 2. Manually move Edges
        for edge_type in raw_graph.edge_types:
            new_graph[edge_type].edge_index = raw_graph[edge_type].edge_index

        # 3. Create the 'dataset' alias inside the NEW graph
        new_graph['dataset'].x = new_graph['data'].x

        model_config = OmegaConf.create({
            "model": {
                "num_channels": input_dim,
                "output_channels": output_dim,
                "trainable_parameters": {
                    "data": True,      # <--- Tell it to train the data nodes
                    "hidden": True     # <--- Tell it to train the hidden nodes
                },
                "model": {
                    "hidden_nodes_name": "hidden",
                    "data_nodes_name": "data",
                    "latent_skip": False
                },
                "encoder": {
                    "num_layers": 1,
                    "num_heads": 4
                },
                "processor": {
                    "num_layers": 4,
                    "num_channels": 128,
                    "num_heads": 4
                },
                "decoder": {
                    "num_layers": 1,
                    "num_heads": 4
                }
            }
        })

        data_indices = {
             "prognostic": list(range(input_dim)),
             "forcing": [],
             "diagnostic": []
        }

        self.gnn = AnemoiModelEncProcDec(
            model_config=model_config,
            data_indices=data_indices,
            graph_data=new_graph,
            statistics={},
            n_step_input=1,
            n_step_output=1
        )

    def forward(self, x):
        batch_size, channels, lat, lon = x.shape

        # Flatten the spatial dimensions: [Batch, Channels, Nodes]
        x_flat = x.view(batch_size, channels, -1)

        # Swap axes for the Anemoi GNN: [Batch, Nodes, Channels]
        x_nodes = x_flat.permute(0, 2, 1)

        # Pass through the Encoder -> Processor -> Decoder
        out_nodes = self.gnn(x_nodes)

        # Swap back to match SPEAR: [Batch, Channels, Nodes]
        out_flat = out_nodes.permute(0, 2, 1)

        # Unflatten to grid: [Batch, Channels, Lat, Lon]
        out_grid = out_flat.view(batch_size, -1, lat, lon)

        return out_grid

def construct_gnn_model(config, input_dim, output_dim):
    return SPEARGNNModel(input_dim=input_dim, output_dim=output_dim, config=config)
