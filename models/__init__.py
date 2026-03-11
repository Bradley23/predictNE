from .lstm import predictNE
from .GNN import GNN_NE

def build_model(config):
    if config["model"] == "lstm":
        return predictNE(**config["lstm_params"])
    elif config["model"] == "gnn":
        return GNN_NE(**config["gnn_params"])
    else:
        raise ValueError("Unknown model type")