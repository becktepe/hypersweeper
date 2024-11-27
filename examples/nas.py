from __future__ import annotations

from torch import nn

import hydra
from neps.search_spaces.architecture.get_example_architecture import get_architecture
from omegaconf import DictConfig



@hydra.main(config_path="configs", config_name="mlp_smac", version_base="1.1")
def nas(cfg: DictConfig):
    arch_string_tree = str(cfg.architecture).replace("'", "")

    architecture = get_architecture()

    architecture.string_tree = arch_string_tree
    architecture._value = architecture.from_stringTree_to_graph_repr(
        architecture.string_tree,
        architecture.grammars[0],
        valid_terminals=architecture.terminal_to_op_names.keys(),
        edge_attr=architecture.edge_attr,
    )

    in_channels = 3
    base_channels = 16
    n_classes = 10
    out_channels_factor = 4
    
    # E.g., in shape = (N, 3, 32, 32) => out shape = (N, 10)
    model = architecture.to_pytorch()
    model = nn.Sequential(
        nn.Conv2d(in_channels, base_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(base_channels),
        model,
        nn.BatchNorm2d(base_channels * out_channels_factor),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(base_channels * out_channels_factor, n_classes),
    )
    return {"loss": 1}

if __name__ == "__main__":
    nas()
