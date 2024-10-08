"""HyperSMAC implementation."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from hydra.utils import get_class
from omegaconf import DictConfig, OmegaConf
from smac import Scenario
from smac.runhistory.dataclasses import TrialInfo, TrialValue

from hydra_plugins.hypersweeper.search_space_encoding import \
    search_space_to_config_space
from hydra_plugins.hypersweeper.utils import Info, convert_to_configuration

if TYPE_CHECKING:
    from ConfigSpace import Configuration


def read_additional_configs(initial_design_fn: str, search_space: DictConfig) -> list[Configuration]:
    """Read configurations from csv-logfile.

    Parameters
    ----------
    initial_design_fn : str
        The path to the log file.
    search_space : DictConfig
        The search space which will be converted to a ConfigSpace.ConfigurationSpace, by default None.
        The search space can be loaded via `search_space = OmegaConf.load(search_space_fn)`.

    Returns.
    --------
    list[Configuration]
        The configurations from the log file.
    """
    configspace = search_space_to_config_space(search_space=search_space)
    initial_design = pd.read_csv(initial_design_fn)
    return initial_design.apply(convert_to_configuration, args=(configspace,), axis=1).to_list()


OmegaConf.register_new_resolver("get_class", get_class, replace=True)
OmegaConf.register_new_resolver("read_additional_configs", read_additional_configs, replace=True)


class HyperSMACAdapter:
    """Adapt SMAC ask/tell interface to HyperSweeper ask/tell interface."""

    def __init__(self, smac):
        """Initialize the adapter."""
        self.smac = smac
        self.config_ids = {}
        self.budget_ids = {}
        self.c_config_id = 0
        self.c_budget_id = 0

        self.last_budget = {}

    @staticmethod
    def config_to_key(config) -> str:
        """Convert a configuration to a unique string key."""
        return "$".join([f"{v}" for v in config.get_dictionary().values()])        
    
    def assign_budget_id(self, budget: float) -> int:
        """Assign a unique ID to the budget."""
        if budget in self.budget_ids:
            return self.budget_ids[budget]
        else:
            self.budget_ids[budget] = self.c_budget_id
            self.c_budget_id += 1
            return self.budget_ids[budget]
    
    def get_load_and_save_paths(self, smac_info) -> tuple[str | None, str | None]:
        """Get the load path for the configuration."""
        config_key = self.config_to_key(smac_info.config)

        # If we have not seen this budget before, 
        # we have to assign a new ID
        budget_id = self.assign_budget_id(smac_info.budget)

        if config_key in self.config_ids:
            # We have already executed this configuration
            # so we need to load the model from the previous run
            config_id = self.config_ids[config_key]
            last_budget_id = self.last_budget[config_key]
            load_path = f"budget_{last_budget_id}_config_{config_id}"
        else:
            self.config_ids[config_key] = self.c_config_id
            self.c_config_id += 1

            self.last_budget[config_key] = budget_id
            load_path = "none"
        
        save_path = f"budget_{budget_id}_config_{self.config_ids[config_key]}"

        return load_path, save_path
        
    def get_save_path(self, smac_info) -> str:
        """Get the save path for the configuration."""
        config_key = self.config_to_key(smac_info.config)
        budget_id = self.last_budget[config_key]
        config_id = self.config_ids[config_key]
        return f"budget_{budget_id}_config_{config_id}"

    def ask(self):
        """Ask for the next configuration."""
        smac_info = self.smac.ask()
        load_path, save_path = self.get_load_and_save_paths(smac_info)

        info = Info(
            config=smac_info.config,
            budget=smac_info.budget,
            save_path=save_path,
            load_path=load_path,
            seed=smac_info.seed
        )

        return info, False

    def tell(self, info, value):
        """Tell the result of the configuration."""
        smac_info = TrialInfo(info.config, seed=info.seed, budget=info.budget)
        smac_value = TrialValue(time=value.cost, cost=value.performance)
        self.smac.tell(smac_info, smac_value)


def make_smac(configspace, smac_args):
    """Make a SMAC instance for optimization."""

    def dummy_func(arg, seed, budget):  # noqa:ARG001
        return 0.0

    if "output_directory" in smac_args["scenario"]:
        smac_args["scenario"]["output_directory"] = Path(smac_args["scenario"]["output_directory"])
    scenario = Scenario(configspace, **smac_args.pop("scenario"))
    smac_kwargs = {}

    if "callbacks" not in smac_args:
        smac_kwargs["callbacks"] = []
    elif "callbacks" in smac_args and isinstance(smac_args["callbacks"], dict):
        smac_kwargs["callbacks"] = list(smac_args["callbacks"].values())
    elif "callbacks" in smac_args and isinstance(smac_args["callbacks"], list):
        smac_kwargs["callbacks"] = smac_args["callbacks"]

    if "acquisition_function" in smac_args and "acquisition_maximizer" in smac_args:
        smac_kwargs["acquisition_maximizer"] = smac_args["acquisition_maximizer"](
            configspace=configspace,
            acquisition_function=smac_args["acquisition_function"],
        )
        if hasattr(smac_args["acquisition_maximizer"], "selector") and hasattr(
            smac_args["acquisition_maximizer"].selector, "expl2callback"
        ):
            smac_kwargs["callbacks"].append(smac_args["acquisition_maximizer"].selector.expl2callback)

    if "config_selector" in smac_args:
        smac_kwargs["config_selector"] = smac_args["config_selector"](scenario=scenario)

    if "initial_design" in smac_args:
        smac_kwargs["initial_design"] = smac_args["initial_design"](scenario=scenario)

    if "intensifier" in smac_args:
        smac_kwargs["intensifier"] = smac_args["intensifier"](scenario)

    smac = smac_args["smac_facade"](scenario, dummy_func, **smac_kwargs)
    return HyperSMACAdapter(smac)


if __name__ == "__main__":
    read_additional_configs()
