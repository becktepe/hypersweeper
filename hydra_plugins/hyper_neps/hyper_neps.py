"""Hypersweeper Interface for NEPS."""

from __future__ import annotations

import importlib
import math
import random
import time
from pathlib import Path

import numpy as np
from neps.runtime import Trial

if (spec := importlib.util.find_spec("neps")) is not None:
    import neps
    import neps.search_spaces

from typing import TYPE_CHECKING

from ConfigSpace.hyperparameters import (CategoricalHyperparameter,
                                         NormalFloatHyperparameter,
                                         NormalIntegerHyperparameter,
                                         UniformFloatHyperparameter,
                                         UniformIntegerHyperparameter)

from ..hyper_adapter import HyperAdapter
from hydra_plugins.hypersweeper import Info, Result
from hydra_plugins.hypersweeper.utils import dynamic_import_and_call
import hydra

if TYPE_CHECKING:
    from ConfigSpace import ConfigurationSpace
    from neps.optimizers.base_optimizer import BaseOptimizer


class HyperNEPS(HyperAdapter):
    """NEPS."""

    def __init__(self, configspace: ConfigurationSpace, optimizer: BaseOptimizer, fidelity_variable: str) -> None:
        """Initialize the optimizer."""
        self.configspace = configspace
        self.optimizer = optimizer
        self.previous_results = {}
        self.pending_evaluations = {}
        self.fidelity_variable = fidelity_variable

    def ask(self) -> tuple[Info, bool]:
        """Sample a new configuration."""
        self.optimizer.load_results(
            previous_results={
                config_id: report.to_config_result(self.optimizer.load_config)
                for config_id, report in self.previous_results.items()
            },
            pending_evaluations={
                config_id: self.optimizer.load_config(trial.config)
                for config_id, trial in self.pending_evaluations.items()
            },
        )

        config, config_id, prev_config_id = self.optimizer.get_config_and_ids()
        previous = None
        if prev_config_id is not None:
            previous = self.previous_results[prev_config_id]

        time_sampled = time.time()

        trial = Trial(
            id=config_id,
            config=config,
            report=None,
            time_sampled=time_sampled,
            pipeline_dir=Path(),  # TODO
            previous=previous,
            metadata={"time_sampled": time_sampled},
        )

        self.pending_evaluations[config_id] = trial

        config_dict = dict(config)
        budget = config_dict.pop(self.fidelity_variable)
        if "architecture" in config:
            config_dict["architecture"] = f'"{config["architecture"].string_tree}"'

        info = Info(
            config=config_dict,
            budget=budget,
            config_id=config_id,
        )
        return info, False

    def tell(self, info: Info, result: Result) -> None:
        """Return the performance."""
        trial = self.pending_evaluations.pop(info.config_id)

        performance = result.performance
        if isinstance(performance, float) or (isinstance(performance, dict) and len(performance) == 1):
            performance = next(iter(performance.values()))

        trial.report = trial.create_success_report(result.performance)

        self.previous_results[info.config_id] = trial


def make_neps(configspace, hyper_neps_args):
    """Make a NEPS instance for optimization."""
    # important for NePS optimizers
    random.seed(hyper_neps_args["seed"])

    np.random.seed(hyper_neps_args["seed"])  # noqa: NPY002

    dict_search_space = get_dict_from_configspace(configspace)

    if "fidelity_variable" in hyper_neps_args:
        dict_search_space[hyper_neps_args["fidelity_variable"]] = neps.FloatParameter(
            lower=hyper_neps_args["min_budget"], upper=hyper_neps_args["max_budget"], is_fidelity=True
        )

    if "architecture" in hyper_neps_args:
        arch_parameter = dynamic_import_and_call(hyper_neps_args["architecture"])
        arch_parameter.default = hyper_neps_args["architecture_default"]
        dict_search_space["architecture"] = arch_parameter

    neps_search_space = neps.search_spaces.SearchSpace(**dict_search_space)

    optimizer = hyper_neps_args["optimizer"](
        pipeline_space=neps_search_space,
    )

    return HyperNEPS(
        configspace=configspace, optimizer=optimizer, fidelity_variable=hyper_neps_args["fidelity_variable"]
    )


def get_dict_from_configspace(configspace: ConfigurationSpace) -> dict:
    """Get a dictionary containing NEPS hyperparameters from a ConfigSpace object."""
    search_space = {}
    for k in configspace:
        param = configspace[k]
        if isinstance(param, NormalFloatHyperparameter | UniformFloatHyperparameter):
            search_space[k] = neps.FloatParameter(
                lower=param.lower,
                upper=param.upper,
                log=param.log,
                default=param.default_value,
                default_confidence="medium",
            )
        elif isinstance(param, NormalIntegerHyperparameter | UniformIntegerHyperparameter):
            search_space[k] = neps.IntegerParameter(
                lower=param.lower,
                upper=param.upper,
                log=param.log,
                default=param.default_value,
                default_confidence="medium",
            )
        elif isinstance(param, CategoricalHyperparameter):
            search_space[k] = neps.CategoricalParameter(
                choices=param.choices, default=param.default_value, default_confidence="medium"
            )
    return search_space


def check_budget_levels(min_epoch, max_epoch, eta):
    """Check the Hyperband budget levels for NEPS."""
    total_budget = 0
    _min = max_epoch
    counter = 0
    fid_level = math.ceil(math.log(max_epoch / min_epoch) / math.log(eta))
    while _min >= min_epoch:
        print(f"Level: {fid_level} -> {_min}")
        total_budget += _min * eta
        _min = _min // eta
        counter += 1
        fid_level -= 1
