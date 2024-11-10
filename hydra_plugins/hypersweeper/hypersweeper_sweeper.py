"""Base class for ask-tell sweepers."""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import wandb
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from hydra_plugins.hypersweeper.utils import Info, Result, RunConfig, read_warmstart_data

if TYPE_CHECKING:
    from ConfigSpace import Configuration, ConfigurationSpace
    from hydra.plugins.launcher import Launcher

log = logging.getLogger(__name__)


class HypersweeperSweeper:
    """Base class for ask-tell sweepers."""

    def __init__(
        self,
        global_config: DictConfig,
        global_overrides: list[str],
        launcher: Launcher,
        make_optimizer: Callable,
        budget_arg_name: str,
        load_arg_name: str,
        save_arg_name: str,
        cs: ConfigurationSpace,
        budget: int = 1_000_000,
        n_trials: int = 1_000_000,
        objectives: list[str] = ["loss"],
        maximize: list[bool] = [False],
        optimizer_kwargs: dict[str, str] | None = None,
        seeds: list[int] | None = None,
        seed_keyword: str = "seed",
        slurm: bool = False,
        slurm_timeout: int = 10,
        max_parallelization: float = 0.1,
        job_array_size_limit: int = 100,
        max_budget: int | None = None,
        base_dir: str | None = None,
        min_budget: str | None = None,
        wandb_project: str | None = None,
        wandb_entity: str | None = None,
        wandb_tags: list[str] | None = None,
        deterministic: bool = True,
        checkpoint_tf: bool = False,
        load_tf: bool = False,
        checkpoint_path_typing: str = ".pt",
        warmstart_file: str | None = None,
    ):
        """Ask-Tell sweeper for hyperparameter optimization.

        Parameters
        ----------
        global_config: DictConfig
            The global configuration
        global_overrides: List[str]
            Global overrides for all jobs
        launcher: Launcher
            A hydra launcher (usually either for local runs or slurm)
        make_optimizer: Callable
            Function to create the optimizer object
        optimizer_kwargs: dict[str, str]
            Optimizer arguments
        budget_arg_name: str
            Name of the argument controlling the budget, e.g. num_steps.
        load_arg_name: str
            Name of the argument controlling the loading of agent parameters.
        save_arg_name: str
            Name of the argument controlling the checkpointing.
        n_trials: int
            Number of trials to run
        cs: ConfigSpace
            Configspace object containing the hyperparameter search space.
        seeds: List[int]
            If not False, optimization will be run and averaged across the given seeds.
        seed_keyword: str = "seed"
            Keyword for the seed argument
        slurm: bool
            Whether to use slurm for parallelization
        slurm_timeout: int
            Timeout for slurm jobs, used for scaling the timeout based on budget
        max_parallelization: float
            Maximum parallelization factor.
            1 will run all jobs in parallel, 0 will run completely sequentially.
        job_array_size_limit: int
            Maximum number of jobs to submit in parallel
        max_budget: int
            Maximum budget for a single trial
        base_dir:
            Base directory for saving checkpoints
        min_budget: int
            Minimum budget for a single trial
        wandb_project: str
            W&B project to log to. If False, W&B logging is disabled.
        wandb_entity: str
            W&B entity to log to
        wandb_tags:
            Tags to log to W&B
        maximize: bool
            Whether to maximize the objective function
        deterministic: bool
            Whether the target function is deterministic

        Returns:
        -------
        None
        """
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
        if wandb_tags is None:
            wandb_tags = ["hypersweeper"]
        self.global_overrides = global_overrides
        self.launcher = launcher
        self.budget_arg_name = budget_arg_name
        self.save_arg_name = save_arg_name
        self.load_arg_name = load_arg_name
        self.checkpoint_tf = checkpoint_tf
        self.load_tf = load_tf

        self.configspace = cs
        self.output_dir = Path(to_absolute_path(base_dir) if base_dir else to_absolute_path("./"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = Path(self.output_dir) / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.job_idx = 0
        self.seeds = seeds
        self.seed_keyword = seed_keyword
        if (seeds or not deterministic) and len(self.global_overrides) > 0:
            for i in range(len(self.global_overrides)):
                if self.global_overrides[i].split("=")[0] == self.seed_keyword:
                    self.global_overrides = self.global_overrides[:i] + self.global_overrides[i + 1 :]
                    break

        assert len(objectives) == len(maximize), "The number of objectives and maximize flags must match."
        self.objectives = objectives
        self.maximize = maximize

        self.slurm = slurm
        self.slurm_timeout = slurm_timeout

        if n_trials is not None:
            self.max_parallel = min(job_array_size_limit, max(1, int(max_parallelization * n_trials)))
        else:
            self.max_parallel = job_array_size_limit

        self.budget = budget
        self.min_budget = min_budget
        self.trials_run = 0
        self.n_trials = n_trials
        self.iteration = 0
        self.opt_time = 0
        self.history = defaultdict(list)

        # We need one incumbent dict for each objective
        self.incumbents = defaultdict(lambda: defaultdict(list))

        self.deterministic = deterministic
        self.max_budget = max_budget
        self.checkpoint_path_typing = checkpoint_path_typing

        self.optimizer = make_optimizer(self.configspace, optimizer_kwargs)
        self.optimizer.checkpoint_dir = self.checkpoint_dir
        self.optimizer.checkpoint_path_typing = self.checkpoint_path_typing
        self.optimizer.seeds = seeds

        self.warmstart_data: list[tuple[Info, Result]] = []

        self._config_ids: dict[str, int] = {}
        self._budget_ids: dict[float, int] = {}
        self._last_budget: dict[int, int] = {}

        if warmstart_file:
            self.warmstart_data = read_warmstart_data(warmstart_filename=warmstart_file, search_space=self.configspace)

        self.wandb_project = wandb_project
        if self.wandb_project:
            wandb_config = OmegaConf.to_container(global_config, resolve=False, throw_on_missing=False)
            assert wandb_entity, "Please provide an entity to log to W&B."
            wandb.init(
                project=self.wandb_project,
                entity=wandb_entity,
                tags=wandb_tags,
                config=wandb_config,
            )

    def run_configs(self, infos):
        """Run a set of overrides.

        Parameters
        ----------
        overrides: List[Tuple]
            A list of overrides to launch

        Returns:
        -------
        List[dict]
            The resulting performances for each objective.
        List[float]
            The incurred costs.
        """
        # TODO fix this. This is a temporary fix to make the code run.
        # We only need this for PBT, in SMAC + HB we only have to load 
        # for the second stage within each bracket

        # if self.load_tf and self.iteration > 0:
        #     assert not any(p.load_path is None for p in infos), """
        #     Load paths must be provided for all configurations
        #     when working with checkpoints. If your optimizer does not support this,
        #     set the 'load_tf' parameter of the sweeper to False."""

        # Generate overrides
        overrides = []
        for i in range(len(infos)):
            run_config = {}

            for k, v in infos[i].config.items():
                run_config[k] = v

            if self.budget_arg_name is not None:
                run_config[self.budget_arg_name] = infos[i].budget

            if self.slurm:
                optimized_timeout = (
                    self.slurm_timeout * (infos[i].budget / self.max_budget) + 0.1 * self.slurm_timeout
                )
                self.launcher.params["timeout_min"] = int(optimized_timeout)

            # The basic load and save paths are the same for all seeeds
            load_path, save_path = self._get_load_and_save_path(infos[i])

            if self.seeds:
                for s in self.seeds:
                    local_run_config = run_config.copy()
                    local_run_config[self.seed_keyword] = s

                    actual_load_path, actual_save_path = self._get_actual_load_and_save_path(load_path, save_path, seed=s)

                    if actual_load_path and self.load_tf:
                        local_run_config[self.load_arg_name] = actual_load_path

                    if self.checkpoint_tf:
                        local_run_config[self.save_arg_name] = actual_save_path

                    job_overrides = tuple(self.global_overrides) + tuple(
                        f"{k}={v}" for k, v in local_run_config.items()
                    )
                    overrides.append(job_overrides)
            elif not self.deterministic:
                assert not any(s.seed is None for s in infos), """
                For non-deterministic target functions, seeds must be provided.
                If the optimizer you chose does not support this,
                manually set the 'seeds' parameter of the sweeper to a list of seeds."""
                run_config[self.seed_keyword] = infos[i].seed

                actual_load_path, actual_save_path = self._get_actual_load_and_save_path(load_path, save_path, seed=infos[i].seed)

                if actual_load_path and self.load_tf:
                    run_config[self.load_arg_name] = actual_load_path

                if self.checkpoint_tf:
                    run_config[self.save_arg_name] = actual_save_path

                job_overrides = tuple(self.global_overrides) + tuple(
                    f"{k}={v}" for k, v in run_config.items()
                )
                overrides.append(job_overrides)
            else:
                actual_load_path, actual_save_path = self._get_actual_load_and_save_path(load_path, save_path)

                if actual_load_path and self.load_tf:
                    run_config[self.load_arg_name] = actual_load_path

                if self.checkpoint_tf:
                    run_config[self.save_arg_name] = actual_save_path

                job_overrides = tuple(self.global_overrides) + tuple(
                    f"{k}={v}" for k, v in run_config.items()
                )
                overrides.append(job_overrides)

        # Run overrides
        res = self.launcher.launch(overrides, initial_job_idx=self.job_idx)
        self.job_idx += len(overrides)
        if self.seeds:
            costs = [infos[i].budget for i in range(len(res) // len(self.seeds))]
        else:
            costs = [infos[i].budget for i in range(len(res))]

        objective_performances = []
        if self.seeds and self.deterministic:
            # When we have seeds, we want to have a list of performances for each config
            n_seeds = len(self.seeds)
            for config_idx in range(len(overrides) // n_seeds):
                objective_performances.append([res[config_idx * n_seeds + seed_idx].return_value for seed_idx in range(n_seeds)])
                self.trials_run += 1
        else:
            for j in range(len(overrides)):
                objective_performances.append(res[j].return_value)
                self.trials_run += 1
        return objective_performances, costs

    def get_incumbent(self) -> tuple[Configuration | dict, dict]:
        """Get the best sequence of configurations so far.

        Returns:
        -------
        List[Configuration]
            Sequence of best hyperparameter configs
        Dict
            Best performance value
        """
        incubments = {}
        configs = {}

        for objective_id, (objective, maximize) in enumerate(zip(self.objectives, self.maximize)):
            if maximize:
                best_run_id = np.argmax(self.history[f"o{objective_id}_{objective}"])
            else:
                best_run_id = np.argmin(self.history[f"o{objective_id}_{objective}"])

            incubments[objective] = self.history[f"o{objective_id}_{objective}"][best_run_id]
            configs[objective] = self.history["config"][best_run_id]

        return configs, incubments

    def _write_csv(self, data: dict, filename: str) -> None:
        """Write a dictionary to a csv file.

        Parameters
        ----------
        data: dict
            The data to write to the csv file
        filename: str
            The name of the csv file
        """
        dataframe = pd.DataFrame(data)

        dataframes_to_concat = []
        if "run_id" not in dataframe.columns:
            dataframes_to_concat += [pd.DataFrame(np.arange(len(dataframe)), columns=["run_id"])]

        # Since some configs might not include values for all hyperparameters
        # (e.g. when using conditions), we need to make sure that the dataframe
        # has all hyperparameters as columns
        hyperparameters = [str(hp) for hp in list(self.configspace.keys())]
        configs_df = pd.DataFrame(list(dataframe["config"]), columns=hyperparameters)

        # Now we merge the basic dataframe with the configs
        dataframes_to_concat += [dataframe.drop(columns="config"), configs_df]
        full_dataframe = pd.concat(dataframes_to_concat, axis=1)
        full_dataframe.to_csv(Path(self.output_dir) / f"{filename}.csv", index=False)

    def write_history(
        self,
        performances: list[list[dict]] | list[dict],
        configs: list[Configuration],
        budgets: list[float],
    ) -> None:
        """Write the history of the optimization to a csv file.

        Parameters
        ----------
        performances: Union[list[list[dict]], list[dict]]
            A list of the latest agent performances, either one value for each config or a list of values for each seed
        configs: list[Configuration],
            A list of the recent configs
        budgets: list[float]
            A list of the recent budgets
        """
        for i in range(len(configs)):
            self.history["config_id"].append(self._get_config_id(dict(configs[i]))[0])
            self.history["config"].append(configs[i])
            if self.seeds:
                # In this case we have a list of performances for each config,
                # one for each seed
                assert isinstance(performances[i], list)

                for objective_id, objective in enumerate(self.objectives):
                    # The mean is calculated over the same objective of didfferent seeds
                    self.history[f"o{objective_id}_{objective}"].append(np.mean([p[objective] for p in performances[i]]))

                    for seed_idx, seed in enumerate(self.seeds):
                        self.history[f"o{objective_id}_{objective}_{self.seed_keyword}_{seed}"].append(performances[i][seed_idx][objective])
            else:
                assert isinstance(performances[i], dict)

                for objective_id, objective in enumerate(self.objectives):
                    self.history[f"o{objective_id}_{objective}"].append(performances[i][objective])

            if budgets[i] is not None:
                self.history["budget"].append(budgets[i])
            else:
                self.history["budget"].append(self.max_budget)

        self._write_csv(self.history, "runhistory")

    def write_incumbents(self) -> None:
        """Write the incumbent configurations to a csv file."""
        for objective_id, (objective, maximize) in enumerate(zip(self.objectives, self.maximize)):
            if maximize:
                best_run_id = np.argmax(self.history[f"o{objective_id}_{objective}"])
            else:
                best_run_id = np.argmin(self.history[f"o{objective_id}_{objective}"])

            self.incumbents[objective_id]["run_id"].append(best_run_id)
            self.incumbents[objective_id]["config_id"].append(self.history["config_id"][best_run_id])
            self.incumbents[objective_id]["config"].append(self.history["config"][best_run_id])
            self.incumbents[objective_id][f"o{objective_id}_{objective}"].append(self.history[f"o{objective_id}_{objective}"][best_run_id])
            self.incumbents[objective_id]["budget"].append(self.history["budget"][best_run_id])
            try:
                self.incumbents[objective_id]["budget_used"].append(sum(self.history["budget"]))
            except:  # noqa:E722
                self.incumbents[objective_id]["budget_used"].append(self.trials_run)
            self.incumbents[objective_id]["total_wallclock_time"].append(time.time() - self.start)
            self.incumbents[objective_id]["total_optimization_time"].append(self.opt_time)

            self._write_csv(self.incumbents[objective_id], f"incumbent_{objective}")

        if self.wandb_project:
            stats = {}
            stats["iteration"] = self.iteration
            stats["total_optimization_time"] = self.incumbents[objective_id]["total_optimization_time"][-1]
            stats["incumbent_performance"] = self.incumbents[objective_id]["performance"][-1]
            best_config = self.incumbents[objective_id]["config"][-1]
            for n in best_config:
                stats[f"incumbent_{n}"] = best_config.get(n)
            wandb.log(stats)

    def _get_config_id(self, config: dict) -> tuple[int, bool]:
        """Get the id of a configuration.

        Parameters
        ----------
        config: dict
            The configuration to get the id of

        Returns:
        -------
        int, bool
            The id of the configuration and whether it was already seen
        """
        # This is a bit hacky, but we need to convert the config to a unique string
        config_str = "$".join([f"{v}" for v in config.values()])    

        if config_str not in self._config_ids:
            self._config_ids[config_str] = len(self._config_ids)
            return self._config_ids[config_str], True
        else:
            return self._config_ids[config_str], False
    
    def _get_budget_id(self, budget: float) -> int:
        """Get the id of a budget.

        Parameters
        ----------
        budget: float
            The budget to get the id of

        Returns:
        -------
        int
            The id of the budget
        """
        if budget not in self._budget_ids:
            self._budget_ids[budget] = len(self._budget_ids)
        
        return self._budget_ids[budget]
    
    def _get_load_and_save_path(self, info: Info) -> tuple[str | None, str]:
        """Get the load path for the configuration."""
        config_id, unseen_config = self._get_config_id(dict(info.config))
        budget_id = self._get_budget_id(info.budget)

        if unseen_config:
            load_path = None
        else:
            # We have already executed this configuration
            # so we need to load the model from the previous run
            last_budget_id = self._last_budget[config_id]
            load_path = f"budget_{last_budget_id}_config_{config_id}"

        self._last_budget[config_id] = budget_id
        
        save_path = f"budget_{budget_id}_config_{config_id}"

        return load_path, save_path
    
    def _get_actual_load_and_save_path(
            self,
            load_path: str | None,
            save_path: str,
            seed: int | None = None
        ) -> tuple[Path | None, Path]:
            if seed:
                final_load_path = self.checkpoint_dir / f"{load_path}_{self.seed_keyword}_{seed}{self.checkpoint_path_typing}" if load_path else None
                final_save_path = self.checkpoint_dir / f"{save_path}_{self.seed_keyword}_{seed}{self.checkpoint_path_typing}"
            else:
                final_load_path = self.checkpoint_dir / f"{load_path}_{self.checkpoint_path_typing}" if load_path else None
                final_save_path = self.checkpoint_dir / f"{save_path}_{self.checkpoint_path_typing}"

            return final_load_path, final_save_path


    def run(self, verbose=False):
        """Actual optimization loop.
        In each iteration:
        - get configs (either randomly upon init or through perturbation)
        - run current configs
        - write history and incbument to a csv file.

        Parameters
        ----------
        verbose: bool
            More logging info

        Returns:
        -------
        List[Configuration]
            The incumbent configurations.
        """
        if verbose:
            log.info("Starting Sweep")
        self.start = time.time()
        trial_termination = False
        budget_termination = False
        done = False
        if len(self.warmstart_data) > 0:
            for info, value in self.warmstart_data:
                self.optimizer.tell(info=info, value=value)
        while not done:
            opt_time_start = time.time()
            configs = []
            budgets = []
            infos = []
            t = 0
            terminate = False
            while t < self.max_parallel and not terminate and not trial_termination and not budget_termination:
                info, terminate = self.optimizer.ask()

                configs.append(info.config)
                t += 1
                if info.budget is not None:
                    budgets.append(info.budget)
                else:
                    budgets.append(self.max_budget)
                infos.append(info)
                if not any(b is None for b in self.history["budget"]) and self.budget is not None:
                    budget_termination = sum(self.history["budget"]) + sum(budgets) >= self.budget
                if self.n_trials is not None:
                    trial_termination = self.trials_run + len(configs) >= self.n_trials
            self.opt_time += time.time() - opt_time_start
            objective_performances, costs = self.run_configs(
                infos
            )
            opt_time_start = time.time()
            for info, performance, cost in zip(infos, objective_performances, costs, strict=True):
                logged_performance = {}

                if self.seeds:
                    assert isinstance(performance, list)
                    for objective, maxizime in zip(self.objectives, self.maximize):
                        performances = [p[objective] for p in performance]
                        logged_performance[objective] = np.mean(performances) if maxizime else -np.mean(performances)
                else:
                    assert isinstance(performance, dict)
                    for objective, maxizime in zip(self.objectives, self.maximize):
                        logged_performance[objective] = performance[objective] if maxizime else -performance[objective]

                value = Result(performance=logged_performance, cost=cost)
                self.optimizer.tell(info=info, result=value)

            self.write_history(
                performances=objective_performances,
                configs=configs,
                budgets=budgets,
            )
            self.write_incumbents()

            if verbose:
                log.info(f"Finished Iteration {self.iteration}!")
                _, inc_performance = self.get_incumbent()
                log.info(f"Current incumbent have performances of { {k: np.round(v, decimals=2) for k, v in inc_performance.items() }}.")

            self.opt_time += time.time() - opt_time_start
            done = trial_termination or budget_termination
            self.iteration += 1

        total_time = time.time() - self.start
        inc_config, inc_performance = self.get_incumbent()
        if verbose:
            log.info(
                f"Finished Sweep! Total duration was {np.round(total_time, decimals=2)}s," \
                "incumbents had a performance of { {k: np.round(v, decimals=2) for k, v in inc_performance.items() }}."
            )
            log.info(f"The incumbent configuration is {inc_config}")
        return [inc_config[objective] for objective in self.objectives]
