"""A custom Optuna sampler for dynamic environments."""

from typing import Sequence

import optuna


class DriftMixedSampler(optuna.samplers.BaseSampler):
    """A custom sampler that interleaves TPE with pure random sampling to combat data drift."""

    def __init__(self, random_mix_ratio: float = 0.20, seed: int | None = None):
        self._tpe_sampler = optuna.samplers.TPESampler(
            seed=seed, constant_liar=True, multivariate=True
        )
        self._random_sampler = optuna.samplers.RandomSampler(seed=seed)
        self._random_mix_ratio = random_mix_ratio

    def reseed_rng(self) -> None:
        self._tpe_sampler.reseed_rng()
        self._random_sampler.reseed_rng()

    def infer_relative_search_space(
        self, study: optuna.Study, trial: optuna.trial.FrozenTrial
    ):
        return self._tpe_sampler.infer_relative_search_space(study, trial)

    def sample_relative(
        self, study: optuna.Study, trial: optuna.trial.FrozenTrial, search_space
    ):
        if (
            trial.number > 0
            and self._random_mix_ratio > 0
            and trial.number % int(1 / self._random_mix_ratio) == 0
        ):
            return self._random_sampler.sample_relative(study, trial, search_space)
        return self._tpe_sampler.sample_relative(study, trial, search_space)

    def sample_independent(
        self,
        study: optuna.Study,
        trial: optuna.trial.FrozenTrial,
        param_name: str,
        param_distribution,
    ):
        if (
            trial.number > 0
            and self._random_mix_ratio > 0
            and trial.number % int(1 / self._random_mix_ratio) == 0
        ):
            return self._random_sampler.sample_independent(
                study, trial, param_name, param_distribution
            )
        return self._tpe_sampler.sample_independent(
            study, trial, param_name, param_distribution
        )

    def before_trial(
        self, study: optuna.Study, trial: optuna.trial.FrozenTrial
    ) -> None:
        if hasattr(self._tpe_sampler, "before_trial"):
            self._tpe_sampler.before_trial(study, trial)
        if hasattr(self._random_sampler, "before_trial"):
            self._random_sampler.before_trial(study, trial)

    def after_trial(
        self,
        study: optuna.Study,
        trial: optuna.trial.FrozenTrial,
        state: optuna.trial.TrialState,
        values: Sequence[float] | None,
    ) -> None:
        if hasattr(self._tpe_sampler, "after_trial"):
            self._tpe_sampler.after_trial(study, trial, state, values)
        if hasattr(self._random_sampler, "after_trial"):
            self._random_sampler.after_trial(study, trial, state, values)
