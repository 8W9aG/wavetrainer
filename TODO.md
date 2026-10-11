# TODO

Changes I'd make to `wavetrainer`, roughly in priority order. Items in the
first section are bugs that change what the optimiser sees or what gets
predicted; the rest are cleanup.

## 1. Correctness bugs

- [ ] **Cached trials return the wrong metric.** When `trial.json` already
      exists for the trial number, `fit_trial` returns `trial_info["output"][0]`
      (F1/R2/CRPS, where higher is better for F1/R2) while the normal path
      returns `loss` (Brier, minimised). Return `[1]`, or better, store only
      the objective value.
- [ ] **The "retry until a good trial" loop never fires.** `Trainer.fit` checks
      `best_trial(study).values == (BAD_OUTPUT, -BAD_OUTPUT)`. The study is
      single-objective now, so `values` is a one-element list and is never
      equal to that tuple. Also `study.best_trial` raises `ValueError` (it
      doesn't return `None`) when no trial completed. Compare
      `best_trial(study).value` against `-BAD_OUTPUT` and handle the no-trials
      case.
- [ ] **Test-set leakage inflates trial scores.** The same `x_test`/`y_test`
      fold is used for early stopping (`eval_x` in `Selector.fit` and
      `model.fit`), for fitting the calibrator, *and* for scoring the trial.
      The score is optimistic, and the optimiser rewards overfitting to the
      fold. Split the fold into calibration and evaluation slices (time-ordered),
      or carve the early-stopping set out of the training window.
- [ ] **Search space differs between parent and worker processes.**
      `pytest_is_running.is_running()` decides the bounds of catboost `depth`
      and xgboost `max_depth` and `n_estimators`. Spawned workers don't see
      pytest, so the test run logs warnings like
      `The value 6 of the parameter 'depth' is out of the range of ... high=2`.
      In production, any env difference would do the same. Make the bounds
      explicit constructor options and drop `pytest-is-running` from
      runtime deps.
- [ ] **`create()` silently overwrites `params.json`.** `create()` defaults
      `walkforward_timedelta` to 1 day, so `Trainer.__init__` always takes the
      `save_params` branch. Calling `create(folder)` to load an existing model
      for inference resets `trials`, `test_size`, `validation_size` and so on to
      defaults. Default it to `None` like `Trainer` does.
- [ ] **Settings needed at inference aren't persisted.** `embedding_cols`,
      `insert_nulls`, `use_power_transformer`, `use_correlation_reducer` and
      `allowed_models` are used to rebuild the pipeline in `TrainerInference`,
      but they aren't in `params.json`. A caller has to pass the same values
      again, and nothing warns them if they don't. Save them in `TrainerParams`.
- [ ] **`cutoff_dt` and target alignment assume sorted input.** `Trainer.fit`
      filters `df` by `dt_index < cutoff_dt`, then does `y.iloc[: len(df)]` and
      `dt_index[: len(df)]`. That is only right if rows are sorted by date.
      Filter `y` and `dt_index` with the same mask, or sort/validate up front.
      Also `cutoff_dt` defaults to naive `datetime.now()`, which raises against
      a tz-aware index.
- [ ] **`feature_importances()` skips the normaliser** (`inference.py`), so with
      `use_power_transformer=True` it runs the model on untransformed features.
      It also keys results by date only (`date_path.split("/")[-1]`), so
      columns sharing a fold date overwrite each other. Key by
      `(column, date)` and use `os.path.basename`.
- [ ] **Multi-class targets will fail scoring.** `f1_score`, `precision_score`
      and `recall_score` use the default `average="binary"`, which raises for
      `MULTI_CLASSIFICATION`. Also note that `determine_model_type` routes an
      `int` 0/1 target to multi-class; only `bool` becomes `BINARY`. Document
      this, or detect two classes.
- [ ] **`CombinedReducer.fit` after `load` crashes.** It opens
      `combined_reducer_removed_columns.json` in read mode and then
      `json.dump`s into it. Open with `"w"`.
- [ ] **`ModelRouter.fit` mutates the caller's weights** (`w += ...`) during
      false-positive reduction. Copy first.
- [ ] **`DriftMixedSampler` ratio maths.** `trial.number % int(1 / ratio)`
      raises `ZeroDivisionError` for `ratio > 1`, and rounds (0.3 gives 33%
      random). Validate the range, and expose `seed` through `Trainer` for
      reproducible runs.
- [ ] `_size_from_json` treats any non-float as a timedelta dict, so an
      integer size saved as JSON `1` crashes on load. `inference.py` also has
      an `int` validation-size branch that the trainer's types don't allow.
      Pick one.

## 2. Performance

- [ ] `CalibratorRouter.fit` renders a 300-dpi matplotlib calibration plot on
      **every** trial, but it is only written when the trial is saved. Render
      lazily in `save()`.
- [ ] `Trainer.fit` rescans `study.trials` (and parses ISO dates) once per
      walk-forward step to find already-processed folds. Build a set of
      processed dates once per column.
- [ ] `inference._assign_bin` is a Python loop per row. Use
      `np.searchsorted`/`pd.cut`.
- [ ] `TrainerInference.transform` runs `joblib.Parallel` with the default
      (loky, multi-process) backend over a closure. Each worker pickles `df`
      and gets an empty `_cached_*` dict, so the model caches never hit, and
      the parent's caches grow without bound. Use the threading backend
      (catboost is already behind a lock), or load once per column and bound
      the caches.
- [ ] `ModelRouter.fit` computes per-row error weights with `DataFrame.apply`.
      Vectorise.
- [ ] The trial-count constants (`1000` max retries, `5` trials per
      walk-forward step, `30s` kaleido timeout) are hard-coded. Make them
      `Trainer` options.

## 3. Code health

- [ ] Replace the roughly 60 `print` calls with `logging` (module loggers), so
      users can silence or route output. There are also a couple of broken
      `print("... %s", name)` calls in the combined reducer/normaliser.
- [ ] Remove the multi-objective leftovers: commented-out `return output, loss`,
      `MAXIMIZE` direction, `pareto_front`, and `target_names = ["F1", "Brier"]`.
      Both plots use `values[0]`, so the "F1" PNG is actually the Brier
      importance.
- [ ] Rename `BAD_OUTPUT` so the sign is obvious (it is returned negated as
      `-BAD_OUTPUT == 1000`), and pull the duplicated "remove new fold folder"
      block in `fit_trial` into a helper.
- [ ] Break up `fit_trial` (~230 lines): pipeline build/fit, scoring per
      `ModelType`, and persistence.
- [ ] `Trainer.fit`'s nested closures (`_optimize`, `_fit_column`,
      `test_objective`, `validate_objective`) would be easier to test as
      methods.
- [ ] Fix `create()` and `Trainer` drift: `insert_null` vs `insert_nulls`;
      `trials` and `random_mix_ratio` aren't exposed by `create()`. Consider
      a single config dataclass instead of 15 positional/keyword args.
- [ ] `CombinedReducer.load` uses an if/elif chain. Use a name → class registry
      like `ModelRouter` and `CalibratorRouter`.
- [ ] Inference swallows `FileNotFoundError` per fold and returns the frame
      without that column's predictions. Raise, or at least log at
      `warning` level with the column name.

## 4. Packaging and dependencies

- [ ] Move from `setup.py` + `requirements.txt` to `pyproject.toml`; single-
      source the version (it is currently duplicated in `setup.py` and
      `__init__.py`); add `python_requires=">=3.11"` (`StrEnum` and
      `typing.Self` require it). The README says 3.11.6 but the venv is 3.14.
- [ ] Prune runtime deps. `hyperopt`, `memory-profiler` and `pytz` are never
      imported, and `pyarrow` is only used by the debug inference dump.
      `kaleido` is only needed for the optional importance PNGs, so it could
      be an extra, with the plot skipped when it is missing. `pytest-is-running`
      shouldn't be a runtime dep (see above). Make `torch` (used only for
      `cuda.is_available()`), `jax` (xgboost focal loss), `xgboost` and
      `lightgbm` optional extras. `joblib` is imported directly but is not
      declared. Neither is `plotly`, which `optuna.visualization` needs.
- [ ] Replace `torch.cuda.is_available()` GPU detection with something
      lighter (e.g. `nvidia-smi`/`CUDA_VISIBLE_DEVICES`, or catboost's own
      `get_gpu_device_count()`), so torch isn't required.
- [ ] `deploy.sh` builds only an sdist via the deprecated `setup.py sdist`.
      Use `python -m build` (sdist + wheel).

## 5. Testing and CI

- [ ] Add regression tests for each bug in section 1 (regression loss, cached
      return value, the bad-trial retry, inference with non-default settings).
- [ ] No tests cover the reducers, normalisers, weights, calibrators,
      windower, selector, xgboost/lightgbm models, or `TrainerInference`
      beyond what `trainer_test.py` reaches end-to-end.
- [ ] The suite takes about 7 minutes (79 tests). Add small-data and unit-level
      tests so most of it can run quickly, and mark the end-to-end ones as `slow`.
- [ ] Add CI (GitHub Actions) running `lint.sh` and `test.sh`. Run lint over
      `tests/` too.
- [ ] Silence or fix the pycaleva `outsample` warning: the calibrator is
      fit and evaluated on the same data (see the leakage item above).
