# Changelog

## 0.3.27

### Fixed

- **pandas 3:** `Trainer.transform()` failed with
  `ValueError: assignment destination is read-only` (raised as a
  `WavetrainException`) for models trained with `use_power_transformer=True`.
  It now works. Training with the power transformer on pandas 3 failed the
  same way and is fixed too.
- **Regression tuning:** regression trials all reported a loss of 0.0, so the
  optimiser couldn't tell them apart. They now return `-R2`.
  Studies trained before this release still hold trials scored 0.0, and these
  beat any new trial with a negative R2. To retune a regression column from
  scratch, delete its `study.db` or `study.journal`.

### Changed

- **Power transformer:** `PowerTransformerNormaliser.transform()` no longer
  does anything. It used to compute the transform and write it into a
  temporary copy, so the result never reached the data: every saved model was
  trained and evaluated on untransformed features. Predictions from existing
  models are unchanged; on pandas 2 they are identical to 0.3.26.
  `fit()`, `save()` and `load()` are unchanged, so existing model folders
  load and new ones contain the same files.
  One side effect: an input with an infinite value in a power-transformer
  column used to raise an error. It is now passed to the model.
- **`Trainer.transform(n_jobs=...)` now defaults to 1, not -1.** Columns are
  inferred in the calling process, so loaded models stay cached on the Trainer
  between calls. Before, each call started a worker process per column, and
  each worker loaded its own models and dropped them when the call ended.
  Pass `n_jobs=-1` to keep the old behaviour.

### Performance

- **Faster `transform()`:** on a single row with 3,000 features and 4 trained
  columns, a call takes 0.07s, down from 3.3s with the old default and 0.35s
  with `n_jobs=1`. Gains grow with the feature count. The changes:
  - no power transform is computed;
  - the check that no input column was dropped is no longer quadratic in the
    column count (about 9s per call at 70,000 columns on pandas 3);
  - the input frame is no longer copied;
  - each column's list of trained folds is read from disk once and cached.
- **Faster training with `use_power_transformer=True`:** the training and test
  data are no longer power-transformed for every trial. The transformer is
  still fitted.

### Notes

- **Folds from other processes:** a Trainer now remembers each column's folds
  after its first `transform()`. Folds it trains itself are picked up. Folds
  another process trains into the same folder are not seen until you create a
  new Trainer.
