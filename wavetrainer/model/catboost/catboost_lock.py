"""A process-wide lock serialising calls into CatBoost's native library."""

import threading

# Optuna runs trials as threads within one process when study.optimize is
# given n_jobs > 1. Running CatBoost's native training concurrently from
# several threads segfaults or deadlocks (seen on Linux, CPU and GPU), so every
# call into CatBoost's native code (fit, Pool construction, prediction, feature
# importances, model save/load) is made while holding this lock. Only those
# calls are serialised; the rest of each trial's pipeline still runs in
# parallel. A plain Lock is enough because no guarded call re-enters another:
# callers must not hold it while calling a wrapper's fit, which takes it
# itself. Kept at module level (rather than on a model or Trainer) so those
# objects stay picklable for joblib's process-based Parallel backend.
CATBOOST_LOCK = threading.Lock()
