"""A function for creating a new trainer."""

# pylint: disable=too-many-arguments,too-many-positional-arguments

import datetime

from .trainer import Trainer


def create(
    folder: str,
    walkforward_timedelta: datetime.timedelta = datetime.timedelta(days=1.0),
    test_size: float | datetime.timedelta | None = None,
    validation_size: float | datetime.timedelta | None = None,
    dt_column: str | None = None,
    max_train_timeout: datetime.timedelta | None = None,
    cutoff_dt: datetime.datetime | None = None,
    embedding_cols: list[list[str]] | None = None,
    allowed_models: set[str] | None = None,
    max_false_positive_reduction_steps: int | None = None,
    correlation_chunk_size: int | None = None,
    insert_null: bool = False,
    use_power_transformer: bool = False,
) -> Trainer:
    """Create a trainer."""
    return Trainer(
        folder,
        walkforward_timedelta,
        test_size=test_size,
        validation_size=validation_size,
        dt_column=dt_column,
        max_train_timeout=max_train_timeout,
        cutoff_dt=cutoff_dt,
        embedding_cols=embedding_cols,
        allowed_models=allowed_models,
        max_false_positive_reduction_steps=max_false_positive_reduction_steps,
        correlation_chunk_size=correlation_chunk_size,
        insert_nulls=insert_null,
        use_power_transformer=use_power_transformer,
    )
