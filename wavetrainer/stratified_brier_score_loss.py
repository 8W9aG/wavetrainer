"""A function for constructing the stratified brier score."""

import numpy as np
from sklearn.metrics import brier_score_loss  # type: ignore


def stratified_brier_score_loss(y_true, y_prob) -> float:
    """Finds the stratified brier score of the boolean classification."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    pos = y_true[y_true is True]
    neg = y_true[y_true is False]

    print(pos)
    b_pos = brier_score_loss(y_true[pos], y_prob[pos])
    b_neg = brier_score_loss(y_true[neg], y_prob[neg])

    return float(0.5 * (b_pos + b_neg))
