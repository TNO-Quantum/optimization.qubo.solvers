"""This module contains utility functions related to qubovert solvers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import qubovert

if TYPE_CHECKING:
    from tno.quantum.optimization.qubo.components import QUBO


def _get_qubovert_model(qubo: QUBO) -> qubovert.PCBO:
    """Get Qubovert model formulation from QUBO.

    Args:
        qubo: QUBO

    Returns:
        Qubovert model
    """
    variables = {i: qubovert.boolean_var(f"x({i})") for i in range(qubo.size)}
    model = qubo.offset
    for i in range(qubo.size):
        for j in range(qubo.size):
            model += variables[i] * variables[j] * qubo.matrix[i, j]
    return model
