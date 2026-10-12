"""This module contains the ``QAOAResult`` class."""

from __future__ import annotations

import itertools
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, SupportsFloat

import numpy as np
from tno.quantum.optimization.qubo.components import Freq, ResultInterface
from tno.quantum.utils.validation import check_arraylike, check_ax

if TYPE_CHECKING:
    from typing import Self

    from matplotlib.axes import Axes
    from numpy.typing import ArrayLike
    from tno.quantum.devices import DeviceConfig
    from tno.quantum.optimization.qubo.components import QUBO
    from tno.quantum.utils import BitVectorLike, OptimizerConfig


class QAOAResult(ResultInterface):
    """Implementation of `ResultInterface` for :py:class:`QAOASolver`."""

    def __init__(  # noqa: PLR0913, PLR0917
        self,
        best_bitvector: BitVectorLike,
        best_value: SupportsFloat,
        freq: Freq,
        init_beta: ArrayLike,
        init_gamma: ArrayLike,
        final_beta: ArrayLike,
        final_gamma: ArrayLike,
        expval_history: ArrayLike,
        training_device: DeviceConfig,
        evaluation_device: DeviceConfig,
        optimizer: OptimizerConfig,
    ) -> None:
        """Init :py:class:`QAOAResult`.

        Args:
            best_bitvector: Bitvector corresponding to the best result.
            best_value: Objective value of the best result.
            freq: Frequency object with the found energies and number of occurrences.
            init_beta: Initial parameters for the mixer layer.
            init_gamma: Initial parameters for the cost layer.
            final_beta: Final parameters for the mixer layer.
            final_gamma: Final parameters for the mixer layer.
            expval_history: Loss values over all optimizing iterations.
            training_device: Training device used.
            evaluation_device: Evaluation device used.
            optimizer: Optimizer used.
        """
        super().__init__(best_bitvector, best_value, freq)

        self.init_beta = check_arraylike(init_beta, "init_beta", ndim=1)
        self.init_gamma = check_arraylike(init_gamma, "init_gamma", ndim=1)
        self.final_beta = check_arraylike(final_beta, "final_beta", ndim=1)
        self.final_gamma = check_arraylike(final_gamma, "final_gamma", ndim=1)
        self.expval_history = check_arraylike(expval_history, "expval_history", ndim=1)
        self.training_device = training_device
        self.evaluation_device = evaluation_device
        self.optimizer = optimizer

    @classmethod
    def from_result(
        cls, qubo: QUBO, raw_result: Mapping[str, int], properties: dict[str, Any]
    ) -> Self:
        """Construct :py:class:`QAOAResult` from `raw_result` for the given `qubo`.

        Args:
            qubo: QUBO to evaluate the given bitvectors.
            raw_result: Mapping with bitstrings as keys and frequencies as values.
            properties: Dictionary containing properties used to solve QUBO.

        Returns:
            A :py:class:`QAOAResult` containing the best bitvector, best value and
            frequency of the best bitvector of `raw_result` based on the given `qubo`.
            The best bitvector has the lowest energy (value) based on the given `qubo`.
            When there are ties, the bitvector with the highest frequency is
            returned.

        Raises:
            ValueError: If `raw_result` is empty.
        """
        freq = Freq(
            bitvectors=raw_result.keys(),
            energies=map(qubo.evaluate, raw_result.keys()),
            num_occurrences=raw_result.values(),
        )

        if not freq.energies:
            msg = "Argument `raw_result` is empty"
            raise ValueError(msg)

        # Find the solution index with the lowest energy. Break ties by returning the
        # solution index with the highest number of occurrences.
        energies = np.array(freq.energies)
        num_occurrences = np.array(freq.num_occurrences)
        (min_indices,) = np.where(energies == energies.min())
        best_idx = min_indices[np.argmax(num_occurrences[min_indices])]

        return cls(
            best_bitvector=freq.bitvectors[best_idx],
            best_value=freq.energies[best_idx],
            freq=freq,
            init_beta=properties["init_beta"],
            init_gamma=properties["init_gamma"],
            final_beta=properties["final_beta"],
            final_gamma=properties["final_gamma"],
            expval_history=properties["expval_history"],
            training_device=properties["training_device"],
            evaluation_device=properties["evaluation_device"],
            optimizer=properties["optimizer"],
        )

    def plot_expval_history(self, ax: Axes | None = None) -> None:
        """Plot the history of the expectation value of the cost function.

        Args:
            ax: Optional matplotlib ``Axes`` to draw on. If ``None`` (default) create a
                new figure with ``Axes`` to draw on.
        """
        ax = check_ax(ax, "ax")

        ax.plot(range(len(self.expval_history)), self.expval_history)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Expectation Value")

    def plot_shots_histogram(self, ax: Axes | None = None) -> None:
        """Plot the histogram of the output of the final circuit.

        Args:
            ax: Optional matplotlib ``Axes`` to draw on. If ``None`` (default) create a
                new figure with ``Axes`` to draw on.
        """
        ax = check_ax(ax, "ax")

        n_bits = len(self.best_bitvector)
        x_values = ["".join(bits) for bits in itertools.product("01", repeat=n_bits)]
        height = [0 for _ in x_values]
        for bitvector, _, n in self.freq:
            i = int(str(bitvector), 2)
            height[i] += n

        ax.bar(x_values, height)

        ax.set_xlabel("Solution")
        ax.set_ylabel("Number of Shots")

    def plot_parameters(self, ax: Axes | None = None) -> None:
        """Plot the final beta and gamma parameters.

        Args:
            ax: Optional matplotlib ``Axes`` to draw on. If ``None`` (default) create a
                new figure with ``Axes`` to draw on.
        """
        ax = check_ax(ax, "ax")

        depth = len(self.final_beta)

        ax.plot(range(depth), self.final_beta, label="beta")
        ax.plot(range(depth), self.final_gamma, label="gamma")

        ax.set_xlabel("Depth")
        ax.set_ylabel("Rotation")
        ax.legend()
