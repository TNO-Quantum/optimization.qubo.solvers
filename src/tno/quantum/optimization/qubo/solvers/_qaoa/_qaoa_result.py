"""This module contains the ``QAOAResult`` class."""

from __future__ import annotations

import itertools
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, SupportsFloat

from tno.quantum.optimization.qubo.components import Freq, ResultInterface
from tno.quantum.utils import BitVector
from tno.quantum.utils.validation import check_arraylike, check_ax

if TYPE_CHECKING:
    from typing import Self

    from matplotlib.axes import Axes
    from numpy.typing import ArrayLike

    from tno.quantum.optimization.qubo.components import QUBO
    from tno.quantum.utils import BackendConfig, BitVectorLike, OptimizerConfig


class QAOAResult(ResultInterface):
    """Implementation of `ResultInterface` for :py:class:`QAOASolver`."""

    def __init__(  # noqa: PLR0913
        self,
        best_bitvector: BitVectorLike,
        best_value: SupportsFloat,
        freq: Freq,
        init_beta: ArrayLike,
        init_gamma: ArrayLike,
        final_beta: ArrayLike,
        final_gamma: ArrayLike,
        expval_history: ArrayLike,
        training_backend: BackendConfig,
        evaluation_backend: BackendConfig,
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
            training_backend: Training backend used.
            evaluation_backend: Evaluation backend used.
            optimizer: Optimizer used.
        """
        super().__init__(best_bitvector, best_value, freq)

        self.init_beta = check_arraylike(init_beta, "init_beta", ndim=1)
        self.init_gamma = check_arraylike(init_gamma, "init_gamma", ndim=1)
        self.final_beta = check_arraylike(final_beta, "final_beta", ndim=1)
        self.final_gamma = check_arraylike(final_gamma, "final_gamma", ndim=1)
        self.expval_history = check_arraylike(expval_history, "expval_history", ndim=1)
        self.training_backend = training_backend
        self.evaluation_backend = evaluation_backend
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
        # Convert `raw_result` to `Freq` object with energies = QUBO evaluation
        bitvectors = [BitVector(s) for s in raw_result]
        energies = [qubo.evaluate(b) for b in bitvectors]
        num_occurrences = [raw_result[s] for s in raw_result]

        freq = Freq(bitvectors, energies, num_occurrences)

        # Find `best_bitvector` and `best_value`
        best_bitvector = None
        best_value = None
        occ_best_bitvector = None
        for bitvector, value, occ in freq:
            if (
                best_value is None
                or occ_best_bitvector is None
                or value < best_value
                or (value == best_value and occ > occ_best_bitvector)
            ):
                best_bitvector = bitvector
                best_value = value
                occ_best_bitvector = occ

        if best_bitvector is None or best_value is None:
            msg = "Argument `raw_result` is empty"
            raise ValueError(msg)

        # Get properties from `properties` (make copies of lists)
        init_beta = list(properties["init_beta"])
        init_gamma = list(properties["init_gamma"])
        final_beta = list(properties["final_beta"])
        final_gamma = list(properties["final_gamma"])
        expval_history = list(properties["expval_history"])

        return cls(
            best_bitvector=best_bitvector,
            best_value=best_value,
            freq=freq,
            init_beta=init_beta,
            init_gamma=init_gamma,
            final_beta=final_beta,
            final_gamma=final_gamma,
            expval_history=expval_history,
            training_backend=properties["training_backend"],
            evaluation_backend=properties["evaluation_backend"],
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
