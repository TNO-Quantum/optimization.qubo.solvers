"""This module contains the ``DAQOResult`` class."""

from __future__ import annotations

import itertools
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, SupportsFloat, SupportsInt

import numpy as np
from tno.quantum.optimization.qubo.components import Freq, ResultInterface
from tno.quantum.utils.validation import check_arraylike, check_ax, check_int

if TYPE_CHECKING:
    from typing import Self

    from matplotlib.axes import Axes
    from numpy.typing import ArrayLike
    from tno.quantum.devices import DeviceConfig
    from tno.quantum.optimization.qubo.components import QUBO
    from tno.quantum.utils import BitVectorLike


class DAQOResult(ResultInterface):
    """Implementation of `ResultInterface` for :py:class:`DAQOResult`."""

    def __init__(  # noqa: PLR0913, PLR0917
        self,
        best_bitvector: BitVectorLike,
        best_value: SupportsFloat,
        freq: Freq,
        n_layers: SupportsInt,
        schedule: ArrayLike,
        device: DeviceConfig,
    ) -> None:
        """Init :py:class:`DAQOResult`.

        Args:
            best_bitvector: Bitvector corresponding to the best result.
            best_value: Objective value of the best result.
            freq: Frequency object with the found energies and number of occurrences.
            n_layers: Number of layers used in the DAQO circuit.
            schedule: Annealing schedule used in the DAQO circuit.
            device: Device used to sample the circuit.
        """
        super().__init__(best_bitvector, best_value, freq)
        self.n_layers = check_int(n_layers, "n_layers", l_bound=1)
        self.device = device
        self.schedule = check_arraylike(
            schedule, "schedule", ndim=2, shape=(2, self.n_layers)
        )

    @classmethod
    def from_result(
        cls, qubo: QUBO, raw_result: Mapping[str, int], properties: dict[str, Any]
    ) -> Self:
        """Construct :py:class:`DAQOResultOAResult` from `raw_result` and the `qubo`.

        Args:
            qubo: QUBO to evaluate the given bitvectors.
            raw_result: Mapping with bitstrings as keys and frequencies as values.
            properties: Dictionary containing properties used to solve QUBO.

        Returns:
            A :py:class:`DAQOResult` containing the best bitvector, best value and
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
            n_layers=properties["n_layers"],
            schedule=properties["schedule"],
            device=properties["device"],
        )

    def plot_shots_histogram(self, ax: Axes | None = None) -> None:
        """Plot the histogram of the output of the final circuit.

        Args:
            ax: Optional matplotlib ``Axes`` to draw on. If ``None`` (default) create a
                new figure with ``Axes`` to draw on.
        """
        ax = check_ax(ax, "ax")

        n_bits = len(self.best_bitvector)
        x_values = ["".join(bits) for bits in itertools.product("01", repeat=n_bits)]
        height = np.zeros_like(x_values)
        for bitvector, _, n in self.freq:
            i = int(str(bitvector), 2)
            height[i] += n

        ax.bar(x_values, height)

        ax.set_xlabel("Solution")
        ax.set_ylabel("Number of Shots")

    def plot_schedule(self, ax: Axes | None = None) -> None:
        """Plot the annealing schedule.

        Args:
            ax: Optional matplotlib ``Axes`` to draw on. If ``None`` (default) create a
                new figure with ``Axes`` to draw on.
        """
        ax = check_ax(ax, "ax")

        x = range(self.n_layers)
        ax.plot(x, self.schedule[0, :], label="schedule initial Hamiltonian")
        ax.plot(x, self.schedule[1, :], label="schedule problem Hamiltonian")

        ax.set_xlabel("Layer")
        ax.set_ylabel("Annealing schedule")
        ax.legend()
