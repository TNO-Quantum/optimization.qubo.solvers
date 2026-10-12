"""This module contains the `DAQOSolver` class.

This class builds quantum circuits for quantum adiabatic optimization.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, SupportsInt

import numpy as np
import pennylane as qml
from tno.quantum.devices import DeviceConfig
from tno.quantum.optimization.qubo.components import Solver
from tno.quantum.utils.validation import check_int

from tno.quantum.optimization.qubo.solvers._dqo._daqo_result import DAQOResult
from tno.quantum.optimization.qubo.solvers._dqo._validate_schedule import (
    check_1d_schedule_bounds,
    check_1d_schedule_start_and_end,
    check_initial_ham_schedule,
    check_length,
    check_ndim,
    check_problem_ham_schedule,
    check_schedule_monotonicity,
    check_shape,
)
from tno.quantum.optimization.qubo.solvers._dqo.layers_lib import (
    CostLayer,
    InitialLayer,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import TypeAlias

    from numpy.typing import ArrayLike, NDArray
    from tno.quantum.optimization.qubo.components import QUBO

    Schedule = Literal["sinusoidal", "linear"] | ArrayLike


class DAQOSolver(Solver[DAQOResult]):
    """Digital Adiabatic Quantum Optimization solver.

    The :py:class:`DAQOSolver` class solves QUBOs using digital adiabatic quantum
    optimization.

    Example:
        >>> from tno.quantum.optimization.qubo.components import QUBO
        >>> from tno.quantum.optimization.qubo.solvers import DAQOSolver
        >>>
        >>> qubo = QUBO([[1, 2, 3], [4, -50, 6], [7, 8, 9]])
        >>> solver = DAQOSolver(n_layers=4)
        >>> result = solver.solve(qubo)
        >>> result.best_bitvector
        BitVector(010)
    """

    non_deterministic = True

    metadata = MappingProxyType({"supported_schedules": ("linear", "sinusoidal")})

    def __init__(
        self,
        n_layers: SupportsInt,
        device: DeviceConfig | Mapping[str, Any] | None = None,
        schedule: Schedule = "sinusoidal",
    ) -> None:
        r"""Init :py:class:`DAQOSolver`.

        Args:
            n_layers: integer value representing the number of layers. The number of
                layers is equal to the number of Trotter steps.
            device: Configuration for device used to sample the circuit.
            schedule: Annealing schedule to use for the adiabatic evolution.
                If ``"sinusoidal"`` or ``"linear"`` is provided, the respective
                predefined sinusoidal or linear schedule is used.
                If a 1D array of length `n_layers` is provided, it is used as the
                schedule for the problem Hamiltonian, and the schedule for the initial
                Hamiltonian will be set complementary, that is, such that they sum to
                one.
                If a 2D array of shape ``(2, n_layers)`` is provided, the two rows
                will be used as schedules for the initial Hamiltonian and problem
                Hamiltonian, respectively.
        """
        self._n_layers = check_int(n_layers, "n_layers", l_bound=1)
        self._device = get_default_device_if_none(device)
        self.schedule = schedule

    def sample_qubo(self, qubo: QUBO) -> dict[str, int]:
        """Sample a QUBO problem.

        The :py:class:`QUBO` problem is transformed to the corresponding Lenz-Ising
        model. This Lenz-Ising model is encoded in the cost and counterdiabatic layers.

        Args:
            qubo: QUBO problem to sample.

        Returns:
            Dictionary with samples. Each key is a bitstring and each correspending
            value is the numner of times that bitstring was measured.
        """
        if qubo.size == 0:
            return {"": 1}
        external_fields, interactions, _ = qubo.to_ising()
        device = self._device.get_instance(wires=range(len(external_fields)))
        quantum_script = self.build_quantum_script(interactions, external_fields)
        return qml.execute([quantum_script], device)[0]  # type: ignore[no-any-return]

    def _solve(self, qubo: QUBO) -> DAQOResult:
        """Solve QUBO using digital adiabatic quantum optimization."""
        samples = self.sample_qubo(qubo)
        properties = {
            "n_layers": self._n_layers,
            "device": self._device,
            "schedule": self._schedule,
        }
        return DAQOResult.from_result(qubo, samples, properties)

    def build_quantum_script(
        self, interactions: NDArray[np.float64], external_fields: NDArray[np.float64]
    ) -> qml.tape.QuantumScript:
        """Build a quantum script for digitized adiabatic quantum optimization.

        This method constructs a :class:`~pennylane.tape.QuantumScript` representing the
        adiabatic evolution of a quantum system under a time-dependent Hamiltonian. It
        normalizes the problem parameters, prepares the initial state, applies initial
        and cost layers according to the provided schedule, and returns a quantum tape
        ready for execution.

        Notes:
          This is a utility function to have direct access to the quantum script. This
          can be used to analyse the circuit or convert it to different formats for
          different devices. If the goal is to sample from a QUBO problem, it is
          recommended to use :class:`~DAQOSolver.solve` or
          :class:`~DAQOSolver.sample_qubo` instead.


        Args:
          interactions: A 2D array representing interaction strengths between qubits
            (e.g., Lenz-Ising couplings).
          external_fields: A 1D array representing external field strengths applied to
            each qubit.

        Returns:
          :class:`~pennylane.tape.QuantumScript` containing the prepared quantum circuit
          for adiabatic quantum optimization, including the
          :py:func:`~pennylane.counts` measurement instruction.

        Example:
          >>> import numpy as np
          >>> solver =  DAQOSolver(10)
          >>> interactions = np.array([[0, 1], [1, 0]], dtype=np.float64)
          >>> external_fields = np.array([0.5, -0.5], dtype=np.float64)
          >>> solver.build_quantum_script(interactions, external_fields)
          <QuantumScript: wires=[0, 1], params=60>
        """
        norm_constant = max(abs(interactions).max(), abs(external_fields).max())
        interactions /= norm_constant
        external_fields /= norm_constant

        init_schedule = self.schedule[0, :]
        cost_schedule = self.schedule[1, :]

        cost_ham_layer = CostLayer(external_fields, interactions)
        initial_ham_layer = InitialLayer(len(external_fields))

        with qml.queuing.AnnotatedQueue() as queue:
            initial_ham_layer.prep()

            for init_strength, cost_strength in zip(
                init_schedule, cost_schedule, strict=False
            ):
                cost_ham_layer(cost_strength)
                initial_ham_layer(init_strength)

            qml.counts()

        shots = check_int(self._device.options.get("shots", 1000), "shots", l_bound=1)
        return qml.tape.QuantumScript.from_queue(queue, shots=shots)

    @property
    def n_layers(self) -> int:
        """Get the number of layers."""
        return self._n_layers

    @property
    def device(self) -> DeviceConfig:
        """Get the device configuration."""
        return self._device

    @property
    def schedule(self) -> NDArray[np.float64]:
        """Get the annealing schedules for the initial and problem Hamiltonian.

        Returns:
            2D array of shape ``(2, n_layers)``, whose first row is the annealing
            schedule for the initial Hamiltonian, and whose second row is the annealing
            schedule for the problem Hamiltonian.
        """
        return self._schedule

    @schedule.setter
    def schedule(self, schedule: Schedule) -> None:
        """Set the annealing schedule."""
        if isinstance(schedule, str):
            self._schedule = self._schedule_from_str(schedule)
            return

        schedule = np.asarray(schedule, dtype=np.float64)

        if schedule.ndim == 1:
            self._schedule = self._schedule_from_1d(schedule)
            return

        if schedule.ndim == 2:
            self._schedule = self._schedule_from_2d(schedule)
            return

        msg = (
            "Invalid schedule value: must either be a string, or a 1D ArrayLike of"
            "length `n_layers`, or 2D ArrayLike object of shape ``(2, n_layers)``."
        )
        raise ValueError(msg)

    def _schedule_from_str(self, schedule: str) -> NDArray[np.float64]:
        """Get annealing schedules from string.

        Returns:
            Tuple of two numpy arrays containing the annealing schedules.
        """
        if schedule == "linear":
            linear = np.linspace(0.0, 1.0, self._n_layers + 2)[1:-1]
            return np.array([1 - linear, linear], dtype=np.float64)
        if schedule == "sinusoidal":
            linear = np.linspace(0.0, 1.0, self._n_layers + 2)[1:-1]
            sinusoidal = np.sin(0.5 * np.pi * np.sin(0.5 * np.pi * linear) ** 2) ** 2
            return np.array([1 - sinusoidal, sinusoidal], dtype=np.float64)

        msg = (
            f"Unsupported schedule '{schedule}'. Supported values are: "
            f"{self.metadata['supported_schedules']}"
        )
        raise ValueError(msg)

    def _schedule_from_1d(self, schedule: NDArray[np.float64]) -> NDArray[np.float64]:
        """Get annealing schedules from 1D array.

        Returns:
            Tuple of two numpy arrays containing the annealing schedules.
        """
        check_ndim(schedule, ndim=1)
        check_length(schedule, self.n_layers)
        check_1d_schedule_bounds(schedule)
        check_1d_schedule_start_and_end(schedule)

        return np.array([1 - schedule, schedule])

    def _schedule_from_2d(self, schedule: NDArray[np.float64]) -> NDArray[np.float64]:
        """Get annealing schedules from 2D array.

        Returns:
            Tuple of two numpy arrays containing the annealing schedules.
        """
        check_ndim(schedule, ndim=2)
        check_shape(schedule, (2, self.n_layers))
        check_initial_ham_schedule(schedule[0])
        check_problem_ham_schedule(schedule[1])
        check_schedule_monotonicity(schedule)

        return schedule


def get_default_device_if_none(
    device: DeviceConfig | Mapping[str, Any] | None = None,
) -> DeviceConfig:
    """Set default device if the one provided is ``None``.

    Default training device ``{"name": "default.qubit", "options": {}}``.

    Args:
        device: Device configuration or ``None``.

    Raises:
        KeyError: If `device` does not contain key ``"name"``.

    Returns:
        Given device or the default training device.
    """
    return DeviceConfig.from_mapping(
        device if device is not None else {"name": "default.qubit"}
    )


DigitalAdiabaticQuantumOptimizationSolver: TypeAlias = DAQOSolver
