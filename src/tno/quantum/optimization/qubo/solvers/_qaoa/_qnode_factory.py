"""This module contains the ``QNodeFactory`` class."""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as np
import pennylane as qml

from tno.quantum.utils import BackendConfig

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from pennylane.measurements import CountsMP, ExpectationMP
    from torchtyping import TensorType


class QNodeFactory:
    """The ``QNodeFactory`` class."""

    def __init__(
        self, external_fields: NDArray[np.float64], interactions: NDArray[np.float64]
    ) -> None:
        """Init of the ``QNodeFactory``.

        The ``QNodeFactory`` makes pennylane ``QNode`` objects representing QAOA
        circuits for a specified Lenz-Ising model.

        Args:
            external_fields: Linear terms in the Lenz-Ising model.
            interactions: Quadratic terms in the Lenz-Ising model.
        """
        self.wires = range(len(external_fields))
        self.cost_h = self._make_cost_hamiltonian(external_fields, interactions)
        self.mixer_h = qml.qaoa.x_mixer(self.wires)

    @staticmethod
    def _make_cost_hamiltonian(
        external_fields: NDArray[np.float64], interactions: NDArray[np.float64]
    ) -> qml.Hamiltonian:
        """Convert the provided Lenz-Ising model to a pennylane Hamiltonian.

        Args:
            external_fields: Linear terms in the Lenz-Ising model.
            interactions: Quadratic terms in the Lenz-Ising model.

        Returns:
            Pennylane Hamiltonian representation of the provided Lenz-Ising model.
        """
        coeffs = []
        observables = []
        for i in external_fields.nonzero()[0]:
            coeffs.append(external_fields[i])
            observables.append(qml.PauliZ(i))

        for i, j in zip(*interactions.nonzero()):
            coeffs.append(interactions[i, j])
            observables.append(qml.PauliZ(i) @ qml.PauliZ(j))

        return qml.Hamiltonian(coeffs, observables)

    def _qaoa_layer(self, gamma: TensorType, beta: TensorType) -> None:
        """Create a QAOA layer.

        Args:
            gamma: Parameters for the cost layer.
            beta: Parameters for the mixer layer.
        """
        qml.qaoa.cost_layer(gamma, self.cost_h)
        qml.qaoa.mixer_layer(beta, self.mixer_h)

    def _circuit(
        self, gamma: TensorType, beta: TensorType, num_layers: int, return_mode: str
    ) -> ExpectationMP | CountsMP:
        """Create a QAOA circuit.

        Args:
            gamma: Parameters for the cost layer.
            beta: Parameters for the mixer layer.
            num_layers: Number of cost and mixer layers.
            return_mode: Whether to use ``qml.expval`` or ``qml.counts``. Choose either
                "expval" or "counts".

        Raises:
            ValueError: If the `return_mode` is not recognized.
        """
        for w in self.wires:
            qml.Hadamard(wires=w)
        qml.layer(self._qaoa_layer, num_layers, gamma, beta)
        if return_mode == "expval":
            return qml.expval(self.cost_h)
        if return_mode == "counts":
            return qml.counts()
        msg = "unknown return_mode"
        raise ValueError(msg)

    def make_qnode(
        self,
        backend: BackendConfig | Mapping[str, Any],
        num_layers: int,
        return_mode: str,
    ) -> qml.QNode:
        """Create a QAOA circuit for the optimization step.

        The QAOA circuit consists of a layer of Hadamard gates on all qubits, followed
        by `num_layers` repeats of alternating cost and mixer layers. Each cost and
        mixer layer is parameterized by one parameter. The final layers consists of
        computing the expectation value of the cost Hamiltonian if `return_mode` is
        `"expval"`. If `return_mode` is `"counts"` the final layer consists of measuring
        all wires in the Z basis and counting the different combinations.

        Args:
            backend: Configuration for backend to execute the tape on.
            num_layers: Number of cost and mixer layers.
            return_mode: Wether to use ``qml.expval`` or ``qml.counts``. Choose either
                "expval" or "counts".

        Returns:
            Pennylane QNode that is parameterized by the gamma (cost) and beta (mixer)
            parameters, which should be vectors of length `num_layers`.
        """
        backend = BackendConfig.from_mapping(backend)
        backend_instance = backend.get_instance(wires=len(self.wires))
        func = partial(self._circuit, num_layers=num_layers, return_mode=return_mode)
        return qml.QNode(func, backend_instance)
