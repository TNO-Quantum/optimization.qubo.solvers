"""Library of layers for simulation based digital quantum optimization."""

from __future__ import annotations

from typing import TYPE_CHECKING, SupportsInt

import pennylane as qml
from tno.quantum.utils.validation import check_arraylike, check_int

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


class CostLayer:
    def __init__(self, external_fields: ArrayLike, interactions: ArrayLike) -> None:
        """Init the CostLayer.

        Args:
            external_fields: 1D-ArrayLike representing the external fields of a
                Lenz-Ising problem. These are the linear terms of the problem.
            interactions: 2D-ArrayLike representing the interaction terms of the
                Lenz-Ising problem. These are the quadratic terms of the problem.
        """
        self._external_fields = check_arraylike(
            external_fields, "external_fields", ndim=1
        )
        self._interactions = check_arraylike(
            interactions, "interactions", shape=(self.n_wires, self.n_wires)
        )

    @property
    def n_wires(self) -> int:
        """Number of wires used by the circuit."""
        return len(self._external_fields)

    def __call__(self, angle: float) -> None:
        r"""Apply the cost layer.

        The cost layer is defined as

        .. math::

            \theta \\left(
                \\sum_{i=1}^N h_i Z_i + \\sum_{i=1}^N\\sum_{j=i+1}^NJ_{ij}Z_iZ_j
            \right).

        **Notation**

        * $h_i$: external field $i$ of the Lenz-Ising model.
        * $J_{ij}$: quadratic interaction term acting on qubits $i$ and $j$.
        * $N$: number of spins in the Lenz-Ising model.
        * $Z_i$: Pauli-Z operator acting on qubit $i$.
        * $\theta$: angle provided to the call of this method.

        The implementation only uses gates that are natively supported by lighting
        qubit device.

        Args:
            angle: $\theta$ to use when applying the cost layer.
        """
        for i, j in zip(*self._interactions.nonzero(), strict=False):
            qml.CNOT([i, j])
            qml.RZ(2 * angle * self._interactions[i, j], wires=j)
            qml.CNOT([i, j])

        for i in self._external_fields.nonzero()[0]:
            qml.RZ(2 * angle * self._external_fields[i], wires=i)


class InitialLayer:
    def __init__(self, n_wires: SupportsInt) -> None:
        """Initialze the InitialLayer.

        Args:
            n_wires: number of wires.
        """
        self._n_wires = check_int(n_wires, "n_wires", l_bound=1)

    def prep(self) -> None:
        r"""State preperation for this InitialLayer.

        For the adiabatic/counterdiabatic quantum optimization protocol, the qubits
        need to be prepared in the ground state of the initial Hamiltonian. For this
        initial Hamiltonian the ground state is a equal superposition, i.e.,

        .. math::

            \\sum_{i=1}^N |+\rangle.

        **Notation**

        * $N$: number of spins in the Lenz-Ising model.
        """
        for i in range(self._n_wires):
            qml.Hadamard(wires=i)

    def __call__(self, angle: float) -> None:
        r"""Apply the layer of an initial Hamiltonian.

        The initial Hamiltonian refers to the initial Hamiltonian in the
        adiabatic/counterdiabatic quantum optimization protocol.

        The initial layer is defined as

        .. math::

            -\theta \sum_{i=1}^N X_i.

        **Notation**

        * $N$: number of spins in the Lenz-Ising model.
        * $X_i$: Pauli-X operator acting on qubit $i$.
        * $\theta$: angle provided to the call of this method.

        Args:
            angle: $\theta$ to use when applying the initial layer.
        """
        for i in range(self._n_wires):
            qml.RX(-2 * angle, wires=i)
