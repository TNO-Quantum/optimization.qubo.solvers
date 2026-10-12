"""Library of layers for simulation based digital quantum optimization."""

from __future__ import annotations

from typing import TYPE_CHECKING, SupportsInt

import numpy as np
import pennylane as qml
from tno.quantum.utils.validation import check_arraylike, check_int

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray


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


class CDLayerY:
    def __init__(self, external_fields: ArrayLike, interactions: ArrayLike) -> None:
        """Init the CDLayerY as described in the work by Hegade *et al.* [1]_.

        Args:
            external_fields: 1D-ArrayLike representing the external field terms of a
                Lenz-Ising problem. These are the linear terms of the problem.
            interactions: 2D-ArrayLike representing the interaction terms of the
                Lenz-Ising problem. These are the quadratic terms of the problem.

        References:
        ----------
        .. [1] `Hegade, N. N., Chen, X., & Solano, E. (2022). \
        *Digitized counterdiabatic quantum optimization*. \
        Physical Review Research, 4(4), L042030. <https://arxiv.org/abs/2201.00790>`_
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

    def __call__(self, strength: float, strength_grad: float) -> None:
        r"""Apply the Y counterdiabatic layer.

        The Y counterdiabatic layer is defined as

        .. math::

            \dot{\lambda(t)}\frac{h_i}{2}
            \sum_{i=1}^NY_i
            \left[
            (\lambda(t)-1)^2
            + \lambda(t)^2\left(h_i^2 + \sum_{j=1}^N(J_{ij} + J_{ji})\right)
            \right]

        **Notation**

        * $h_i$: linear bias term $i$ of the Lenz-Ising model.
        * $J_{ij}$: quadratic interaction term acting on qubits $i$ and $j$.
        * $N$: number of spins in the Lenz-Ising model.
        * $Y_i$: Pauli-Y operator acting on qubit $i$.
        * $\lambda$: strength schedule.

        Args:
            strength: strength of the schedule at timestep $t$.
            strength_grad: gradient strength of the schedule at timestep $t$.
        """
        for i, bias in enumerate(self._external_fields):
            angle = bias**2
            angle += np.sum(self._interactions[i] ** 2)
            angle += np.sum(self._interactions[:, i] ** 2)
            angle *= strength**2
            angle += (strength - 1) ** 2
            angle *= bias * strength_grad

            qml.RY(angle, wires=i)


class CDLayerYZY:
    def __init__(self, external_fields: ArrayLike, interactions: ArrayLike) -> None:
        """Init the CDLayerYZY as described in the work by Hegade *et al.* [1]_.

        Args:
            external_fields: 1D-ArrayLike representing the external field terms of a
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

        const1, const2, const3 = self._compute_schedule_consts(
            self._external_fields,
            self._interactions,
        )
        self._const1 = const1
        self._const2 = const2
        self._const3 = const3

    @property
    def n_wires(self) -> int:
        """Number of wires used by the circuit."""
        return len(self._external_fields)

    @staticmethod
    def _compute_schedule_consts(
        external_fields: NDArray[np.float64],
        interactions: NDArray[np.float64],
    ) -> tuple[float, float, float]:
        n_wires = len(external_fields)
        const1 = np.sum(external_fields**2) + 8 * np.sum(interactions**2)
        const2 = const1 + np.sum(external_fields**4) + 2 * np.sum(interactions**4)
        for i in range(n_wires):
            for j in range(i + 1, n_wires):
                const2 += 6 * external_fields[i] ** 2 * interactions[i, j] ** 2

        for i in range(n_wires):
            for j in range(i + 1, n_wires):
                for k in range(n_wires):
                    for l in range(k + 1, n_wires):  # noqa: E741 used to stay true to the original paper
                        if i != k and j != l:
                            continue
                        const2 += 6 * interactions[i, j] ** 2 * interactions[k, l] ** 2

        const3 = -0.25 * (np.sum(external_fields**2) + 2 * np.sum(interactions**2))
        return const1, const2, const3

    def __call__(self, strength: float, strength_grad: float) -> None:
        """Add the Counterdiabatic YZY layer."""
        R_ts = (1 - 2 * strength) * self._const1 + strength**2 * self._const2  # noqa: N806 used to stay true to the original paper
        alpha = self._const3 / R_ts

        angle = -2 * strength_grad * alpha

        for i, bias in enumerate(self._external_fields):
            if bias == 0:
                continue
            qml.RY(2 * angle * bias, wires=i)

        for i in range(self.n_wires):
            for j in range(i + 1, self.n_wires):
                interaction = self._interactions[i, j]
                if interaction == 0:
                    continue
                # Benchmarking shows that PauliRot works faster with Pennylane
                # simulators compared to manual decomposition. I don't understand why
                # though. It is not natively supported by the simulator.
                qml.PauliRot(2 * angle * interaction, "YZ", wires=[i, j])
                qml.PauliRot(2 * angle * interaction, "ZY", wires=[i, j])
