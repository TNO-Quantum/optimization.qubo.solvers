"""This module contains the ``QAOASolver`` class."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, SupportsInt

import numpy as np
import torch
from numpy.random import RandomState
from numpy.typing import ArrayLike, NDArray
from tqdm import trange

from tno.quantum.optimization.qubo.components import QUBO, Solver
from tno.quantum.optimization.qubo.solvers._qaoa._qaoa_result import QAOAResult
from tno.quantum.optimization.qubo.solvers._qaoa._qnode_factory import QNodeFactory
from tno.quantum.utils import BackendConfig, OptimizerConfig
from tno.quantum.utils.validation import (
    check_arraylike,
    check_bool,
    check_int,
    check_random_state,
)

if TYPE_CHECKING:
    import pennylane as qml
    from torch.optim.optimizer import Optimizer


class QAOASolver(Solver[QAOAResult]):
    """Solver implementing a basic Quantum Approximate Optimization Algorithm (QAOA).

    The Mixer Hamiltonian is a Pauli-X Hamiltonian and the cost Hamiltonian is the
    Lenz-Ising Hamiltonian retrieved from the QUBO problem.

    Example:
        >>> from tno.quantum.optimization.qubo.components import QUBO
        >>> from tno.quantum.optimization.qubo.solvers import QAOASolver
        >>>
        >>> qubo = QUBO([[1, 2, 3], [4, -50, 6], [7, 8, 9]])
        >>> solver = QAOASolver(
        ...     num_layers=4,
        ...     num_iter=100,
        ... )
        >>> result = solver.solve(qubo)
        >>> result.best_bitvector
        BitVector(010)
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        random_state: int | RandomState | None = None,
        num_layers: SupportsInt = 1,
        num_iter: SupportsInt = 100,
        init_beta: ArrayLike | None = None,
        init_gamma: ArrayLike | None = None,
        training_backend: BackendConfig | Mapping[str, Any] | None = None,
        evaluation_backend: BackendConfig | Mapping[str, Any] | None = None,
        optimizer: OptimizerConfig | Mapping[str, Any] | None = None,
        verbose: bool = False,
    ) -> None:
        r"""Init :py:class:`QAOASolver`.

        Args:
            random_state: Random state to seed. The seed is used when creating random
                parameters (gamma and beta). Note that this seed *does not* seed the
                backends or optimizers. These should be seeded separately.
            num_layers: Number of layers to use. Each layer contains one mixer block and
                one cost block. Default is 1.
            num_iter: Number of optimization iterations to perform. Default is 100.
            init_beta: Initial parameters for the mixer layer. Should be an
                1-D ArrayLike of length `num_layers` with values in the range
                $[0, \pi)$. If ``None`` (default) is provided the random values will be
                generated.
            init_gamma: Initial parameters for the cost layer. Should be an
                1-D ArrayLike of length `num_layers` with values in range $[0, 2\pi)$.
                If ``None`` (default) is provided the random values will be generated.
            training_backend: Training backend to use. Must be a ``BackendConfig`` or a
                mapping with the ``"name"`` and ``"options"`` keys. If ``None``
                (default) is provided, ``{"name": "default.qubit", "options": {}}`` is
                used.
            evaluation_backend: Evaluation backend to use after the circuit has been
                optimized. The chosen backend must have the `shots` attribute. Must be
                a ``BackendConfig`` or a mapping with the "name" and "options" keys. If
                ``None`` (default) is provided,
                ``{"name": "default.qubit", "options": {"shots": 100}}`` is used.
            optimizer: Optimizer to use. Must be an ``OptimizerConfig`` or a mapping
                with the "name" and "options" keys. If ``None`` (default) is provided,
                ``{"name": "adagrad", "options": {}}`` is used.
            verbose: If ``True`` print additional information. Default is ``False``.

        Raises:
            ValueError: If `num_layers` or `num_iter` is less than 1, or if `init_beta`
                or `init_gamma` is not 1-dimensional of length `num_layers`.
            TypeError: If `random_state`, `num_layers`, `num_iter`, `init_beta` or
                `init_gamma` has invalid type.
        """
        self._random_state = check_random_state(random_state, "random_state")
        self._num_layers = check_int(num_layers, "num_layers", l_bound=1)
        self.num_iter = num_iter  # type: ignore
        self.init_beta = (
            init_beta  # type: ignore
            if init_beta is not None
            else self._random_state.uniform(high=np.pi, size=self.num_layers)
        )
        self.init_gamma = (
            init_gamma  # type: ignore
            if init_gamma is not None
            else self._random_state.uniform(high=2.0 * np.pi, size=self.num_layers)
        )
        self.training_backend = training_backend  # type: ignore
        self.evaluation_backend = evaluation_backend  # type: ignore
        self.optimizer = optimizer  # type: ignore
        self.verbose = verbose

    @property
    def num_layers(self) -> int:
        """Number of layers to use."""
        return self._num_layers

    @property
    def num_iter(self) -> int:
        """Number of optimization iterations to perform."""
        return self._num_iter

    @num_iter.setter
    def num_iter(self, value: SupportsInt) -> None:
        self._num_iter = check_int(value, "num_iter", l_bound=1)

    @property
    def init_beta(self) -> NDArray[np.float64]:
        """Initial parameters for the mixer layer."""
        return self._init_beta

    @init_beta.setter
    def init_beta(self, value: ArrayLike) -> None:
        self._init_beta = check_arraylike(
            value, "init_beta", ndim=1, shape=(self.num_layers,)
        ).astype(np.float64)

    @property
    def init_gamma(self) -> NDArray[np.float64]:
        """Initial parameters for the cost layer."""
        return self._init_gamma

    @init_gamma.setter
    def init_gamma(self, value: ArrayLike) -> None:
        self._init_gamma = check_arraylike(
            value, "init_gamma", ndim=1, shape=(self.num_layers,)
        ).astype(np.float64)

    @property
    def training_backend(self) -> BackendConfig:
        """Configuration for training backend to use."""
        return self._training_backend

    @training_backend.setter
    def training_backend(self, value: BackendConfig | Mapping[str, Any]) -> None:
        self._training_backend = get_default_training_backend_if_none(value)

    @property
    def evaluation_backend(self) -> BackendConfig:
        """Configuration for evaluation backend to use."""
        return self._evaluation_backend

    @evaluation_backend.setter
    def evaluation_backend(self, value: BackendConfig | Mapping[str, Any]) -> None:
        self._evaluation_backend = get_default_evaluation_backend_if_none(value)

    @property
    def optimizer(self) -> OptimizerConfig:
        """Configuration for optimizer to use."""
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value: OptimizerConfig | Mapping[str, Any]) -> None:
        self._optimizer = get_default_optimizer_if_none(value)

    @property
    def verbose(self) -> bool:
        """Flag to indicate whether to print additional information."""
        return self._verbose

    @verbose.setter
    def verbose(self, value: bool) -> None:
        self._verbose = check_bool(value, "verbose")

    def _solve(self, qubo: QUBO) -> QAOAResult:
        """Solve the given QUBO using the QAOA algorithm."""
        external_fields, interactions, _ = qubo.to_ising()
        qnode_factory = QNodeFactory(external_fields, interactions)
        qnode_cost = qnode_factory.make_qnode(
            self.training_backend, self.num_layers, "expval"
        )
        qnode_counts = qnode_factory.make_qnode(
            self.evaluation_backend, self.num_layers, "counts"
        )

        properties = {
            "num_iter": self.num_iter,
            "init_beta": self._init_beta,
            "init_gamma": self._init_gamma,
            "training_backend": self.training_backend,
            "evaluation_backend": self.evaluation_backend,
            "optimizer": self.optimizer,
        }

        properties.update(
            self._optimize_parameters(
                qnode_cost=qnode_cost,
                num_iter=self.num_iter,
                init_beta=self._init_beta,
                init_gamma=self._init_gamma,
                optimizer=self.optimizer,
                verbose=self.verbose,
            )
        )

        raw_result = self._run_optimized_circuit(qnode_counts, properties)

        return QAOAResult.from_result(qubo, raw_result, properties)

    @staticmethod
    def _optimize_parameters(  # noqa: PLR0913
        qnode_cost: qml.QNode,
        num_iter: int,
        init_beta: NDArray[np.float64],
        init_gamma: NDArray[np.float64],
        optimizer: OptimizerConfig,
        *,
        verbose: bool = False,
    ) -> dict[str, Any]:
        """Optimize the parameters for the cost circuit.

        Args:
            qnode_cost: QNode encoding the cost function to optimize.
            num_layers: Number of layers to use.
            num_iter: Number of optimization iterations to perform.
            init_beta: Initial parameters for the mixer layer.
            init_gamma: Initial parameters for the cost layer.
            optimizer: Configuration for optimizer to use.
            verbose: If ``True`` print additional information. Default is ``False``.

        Returns:
            Dictionary containing three keys: ``"final_gamma"``, ``"final_beta"`` and
            ``"expval_history"``.
        """
        beta = torch.tensor(init_beta, requires_grad=True)
        gamma = torch.tensor(init_gamma, requires_grad=True)

        optimizer_instance: Optimizer = optimizer.get_instance([gamma, beta])
        expval_history = np.zeros(num_iter)

        # Optimize gamma and beta
        for i in trange(num_iter) if verbose else range(num_iter):
            loss = qnode_cost(gamma, beta)
            expval_history[i] = loss.detach()

            optimizer_instance.zero_grad()
            loss.backward()
            optimizer_instance.step()

        final_gamma, final_beta = optimizer_instance.param_groups[0]["params"]

        return {
            "final_gamma": final_gamma.tolist(),
            "final_beta": final_beta.tolist(),
            "expval_history": expval_history.tolist(),
        }

    @classmethod
    def _run_optimized_circuit(
        cls, qnode_counts: qml.QNode, properties: dict[str, Any]
    ) -> Mapping[str, int]:
        """Optimize the parameters for the cost circuit.

        Args:
            qnode_counts: QNode of the QAOA circuit.
            properties: Dictionary containing at least the keys ``"final_gamma"`` and
                ``"final_beta"``.

        Returns:
            A mapping with bitstrings as keys and frequencies as values.
        """
        final_gamma = properties["final_gamma"]
        final_beta = properties["final_beta"]

        raw_result: Mapping[str, int] = qnode_counts(final_gamma, final_beta)
        return raw_result


def get_default_training_backend_if_none(
    backend: BackendConfig | Mapping[str, Any] | None = None,
) -> BackendConfig:
    """Set default training backend if the one provided is ``None``.

    Default training backend ``{"name": "default.qubit", "options": {}}``.

    Args:
        backend: backend configuration or ``None``.

    Raises:
        KeyError: If `backend` does not contain key ``"name"``.

    Returns:
        Given backend or the default training backend.
    """
    return BackendConfig.from_mapping(
        backend if backend is not None else {"name": "default.qubit"}
    )


def get_default_evaluation_backend_if_none(
    backend: BackendConfig | Mapping[str, Any] | None = None,
) -> BackendConfig:
    """Set default evaluation backend if the one provided is ``None``.

    Default evaluation backend ``{"name": "default.qubit", "options": {"shots": 100}}``.

    Args:
        backend: evaluation configuration or ``None``.

    Raises:
        KeyError: If `backend` does not contain key ``"name"``.

    Returns:
        Given backend or the default evaluation backend.
    """
    return BackendConfig.from_mapping(
        backend
        if backend is not None
        else {"name": "default.qubit", "options": {"shots": 100}}
    )


def get_default_optimizer_if_none(
    optimizer: OptimizerConfig | Mapping[str, Any] | None = None,
) -> OptimizerConfig:
    """Set default optimizer if the one provided is ``None``.

    Default optimizer ``{"name": "adagrad", "options": {}}``.

    Args:
        optimizer: optimizer configuration or ``None``.

    Raises:
        KeyError: If `optimizer` does not contain key ``"name"``.

    Returns:
        Given optimizer or the default optimizer.
    """
    return OptimizerConfig.from_mapping(
        optimizer if optimizer is not None else {"name": "adagrad"}
    )
