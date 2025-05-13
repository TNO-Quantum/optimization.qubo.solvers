"""Test module for the ``QNodeFactory`` class."""

import numpy as np
import pennylane as qml
import pytest

from tno.quantum.optimization.qubo.solvers._qaoa._qnode_factory import QNodeFactory


class TestQNodeFactory:
    @pytest.fixture(name="factory")
    def factory_fixture(self) -> QNodeFactory:
        external_fields = np.array([1, 2, 3])
        interactions = np.array([[0, 10, 20], [0, 0, 30], [0, 0, 0]])
        return QNodeFactory(external_fields, interactions)

    def test_attributes(self, factory: QNodeFactory) -> None:
        expected_wires = range(3)
        expected_cost_h = qml.Hamiltonian(
            [1, 2, 3, 10, 20, 30],
            [
                qml.Z(0),
                qml.Z(1),
                qml.Z(2),
                qml.Z(0) @ qml.Z(1),
                qml.Z(0) @ qml.Z(2),
                qml.Z(1) @ qml.Z(2),
            ],
        )
        expected_mixer_h = qml.Hamiltonian([1, 1, 1], [qml.X(0), qml.X(1), qml.X(2)])

        assert factory.wires == expected_wires
        assert factory.cost_h == expected_cost_h
        assert factory.mixer_h == expected_mixer_h

    def test_circuit_error(self, factory: QNodeFactory) -> None:
        with pytest.raises(ValueError, match="unknown return_mode"):
            factory._circuit([1], [2], 1, "test")

    @pytest.mark.filterwarnings("ignore:Attempted to compute the gradient of a tape")
    @pytest.mark.parametrize("return_mode", ["expval", "counts"])
    def test_make_qnode(self, factory: QNodeFactory, return_mode: str) -> None:
        beta = [1, 2]
        gamma = [3, 4]
        device = {"name": "default.qubit", "options": {"shots": 1000}}
        qnode = factory.make_qnode(device, 2, return_mode)

        specs = qml.specs(qnode)(gamma, beta)
        assert specs["device_name"] == "default.qubit"
        resources = specs["resources"]
        assert resources.num_wires == 3
        assert resources.num_gates == 7
        assert resources.gate_types == {"Hadamard": 3, "ApproxTimeEvolution": 4}
