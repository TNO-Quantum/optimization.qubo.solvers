"""This module contains the ``QubovertAnnealResult`` class."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any, SupportsFloat

from qubovert.sim import AnnealResult, AnnealResults

from tno.quantum.optimization.qubo.components import Freq, ResultInterface
from tno.quantum.utils import BitVector, BitVectorLike
from tno.quantum.utils.serialization import Serializable

if TYPE_CHECKING:
    from typing import Self


class QubovertAnnealResult(ResultInterface):
    """Implementation of :py:class:`ResultInterface` for quboverts :py:class:`AnnealResults`."""  # noqa: E501

    def __init__(
        self,
        best_bitvector: BitVectorLike,
        best_value: SupportsFloat,
        freq: Freq,
        anneal_results: AnnealResults,
    ) -> None:
        """Init :py:class:`QubovertAnnealResult`."""
        super().__init__(best_bitvector, best_value, freq)
        self._anneal_results = anneal_results

    @property
    def anneal_results(self) -> AnnealResult:
        """Underlying :py:class:`AnnealResults` object."""
        return self._anneal_results

    @classmethod
    def from_result(cls, result: AnnealResults) -> Self:
        """Create :py:class:`QubovertAnnealResult` from :py:class:`AnnealResults`.

        Args:
            result: :py:class:`AnnealResults` as obtained from a qubovert solver.
        """
        best_bit_vector = BitVector(result.best.state)
        best_value = result.best.value
        anneal_results = result

        counter = Counter((BitVector(item.state), item.value) for item in result)
        bit_vectors = [bit_vector for (bit_vector, _) in counter]
        energies = [energies for (_, energies) in counter]
        num_occurrences = list(counter.values())
        freq = Freq(bit_vectors, energies, num_occurrences)

        return cls(best_bit_vector, best_value, freq, anneal_results)


# Register `AnnealResult` as serializable
def _serialize_anneal_result(anneal_result: AnnealResult) -> dict[str, Any]:
    return {
        "state": anneal_result.state,
        "value": anneal_result.value,
        "spin": anneal_result.spin,
    }


def _deserialize_anneal_result(data: dict[str, Any]) -> AnnealResult:
    return AnnealResult(**data)


Serializable.register(
    AnnealResult, _serialize_anneal_result, _deserialize_anneal_result
)


# Register `AnnealResults` as serializable
def _serialize_anneal_results(anneal_results: AnnealResults) -> dict[str, Any]:
    return {"anneal_results": [Serializable.serialize(ar) for ar in anneal_results]}


def _deserialize_anneal_results(data: dict[str, Any]) -> AnnealResults:
    return AnnealResults(
        Serializable.deserialize(sample) for sample in data["anneal_results"]
    )


Serializable.register(
    AnnealResults, _serialize_anneal_results, _deserialize_anneal_results
)
