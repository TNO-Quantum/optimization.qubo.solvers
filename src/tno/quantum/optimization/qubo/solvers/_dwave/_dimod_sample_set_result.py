"""This module contains the result object for dimod ``SampleSet``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, SupportsFloat, cast

from dimod.sampleset import SampleSet

from tno.quantum.optimization.qubo.components import Freq, ResultInterface
from tno.quantum.utils import BitVector
from tno.quantum.utils.serialization import Serializable

if TYPE_CHECKING:
    from typing import Self

    from tno.quantum.optimization.qubo.components import QUBO
    from tno.quantum.utils import BitVectorLike


class DimodSampleSetResult(ResultInterface):
    """Implementation of `ResultInterface` for the dimod sample sets."""

    def __init__(
        self,
        best_bitvector: BitVectorLike,
        best_value: SupportsFloat,
        freq: Freq,
        sampleset: SampleSet,
    ) -> None:
        """Init :py:class:`DimodSampleSetResult`."""
        super().__init__(best_bitvector, best_value, freq)
        self.sampleset = sampleset

    @classmethod
    def from_result(cls, qubo: QUBO, sampleset: SampleSet) -> Self:
        """Construct a :py:class:`DimodSampleSetResult` from a :py:class:`~dimod.SampleSet`.

        Args:
            qubo: QUBO that was solved, used to evaluate best sample.
            sampleset: Sample set.

        Returns:
            A :py:class:`DimodSampleSetResult` instance.
        """  # noqa: E501
        sampleset = sampleset.aggregate()  # type: ignore[no-untyped-call]
        sampleset.record.sort(order="energy")
        data_vectors = sampleset.data_vectors
        freq = Freq(
            sampleset.record["sample"],
            data_vectors["energy"],
            data_vectors["num_occurrences"],
        )
        best_bitvector = BitVector(sampleset.record["sample"][0])
        best_value = sampleset.first.energy
        return cls(best_bitvector, best_value, freq, sampleset)


# Register `SampleSet` as serializable
def _serialize_sampleset(sampleset: SampleSet) -> dict[str, Any]:
    return cast("dict[str, Any]", sampleset.to_serializable())  # type: ignore[no-untyped-call]


def _deserialize_sampleset(data: dict[str, Any]) -> SampleSet:
    return cast("SampleSet", SampleSet.from_serializable(data))  # type: ignore[no-untyped-call]


Serializable.register(SampleSet, _serialize_sampleset, _deserialize_sampleset)
