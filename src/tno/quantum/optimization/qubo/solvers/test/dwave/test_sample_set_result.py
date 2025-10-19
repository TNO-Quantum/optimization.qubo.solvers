"""This module contains tests for the ``DimodSampleSetResult`` class."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
from dimod.sampleset import SampleSet
from tno.quantum.optimization.qubo.components import QUBO, Freq
from tno.quantum.utils import BitVector

from tno.quantum.optimization.qubo.solvers import (
    DimodSampleSetResult,
)

if TYPE_CHECKING:
    from pathlib import Path

DUMMY_PROPERTIES = {"dummy_properties": 1}


class TestWithSampleSet:
    """Test the ``DimodSampleSetResult`` with SampleSet."""

    @pytest.fixture
    def sampleset(self) -> SampleSet:
        """Return an example of SampleSet similar to real-world execution."""
        sampleset = SampleSet.from_samples(  # type: ignore[no-untyped-call]
            [[0, 1, 0], [0, 0, 0]],
            "BINARY",
            [-50, 0],
            num_occurrences=[9, 1],
            chain_break_fraction=[0.0, 0.0],
            info={
                "timing": {
                    "qpu_sampling_time": 989.0,
                    "qpu_anneal_time_per_sample": 20.0,
                    "qpu_readout_time_per_sample": 58.36,
                    "qpu_access_time": 16054.16,
                    "qpu_access_overhead_time": 1705.84,
                    "qpu_programming_time": 15065.16,
                    "qpu_delay_time_per_sample": 20.54,
                    "total_post_processing_time": 1960.0,
                    "post_processing_overhead_time": 1960.0,
                },
                "problem_id": "637d7174-c4c8-4263-bc52-7bbafb67803b",
            },
        )
        return cast("SampleSet", sampleset)

    @pytest.fixture
    def sampleset_qubo(self) -> QUBO:
        return QUBO([[1, 2, 3], [4, -50, 6], [7, 8, 9]])

    @pytest.fixture
    def bitvector(self) -> BitVector:
        return BitVector("010")

    @pytest.fixture
    def value(self) -> float:
        return -50.0

    @pytest.fixture
    def freq(self) -> Freq:
        return Freq(["010", "000"], [-50, 0], [9, 1])

    def test_init(
        self,
        sampleset_qubo: QUBO,
        bitvector: BitVector,
        value: float,
        freq: Freq,
        sampleset: SampleSet,
    ) -> None:
        """Check if the SampleSetResult parses the result correctly."""
        result = DimodSampleSetResult.from_result(sampleset_qubo, sampleset)
        assert np.array_equiv(result.best_bitvector, bitvector)
        assert result.best_value == value
        assert result.freq == freq
        assert result.sampleset == sampleset

    def test_dump_load(self, qubo: QUBO, sampleset: SampleSet, tmp_path: Path) -> None:
        """Check if the result can be stored and read from file"""
        result = DimodSampleSetResult.from_result(qubo, sampleset)
        result.to_json_file(tmp_path / "sampleset.json")
        loaded_sro = DimodSampleSetResult.from_json_file(tmp_path / "sampleset.json")

        assert np.array_equiv(result.best_bitvector, loaded_sro.best_bitvector)
        assert result.best_value == loaded_sro.best_value
        assert result.freq == loaded_sro.freq
        assert result.sampleset == loaded_sro.sampleset
