"""Test module for the ``QubovertAnnealResult`` class."""

from pathlib import Path

import pytest
from qubovert.sim._anneal_results import AnnealResult, AnnealResults

from tno.quantum.optimization.qubo.components import Freq
from tno.quantum.optimization.qubo.solvers._qubovert._qubovert_anneal_result import (
    QubovertAnnealResult,
)
from tno.quantum.utils import BitVector


class TestWithAnnealResults:
    """Test the ``ResultInterface`` of AnnealResults."""

    @pytest.fixture
    def sa_results(self) -> AnnealResults:
        """Create an example result"""
        return AnnealResults(
            [AnnealResult({"x(0)": 0, "x(1)": 1, "x(2)": 0}, -50, False)] * 9
            + [AnnealResult({"x(0)": 0, "x(1)": 0, "x(2)": 0}, 0, False)]
        )

    @pytest.fixture
    def expected_bitvector(self) -> BitVector:
        return BitVector("010")

    @pytest.fixture
    def expected_value(self) -> float:
        return -50

    def test_init(
        self,
        sa_results: AnnealResults,
        expected_bitvector: BitVector,
        expected_value: float,
    ) -> None:
        """Test if the object can parse the result correctly"""
        result = QubovertAnnealResult.from_result(sa_results)

        expected_freq = Freq(["010", "000"], [-50, 0], [9, 1])

        assert result.best_bitvector == expected_bitvector
        assert result.best_value == expected_value
        assert result.freq == expected_freq
        assert result.anneal_results == sa_results

    def test_dump_load(self, sa_results: AnnealResults, tmp_path: Path) -> None:
        """Test if the results can be stored and loaded from a file"""
        result = QubovertAnnealResult.from_result(sa_results)
        result.to_json_file(tmp_path / "qubovert.json")
        loaded_sro: QubovertAnnealResult = QubovertAnnealResult.from_json_file(
            tmp_path / "qubovert.json"
        )

        assert result.best_bitvector == loaded_sro.best_bitvector
        assert result.best_value == loaded_sro.best_value
        assert result.freq == loaded_sro.freq
        assert result.anneal_results == loaded_sro.anneal_results
