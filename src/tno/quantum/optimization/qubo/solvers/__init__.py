"""This package contains implementations of QUBO :py:class:`~tno.quantum.optimization.qubo.components.Solver`.

The :py:mod:`~tno.quantum.optimization.qubo.solvers` package can be installed using pip::

    pip install tno.quantum.optimization.qubo.solvers

By default, the package is installed without external solver dependencies. You can
specify which QUBO solvers you would like to install. Available options are
``[dwave, qubovert, qaoa]``. Alternatively, you can install all solvers
simultaneously using the ``[all]`` option::

    pip install tno.quantum.optimization.qubo.solvers[all]


Example:
-------
The following example shows how to list the available solvers and how to instantiate them.

>>> from tno.quantum.optimization.qubo.components import SolverConfig
>>> list(SolverConfig.supported_items())  # doctest: +NORMALIZE_WHITESPACE
['bf_solver',
 'custom_solver',
 'd_wave_clique_sampler_solver',
 'd_wave_sampler_solver',
 'exact_sampler_solver',
 'kerberos_sampler_solver',
 'leap_hybrid_solver',
 'neighborhood_solver',
 'pipeline_solver',
 'qaoa_solver',
 'rs_solver',
 'random_sampler_solver',
 'sa2_solver',
 'simulated_annealing_solver',
 'steepest_descent_solver',
 'tabu_solver',
 'tree_decomposition_solver']
>>> solver = SolverConfig(name='bf_solver').get_instance()

Once a solver is instantiated, it can be used to solve a
:py:class:`~tno.quantum.optimization.qubo.components.QUBO` as follows.

>>> from tno.quantum.optimization.qubo.components import QUBO
>>> # Construct QUBO
>>> qubo = QUBO([
...     [1,   2, 3],
...     [4, -50, 6],
...     [7,   8, 9]
... ])
>>>
>>> # Solve QUBO
>>> result = solver.solve(qubo)
>>> result.best_bitvector
BitVector(010)
>>> result.best_value
-50.0
"""  # noqa: E501

import importlib.util

from tno.quantum.optimization.qubo.solvers._classical._iterative_result import (
    IterativeResult,
)
from tno.quantum.optimization.qubo.solvers._classical._neighborhood_solver import (
    NeighborhoodSolver,
)
from tno.quantum.optimization.qubo.solvers._classical._rs_solver import (
    RSSolver,
)
from tno.quantum.optimization.qubo.solvers._pipeline._pipeline_result import (
    PipelineResult,
)
from tno.quantum.optimization.qubo.solvers._pipeline._pipeline_solver import (
    PipelineSolver,
)
from tno.quantum.optimization.qubo.solvers._pipeline._preprocess_result import (
    PreprocessResult,
)

__all__ = [
    "IterativeResult",
    "NeighborhoodSolver",
    "PipelineResult",
    "PipelineSolver",
    "PreprocessResult",
    "RSSolver",
]

if importlib.util.find_spec("dwave"):
    from tno.quantum.optimization.qubo.solvers._dwave._custom_solver import CustomSolver
    from tno.quantum.optimization.qubo.solvers._dwave._dimod_sample_set_result import (
        DimodSampleSetResult,
    )
    from tno.quantum.optimization.qubo.solvers._dwave._dwave_clique_sampler_solver import (  # noqa: E501
        DWaveCliqueSamplerSolver,
    )
    from tno.quantum.optimization.qubo.solvers._dwave._dwave_sampler_solver import (
        DWaveSamplerSolver,
    )
    from tno.quantum.optimization.qubo.solvers._dwave._exact_sampler_solver import (
        ExactSamplerSolver,
    )
    from tno.quantum.optimization.qubo.solvers._dwave._kerberos_sampler_solver import (
        KerberosSamplerSolver,
    )
    from tno.quantum.optimization.qubo.solvers._dwave._leap_hybrid_solver import (
        LeapHybridSolver,
    )
    from tno.quantum.optimization.qubo.solvers._dwave._random_sampler_solver import (
        RandomSamplerSolver,
    )
    from tno.quantum.optimization.qubo.solvers._dwave._simulated_annealing_solver import (  # noqa: E501
        SimulatedAnnealingSolver,
    )
    from tno.quantum.optimization.qubo.solvers._dwave._steepest_descent_solver import (
        SteepestDescentSolver,
    )
    from tno.quantum.optimization.qubo.solvers._dwave._tabu_solver import TabuSolver
    from tno.quantum.optimization.qubo.solvers._dwave._tree_decomposition_solver import (  # noqa: E501
        TreeDecompositionSolver,
    )

    __all__ += [
        "CustomSolver",
        "DWaveCliqueSamplerSolver",
        "DWaveSamplerSolver",
        "DimodSampleSetResult",
        "ExactSamplerSolver",
        "KerberosSamplerSolver",
        "LeapHybridSolver",
        "RandomSamplerSolver",
        "SimulatedAnnealingSolver",
        "SteepestDescentSolver",
        "TabuSolver",
        "TreeDecompositionSolver",
    ]

if importlib.util.find_spec("qubovert"):
    from tno.quantum.optimization.qubo.solvers._qubovert._bf_solver import BFSolver
    from tno.quantum.optimization.qubo.solvers._qubovert._qubovert_anneal_result import (  # noqa: E501
        QubovertAnnealResult,
    )
    from tno.quantum.optimization.qubo.solvers._qubovert._sa2_solver import SA2Solver

    __all__ += [
        "BFSolver",
        "QubovertAnnealResult",
        "SA2Solver",
    ]

if all(
    importlib.util.find_spec(package)
    for package in ["torch", "pennylane", "tqdm", "matplotlib"]
):
    from tno.quantum.optimization.qubo.solvers._qaoa._qaoa_result import QAOAResult
    from tno.quantum.optimization.qubo.solvers._qaoa._qaoa_solver import QAOASolver

    __all__ += [
        "QAOAResult",
        "QAOASolver",
    ]

__version__ = "1.0.0"
