# 1.5.0 (2025 - 10 - 19)

Support for `python3.9` is dropped.

### Features

* **SpinReversalTransform:** Added option to perform spin reversal transform to `DWaveSamplerSolver`, 
`DWaveCliqueSamplerSolver` and `DWaveParallelEmbeddingSolver`.
* **Parallel qpu solving:** Added `DWaveParallelEmbeddingSolver`.
* **Embedded simulated annealing:** Added `DWaveEmbeddedSimulatedAnnealingSolver` and `DWaveCliqueEmbeddedSimulatedAnnealingSolver`.
* **Digital Adiabatic Quantum Optimization solver :** Added `DAQOSolver`.

### Deprecations

* **DWaveTilingSolver**: Solver introduced in `v1.1.0` will be removed in `v2.0.0`, use `DWaveParallelEmbeddingSolver` instead.

# 1.0.0 (2025 - 05 - 13)

Initial public release.

### Features

* **Classical:** Classical neighborhood and relaxation sampler solvers.
* **Dwave:** Wrappers around D-Wave solvers.
* **QAOA:** PennyLane implementation of QAOA QUBO solver.
* **Qubovert:** Classical brute force and simulated annealing solvers.
