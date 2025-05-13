# TNO Quantum: Optimization - QUBO - Solvers

TNO Quantum provides generic software components aimed at facilitating the development
of quantum applications.

This package contains implementations of QUBO solvers.

## Documentation

Documentation of the `tno.quantum.optimization.qubo.solvers` package can be found [here](https://tno-quantum.github.io/documentation/).


## Install

Easily install the `tno.quantum.optimization.qubo.solvers` package using pip:

```console
$ python -m pip install tno.quantum.optimization.qubo.solvers
```

By default, the package is installed without external solver dependencies. You can
specify which QUBO solvers you would like to install. Available options are
``[dwave, qubovert, qaoa]``. Alternatively, you can install all solvers
simultaneously using the ``[all]`` option

```console
$ python -m pip install tno.quantum.optimization.qubo.solvers[all]
```

## Usage

The following example shows how to list the available solvers and how to instantiate them.

```python
from tno.quantum.optimization.qubo.components import SolverConfig

supported_solvers = list(SolverConfig.supported_items())
solver = SolverConfig(name='bf_solver').get_instance()
```

Once a solver is instantiated, it can be used to solve a `QUBO` as follows.

```python
from tno.quantum.optimization.qubo.components import QUBO

# Construct QUBO
qubo = QUBO([
     [1,   2, 3],
     [4, -50, 6],
     [7,   8, 9]
 ])

# Solve QUBO
result = solver.solve(qubo)
result.best_bitvector # BitVector(010)
result.best_value # -50.0
```

## (End)use limitations
The content of this software may solely be used for applications that comply with international export control laws.