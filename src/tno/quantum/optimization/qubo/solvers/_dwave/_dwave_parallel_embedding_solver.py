"""This module contains the ``DWaveParallelEmbeddingSolver`` class."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any, SupportsInt

import dimod
from dimod import child_structure_dfs
from dwave.preprocessing import SpinReversalTransformComposite
from dwave.system import (
    AutoEmbeddingComposite,
    DWaveSampler,
    ParallelEmbeddingComposite,
)
from minorminer import find_embedding
from tno.quantum.optimization.qubo.components import QUBO, Solver
from tno.quantum.utils.validation import (
    check_bool,
    check_int,
    check_string,
)

from tno.quantum.optimization.qubo.solvers._dwave._dimod_sample_set_result import (
    DimodSampleSetResult,
)
from tno.quantum.optimization.qubo.solvers._dwave._utils import (
    get_singleton,
    retry_on_network_errors,
)


class DWaveParallelEmbeddingSolver(Solver[DimodSampleSetResult]):
    """D-Wave parallel embedding composite solver.

    The :py:class:`DWaveParallelEmbeddingSolver` class solves QUBOs using the
    :py:class:`~dwave.system.composites.ParallelEmbeddingComposite`. This class enables
    parallel sampling on a structured sampler by use of multiple disjoint embeddings.

    Enables parallel sampling on a target sampler by use of multiple disjoint embeddings.
    If a list of embeddings is not provided, the function `find_multiple_embeddings()` is
    called by default to attempt a maximum number of embeddings. Parallelization of job
    submissions can mitigate for network latency, programming time and readout time in
    the case of QPU samplers, subject to additional complexity in the embedding process.

    Example:
        >>> from tno.quantum.optimization.qubo.components import QUBO
        >>> from tno.quantum.optimization.qubo.solvers import DWaveParallelEmbeddingSolver
        >>>
        >>> qubo = QUBO([[1, 2, 3], [4, -50, 6], [7, 8, 9]])
        >>> solver = DWaveParallelEmbeddingSolver(num_reads = 100, max_num_emb = 5)
        >>> result = solver.solve(qubo)  # doctest: +SKIP
        >>> result.best_bitvector # doctest: +SKIP
        BitVector(010)
    """  # noqa: E501

    non_deterministic = True

    def __init__(  # noqa: PLR0913
        self,
        *,
        num_attempts: int = 1,
        num_reads: SupportsInt = 1,
        backend_id: str | None = None,
        embeddings: list[dict[int, list[int]]] | None = None,
        embedder: Callable[..., Any] | None = None,
        embedder_kwargs: dict[str, Any] | None = None,
        one_to_iterable: bool = True,
        max_num_emb: int | None = None,
        child_structure_search: Callable[..., Any] | None = None,
        reuse_embedding: bool = False,
        num_spin_reversal_transforms: int = 0,
        **sample_kwargs: Any,
    ) -> None:
        """Init :py:class:`DWaveParallelEmbeddingSolver`.

        Args:
            num_attempts: Number of solve attempts whenever a
                :py:exc:`~http.client.RemoteDisconnected`, or
                :py:exc:`~urllib3.exceptions.ProtocolError` or
                :py:exc:`~requests.ConnectionError` errors has occurred. Waits 15
                seconds in between attempts.
            num_reads: Number reads to sample. Default is 1.
            backend_id: Id of the sampler to use (e.g. ``'Advantage_system4.1'``). If
                ``None`` is given, the D-Wave default sampler is used. Default is
                ``None``.
            embeddings: A list of embeddings. Each embedding is assumed to be a
                dictionary with source-graph nodes as keys and iterables on target-graph
                nodes as values. The embeddings can include keys not required by the
                source graph. Note that one_to_iterable is ignored (assumed True). Here
                the source graph is the graph corresponding to the qubo.
            embedder: The outer embedder function that finds multiple disjoint
                embeddings for parallel sampling. This function should return a list of
                embeddings, where each embedding maps the source graph to a different
                region of the target hardware graph. If ``None`` (default), uses
                :py:func:`~minorminer.utils.parallel_embeddings.find_multiple_embeddings`.
                This outer embedder internally uses an "inner embedder" (specified via
                ``embedder_kwargs["embedder"]``) to find individual embeddings. The
                inner embedder defaults to :py:func:`~minorminer.find_embedding` in this
                solver (which allows chains), rather than D-Wave's default
                :py:func:`~minorminer.find_subgraph` (which requires exact subgraph
                matches and often fails for dense graphs).
            embedder_kwargs: Keyword arguments for the embedder function. The default
                is an empty dictionary.
            one_to_iterable: This parameter should be fixed to match the value type
                returned by embedder. If ``False`` the values in every dictionary are
                target nodes (defining a subgraph embedding), these are transformed to
                tuples for compatibility with :py:func:`~dwave.embedding.embed_bqm` and
                :py:func:`~dwave.embedding.unembed_sampleset`. If ``True``, the
                values are iterables over target nodes and no transformation is
                required. When no custom embedder is provided, this is automatically
                set to ``True`` to work with :py:func:`~minorminer.find_embedding`.
                Default is ``True``.
            max_num_emb: Maximum number of embeddings to find for parallel sampling.
                This parameter is to be used within the `embedder_kwargs`. If it is
                provided as ``None`` (the default), the embedder will try to find
                as many embeddings as possible. If this value is also specified
                insde the `embedder_kwargs`, the solver will give priority to the
                value set in the `embedder_kwargs`.
            child_structure_search: A function that accepts a sampler and returns the
                structure attribute. Defaults to
                :py:func:`~dimod.utilities.child_structure_dfs`.
            reuse_embedding: If ``True``, the embedding is reused in subsequent calls to
                the solver. If ``False``, a new embedding is created for each call to
                the solver. Default is ``False``. Note that this option locks the solver
                to solve just one QUBO 'shape'.
            num_spin_reversal_transforms: Number of spin reversal transform runs. A
                value of ``0`` will not transform the problem. If you specify a
                nonzero value, each spin reversal transform will result in an
                independent run of the child sampler, with all results aggregated
                into a single sample set containing
                ``num_emb*num_reads*num_spin_reversal_transforms`` samples, where
                ``num_emb <= max_num_emb`` is the number of times the problem could be
                embedded in parallel. Default is ``0``.
            sample_kwargs: Additional keyword arguments that can be used for the
                sampler. See the D-Wave documentation of
                :py:attr:`~dwave.system.samplers.DWaveSampler.parameters` for possible
                additional keyword definitions.

        Raises:
            ValueError: For invalid values in `num_reads`. If the child_sampler is not
                structured, and the structure cannot be inferred from
                child_structure_search. If the embeddings provided are an empty list,
                or no embeddings are found. If embeddings and source graph nodes are
                inconsistent. If embeddings and target graph nodes are inconsistent.
            TypeError: For incorrect types in `num_reads`.
        """
        self.num_attempts = check_int(num_attempts, "num_attempts", l_bound=1)
        self.num_reads = check_int(num_reads, "num_reads", l_bound=1)
        self.backend_id = (
            check_string(backend_id, "backend_id") if backend_id is not None else None
        )
        self.embeddings = embeddings
        self.embedder = embedder
        self.embedder_kwargs = embedder_kwargs or {}
        self.one_to_iterable = one_to_iterable
        if embedder is None:
            self.embedder_kwargs["embedder"] = find_embedding
            self.embedder_kwargs["one_to_iterable"] = True

            if self.one_to_iterable is False:
                warnings.warn(
                    "`one_to_iterable` was provided as `False` and `embedder` as `None`"
                    " Therefore it will be set to `True` instead.",
                    UserWarning,
                    stacklevel=2,
                )
            self.one_to_iterable = True

        else:
            self.one_to_iterable = one_to_iterable
        if "max_num_emb" not in self.embedder_kwargs:
            self.embedder_kwargs["max_num_emb"] = max_num_emb
        elif max_num_emb is not None and "max_num_emb" in self.embedder_kwargs:
            warn_msg = (
                "Multiple declarations of parameter 'max_num_emb': "
                "provided both as function argument and inside 'embedder_kwargs'. "
                "Priority will be given to the value set inside 'embedder_kwargs'."
            )
            warnings.warn(warn_msg, UserWarning, stacklevel=2)
        self.child_structure_search = child_structure_search or child_structure_dfs
        self.reuse_embedding = check_bool(reuse_embedding, "reuse_embedding", safe=True)
        self.num_spin_reversal_transforms = check_int(
            num_spin_reversal_transforms, "num_spin_reversal_transforms", l_bound=0
        )
        self.sample_kwargs = sample_kwargs

    @retry_on_network_errors
    def _solve(self, qubo: QUBO) -> DimodSampleSetResult:
        sampler = get_singleton(DWaveSampler, solver=self.backend_id)
        sampler = SpinReversalTransformComposite(sampler)

        if qubo.size < 2:
            warnings.warn(
                f"QUBO size ({qubo.size}) is too small for this solver."
                " Consider using another solver."
                " Defaulting to AutoEmbeddingComposite instead.",
                stacklevel=2,
            )
            sampler = AutoEmbeddingComposite(sampler)
            response = sampler.sample(
                qubo.to_bqm(),
                num_reads=self.num_reads,
                num_spin_reversal_transforms=self.num_spin_reversal_transforms,
                **self.sample_kwargs,
            )
            return DimodSampleSetResult.from_result(qubo, response)

        if self.embeddings is None:
            source = dimod.to_networkx_graph(qubo.to_bqm())  # type: ignore[no-untyped-call]
        else:
            source = None

        parallel_sampler = ParallelEmbeddingComposite(
            sampler,
            embeddings=self.embeddings,
            source=source,
            embedder=self.embedder,
            embedder_kwargs=self.embedder_kwargs,
            one_to_iterable=self.one_to_iterable,
            child_structure_search=self.child_structure_search,
        )

        response = parallel_sampler.sample(
            qubo.to_bqm(),
            num_reads=self.num_reads,
            num_spin_reversal_transforms=self.num_spin_reversal_transforms,
            **self.sample_kwargs,
        )

        if self.reuse_embedding and self.embeddings is None:
            self.embeddings = parallel_sampler.embeddings
        response.info["embedding_context"] = {"embedding": parallel_sampler.embeddings}
        return DimodSampleSetResult.from_result(qubo, response)
